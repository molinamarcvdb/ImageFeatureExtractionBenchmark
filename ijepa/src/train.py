# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import copy
import logging
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import init_distributed, AllReduce
from src.utils.logging import CSVLogger, gpu_timer, grad_logger, AverageMeter
from src.utils.tensors import repeat_interleave_batch
from src.utils.moco import MocoLoss
from src.datasets.imagenet1k import make_imagenet1k
from src.datasets.imageDataset import make_contrastive_data

from src.helper import load_checkpoint, init_model, init_opt
from src.transforms import make_transforms

# --
log_timings = True
log_freq = 10
checkpoint_freq = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def setup_distributed():
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        world_size = 1
        rank = 0
        local_rank = 0

    torch.cuda.set_device(local_rank)

    # Always initialize the process group, even for single-GPU setups
    try:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    except RuntimeError as e:
        print(f"RuntimeError due to: {e}")
    return world_size, rank, local_rank


def main(args, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args["meta"]["use_bfloat16"]
    model_name = args["meta"]["model_name"]
    load_model = args["meta"]["load_checkpoint"] or resume_preempt
    r_file = args["meta"]["read_checkpoint"]
    copy_data = args["meta"]["copy_data"]
    pred_depth = args["meta"]["pred_depth"]
    pred_emb_dim = args["meta"]["pred_emb_dim"]
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- DATA
    use_gaussian_blur = args["data"]["use_gaussian_blur"]
    use_horizontal_flip = args["data"]["use_horizontal_flip"]
    use_vertical_flip = args["data"]["use_vertical_flip"]
    use_color_distortion = args["data"]["use_color_distortion"]
    color_jitter = args["data"]["color_jitter_strength"]
    # --
    batch_size = args["data"]["batch_size"]
    pin_mem = args["data"]["pin_mem"]
    num_workers = args["data"]["num_workers"]
    root_path = args["data"]["root_path"]
    image_folder = args["data"]["image_folder"]
    crop_size = args["data"]["crop_size"]
    crop_scale = args["data"]["crop_scale"]
    dataset_info = args["data"]["dataset_info"]
    unique_individual_id = args["data"]["unique_individual_id"]
    unique_image_id = args["data"]["unique_image_id"]
    split_ratio = args["data"]["split_ratio"]
    image_extension = args["data"]["image_extension"]
    seed = args["data"]["seed"]
    secondary_ids = args["data"]["secondary_ids"]
    # --

    # -- MASK
    allow_overlap = args["mask"][
        "allow_overlap"
    ]  # whether to allow overlap b/w context and target blocks
    patch_size = args["mask"]["patch_size"]  # patch-size for model training
    num_enc_masks = args["mask"]["num_enc_masks"]  # number of context blocks
    min_keep = args["mask"]["min_keep"]  # min number of patches in context block
    enc_mask_scale = args["mask"]["enc_mask_scale"]  # scale of context blocks
    num_pred_masks = args["mask"]["num_pred_masks"]  # number of target blocks
    pred_mask_scale = args["mask"]["pred_mask_scale"]  # scale of target blocks
    aspect_ratio = args["mask"]["aspect_ratio"]  # aspect ratio of target blocks
    # --

    # -- OPTIMIZATION
    ema = args["optimization"]["ema"]
    ipe_scale = args["optimization"]["ipe_scale"]  # scheduler scale factor (def: 1.0)
    wd = float(args["optimization"]["weight_decay"])
    final_wd = float(args["optimization"]["final_weight_decay"])
    num_epochs = args["optimization"]["epochs"]
    warmup = args["optimization"]["warmup"]
    start_lr = args["optimization"]["start_lr"]
    lr = args["optimization"]["lr"]
    final_lr = args["optimization"]["final_lr"]
    moco_temp = args["optimization"]["temp_moco"]
    lambda_loss = args["optimization"]["lambda_loss"]
    loss_config = args["optimization"]["loss_config"]

    # -- LOGGINIG
    folder = args["logging"]["folder"]
    tag = args["logging"]["write_tag"]

    dump = os.path.join(folder, "params-ijepa.yaml")
    with open(dump, "w") as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank, local_rank = setup_distributed()

    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"{tag}_r{rank}.csv")
    save_path = os.path.join(folder, f"{tag}" + "-ep{epoch}.pth.tar")
    latest_path = os.path.join(folder, f"{tag}-latest.pth.tar")
    load_path = None
    if load_model:
        load_path = r_file if r_file is not None else latest_path
        print(load_path)
    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%.5f", "mask-A"),
        ("%.5f", "mask-B"),
        ("%d", "time (ms)"),
    )

    # -- init model
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
    )
    target_encoder = copy.deepcopy(encoder)

    # -- make data transforms
    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep,
    )

    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        vertical_flip=use_vertical_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter,
    )

    # -- init data-loaders/samplers
    (
        _,
        _,
        unsupervised_loader,
        val_unsupervised_loader,
        unsupervised_sampler,
        val_unsupervised_loader,
    ) = make_contrastive_data(
        transform=transform,
        batch_size=batch_size,
        collator=mask_collator,
        pin_mem=pin_mem,
        training=True,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        root_path=root_path,
        image_folder=image_folder,
        copy_data=copy_data,
        drop_last=True,
        target_res=crop_size,
        folder=folder,
        dataset_info=dataset_info,
        unique_individual_id=unique_individual_id,
        unique_image_id=unique_image_id,
        split_ratio=split_ratio,
        image_extension=image_extension,
        seed=seed,
        secondary_ids=secondary_ids,
    )
    ipe = len(unsupervised_loader)

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16,
    )
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs * ipe_scale) + 1)
    )
    # -- instantaniate Moco loss
    moco_loss = MocoLoss(moco_temp)

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        print(load_path)
        (
            encoder,
            predictor,
            target_encoder,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    def save_checkpoint(epoch):
        save_dict = {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f"{epoch + 1}"))

    for param in encoder.parameters():
        param.requires_grad = True
    for param in predictor.parameters():
        param.requires_grad = True
    for param in target_encoder.parameters():
        param.requires_grad = False

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        loss_f1_meter = AverageMeter()
        loss_mocco_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):

            def load_imgs():
                # -- unsupervised imgs
                imgs = udata["data_pos"].to(device, non_blocking=True)
                imgs_natural = udata["data"].to(device, non_blocking=True)
                # if itr <=1:
                #    print('normal', imgs.shape, type(imgs), imgs.dtype)
                #    print('target', imgs_natural.shape, type(imgs_natural), imgs_natural.dtype)
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (imgs, imgs_natural, masks_1, masks_2)

            imgs, imgs_natural, masks_enc, masks_pred = load_imgs()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                def forward_target():
                    with torch.no_grad():
                        h0 = target_encoder(imgs_natural)
                        h = F.layer_norm(
                            h0, (h0.size(-1),)
                        )  # normalize over feature-dim
                        h0 = h0.mean(dim=1)
                        B = len(h)
                        # -- create targets (masked regions of h)
                        h = apply_masks(h, masks_pred)
                        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                        return h, h0

                def forward_context():
                    # z0 = encoder(imgs).mean(dim=1)
                    z_enc = encoder(imgs, masks_enc)
                    z = predictor(z_enc, masks_enc, masks_pred)
                    return z, z_enc

                def loss_fn(z, h, z0, h0):
                    loss = F.smooth_l1_loss(z, h)
                    loss_moco = moco_loss(z0, h0)
                    loss = (1 - lambda_loss) * loss + lambda_loss * loss_moco
                    loss = AllReduce.apply(loss)
                    return loss

                with torch.cuda.amp.autocast(
                    dtype=torch.bfloat16, enabled=use_bfloat16
                ):
                    h, h0 = forward_target()
                    z, z_enc = forward_context()

                    # Compute losses separately
                    f1_loss = F.smooth_l1_loss(z, h)
                    moco_loss_val = moco_loss(z_enc.mean(dim=1), h0)

                if loss_config == "linear_combination":
                    # Linear combination mode
                    total_loss = (
                        1 - lambda_loss
                    ) * f1_loss + lambda_loss * moco_loss_val
                    if use_bfloat16:
                        scaler.scale(total_loss).backward()
                    else:
                        total_loss.backward()

                elif loss_config == "mixed_backward":
                    # Mixed backward mode
                    predictor_params = [
                        p for p in predictor.parameters() if p.requires_grad
                    ]
                    if use_bfloat16:
                        scaler.scale(f1_loss).backward(
                            inputs=predictor_params, retain_graph=True
                        )
                    else:
                        f1_loss.backward(inputs=predictor_params, retain_graph=True)

                    encoder_loss = (
                        1 - lambda_loss
                    ) * f1_loss + lambda_loss * moco_loss_val
                    encoder_params = [
                        p for p in encoder.parameters() if p.requires_grad
                    ]
                    if use_bfloat16:
                        scaler.scale(encoder_loss).backward(inputs=encoder_params)
                    else:
                        encoder_loss.backward(inputs=encoder_params)
                else:
                    raise ValueError(f"Unknown loss_config: {loss_config}")

                # Optimizer step
                if use_bfloat16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(
                        encoder.parameters(), target_encoder.parameters()
                    ):
                        param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

                total_loss = (
                    1 - lambda_loss
                ) * f1_loss + lambda_loss * moco_loss_val  # for logging purposes
                return (
                    float(total_loss),
                    float(f1_loss),
                    float(moco_loss_val),
                    _new_lr,
                    _new_wd,
                    grad_stats,
                )

            (
                loss,
                f1_loss,
                moco_loss_val,
                _new_lr,
                _new_wd,
                grad_stats,
            ), etime = gpu_timer(train_step)

            loss_meter.update(loss)
            loss_f1_meter.update(f1_loss)
            loss_mocco_meter.update(moco_loss_val)
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                csv_logger.log(
                    epoch + 1,
                    itr,
                    loss,
                    f1_loss,
                    moco_loss_val,
                    maskA_meter.val,
                    maskB_meter.val,
                    etime,
                )
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        "[%d, %5d] loss: %.3f "
                        "loss f1: %.3f "
                        "loss moco: %.3f "
                        "masks: %.1f %.1f "
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2e] "
                        "(%.1f ms)"
                        % (
                            epoch + 1,
                            itr,
                            loss_meter.avg,
                            loss_f1_meter.avg,
                            loss_mocco_meter.avg,
                            maskA_meter.avg,
                            maskB_meter.avg,
                            _new_wd,
                            _new_lr,
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            time_meter.avg,
                        )
                    )

                # if grad_stats is not None:
                #    logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                #                % (epoch + 1, itr,
                #                   grad_stats.first_layer,
                #                   grad_stats.last_layer,
                #                   grad_stats.min,
                #                   grad_stats.max))

            log_stats()
            assert not np.isnan(loss), "loss is nan"

        # -- Save Checkpoint after every epoch
        logger.info("avg. loss %.3f" % loss_meter.avg)
        save_checkpoint(epoch + 1)


if __name__ == "__main__":
    main()
