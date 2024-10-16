import torch
import math
import sys
import os
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPConfig

# Get the parent directory of the current script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Append the parent directory to sys.path
sys.path.append(parent_dir)


from ijepa.src.helper import init_model
from ijepa.src.models.vision_transformer import vit_huge


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.proj_drop_prob
                )
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        grid_size=None,
        grid_depth=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, return_attention=False, mask=None):
        y, attn = self.attn(self.norm1(x), mask=mask)
        if return_attention:
            return attn
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, use_sdpa=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, int(dim * 2), bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_sdpa = use_sdpa

    def forward(self, q, x):
        B, n, C = q.shape
        q = (
            self.q(q)
            .reshape(B, n, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        B, N, C = x.shape
        kv = (
            self.kv(x)
            .reshape(B, N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]  # (batch_size, num_heads, seq_len, feature_dim_per_head)

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                q = F.scaled_dot_product_attention(q, k, v)
        else:
            xattn = (q @ k.transpose(-2, -1)) * self.scale
            xattn = xattn.softmax(dim=-1)  # (batch_size, num_heads, query_len, seq_len)
            q = xattn @ v

        q = q.transpose(1, 2).reshape(B, n, C)
        q = self.proj(q)

        return q


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.xattn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer
        )

    def forward(self, q, x):
        y = self.xattn(q, self.norm1(x))
        q = q + y
        q = q + self.mlp(self.norm2(q))
        return q


class AttentivePooler(nn.Module):
    def __init__(
        self,
        num_queries=1,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True,
    ):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        self.complete_block = complete_block
        if complete_block:
            self.cross_attention_block = CrossAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
            )
        else:
            self.cross_attention_block = CrossAttention(
                dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias
            )
        self.blocks = None
        if depth > 1:
            self.blocks = nn.ModuleList(
                [
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=None,
                        norm_layer=norm_layer,
                    )
                    for i in range(depth - 1)
                ]
            )
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        if self.complete_block:
            rescale(self.cross_attention_block.xattn.proj.weight.data, 1)
            rescale(self.cross_attention_block.mlp.fc2.weight.data, 1)
        else:
            rescale(self.cross_attention_block.proj.weight.data, 1)
        if self.blocks is not None:
            for layer_id, layer in enumerate(self.blocks, 1):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        q = self.query_tokens.repeat(len(x), 1, 1)
        q = self.cross_attention_block(q, x)
        if self.blocks is not None:
            for blk in self.blocks:
                q = blk(q)
        return q


class IJEPAEncoderWithProbe(nn.Module):
    def __init__(
        self,
        device="cuda",
        patch_size=14,
        crop_size=224,
        pred_depth=12,
        pred_emb_dim=384,
        model_name="vit_huge",
        output_dim=None,
        use_attentive_pooling=False,
        num_queries=1,
        num_heads=16,
        mlp_ratio=4.0,
        pooler_depth=1,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True,
    ):
        super().__init__()
        self.device = device
        self.use_attentive_pooling = use_attentive_pooling

        # Initialize the model
        self.encoder, _ = init_model(
            device=self.device,
            patch_size=patch_size,
            crop_size=crop_size,
            pred_depth=pred_depth,
            pred_emb_dim=pred_emb_dim,
            model_name=model_name,
        )

        # Get the output dimension of the LayerNorm
        self.encoder_output_dim = self.encoder.norm.normalized_shape[0]

        # Add attentive pooling if specified
        if self.use_attentive_pooling:
            self.attentive_pooler = AttentivePooler(
                num_queries=num_queries,
                embed_dim=self.encoder_output_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                depth=pooler_depth,
                norm_layer=nn.LayerNorm,
                init_std=init_std,
                qkv_bias=qkv_bias,
                complete_block=complete_block,
            )

        # Add output layer if output_dim is specified
        self.output_layer = (
            nn.Linear(self.encoder_output_dim, output_dim) if output_dim else None
        )

        self.to(self.device)

    def load_encoder(self, load_path):
        ckpt = torch.load(load_path, map_location=torch.device("cpu"))
        pretrained_dict = ckpt["encoder"]

        # Loading encoder
        model_dict = self.encoder.state_dict()
        for k, v in pretrained_dict.items():
            if k.startswith("module."):
                k = k[len("module.") :]
            if k in model_dict:
                model_dict[k].copy_(v)

        self.encoder.load_state_dict(model_dict)

    def forward(self, x):
        x = self.encoder(x.to(self.device))

        if self.use_attentive_pooling:
            x = self.attentive_pooler(x)
            x = x.mean(dim=1)  # Average over queries if more than one
        else:
            x = x.mean(dim=1)  # Global average pooling

        if self.output_layer:
            x = self.output_layer(x)

        return x


class ContrastiveNetwork(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        feature_dim: int = 768,
        attentive_probing: bool = False,
        device: str = "cuda",
    ):
        super(ContrastiveNetwork, self).__init__()

        self.attentive_probing = attentive_probing
        self.feature_dim = feature_dim
        self.device = device

        self.init_backbone(backbone_name)
        self.append_head()

        # print(self.backbone)

    def init_backbone(self, backbone_name):
        self.backbone, self.backbone_type, self.processor = initialize_model(
            backbone_name
        )

    def append_head(self):

        self.get_backbone_out_dim()

        print(f"Backbone output dimension: {self.backbone_output_dim}")

        if self.attentive_probing:
            self.output_layer = AttentivePooler(
                embed_dim=self.backbone_output_dim,
                num_queries=1,
                num_heads=8,
                mlp_ratio=4.0,
                depth=1,
            )
            self.final_projection = nn.Linear(
                self.backbone_output_dim, self.feature_dim
            )
        else:
            self.output_layer = nn.Linear(self.backbone_output_dim, self.feature_dim)

    def get_backbone_out_dim(self):
        if hasattr(self.backbone, "fc") and hasattr(self.backbone.fc, "in_features"):
            self.backbone_output_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            # self.backbone.avgpool = nn.Identity()
            if hasattr(self.backbone, "AuxLogits"):
                self.backbone.AuxLogits = None

        elif hasattr(self.backbone, "classifier") and hasattr(
            self.backbone.classifier, "in_features"
        ):
            self.backbone_output_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()

        elif (
            hasattr(self.backbone, "config")
            and hasattr(self.backbone.config, "vision_config")
            and hasattr(self.backbone.config.vision_config, "hidden_size")
        ):
            self.backbone_output_dim = self.backbone.config.vision_config.hidden_size

        elif hasattr(self.backbone, "encoder_output_dim"):
            self.backbone_output_dim = self.backbone.encoder_output_dim

        elif hasattr(self.backbone, "config") and hasattr(
            self.backbone.config, "hidden_size"
        ):
            self.backbone_output_dim = self.backbone.config.hidden_size

        else:
            raise "No ouput dim feture found"

    def _get_features(self, x):
        """Helper method to flexibly extract features from various model outputs"""
        if hasattr(x, "logits"):
            return x.logits
        elif isinstance(x, dict) and "logits" in x:
            return x["logits"]
        elif isinstance(x, (tuple, list)):
            return x[0]  # Assume the first element contains the main output
        elif hasattr(x, "last_hidden_state"):
            return x.last_hidden_state
        else:
            return x  # Assume x is already the feature tensor we want

    def forward(self, x):

        if hasattr(self.backbone, "config") and isinstance(
            self.backbone.config, CLIPConfig
        ):

            features = self._get_features(self.backbone.vision_model(x.to(self.device)))

        else:
            features = self._get_features(self.backbone(x.to(self.device)))
        print(features.shape)
        if self.attentive_probing:
            if len(features.shape) == 2:
                features = features.unsqueeze(1)

            features = self.output_layer(
                features
            )  # This should output [batch_size, num_queries, backbone_output_dim]
            features = features.squeeze(1)  # Remove the num_queries dimension
            features = self.final_projection(features)
        else:
            if len(features.shape) == 3:
                features = features.mean(dim=1)  # Global average pooling
            features = self.output_layer(features)
        return features


# Usage example:
if __name__ == "__main__":
    # Initialize the model
    # model = IJEPAEncoder(
    #    output_dim=256,
    #    use_attentive_pooling=True,
    #    num_queries=1,
    #    pooler_depth=2
    # )  # Change parameters as needed

    ## Load pre-trained weights
    # model.load_encoder('/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/syntheva/pretrained/jepa/IN22K-vit.h.14-900e.pth.tar')
    for model_name in [
        "inception",
        "resnet50",
        "resnet18",
        "clip",
        "densenet121",
        "rad_clip",
        "rad_dino",
        "dino",
        "rad_inception",
        "rad_resnet50",
        "rad_densenet",
        "ijepa",
    ]:

        print(f"Processing model {model_name}")
        model = ContrastiveNetwork(model_name, attentive_probing=True).eval()
        model.to("cuda")
        # Test the model
        img = torch.rand([1, 3, 224, 224])
        with torch.no_grad():
            feature = model(img)

            print(feature.shape)

        model = ContrastiveNetwork(model_name, attentive_probing=False).eval()
        model.to("cuda")
        # Test the model
        img = torch.rand([1, 3, 224, 224])
        with torch.no_grad():
            feature = model(img)
            print(feature.shape)

            print()
