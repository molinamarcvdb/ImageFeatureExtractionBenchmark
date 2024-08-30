"""
Script to transform the weights of the ResNetmodel from Keras to Pytorch.

Script based on https://github.com/BMEII-AI/RadImageNet/issues/3#issuecomment-1232417600
and https://discuss.pytorch.org/t/transferring-weights-from-keras-to-pytorch/9889
"""
import argparse

import numpy as np
import tensorflow as tf
import torch

import argparse
import numpy as np
import tensorflow as tf
import torch
from torchvision.models import densenet121

torch.set_printoptions(precision=10)
#from radimagenet_models.models.resnet import ResNet50

    


def convert_conv(pytorch_conv, tf_conv):
    pytorch_conv.weight.data = torch.tensor(np.transpose(tf_conv.kernel.numpy(), (3, 2, 0, 1)))
    pytorch_conv.bias.data = torch.tensor(tf_conv.bias.numpy())
    return pytorch_conv


def convert_bn(pytorch_bn, tf_bn):
    pytorch_bn.weight.data = torch.tensor(tf_bn.gamma.numpy())
    pytorch_bn.bias.data = torch.tensor(tf_bn.beta.numpy())
    pytorch_bn.running_mean.data = torch.tensor(tf_bn.moving_mean.numpy())
    pytorch_bn.running_var.data = torch.tensor(tf_bn.moving_variance.numpy())
    return pytorch_bn


def convert_stack(pytorch_stack, keras_model, stack_name, num_blocks):
    layers_list = []
    for layer in keras_model.layers:
        if stack_name in layer.get_config()["name"]:
            layers_list.append(layer)

    for i in range(1, num_blocks + 1):
        pytorch_block = pytorch_stack[i - 1]
        for layer in layers_list:
            if f"{stack_name}_block{str(i)}_0_conv" in layer.get_config()["name"]:
                pytorch_block.downsample[0] = convert_conv(pytorch_block.downsample[0], layer)
            elif f"{stack_name}_block{str(i)}_0_bn" in layer.get_config()["name"]:
                pytorch_block.downsample[1] = convert_bn(pytorch_block.downsample[1], layer)
            elif f"{stack_name}_block{str(i)}_1_conv" in layer.get_config()["name"]:
                pytorch_block.conv1 = convert_conv(pytorch_block.conv1, layer)
            elif f"{stack_name}_block{str(i)}_1_bn" in layer.get_config()["name"]:
                pytorch_block.bn1 = convert_bn(pytorch_block.bn1, layer)
            elif f"{stack_name}_block{str(i)}_2_conv" in layer.get_config()["name"]:
                pytorch_block.conv2 = convert_conv(pytorch_block.conv2, layer)
            elif f"{stack_name}_block{str(i)}_2_bn" in layer.get_config()["name"]:
                pytorch_block.bn2 = convert_bn(pytorch_block.bn2, layer)
            elif f"{stack_name}_block{str(i)}_3_conv" in layer.get_config()["name"]:
                pytorch_block.conv3 = convert_conv(pytorch_block.conv3, layer)
            elif f"{stack_name}_block{str(i)}_3_bn" in layer.get_config()["name"]:
                pytorch_block.bn3 = convert_bn(pytorch_block.bn3, layer)

        pytorch_stack[i - 1] = pytorch_block
    return pytorch_stack


def main_resnet50(args):
    pytorch_model = ResNet50()
    keras_model = tf.keras.models.load_model(args.input_path)

    # Convert weights
    pytorch_model.conv1 = convert_conv(pytorch_model.conv1, keras_model.get_layer("conv1_conv"))
    pytorch_model.bn1 = convert_bn(pytorch_model.bn1, keras_model.get_layer("conv1_bn"))

    pytorch_model.layer1 = convert_stack(pytorch_model.layer1, keras_model, "conv2", num_blocks=3)
    pytorch_model.layer2 = convert_stack(pytorch_model.layer2, keras_model, "conv3", num_blocks=4)
    pytorch_model.layer3 = convert_stack(pytorch_model.layer3, keras_model, "conv4", num_blocks=6)
    pytorch_model.layer4 = convert_stack(pytorch_model.layer4, keras_model, "conv5", num_blocks=3)

    # Test converted model
    x = np.random.rand(1, 224, 224, 3)
    x_pt = torch.from_numpy(np.transpose(x, (0, 3, 1, 2))).float()

    pytorch_model.eval()
    with torch.no_grad():
        outputs_pt = pytorch_model(x_pt)
        outputs_pt = np.transpose(outputs_pt.numpy(), (0, 2, 3, 1))

    x_tf = tf.convert_to_tensor(x)
    outputs_tf = keras_model(x_tf, training=False)
    outputs_tf = outputs_tf.numpy()

    print(f"Are the outputs all close (absolute tolerance = 1e-04)? {np.allclose(outputs_tf, outputs_pt, atol=1e-04)}")
    print("Pytorch output")
    print(outputs_pt[0, :30, 0, 0])
    print("Tensoflow Keras output")
    print(outputs_tf[0, :30, 0, 0])

    # Saving model
    torch.save(pytorch_model.state_dict(), args.output_path)
import numpy as np
import h5py
import torch
from torchvision.models import densenet121

def load_keras_weights(h5_path):
    with h5py.File(h5_path, 'r') as f:
        return {name: np.array(val) for name, val in f['model_weights'].items()}

def find_matching_key(keras_weights, patterns):
    for pattern in patterns:
        for key in keras_weights.keys():
            if pattern in key:
                return key
    return None

def convert_conv(pytorch_conv, keras_weights, layer_name):
    possible_patterns = [f"{layer_name}/kernel:", f"{layer_name}_kernel:", f"{layer_name}/weights:", f"{layer_name}_weights:", f"{layer_name}/"]
    weight_key = find_matching_key(keras_weights, possible_patterns)
    if weight_key is None:
        return False, f"Couldn't find weight for layer: {layer_name}"
    
    pytorch_conv.weight.data = torch.tensor(np.transpose(keras_weights[weight_key], (3, 2, 0, 1)))
    
    bias_patterns = [f"{layer_name}/bias:", f"{layer_name}_bias:", f"{layer_name}/"]
    bias_key = find_matching_key(keras_weights, bias_patterns)
    if pytorch_conv.bias is not None and bias_key:
        pytorch_conv.bias.data = torch.tensor(keras_weights[bias_key])
    
    return True, ""

def convert_bn(pytorch_bn, keras_weights, layer_name):
    param_names = ['gamma', 'beta', 'moving_mean', 'moving_variance']
    pytorch_names = ['weight', 'bias', 'running_mean', 'running_var']
    
    for k_name, p_name in zip(param_names, pytorch_names):
        possible_patterns = [f"{layer_name}/{k_name}:", f"{layer_name}_{k_name}:", f"{layer_name}/"]
        key = find_matching_key(keras_weights, possible_patterns)
        if key:
            getattr(pytorch_bn, p_name).data = torch.tensor(keras_weights[key])
        else:
            return False, f"Couldn't find {k_name} for layer: {layer_name}"
    
    return True, ""

def convert_dense_block(pytorch_block, keras_weights, block_name):
    warnings = []
    for i, layer in enumerate(pytorch_block):
        success, warning = False, ""
        if isinstance(layer, torch.nn.BatchNorm2d):
            success, warning = convert_bn(layer, keras_weights, f'{block_name}_block{i//2+1}_1_bn')
        elif isinstance(layer, torch.nn.Conv2d):
            success, warning = convert_conv(layer, keras_weights, f'{block_name}_block{i//2+1}_1_conv')
        if not success:
            warnings.append(warning)
    return warnings

def convert_transition(pytorch_trans, keras_weights, trans_name):
    warnings = []
    success, warning = convert_bn(pytorch_trans.norm, keras_weights, f'{trans_name}_bn')
    if not success:
        warnings.append(warning)
    success, warning = convert_conv(pytorch_trans.conv, keras_weights, f'{trans_name}_conv')
    if not success:
        warnings.append(warning)
    return warnings

class DenseNet121NoTop(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.features = original_model.features

    def forward(self, x):
        features = self.features(x)
        out = torch.nn.functional.relu(features, inplace=True)
        return out

def main():
    input_path = "/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/syntheva/pretrained/RadImageNet_models/RadImageNet-DenseNet121_notop.h5"
    output_path = "/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/syntheva/pretrained/RadImageNet_models/RadImageNet-DenseNet121_notop.pth"

    original_model = densenet121(pretrained=False)
    pytorch_model = DenseNet121NoTop(original_model)
    keras_weights = load_keras_weights(input_path)

    all_warnings = []

    # Convert initial convolution and batch norm
    success, warning = convert_conv(pytorch_model.features.conv0, keras_weights, 'conv1')
    if not success:
        all_warnings.append(warning)
    success, warning = convert_bn(pytorch_model.features.norm0, keras_weights, 'conv1')
    if not success:
        all_warnings.append(warning)

    # Convert dense blocks and transitions
    for i in range(4):
        block_name = f'conv{i+2}'
        warnings = convert_dense_block(getattr(pytorch_model.features, f'denseblock{i+1}'), keras_weights, block_name)
        all_warnings.extend(warnings)
        if i < 3:  # No transition after the last dense block
            trans_name = f'pool{i+2}'
            warnings = convert_transition(getattr(pytorch_model.features, f'transition{i+1}'), keras_weights, trans_name)
            all_warnings.extend(warnings)

    # Convert final batch norm
    success, warning = convert_bn(pytorch_model.features.norm5, keras_weights, 'bn')
    if not success:
        all_warnings.append(warning)

    # Save the converted model
    torch.save(pytorch_model.state_dict(), output_path)
    print(f"Converted model saved to {output_path}")

    # Print warnings
    if all_warnings:
        print("\nWarnings during conversion:")
        for warning in all_warnings:
            print(f"- {warning}")
    else:
        print("\nNo warnings during conversion.")

    # Perform a forward pass and print the shape of the output
    pytorch_model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 color channels, 224x224 image
        output = pytorch_model(dummy_input)
        print(f"\nShape of the generated feature map: {output.shape}")

if __name__ == "__main__":
    main()
