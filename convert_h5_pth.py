import argparse
import numpy as np
import h5py
import torch
from torchvision.models import resnet50, inception_v3, densenet121

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

def convert_inception(pytorch_model, keras_weights):
    warnings = []
    
    # Define a mapping between Keras layer names and PyTorch module names
    layer_mapping = {
        'conv2d_1': 'Conv2d_1a_3x3.conv',
        'batch_normalization_1': 'Conv2d_1a_3x3.bn',
        'conv2d_2': 'Conv2d_2a_3x3.conv',
        'batch_normalization_2': 'Conv2d_2a_3x3.bn',
        'conv2d_3': 'Conv2d_2b_3x3.conv',
        'batch_normalization_3': 'Conv2d_2b_3x3.bn',
        'conv2d_4': 'Conv2d_3b_1x1.conv',
        'batch_normalization_4': 'Conv2d_3b_1x1.bn',
        'conv2d_5': 'Conv2d_4a_3x3.conv',
        'batch_normalization_5': 'Conv2d_4a_3x3.bn',
        # Add more mappings for other layers...
    }

    # Iterate through the Keras weights
    for keras_name, keras_weight in keras_weights.items():
        if 'kernel' in keras_name or 'gamma' in keras_name:
            # Extract the base name (remove _kernel or _gamma)
            base_name = keras_name.split('_kernel')[0].split('_gamma')[0]
            
            if base_name in layer_mapping:
                pytorch_name = layer_mapping[base_name]
                module = pytorch_model
                for part in pytorch_name.split('.'):
                    module = getattr(module, part)
                
                if isinstance(module, torch.nn.Conv2d) and 'kernel' in keras_name:
                    module.weight.data = torch.tensor(np.transpose(keras_weight, (3, 2, 0, 1)))
                elif isinstance(module, torch.nn.BatchNorm2d) and 'gamma' in keras_name:
                    module.weight.data = torch.tensor(keras_weight)
                else:
                    warnings.append(f"Unexpected layer type for {keras_name}")
            else:
                warnings.append(f"Couldn't find PyTorch equivalent for Keras layer: {base_name}")

    # Handle bias and other BatchNorm parameters
    for keras_name, keras_weight in keras_weights.items():
        if 'bias' in keras_name or 'beta' in keras_name or 'moving_mean' in keras_name or 'moving_variance' in keras_name:
            base_name = keras_name.split('_bias')[0].split('_beta')[0].split('_moving_mean')[0].split('_moving_variance')[0]
            
            if base_name in layer_mapping:
                pytorch_name = layer_mapping[base_name]
                module = pytorch_model
                for part in pytorch_name.split('.'):
                    module = getattr(module, part)
                
                if isinstance(module, torch.nn.Conv2d) and 'bias' in keras_name:
                    module.bias.data = torch.tensor(keras_weight)
                elif isinstance(module, torch.nn.BatchNorm2d):
                    if 'beta' in keras_name:
                        module.bias.data = torch.tensor(keras_weight)
                    elif 'moving_mean' in keras_name:
                        module.running_mean.data = torch.tensor(keras_weight)
                    elif 'moving_variance' in keras_name:
                        module.running_var.data = torch.tensor(keras_weight)
                else:
                    warnings.append(f"Unexpected layer type for {keras_name}")
            else:
                warnings.append(f"Couldn't find PyTorch equivalent for Keras layer: {base_name}")

    return warnings
def convert_resnet(pytorch_model, keras_weights):
    warnings = []

    # Convert initial layers
    success, warning = convert_conv(pytorch_model.conv1, keras_weights, 'conv1')
    if not success:
        warnings.append(warning)
    success, warning = convert_bn(pytorch_model.bn1, keras_weights, 'bn_conv1')
    if not success:
        warnings.append(warning)

    # Convert residual blocks
    for i, layer_name in enumerate(['conv2', 'conv3', 'conv4', 'conv5']):
        layer = getattr(pytorch_model, f'layer{i+1}')
        for j, block in enumerate(layer):
            for k in range(3):  # Each ResNet block has 3 conv layers
                conv_name = f'{layer_name}_{chr(97+j)}_branch2{chr(97+k)}'
                bn_name = f'bn{i+2}{chr(97+j)}_branch2{chr(97+k)}'
                success, warning = convert_conv(getattr(block, f'conv{k+1}'), keras_weights, conv_name)
                if not success:
                    warnings.append(warning)
                success, warning = convert_bn(getattr(block, f'bn{k+1}'), keras_weights, bn_name)
                if not success:
                    warnings.append(warning)

    return warnings
def convert_densenet(pytorch_model, keras_weights):
    warnings = []

    # Convert initial convolution and batch norm
    success, warning = convert_conv(pytorch_model.features.conv0, keras_weights, 'conv1')
    if not success:
        warnings.append(warning)
    success, warning = convert_bn(pytorch_model.features.norm0, keras_weights, 'conv1/bn')
    if not success:
        warnings.append(warning)

    # Convert dense blocks and transitions
    for i in range(4):
        block_name = f'conv{i+2}'
        block = getattr(pytorch_model.features, f'denseblock{i+1}')
        for j, module in enumerate(block):
            if isinstance(module, torch.nn.BatchNorm2d):
                success, warning = convert_bn(module, keras_weights, f'{block_name}_block{j//2+1}_1_bn')
            elif isinstance(module, torch.nn.Conv2d):
                success, warning = convert_conv(module, keras_weights, f'{block_name}_block{j//2+1}_1_conv')
            if not success:
                warnings.append(warning)
        
        if i < 3:  # No transition after the last dense block
            trans_name = f'pool{i+2}'
            trans = getattr(pytorch_model.features, f'transition{i+1}')
            success, warning = convert_bn(trans.norm, keras_weights, f'{trans_name}_bn')
            if not success:
                warnings.append(warning)
            success, warning = convert_conv(trans.conv, keras_weights, f'{trans_name}_conv')
            if not success:
                warnings.append(warning)

    # Convert final batch norm
    success, warning = convert_bn(pytorch_model.features.norm5, keras_weights, 'bn')
    if not success:
        warnings.append(warning)

    return warnings

def detect_model_type(keras_weights, input_path):
    print("Detecting model type...")
    print("Keys in keras_weights:", list(keras_weights.keys())[:10])  # Print first 10 keys for debugging
    print(input_path.lower())
    if any('res' in key.lower() for key in keras_weights.keys()) or 'resnet' in input_path.lower():
        return 'resnet'
    elif any('inception' in key.lower() for key in keras_weights.keys()) or 'inception' in input_path.lower():

        return 'inception'
    elif any('dense' in key.lower() for key in keras_weights.keys()) or 'densenet' in input_path.lower():
        return 'densenet'
    else:
        return 'unknown'

def main(input_path, output_path):
    keras_weights = load_keras_weights(input_path)
    model_type = detect_model_type(keras_weights, input_path)
    print(f"Detected model type: {model_type}")

    if model_type == 'resnet':
        pytorch_model = resnet50(pretrained=False)
        warnings = convert_resnet(pytorch_model, keras_weights)
    elif model_type == 'inception':
        pytorch_model = inception_v3(pretrained=False, aux_logits=False)
        warnings = convert_inception(pytorch_model, keras_weights)
    elif model_type == 'densenet':
        pytorch_model = densenet121(pretrained=False)
        warnings = convert_densenet(pytorch_model, keras_weights)
    else:
        raise ValueError(f"Unknown model type in {input_path}")

    # Save the converted model
    torch.save(pytorch_model.state_dict(), output_path)
    print(f"Converted {model_type} model saved to {output_path}")

    # Print warnings
    if warnings:
        print("\nWarnings during conversion:")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("\nNo warnings during conversion.")

    # Perform a forward pass and print the shape of the output
    pytorch_model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 299, 299) if model_type == 'inception' else torch.randn(1, 3, 224, 224)
        output = pytorch_model(dummy_input)
        print(f"\nShape of the generated feature map: {output.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Keras H5 model to PyTorch PTH")
    parser.add_argument("input_path", help="Path to input Keras H5 file")
    parser.add_argument("output_path", help="Path to output PyTorch PTH file")
    args = parser.parse_args()

    main(args.input_path, args.output_path)
