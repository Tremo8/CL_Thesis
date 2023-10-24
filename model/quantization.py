import torch
import torch.nn as nn
import torch.ao.quantization as nnq

from micromind.networks.phinet import PhiNetConvBlock, SeparableConv2d, DepthwiseConv2d

def phinet_fuse_modules(model):
    """
    Fuse the convolutional layers of a PhiNet model using PyTorch's quantization-aware training (QAT) API.

    Args:
        model (nn.Module): The PhiNet model to be quantized.

    Returns:
        None
    """
    for basic_block_name, basic_block in model.named_children():
        if isinstance(basic_block, SeparableConv2d):
            torch.ao.quantization.fuse_modules(basic_block, [["_layers.1", "_layers.2"]], inplace=True)
        if isinstance(basic_block, PhiNetConvBlock) and len(basic_block._layers) == 6:
            torch.ao.quantization.fuse_modules(basic_block, [["_layers.1", "_layers.2"], ["_layers.4", "_layers.5"]], inplace=True)
        elif isinstance(basic_block, PhiNetConvBlock):
            torch.ao.quantization.fuse_modules(basic_block, [["_layers.0", "_layers.1"], ["_layers.4", "_layers.5"], ["_layers.8", "_layers.9"]], inplace=True)

def remove_depthwise(model):
    def convert_to_conv2d(depthwise_conv2d):
        in_channels = depthwise_conv2d.in_channels
        depth_multiplier = depthwise_conv2d.out_channels // in_channels
        kernel_size = depthwise_conv2d.kernel_size
        stride = depthwise_conv2d.stride
        padding = depthwise_conv2d.padding
        dilation = depthwise_conv2d.dilation
        bias = depthwise_conv2d.bias is not None
        padding_mode = depthwise_conv2d.padding_mode

        # Create an equivalent nn.Conv2d layer
        conv2d_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * depth_multiplier,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # Set groups to in_channels for depthwise convolution
            bias=bias,
            padding_mode=padding_mode,
        )

        
        # Assuming you want to copy the weights from conv1 to conv2
        with torch.no_grad():
            conv2d_layer.weight.copy_(depthwise_conv2d.weight)
           

        # If bias was not used in the original depthwise_conv2d, set bias to None in conv2d_layer
        if not bias:
            conv2d_layer.bias = None
        else:
             conv2d_layer.bias.copy_(depthwise_conv2d.bias)

        return conv2d_layer

    for name, module in model._layers.named_children():
        if isinstance(module, PhiNetConvBlock):
            for i, layer in enumerate(module._layers.children()):
                if isinstance(layer, DepthwiseConv2d):
                    module._layers[i] = convert_to_conv2d(layer)

def calibrate_model(model, loader, device=torch.device("cpu")):
    """
    Calibrates a PyTorch model for quantization by running it on a calibration dataset.

    Args:
        model (nn.Module): The PyTorch model to calibrate.
        loader (DataLoader): The DataLoader containing the calibration dataset.
        device (torch.device, optional): The device to run the model on. Defaults to CPU.

    Returns:
        None
    """
    model.to(device)
    model.eval()

    for inputs, labels, _ in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        _ = model(inputs)


def quantize_model(model, fuse_function, calibration_function, calibration_data, quantized_output=False):
    """
    Quantizes a PyTorch model using static quantization.

    Args:
        model (nn.Module): The PyTorch model to be quantized.
        fuse_function (function): The function used to fuse Conv and BN modules in the model.
        calibration_function (function): The function used to calibrate the quantized model.
        calibration_data (torch.utils.data.DataLoader): The calibration dataset.
        quantized_output (bool, optional): Whether to return the output in int8 or fp32. Defaults to False.

    Returns:
        nn.Module: The quantized PyTorch model.
    """
    # Set the model in eval mode
    model.eval()

    # Fuse Conv, BN modules in the model
    fuse_function(model)

    # Insert stubs for quantization
    if quantized_output:
        model = nn.Sequential(nnq.QuantStub(), model)
    else:
        model = nn.Sequential(nnq.QuantStub(), model, nnq.DeQuantStub())
 
    model.qconfig = nnq.get_default_qconfig("x86")

    # Prepare the model for static quantization. 
    nnq.prepare(model, inplace=True)

    calibration_function(model, calibration_data)

    # Convert the observed model to a quantized model.
    model = nnq.convert(model, inplace=True)

    return model
