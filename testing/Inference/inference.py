import time
import numpy as np
import torch
from micromind import PhiNet
from mobilenetv1 import MobilenetV1
from mobilenetv2 import MobilenetV2
from torchinfo import summary

def measure_inference_time(model, input_shape):
    """
    Measures the inference time of the provided neural network model.

    Args:
        model: The neural network model to evaluate.
        input_shape: The shape of the input data expected by the model.

    Returns:
        tuple: A tuple containing the mean and standard deviation of the inference time
               measured in milliseconds.
    """
    device = next(model.parameters()).device  # Get the device where the model is located
    
    dummy_input = torch.randn(1, *input_shape, dtype=torch.float).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # GPU warm-up
    for _ in range(10):
        _ = model(dummy_input)

    # Measure inference time
    repetitions = 300
    timings = []
    with torch.no_grad():
        if device.type == 'cuda':
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for rep in range(repetitions):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings.append(curr_time)
        else:  # CPU
            for rep in range(repetitions):
                start_time = time.time()
                _ = model(dummy_input)
                end_time = time.time()
                elapsed_time = (end_time - start_time) * 1000.0  # Convert to milliseconds
                timings.append(elapsed_time)

    # Calculate mean and std
    mean_time = np.mean(timings)
    std_time = np.std(timings)

    return mean_time, std_time

def get_info(model, input_shape):
    """
    Get the number of multiply-accumulate operations and the number of parameters for the given model and input shape.

    Args:
        model: The neural network model.
        input_shape: The shape of the input data.

    Returns:
        The number of multiply-accumulate operations and parameters.
    """

    # Get the device where the model is located
    device = next(model.parameters()).device

    # Create a dummy input
    input_data = torch.zeros([1] + list(input_shape)).to(device)

    # Get the number of multiply-accumulate operations
    temp = summary(model, input_data=input_data, verbose = 0)

    return temp.total_mult_adds, temp.total_params

if __name__ == "__main__":

    input_shape = (3, 224, 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    phinet = PhiNet(input_shape = input_shape, alpha = 3, beta = 0.75, t_zero = 6, num_layers=7 ,include_top = True, num_classes = 1000).to(device)
    mean, std = measure_inference_time(phinet, input_shape)
    mac, parm_nr = get_info(phinet, input_shape)
    print(f"Average inference time of PhiNet: {mean:.3f} +/- {std:.3f} ms")
    print(f"Number of multiply-accumulate operations of PhiNet: {mac:,}")
    print(f"Number of parameters of PhiNet: {parm_nr:,}")

    print("")

    mobilenet = MobilenetV1(True, latent_layer_num=20).to(device)
    mean, std = measure_inference_time(mobilenet, input_shape)
    mac, parm_nr = get_info(mobilenet, input_shape)    
    print(f"Average inference time of MobileNetV1: {mean:.3f} +/- {std:.3f} ms")
    print(f"Number of multiply-accumulate operations of MobileNetV1: {mac:,}")
    print(f"Number of parameters of MobileNetV1: {parm_nr:,}")

    print("")

    mobilenetv2 = MobilenetV2(True, latent_layer_num=15).to(device)
    mean, std = measure_inference_time(mobilenetv2, input_shape)
    mac, parm_nr = get_info(mobilenetv2, input_shape)    
    print(f"Average inference time of MobileNetV2: {mean:.3f} +/- {std:.3f} ms")
    print(f"Number of multiply-accumulate operations of MobileNetV2: {mac:,}")
    print(f"Number of parameters of MobileNetV2: {parm_nr:,}")