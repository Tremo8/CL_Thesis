import time

import numpy as np

import torch
from torchinfo import summary

def measure_inference_time(model, input_shape):
    """
    Measure the inference time of the given model.

    Args:
        input_shape: The shape of the input data.
        model: The neural network model.
        device: The device to perform the computations on (e.g., CPU or GPU).

    Returns:
        The mean and standard deviation of the inference time.
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
    mean_syn = np.mean(timings)
    std_syn = np.std(timings)

    return mean_syn, std_syn

def get_MAC(model, input_shape):
    """
    Get the number of multiply-accumulate operations for the given model and input shape.

    Args:
        model: The neural network model.
        input_shape: The shape of the input data.

    Returns:
        The number of multiply-accumulate operations.
    """

    # Get the device where the model is located
    device = next(model.parameters()).device

    # Create a dummy input
    input_data = torch.zeros([1] + list(input_shape)).to(device)

    # Get the number of multiply-accumulate operations
    temp = summary(model, input_data=input_data, verbose=0)

    return temp.total_mult_adds