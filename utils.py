import time

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset

from torchinfo import summary

import numpy as np

import matplotlib.pyplot as plt

from avalanche.evaluation.metrics import MAC

def train(model, optimizer, criterion, train_loader, device):
    """
    Train the model using the provided optimizer and criterion.

    Args:
        model: The neural network model to train.
        optimizer: The optimizer used to update the model's parameters.
        criterion: The loss function used to calculate the loss.
        train_loader: The data loader providing the training data.
        device: The device to perform the computations on (e.g., CPU or GPU).

    Returns:
        accuracy: The training accuracy as a percentage.
        average_loss: The average training loss.
    """
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        prob = F.softmax(outputs, dim=1)
        _, predicted = prob.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    average_loss = train_loss / len(train_loader)

    return accuracy, average_loss

def test(model, criterion, test_loader, device):
    """
    Evaluate the model using the provided criterion on the test data.

    Args:
        model: The neural network model to evaluate.
        criterion: The loss function used to calculate the loss.
        test_loader: The data loader providing the test data.
        device: The device to perform the computations on (e.g., CPU or GPU).

    Returns:
        accuracy: The test accuracy as a percentage.
        average_loss: The average test loss.
    """

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            prob = F.softmax(outputs, dim=1)
            _, predicted = prob.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    average_loss = test_loss / len(test_loader)

    return accuracy, average_loss

def concat_experience(data_stream):
    """
    Concatenate multiple datasets into a single dataset.

    Args:
        data_stream: A list of data streams, where each data stream contains a dataset.

    Returns:
        concat_data: The concatenated dataset.
    """

    concat_data = data_stream[0].dataset

    for i in range(1, len(data_stream)):
        concat_data = ConcatDataset([concat_data, data_stream[i].dataset])

    return concat_data

def get_MAC(model, input_shape):
    """
    Get the number of multiply-accumulate operations for the given model and input shape.

    Args:
        model: The neural network model.
        input_shape: The shape of the input data.

    Returns:
        The number of multiply-accumulate operations.
    """
    
    device = next(model.parameters()).device  # Get the device where the model is located

    input_data = torch.zeros([1] + list(input_shape)).to(device)  # Move input_data to the same device as the model

    temp = summary(model, input_data=input_data, verbose=0)
    return temp.total_mult_adds

def plot_individual_task_accuracy(task_acc):
    """
    Plot the accuracy of each individual task.

    Args:
        task_acc (dict): A dictionary containing the accuracy of each task.
    """
    num_tasks = len(task_acc)
    num_rows = (num_tasks + 1) // 2  # Number of rows in the subplots grid
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))
    axes = axes.flatten()
    plot_num = 0

    for idx, (key, ax) in enumerate(zip(task_acc.keys(), axes)):
        ax.plot(task_acc[key], label=f"Task {key}", marker='.')
        ax.grid(True)
        ax.set_xlabel('Task')
        ax.set_ylabel('Accuracy')
        ax.set_yticks(np.arange(0, 101, 5))
        ax.set_ylim(-1, 101)  # Set y-axis limits
        ax.set_title(f"Task {key}", loc='center')
        ax.set_xticks(list(task_acc.keys()))
        plot_num += 1
    # Remove any unused subplots if num_tasks is odd
    for i in range(num_tasks, num_rows * 2):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.show()

def plot_average_accuracy(task_acc):
    """
    Plot the average accuracy of all tasks.

    Args:
        task_acc (dict): A dictionary containing the accuracy of each task.
    """
    num_tasks = len(task_acc)
    averages = [sum(values) / num_tasks for values in zip(*task_acc.values())]
    plt.plot(averages, marker='.')
    plt.grid(True)
    plt.xlabel('Task')
    plt.ylabel('Accuracy')
    plt.yticks(np.arange(0, 101, 5))
    plt.ylim(-1, 101) 
    plt.xticks(list(task_acc.keys()))
    plt.title('Average Accuracy')
    plt.show()

def plot_encountered_tasks_accuracy(task_acc):
    """
    Plot the average accuracy of only encountered tasks.

    Args:
        task_acc (dict): A dictionary containing the accuracy of each task.
    """
    encountered_averages = []
    for last_key in range(len(next(iter(task_acc.values())))):
        total_last_element = 0
        num_keys = 0

        for key, value in task_acc.items():
            if int(key) <= int(last_key):
                last_element = value[int(last_key)]
                total_last_element += last_element
                num_keys += 1

        avg = total_last_element / num_keys
        encountered_averages.append(avg)

    plt.plot(encountered_averages, marker='.')
    plt.grid(True)
    plt.xlabel('Task')
    plt.ylabel('Accuracy')
    plt.yticks(np.arange(0, 101, 5))
    plt.ylim(-1, 101) 
    plt.xticks(list(task_acc.keys()))
    plt.title('Average Accuracy of Encountered Tasks')
    plt.show()

def plot_task_accuracy(task_acc, plot_task_acc=True, plot_avg_acc=True, plot_encountered_avg=True):
    """
    Plot the accuracy of each task, the average accuracy of all tasks, and the average accuracy of only encountered tasks.

    Args:
        task_acc (dict): A dictionary containing the accuracy of each task.
        plot_task_acc (bool): Whether to plot individual task accuracies. Default is True.
        plot_avg_acc (bool): Whether to plot average accuracy of all tasks. Default is True.
        plot_encountered_avg (bool): Whether to plot average accuracy of only encountered tasks. Default is True.
    """
    if plot_task_acc:
        plot_individual_task_accuracy(task_acc)
    if plot_avg_acc:
        plot_average_accuracy(task_acc)
    if plot_encountered_avg:
        plot_encountered_tasks_accuracy(task_acc)

def plot_accs(accs):
    
    num_tasks = len(next(iter(accs.values())))
    num_rows = (num_tasks + 1) // 2  # Number of rows in the subplots grid
    fig, axes = plt.subplots(num_rows, 2, figsize=(10, 5 * num_rows))

    # Flatten the axes array if there is only one row
    axes = axes.flatten() if num_rows == 1 else axes.ravel()

    for task_key in range(num_tasks):
        ax = axes[task_key]

        for strategy, strategy_dic in accs.items():
            ax.plot(strategy_dic[task_key], label=strategy, marker='.')

        ax.grid(True)
        ax.set_xlabel('Task')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 100)  # Set y-axis limits
        ax.set_title(f"Task {task_key}", loc='center')
        ax.legend()
        ax.set_xticks(list(strategy_dic.keys()))

    # Calculate and plot average accuracy of all tasks
    ax = axes[-1]
    for strategy, strategy_dic in accs.items():
        averages = [sum(values) / len(strategy_dic) for values in zip(*strategy_dic.values())]
        ax.plot(averages, label=strategy, marker='.')

    ax.grid(True)
    ax.set_xlabel('Task')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 100)  # Set y-axis limits
    ax.set_title('Average Accuracy', loc='center')
    ax.legend()
    ax.set_xticks(list(strategy_dic.keys()))

    plt.tight_layout()
    plt.show()


def measure_inference_time(input_shape, model, device):
    """
    Measure the inference time of the given model.

    Args:
        input_shape: The shape of the input data.
        model: The neural network model.
        device: The device to perform the computations on (e.g., CPU or GPU).

    Returns:
        The average inference time in milliseconds.
    """

    dummy_input = torch.randn(1, *input_shape, dtype=torch.float).to(device)
    
    # MODEL TO EVAL MODE
    model.eval()
    
    # GPU/CPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)

    # MEASURE PERFORMANCE
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

    mean_syn = np.mean(timings)
    std_syn = np.std(timings)

    print(f"Average inference time: {mean_syn:.3f} +/- {std_syn:.3f} ms")

    return mean_syn

def plot_task_accuracy_multiple(task_acc_list, plot_task_acc=True, plot_avg_acc=True, plot_encountered_avg=True):
    """
    Plot the accuracy of each task, the average accuracy of all tasks, and the average accuracy of only encountered tasks
    for multiple dictionaries.

    Args:
        task_acc_list (list): A list of dictionaries, each containing the accuracy of each task.
        plot_task_acc (bool): Whether to plot individual task accuracies. Default is True.
        plot_avg_acc (bool): Whether to plot average accuracy of all tasks. Default is True.
        plot_encountered_avg (bool): Whether to plot average accuracy of only encountered tasks. Default is True.
    """
    def plot_individual_task_accuracy(task_acc_list):
        num_tasks = max(len(d) for d in task_acc_list)
        num_rows = (num_tasks + 1) // 2  # Number of rows in the subplots grid
        fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))
        axes = axes.flatten()
        plot_num = 0

        for idx, (key, ax) in enumerate(zip(range(0, num_tasks), axes)):
            for i, task_acc in enumerate(task_acc_list):
                if key in task_acc:
                    ax.plot(task_acc[key], label=f"Task {key} (Dict {i + 1})", marker='.')
            ax.grid(True)
            ax.set_xlabel('Task')
            ax.set_ylabel('Accuracy')
            ax.set_yticks(np.arange(0, 101, 5))
            ax.set_ylim(-1, 101)  # Set y-axis limits
            ax.set_title(f"Task {key}", loc='center')
            ax.set_xticks(range(1, num_tasks + 1))
            plot_num += 1

        # Remove any unused subplots if num_tasks is odd
        for i in range(num_tasks, num_rows * 2):
            fig.delaxes(axes[i])
        plt.tight_layout()
        plt.legend()
        plt.show()

    def plot_average_accuracy(task_acc_list):
        num_tasks = max(len(d) for d in task_acc_list)
        for i, task_acc in enumerate(task_acc_list):
            averages = [sum(values) / num_tasks for values in zip(*task_acc.values())]
            plt.plot(averages, label=f"Dict {i + 1}", marker='.')
        plt.grid(True)
        plt.xlabel('Task')
        plt.ylabel('Accuracy')
        plt.yticks(np.arange(0, 101, 5))
        plt.ylim(-1, 101)
        plt.xticks(range(1, num_tasks + 1))
        plt.title('Average Accuracy')
        plt.legend()
        plt.show()

    def plot_encountered_tasks_accuracy(task_acc_list):
        num_tasks = max(len(d) for d in task_acc_list)
        for i, task_acc in enumerate(task_acc_list):
            encountered_averages = []
            for last_key in range(max(len(next(iter(task_acc.values()))), num_tasks)):
                total_last_element = 0
                num_keys = 0

                for key, value in task_acc.items():
                    if int(key) <= int(last_key):
                        last_element = value[int(last_key)]
                        total_last_element += last_element
                        num_keys += 1

                avg = total_last_element / num_keys if num_keys > 0 else 0
                encountered_averages.append(avg)

            plt.plot(range(1, len(encountered_averages) + 1), encountered_averages, label=f"Dict {i + 1}", marker='.')
        plt.grid(True)
        plt.xlabel('Task')
        plt.ylabel('Accuracy')
        plt.yticks(np.arange(0, 101, 5))
        plt.ylim(-1, 101)
        plt.xticks(range(1, num_tasks + 1))
        plt.title('Average Accuracy of Encountered Tasks')
        plt.legend()
        plt.show()

    if plot_task_acc:
        plot_individual_task_accuracy(task_acc_list)
    if plot_avg_acc:
        plot_average_accuracy(task_acc_list)
    if plot_encountered_avg:
        plot_encountered_tasks_accuracy(task_acc_list)
