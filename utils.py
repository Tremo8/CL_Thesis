import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
from avalanche.evaluation.metrics import MAC
import matplotlib.pyplot as plt
from torchinfo import summary

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


def plot_task_accuracy(task_acc):
    """
    Plot the accuracy of each task and the average accuracy of all tasks encountered so far.

    Args:
        task_acc: A dictionary containing the accuracy of each task.
    """

    num_tasks = len(task_acc)
    num_rows = (num_tasks + 1) // 2  # Number of rows in the subplots grid
    fig, axes = plt.subplots(num_rows, 2, figsize=(10, 5 * num_rows))

    # Flatten the axes array if there is only one row
    axes = axes.flatten() if num_rows == 1 else axes.ravel()

    # Plot individual task accuracies
    for idx, (key, ax) in enumerate(zip(task_acc.keys(), axes)):
        ax.plot(task_acc[key], label=f"Task {key}", marker='.')
        ax.grid(True)
        ax.set_xlabel('Task')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 100)  # Set y-axis limits
        ax.set_title(f"Task {key}", loc='center')
        ax.set_xticks(list(task_acc.keys()))

    averages = []
    values_count = len(next(iter(task_acc.values())))  # Assuming all lists have the same length

    for i in range(values_count):
        sum_values = 0
        for key in task_acc:
            sum_values += task_acc[key][i]

        average = sum_values / len(task_acc)
        averages.append(average)

    # Plot average accuracy of all tasks
    ax = axes[-1]
    ax.plot(averages, label='Average Accuracy', marker='.')
    ax.grid(True)
    ax.set_xlabel('Task')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 100)  # Set y-axis limits
    ax.set_title('Average Accuracy', loc='center')
    ax.set_xticks(list(task_acc.keys()))

    plt.tight_layout()
    plt.show()  

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

