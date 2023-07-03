import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset

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