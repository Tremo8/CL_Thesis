import torch
import torch.nn.functional as F
from torch.nn import Module, BatchNorm2d
from torch.utils.data import ConcatDataset, random_split

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from avalanche.models.batch_renorm import BatchRenorm2D
from avalanche.benchmarks.datasets import CORe50Dataset
from avalanche.benchmarks.classic import SplitMNIST, SplitCIFAR10
from avalanche.benchmarks.generators import nc_benchmark

from avalanche.models import MobilenetV1
#from micromind import PhiNet
#from model.phinet_v3 import PhiNetV3

from pytorchcv.models.mobilenetv2 import mobilenetv2_w1, mobilenetv2_w3d4, mobilenetv2_wd2

from micromind import PhiNet
from model.phinetv1 import PhiNetV1
from model.mobilenetv2 import MobilenetV2

def benchmark_selction(name, n_experiences=5, seed=0, return_task_id = True, train_transform = None, eval_transform = None, dataset_root = None):
    """
    Select the benchmark to be used for training and testing.

    Args:
        name (str): The name of the benchmark to be used.
        n_experiences (int): The number of experiences in the benchmark. (default: 5)
        seed (int): The seed to be used for reproducibility. (default: 0)
        return_task_id (bool): Whether to return the task ID. (default: True)
        train_transform (torchvision.transforms): The transformations to be applied to the training data. (default: None)
        eval_transform (torchvision.transforms): The transformations to be applied to the evaluation data. (default: None)
        dataset_root (str): The root directory of the dataset. (default: None)

    Returns:
        Benchmark: The benchmark to be used for training and testing.
    """
    if name == 'core50':
        train_data = CORe50Dataset(root= dataset_root, transform=train_transform, object_level = False)
        test_data = CORe50Dataset(root= dataset_root, train=False, transform=eval_transform, object_level = False)
        return nc_benchmark(train_data, test_data, n_experiences=n_experiences, task_labels=return_task_id, seed=seed)
    elif name == 'split_mnist':
        return SplitMNIST(n_experiences=n_experiences, seed=seed, return_task_id = return_task_id, train_transform = train_transform, eval_transform = eval_transform, dataset_root = dataset_root)
    elif name == 'split_cifar10':
        return SplitCIFAR10(n_experiences=n_experiences, seed=seed, return_task_id = return_task_id, train_transform = train_transform, eval_transform = eval_transform, dataset_root = dataset_root)
    else:
        raise Exception('Invalid benchmark name')
    
def model_selection(name, latent_layer, pretrained=True):
    """
    Selects a model based on the given name and latent layer.

    Args:
        name (str): The name of the model to select.
        latent_layer (int): The number of the latent layer to use.
        pretrained (bool, optional): Whether to use a pretrained model. Defaults to True.

    Returns:
        torch.nn.Module: The selected model.

    Raises:
        ValueError: If an invalid model name is provided.
    """
    if name == 'mobilenetv1':
        model = MobilenetV1(pretrained=pretrained, latent_layer_num = latent_layer)
        return model
    elif name == 'mobilenetv2':
        model = mobilenetv2_w1(pretrained=pretrained)
        model = MobilenetV2(model=model,latent_layer_num = latent_layer)
        return model
    elif name == '0.75_mobilenetv2':
        model = mobilenetv2_w3d4(pretrained=pretrained)
        model = MobilenetV2(model=model,latent_layer_num = latent_layer)
        return model
    elif name == '0.5_mobilenetv2':
        model = mobilenetv2_wd2(pretrained=pretrained)
        model = MobilenetV2(model=model,latent_layer_num = latent_layer)
        return model
    elif name == 'phinet_2.3_0.75_5':
        model = PhiNet(input_shape=(3,224,224), alpha = 2.3, beta = 0.75, t_zero = 5, include_top = True,num_classes = 1000, divisor = 8)
        model.load_state_dict(torch.load('./new_phinet_small_71.pth.tar')["state_dict"])
        model = PhiNetV1(model=model, latent_layer_num=latent_layer)
        return model
    elif name == 'phinet_1.2_0.5_6_downsampling':
        model = PhiNet(input_shape=(3,224,224), num_layers=7, alpha = 1.2, beta = 0.5, t_zero = 6, downsampling_layers=[4,5,7], include_top = True,num_classes = 1000, divisor = 8)
        model.load_state_dict(torch.load('./new_phinet_divisor8_v2_downsampl.pth.tar')["state_dict"])
        model = PhiNetV1(model=model, latent_layer_num=latent_layer)
        return model
    elif name == 'phinet_0.8_0.75_8_downsampling':
        model = PhiNet(input_shape=(3,224,224), num_layers=7, alpha = 0.8, beta = 0.75, t_zero = 8, downsampling_layers=[4,5,7], include_top = True,num_classes = 1000, divisor = 8)
        model.load_state_dict(torch.load('./new_phinet_divisor8_v3.pth.tar')["state_dict"])
        model = PhiNetV1(model=model, latent_layer_num=latent_layer)
        return model
    elif name == 'phinet_1.3_0.5_7_downsampling':
        model = PhiNet(input_shape=(3,224,224), num_layers=7, alpha = 1.3, beta = 0.5, t_zero = 7, downsampling_layers=[4,5,7], include_top = True,num_classes = 1000, divisor = 8)
        model.load_state_dict(torch.load('./phinet_13057DS.pth.tar')["state_dict"])
        model = PhiNetV1(model=model, latent_layer_num=latent_layer)
        return model
    elif name == 'phinet_0.9_0.5_4_downsampling_deep':
        model = PhiNet(input_shape=(3,224,224), num_layers=9, alpha = 0.9, beta = 0.5, t_zero = 4, downsampling_layers=[4,5,7], include_top = True,num_classes = 1000, divisor = 8)
        model.load_state_dict(torch.load('./phinet_09054DSDE.pth.tar')["state_dict"])
        model = PhiNetV1(model=model, latent_layer_num=latent_layer)
        return model
    elif name == 'phinet_0.9_0.5_4_downsampling':
        model = PhiNet(input_shape=(3,224,224), num_layers=7, alpha = 0.9, beta = 0.5, t_zero = 4, downsampling_layers=[4,5,7], include_top = True,num_classes = 1000, divisor = 8)
        model.load_state_dict(torch.load('./phinet_09054DS.pth.tar')["state_dict"])
        model = PhiNetV1(model=model, latent_layer_num=latent_layer)
        return model
    else:
        raise ValueError(f"Invalid model name: {name}")
    
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

def split_dataset(dataset, split_ratio=0.8):
    """
    Split the dataset into train and validation sets using random_split.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to be split.
        split_ratio (float): The ratio of the dataset to be used for training. (default: 0.8)

    Returns:
        torch.utils.data.Dataset: Training dataset.
        torch.utils.data.Dataset: Validation dataset.
    """
    # Calculate the size of the training dataset based on the split_ratio
    train_size = int(len(dataset) * split_ratio)

    # Calculate the size of the validation dataset
    val_size = len(dataset) - train_size

    # Use random_split to split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset

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

    shuffled_indices = torch.randperm(len(concat_data))
    shuffled_dataset = torch.utils.data.Subset(concat_data, shuffled_indices)
    
    return shuffled_dataset.dataset
            
# momentum=0.1, r_d_max_inc_step=0.0001, max_r_max=3.0, max_d_max=5.0
def replace_bn_with_brn(m: Module, momentum=0.00005, r_d_max_inc_step=0,
                        r_max=1.0, d_max=0.0, max_r_max=1.25, max_d_max=0.5):
    """
    Recursively replace all instances of BatchNorm2d with BatchRenorm2d.

    Args:
        m: The module to be transformed.
        momentum: The momentum for the running mean and variance.
        r_d_max_inc_step: The step size for increasing r_max and d_max.
        r_max: The maximum value for r_max.
        d_max: The maximum value for d_max.
        max_r_max: The maximum value for the maximum value of r_max.
        max_d_max: The maximum value for the maximum value of d_max.
    """

    for name, child_module in m.named_children():
        if isinstance(child_module, torch.nn.BatchNorm2d):
            setattr(
                m,
                name,
                BatchRenorm2D(
                    child_module.num_features,
                    gamma=child_module.weight,
                    beta=child_module.bias,
                    running_mean=child_module.running_mean,
                    running_var=child_module.running_var,
                    eps=child_module.eps,
                    momentum=momentum,
                    r_d_max_inc_step=r_d_max_inc_step,
                    r_max=r_max,
                    d_max=d_max,
                    max_r_max=max_r_max,
                    max_d_max=max_d_max,
                ),
            )
        else:
            replace_bn_with_brn(child_module, momentum, r_d_max_inc_step,
                                r_max, d_max, max_r_max, max_d_max)

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

class TaskAccuracyPlotter:
    """
    Class for plotting the accuracy of each task, the average accuracy of all tasks, and the average accuracy of only encountered tasks.
    """

    def __init__(self):
        self.fig1 = None
        self.fig2 = None
        self.fig3 = None
        self.label = None

    def plot_individual_task_accuracy(self, task_acc):
        """
        Plot the accuracy of each individual task.

        Args:
            task_acc (dict): A dictionary containing the accuracy of each task.
            label (str, optional): Label for the current plot.

        Returns:
            matplotlib.figure.Figure: The updated figure containing the plots.
        """
        if self.fig1 is None:
            num_tasks = len(task_acc)
            num_rows = (num_tasks + 1) // 2  # Number of rows in the subplots grid
            self.fig1, axes = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))
            axes = axes.flatten()
            plot_num = 0
        else:
            axes = self.fig1.get_axes()
            plot_num = 0

        for idx, (key, ax) in enumerate(zip(task_acc.keys(), axes[plot_num:])):
            ax.plot(task_acc[key], label=self.label, marker='.')
            ax.grid(True)
            ax.set_xlabel('Task')
            ax.set_ylabel('Accuracy')
            ax.set_yticks(np.arange(0, 101, 5))
            ax.set_ylim(-1, 101)  # Set y-axis limits
            ax.set_title(f"Task {key}", loc='center')
            ax.set_xticks(list(task_acc.keys()))
            ax.legend() if ax.get_legend_handles_labels()[1] else None
            plot_num += 1

        # Remove any unused subplots if num_tasks is odd
        for i in range(plot_num, len(axes)):
            self.fig1.delaxes(axes[i])

        plt.tight_layout()

        return self.fig1

    def plot_average_accuracy(self, task_acc):
        """
        Plot the average accuracy of all tasks.

        Args:
            task_acc (dict): A dictionary containing the accuracy of each task.
            label (str, optional): Label for the current plot.

        Returns:
            matplotlib.figure.Figure: The figure containing the plot.
        """
        if self.fig2 is None:
            self.fig2, ax = plt.subplots(figsize=(10, 6))
        else:
            ax = self.fig2.get_axes()[0]

        num_tasks = len(task_acc)
        averages = [sum(values) / num_tasks for values in zip(*task_acc.values())]
        ax.plot(averages, marker='.', label=self.label)
        ax.grid(True)
        ax.set_xlabel('Task')
        ax.set_ylabel('Accuracy')
        ax.set_yticks(np.arange(0, 101, 5))
        ax.set_ylim(-1, 101)
        ax.set_xticks(list(task_acc.keys()))
        ax.set_title('Average Accuracy')
        ax.legend() if ax.get_legend_handles_labels()[1] else None

        plt.tight_layout()

        return self.fig2

    def plot_encountered_tasks_accuracy(self, task_acc):
        """
        Plot the average accuracy of only encountered tasks.

        Args:
            task_acc (dict): A dictionary containing the accuracy of each task.
            label (str, optional): Label for the current plot.

        Returns:
            matplotlib.figure.Figure: The figure containing the plot.
        """
        if self.fig3 is None:
            self.fig3, ax = plt.subplots(figsize=(10, 6))
        else:
            ax = self.fig3.get_axes()[0]

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

        ax.plot(encountered_averages, marker='.', label=self.label)
        ax.grid(True)
        ax.set_xlabel('Task')
        ax.set_ylabel('Accuracy')
        ax.set_yticks(np.arange(0, 101, 5))
        ax.set_ylim(-1, 101)
        ax.set_xticks(list(task_acc.keys()))
        ax.set_title('Average Accuracy of Encountered Tasks')
        ax.legend() if ax.get_legend_handles_labels()[1] else None

        plt.tight_layout()

        return self.fig3


    def show_figures(self):
        """
        Show all figures.
        """
        if self.fig1 is not None:
            self.fig1.show()
        if self.fig2 is not None:
            self.fig2.show()
        if self.fig3 is not None:
            self.fig3.show()

        
    def plot_task_accuracy(self, task_acc,  label=None, plot_task_acc=True, plot_avg_acc=True, plot_encountered_avg=True):
        """
        Plot the accuracy of each task, the average accuracy of all tasks, and the average accuracy of only encountered tasks.

        Args:
            task_acc (dict): A dictionary containing the accuracy of each task.
            label (str, optional): Label for the current plot.
            plot_task_acc (bool): Whether to plot individual task accuracies. Default is True.
            plot_avg_acc (bool): Whether to plot average accuracy of all tasks. Default is True.
            plot_encountered_avg (bool): Whether to plot average accuracy of only encountered tasks. Default is True.

        Returns:
            matplotlib.figure.Figure: The figure containing the plot.
        """
        self.label = label
        
        if plot_task_acc:
            fig1 = self.plot_individual_task_accuracy(task_acc)
        if plot_avg_acc:
            fig2 = self.plot_average_accuracy(task_acc)
        if plot_encountered_avg:
            fig3 = self.plot_encountered_tasks_accuracy(task_acc)

        return self.fig1, self.fig2, self.fig3