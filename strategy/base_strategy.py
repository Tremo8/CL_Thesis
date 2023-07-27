import torch
from torch.utils.data import DataLoader

import utility.utils as utils

from utility.pytorchtools import EarlyStopping

class BaseStrategy():
    def __init__(self, model, optimizer, criterion, train_mb_size, train_epochs, eval_mb_size, split_ratio = 0, patience = 5, device="cpu"):
        """Init.

        Args:
            model: PyTorch model.
            optimizer: PyTorch optimizer.
            criterion: PyTorch criterion.
            train_mb_size: training mini-batch size.
            train_epochs: number of training epochs.
            eval_mb_size: evaluation mini-batch size.
            split_ratio: ratio to split the dataset into training and validation.
            patience: patience for early stopping.
            device: PyTorch device where the model will be allocated.
        """
        self.model = model
        """ PyTorch model. """

        self.optimizer = optimizer
        """ PyTorch optimizer. """

        self.criterion = criterion
        """ Criterion. """

        self.train_epochs = train_epochs
        """ Number of training epochs. """

        self.train_mb_size = train_mb_size
        """ Training mini-batch size. """

        self.eval_mb_size = train_mb_size if eval_mb_size is None else eval_mb_size

        # JointTraining can be trained only once.
        self._is_fitted = False

        self.device = torch.device(device)
        """ PyTorch device where the model will be allocated. """

        self.tasks_acc = dict()
        """ Dictionary with the accuracy of each task. """

        self.early_stopping = EarlyStopping(patience=patience, verbose=2)
        """ Early stopping. """

        self.split_ratio = split_ratio
        """ Ratio to split the dataset into training and validation. """


    def train(self, train_loader, valid_loader = None):
        """
        Training loop. If validation data loader is provided, it will be used to perform early stopping.

        Args:
            train_loader: training data loader.
            valid_loader: validation data loader.
        """
        
        for epoch in range(self.train_epochs):
            train_acc, train_loss = utils.train(self.model, self.optimizer, self.criterion, train_loader, self.device)

            print(f"Epoch: {epoch+1}/{self.train_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
            if valid_loader is not None:
                early_stopped = self.validate_and_early_stop(valid_loader)
                if early_stopped:
                    self.early_stopping.reset_counter()
                    print("Early stopping")
                    break
    

    def test(self, dataset):
        """
        Testing loop. It will test the model on each task in the dataset.

        Args:
            dataset: dataset containing the tasks to test.

        Returns:
            Dictionary with the accuracy of each task and the average accuracy.
        """
        print("Starting the testing...")
        sum = 0
        exps_acc = dict()
        for exp in dataset:
            print("Testing task ", exp.task_label)
            print('Classes in this task:', exp.classes_in_this_experience)

            experience_dataloader = DataLoader(exp.dataset, batch_size=self.eval_mb_size, shuffle=False)
            test_acc, test_loss = utils.test(self.model, self.criterion, experience_dataloader, self.device)
            sum += test_acc
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
            exps_acc[exp.task_label] = test_acc
        avg_acc = sum/len(dataset)
        print(f"Average accuracy: {avg_acc:.2f}%")
     
        return exps_acc, avg_acc
    
    def validate_and_early_stop(self, valid_loader):
        """
        Validate the model on the validation set and perform early stopping.

        Args:
            valid_loader: validation data loader.

        Returns:
            Flag indicating whether early stopping condition is met or not.
        """

        _, valid_loss = utils.test(self.model, self.criterion, valid_loader, self.device)
        # early_stopping needs the validation loss to check if it has decreased, 
        # and if it has, it will make a checkpoint of the current model
        self.early_stopping(valid_loss, self.model)
        
        # Return a flag indicating whether early stopping condition is met or not
        return self.early_stopping.early_stop
    
    def update_tasks_acc(self, exps_acc):
        """Update the dictionary with the accuracy of each task.

        Args:
            exps_acc: dictionary with the accuracy of each task.
        """
        
        if self.tasks_acc:
            for key in exps_acc.keys():
                self.tasks_acc[key].append(exps_acc[key])
        else:
            for key in exps_acc.keys():
                self.tasks_acc[key] = [exps_acc[key]]

    def get_tasks_acc(self):
        """Get the dictionary with the accuracy of each task.

        Returns:
            Dictionary with the accuracy of each task.
        """
        return self.tasks_acc

    