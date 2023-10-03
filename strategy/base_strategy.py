import torch
from torch.utils.data import DataLoader

import utility.utils as utils
from utility.CSVsave import save_results_to_csv

from utility.pytorchtools import EarlyStopping

class BaseStrategy():
    def __init__(self, model, optimizer, criterion, train_mb_size, train_epochs, eval_mb_size, split_ratio = 0, patience = 5, device="cpu", file_name = None, path = None):
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
            file_name: file name to save the results. If None, no results are saved.
            path: path to save the model.
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

        self.path = path
        """ Path to save the model. """

        self.file_name = file_name
        """ File name to save the results. """

        self.early_stopping = EarlyStopping(patience=patience, verbose=2, path=None)
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

        tot_exp_time = 0
        train_losses = []
        train_accuracies = []

        for self.epoch in range(self.train_epochs):
            
            # Initialize the timer
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

            # Start Recording the time
            starter.record()

            train_acc, train_loss = utils.train(self.model, self.optimizer, self.criterion, train_loader, self.device)

            ender.record()

            torch.cuda.synchronize()

            curr_time = starter.elapsed_time(ender)

            # Save some statics the be saved in output
            tot_exp_time += curr_time
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            print(f"Epoch: {self.epoch+1}/{self.train_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Training Time: {curr_time/1000:.3f} s")

            if valid_loader is not None:
                early_stopped = self.validate_and_early_stop(valid_loader)
                if early_stopped:
                    if self.file_name is not None:
                            save_results_to_csv([["Stop Epoch"],[self.epoch+1]], self.file_name)
                    print("Early stopping")
                    break

        if self.file_name is not None:
                # Calculate the average train loss and accuracy
                avg_train_loss = sum(train_losses) / len(train_losses)
                avg_train_acc = sum(train_accuracies) / len(train_accuracies)
                save_results_to_csv([[f"Training Time", "Avg Training Loss", "Avg Training Acc"],[tot_exp_time/1000, avg_train_loss, avg_train_acc]], self.file_name)   
        
        if valid_loader is not None:
            self.early_stopping.reset_counter()
    
    def test(self, dataset):
        """
        Testing loop. It will test the model on each task in the dataset.

        Args:
            dataset: dataset containing the tasks to test.

        Returns:
            List of dictionaries containing the results of each task and the average accuracy.
        """
        
        print("\nStart of the testing process...")

        sum_accuracy = 0
        exps_acc = dict()
        results = [[],[]]

        for exp in dataset:
            print("Testing task ", exp.task_label)
            print('Classes in this task:', exp.classes_in_this_experience)

            experience_dataloader = DataLoader(exp.dataset, batch_size=self.eval_mb_size, shuffle=False)

            test_acc, test_loss = utils.test(self.model, self.criterion, experience_dataloader, self.device)
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

            sum_accuracy += test_acc
            exps_acc[exp.task_label] = test_acc

            results[0].append(f"Task {exp.task_label}")
            results[1].append(test_acc/100)

        # Calculate and add average accuracy
        avg_accuracy = sum_accuracy / len(dataset)
        print(f"Average accuracy: {avg_accuracy:.2f}%")

        results[0].append(f"Avg Acc")
        results[1].append(avg_accuracy/100)
        
        if self.file_name is not None:
            save_results_to_csv(results, self.file_name)

        return exps_acc, avg_accuracy
    
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

    