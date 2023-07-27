import torch
from torch.utils.data import DataLoader, ConcatDataset

import utility.utils as utils

from strategy.base_strategy import BaseStrategy

class Comulative(BaseStrategy):
    """ Comulative strategy."""
    def __init__(self, model, optimizer, criterion, train_mb_size, train_epochs, eval_mb_size, split_ratio = 0, patience = 5, device="cpu"):
        """Init.
        
        Args:
            model: PyTorch model.
            optimizer: PyTorch optimizer.
            criterion: PyTorch criterion.
            train_mb_size: training mini-batch size.
            train_epochs: number of training epochs.
            eval_mb_size: evaluation mini-batch size.
            split_ratio: ratio to split the dataset into training and validation.  If 0, no early stopping is performed.
            patience: patience for early stopping.
            device: PyTorch device where the model will be allocated.
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            split_ratio = split_ratio,
            patience= patience,
            device= device,
        )

    def train(self, dataset, test_data = None, plotting = False):
        """
        Training loop. If test data loader is provided, it will be used to test the model.

        Args:
            dataset: dataset to train on.
            test_data: dataset to test on.
            plotting: flag to plot the task accuracy.

        """
        print("Start of the training process...")
        cumulative_train = None
        for exp in dataset:
            print("Training of the experience with class: ", exp.classes_in_this_experience)
            
            # Create the dataloader
            if self.split_ratio != 0:
                train_dataset, val_dataset = utils.split_dataset(exp.dataset, split_ratio=self.split_ratio)
            else:
                train_dataset = exp.dataset
                val_loader = None

            if cumulative_train is None:
                # First experience
                cumulative_train = train_dataset
                if self.split_ratio != 0:
                    comulative_val = val_dataset
                    val_loader = DataLoader(comulative_val, batch_size=self.eval_mb_size, shuffle=True)
            else:
                # Concatenate the new dataset with the previous one
                cumulative_train = ConcatDataset([cumulative_train, train_dataset])
                if self.split_ratio != 0:
                    comulative_val = ConcatDataset([comulative_val, val_dataset])
                    val_loader = DataLoader(comulative_val, batch_size=self.eval_mb_size, shuffle=True)

            train_loader = DataLoader(cumulative_train, batch_size=self.train_mb_size, shuffle=True)

            super().train(train_loader, val_loader)

            if test_data is not None:
                print("")
                print("Test after the training of the experience with class: ", exp.classes_in_this_experience)
                exps_acc, _ = self.test(test_data)
                self.update_tasks_acc(exps_acc)
            print("-----------------------------------------------------------------------------------")
            
        if plotting:
            plotter = utils.TaskAccuracyPlotter()
            _ = plotter.plot_task_accuracy(self.tasks_acc, plot_task_acc=True, plot_avg_acc=True, plot_encountered_avg=True)
            plotter.show_figures()

    def test(self, dataset):
        """ 
        Test the model on the given dataset.

        Args:
            dataset: dataset to test on.

        Returns:
            exps_acc: dictionary with the accuracy of each experience.
            avg_acc: average accuracy.
        """
        exps_acc, avg_acc = super().test(dataset)
            
        return exps_acc, avg_acc