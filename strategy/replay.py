import torch
from torch.utils.data import DataLoader, ConcatDataset

from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.benchmarks.utils import classification_subset

from strategy.base_strategy import BaseStrategy

import utility.utils as utils

class Replay(BaseStrategy):
    """ Experience replay strategy. """
    def __init__(self, model, optimizer, criterion, train_mb_size, train_epochs, eval_mb_size, rm_size, mem_mb_size = None, storage_policy = None, split_ratio = 0, patience = 5, device="cpu", file_name = None, path = None):
        """Init.

        Args:
            model: PyTorch model.
            optimizer: PyTorch optimizer.
            criterion: PyTorch criterion.
            train_mb_size: training mini-batch size.
            train_epochs: number of training epochs.
            eval_mb_size: evaluation mini-batch size.
            rm_size: size of the memory.
            storage_policy: storage policy.
            split_ratio: ratio to split the dataset into training and validation.  If 0, no early stopping is performed.
            patience: patience for early stopping.
            device: PyTorch device where the model will be allocated.
            file_name: name of the file to save the results.
            path: path to save the model.
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            split_ratio = split_ratio,
            patience=patience,
            device=device,
            file_name = file_name,
            path = path
        )

        if storage_policy is not None:
            self.storage_policy = storage_policy
        else:
            self.storage_policy = ClassBalancedBuffer(rm_size)
        """ Storage policy. """

        self.mem_mb_size = mem_mb_size

    def before_training_exp(self, dataset, shuffle: bool = True):
        """ 
        This method is called before training on a new experience. It returns
        the dataloader that will be used to train the model on the new
        experience.

        Args:
            dataset: dataset to train on.
            shuffle: if True, shuffle the dataset.

        Returns:
            dataloader to use to train on the new experience.
        """

        if len(self.storage_policy.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return DataLoader(dataset, batch_size=self.train_mb_size, shuffle=shuffle)

        # replay dataloader samples mini-batches from the memory and current
        # data separately and combines them together.
        if self.mem_mb_size is None:
            self.mem_mb_size = self.train_mb_size

        print("Override the dataloader.")
        dataloader = ReplayDataLoader(
            dataset,
            self.storage_policy.buffer,
            oversample_small_tasks=True,
            batch_size = self.train_mb_size,
            batch_size_mem  = self.mem_mb_size,
            shuffle=shuffle)
        
        return dataloader

    def train(self, dataset, test_data = None, plotting = False):
        """
        Training loop. If test data loader is provided, it will be used to test the model.
        
        Args:
            dataset: dataset to train on.
            test_data: test dataset to test the model.
            plotting: if True, plot the accuracy after each experience.
        """

        print("Start of the training process...")
        for exp in dataset:
            # Print start training of experince exp
            print("Training of the experience with class: ", exp.classes_in_this_experience)
            if self.split_ratio != 0:
                train_dataset, val_dataset = utils.split_dataset(exp.dataset, split_ratio=self.split_ratio)
                train_dataset, val_dataset = classification_subset(dataset=exp.dataset, indices=train_dataset.indices), classification_subset(dataset=exp.dataset, indices=val_dataset.indices)
                print("Train dataset size: ", len(train_dataset))
                print("Validation dataset size: ", len(val_dataset))
                val_loader = DataLoader(val_dataset, batch_size=self.eval_mb_size, shuffle=True)
            else:
                train_dataset = exp.dataset
                val_loader = None

            train_loader = self.before_training_exp(train_dataset, shuffle=True)

            super().train(train_loader, val_loader)

            self.storage_policy.update_from_dataset(train_dataset)

            if test_data is not None:
                exps_acc, _ = self.test(test_data)
                self.update_tasks_acc(exps_acc)

        if self.path is not None:
            torch.save(self.model.state_dict(), self.path)

        if plotting:
            plotter = utils.TaskAccuracyPlotter()
            _ = plotter.plot_task_accuracy(self.tasks_acc, plot_task_acc=True, plot_avg_acc=True, plot_encountered_avg=True)
            plotter.show_figures() 

    def test(self, dataset):
        """
        Test the model on the given dataset.
        
        Args:
            dataset: dataset to test the model on.

        Returns:
            exps_acc: list of accuracies for each experience.
            avg_acc: average accuracy over all the experiences.
        """
        exps_acc, avg_acc = super().test(dataset)
            
        return exps_acc, avg_acc