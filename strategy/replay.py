from strategy.base_strategy import BaseStrategy
from torch.utils.data import DataLoader, ConcatDataset
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
import utils
import torch

class Replay(BaseStrategy):
    def __init__(self, model, optimizer, criterion, train_mb_size, train_epochs, eval_mb_size, storage_policy, device="cpu"):
        """
        Init.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param train_mb_size: mini-batch size for training.
        :param train_epochs: number of training epochs.
        :param eval_mb_size: mini-batch size for eval.
        :param storage_policy: storage policy.
        :param device: PyTorch device to run the model.
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
        )

        self.storage_policy = storage_policy
        """ Storage policy. """

    def before_training_exp(self, dataset, shuffle: bool = True):
            """ Here we set the dataloader. """
            if len(self.storage_policy.buffer) == 0:
                # first experience. We don't use the buffer, no need to change
                # the dataloader.
                return DataLoader(dataset, batch_size=self.train_mb_size, shuffle=shuffle)

            # replay dataloader samples mini-batches from the memory and current
            # data separately and combines them together.
            print("Override the dataloader.")
            dataloader = ReplayDataLoader(
                dataset,
                self.storage_policy.buffer,
                oversample_small_tasks=True,
                batch_size=self.train_mb_size,
                batch_size_mem  = self.train_mb_size,
                shuffle=shuffle)
            
            return dataloader

    def train(self, dataset, test_data = None, plotting = False):
        """
        Training loop.

        :param dataset: dataset to train the model.
        """
        print("Start of the training process...")
        for exp in dataset:
            # Print start training of experince exp
            print("Training of the experience with class: ", exp.classes_in_this_experience)
            subset_indices = torch.randperm(len(exp.dataset))[:1500]
            train_dataset_subset = torch.utils.data.Subset(exp.dataset, subset_indices)
            train_loader = self.before_training_exp(train_dataset_subset, shuffle=True)

            super().train(train_loader)

            self.storage_policy.update_from_dataset(train_dataset_subset)
            if test_data is not None:
                print("")
                print("Test after the training of the experience with class: ", exp.classes_in_this_experience)
                exps_acc, _ = self.test(test_data)
                self.update_tasks_acc(exps_acc)
            print("-----------------------------------------------------------------------------------")
            if plotting:
                utils.plot_task_accuracy(self.tasks_acc)      

    def test(self, dataset):
        """
        Testing loop.

        :param dataset: dataset to test on.

        """
        exps_acc, avg_acc = super().test(dataset)
            
        return exps_acc, avg_acc