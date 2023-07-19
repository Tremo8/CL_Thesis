from strategy.base_strategy import BaseStrategy
from torch.utils.data import DataLoader, ConcatDataset
import utils
import torch

class Comulative(BaseStrategy):
    def __init__(self, model, optimizer, criterion, train_mb_size, train_epochs, eval_mb_size, device="cpu"):
        """Init.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param train_mb_size: mini-batch size for training.
        :param train_epochs: number of training epochs.
        :param eval_mb_size: mini-batch size for eval.
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

    def train(self, dataset, test_data = None, plotting = False):
        """
        Training loop.

        :param dataset: dataset to train the model.
        """
        print("Start of the training process...")
        cumulative_data = None
        for exp in dataset:
            subset_indices = torch.randperm(len(exp.dataset))[:1500]
            train_dataset_subset = torch.utils.data.Subset(exp.dataset, subset_indices)
            print("Training of the experience with class: ", exp.classes_in_this_experience)
            if cumulative_data is None:
                # First experience
                cumulative_data = train_dataset_subset
            else:
                # Concatenate the new dataset with the previous one
                cumulative_data = ConcatDataset([cumulative_data, train_dataset_subset])

            # Create the dataloader
            train_loader = DataLoader(cumulative_data, batch_size=self.train_mb_size, shuffle=True)

            super().train(train_loader)
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