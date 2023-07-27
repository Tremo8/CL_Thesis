import utils
from strategy.base_strategy import BaseStrategy
from torch.utils.data import DataLoader
import torch

class JointTraining(BaseStrategy):
    def __init__(self, model, optimizer, criterion, train_mb_size, train_epochs, eval_mb_size, split_ratio = 0, patience = 5, device="cpu"):
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
            split_ratio = split_ratio,         
            patience = patience,
            device=device,
        )

    def train(self, dataset,test_data = None, plotting = False):
        """
        Training loop.

        :param dataset: dataset to train the model.
        """
        print("Start of the training process...")
        # Concatenate all the experiencess
        concat_set = utils.concat_experience(dataset)
        # Create the dataloader
        if self.split_ratio != 0:
            train_dataset, val_dataset = utils.split_dataset(concat_set, split_ratio=self.split_ratio)
            print("Train dataset size: ", len(train_dataset))
            print("Validation dataset size: ", len(val_dataset))
            train_loader = DataLoader(train_dataset, batch_size=self.train_mb_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.eval_mb_size, shuffle=True)
        else:
            train_loader = DataLoader(concat_set, batch_size=self.train_mb_size, shuffle=True)
            val_loader = None

        super().train(train_loader, val_loader)
        if test_data is not None:
            print("")
            print("Test after the joint training")
            exps_acc, _ = self.test(test_data)
            self.update_tasks_acc(exps_acc)
        print("-----------------------------------------------------------------------------------")
        if plotting:
            plotter = utils.TaskAccuracyPlotter()
            _ = plotter.plot_task_accuracy(self.tasks_acc, plot_task_acc=True, plot_avg_acc=True, plot_encountered_avg=True)
            plotter.show_figures()
            
    def test(self, dataset):
        """
        Testing loop.

        :param dataset: dataset to test on.

        """
        exps_acc, avg_acc = super().test(dataset)
            
        return exps_acc, avg_acc