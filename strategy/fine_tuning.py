from strategy.base_strategy import BaseStrategy
from torch.utils.data import DataLoader
import utils

class FineTuning(BaseStrategy):
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

        self.tasks_acc = {}

    def train(self, dataset):
        """
        Training loop.

        :param dataset: dataset to train the model.
        """
        print("Start of the training process...")
        for exp in dataset:
            print("Training of the experience with class: ", exp.classes_in_this_experience)
            # Create the dataloader
            train_loader = DataLoader(exp.dataset, batch_size=self.train_mb_size, shuffle=True)

            super().train(train_loader)

            print("")
            print("Test after the training of the experience with class: ", exp.classes_in_this_experience)
            self.test(dataset, Plotting=True)
            print("-----------------------------------------------------------------------------------")

    def test(self, dataset, Plotting=False):
        """
        Testing loop.
        """
        exps_acc, avg_acc = super().test(dataset)
        if self.tasks_acc:
            for key in exps_acc.keys():
                self.tasks_acc[key].append(exps_acc[key])
        else:
            for key in exps_acc.keys():
                self.tasks_acc[key] = [exps_acc[key]]
                
        if Plotting:
            utils.plot_task_accuracy(self.tasks_acc)
        return exps_acc, avg_acc  