from base_strategy import BaseStrategy
from torch.utils.data import DataLoader

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
            self.test(dataset)
            print("-----------------------------------------------------------------------------------")

    def test(self, dataset):
        """
        Testing loop.

        :param dataset: dataset to test the model.
        """
        super().test(dataset)