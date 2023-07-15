import torch
import torch.nn as nn
import torch.optim as optim
import utils
from torch.utils.data import DataLoader

class BaseStrategy():
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


    def train(self, dataset):
        """
        Training loop.

        :param dataset: dataset to train the model.

        """
        for epoch in range(self.train_epochs):
            train_acc, train_loss = utils.train(self.model, self.optimizer, self.criterion, dataset, self.device)

            print(f"Epoch: {epoch+1}/{self.train_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
    

    def test(self, dataset):
        """
        Testing loop.

        :param dataset: dataset to test the model.
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
    