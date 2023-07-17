import utils
from strategy.base_strategy import BaseStrategy
from torch.utils.data import DataLoader

class JointTraining(BaseStrategy):
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

    def train(self, dataset,test_data = None, plotting = False):
        """
        Training loop.

        :param dataset: dataset to train the model.
        """
        print("Start of the training process...")
        # Concatenate all the experiencess
        train_set = utils.concat_experience(dataset)
        
        # Create the dataloader
        train_loader = DataLoader(train_set, batch_size=self.train_mb_size, shuffle=True)
        
        super().train(train_loader)
        if test_data is not None:
            print("")
            print("Test after the joint training")
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