import utility.utils as utils
from utility.CSVsave import save_results_to_csv
from strategy.base_strategy import BaseStrategy
from torch.utils.data import DataLoader
import torch

class JointTraining(BaseStrategy):
    """ JointTraining strategy."""
    def __init__(self, model, optimizer, criterion, train_mb_size, train_epochs, eval_mb_size, split_ratio = 0, patience = 5, device="cpu", file_name = None, path = None):
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
            patience = patience,
            device=device,
            path = path
        )

    def train(self, dataset,test_data = None, plotting = False):
        """
        Training loop. If test data loader is provided, it will be used to test the model.

        Args:
            dataset: dataset to train on.
            test_data: dataset to test on.
            plotting: flag to plot the task accuracy.

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

        # Train the model
        super().train(train_loader, val_loader)

        if test_data is not None:
            print("")
            print("Test after the joint training")
            exps_acc, _ = self.test(test_data)
            self.update_tasks_acc(exps_acc)
        
        if self.path is not None:
            torch.save(self.model.state_dict(), self.path)

        print("-----------------------------------------------------------------------------------")
        if plotting:
            plotter = utils.TaskAccuracyPlotter()
            _ = plotter.plot_task_accuracy(self.tasks_acc, plot_task_acc=True, plot_avg_acc=True, plot_encountered_avg=True)
            plotter.show_figures()
            
    def test(self, dataset):
        """Test the model on the given dataset.

        Args:
            dataset: dataset to test on.

        Returns:
            exps_acc: list of accuracies for each experience.
            avg_acc: average accuracy over all the experiences.
        """
        exps_acc, avg_acc = super().test(dataset)
            
        return exps_acc, avg_acc