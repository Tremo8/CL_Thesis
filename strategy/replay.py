from strategy.base_strategy import BaseStrategy
from torch.utils.data import DataLoader, ConcatDataset
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader

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

    def train(self, dataset):
        """
        Training loop.

        :param dataset: dataset to train the model.
        """
        print("Start of the training process...")
        for exp in dataset:
            # Print start training of experince exp
            print("Training of the experience with class: ", exp.classes_in_this_experience)
            train_loader = self.before_training_exp(exp.dataset, shuffle=True)

            super().train(train_loader)

            self.storage_policy.update_from_dataset(exp.dataset)
            print("")
            print("Test after the training of the experience with class: ", exp.classes_in_this_experience)
            self.test(dataset)
            print("-----------------------------------------------------------------------------------")
      

    def test(self, dataset):
        """
        Testing loop.
        """
        super().test(dataset)   