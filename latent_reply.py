import torch
from torch import nn
from base_strategy import BaseStrategy
from torch.utils.data import DataLoader, ConcatDataset
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from typing import NamedTuple, List, Callable
from torch import Tensor
from avalanche.training.utils import freeze_up_to


class LatentReplay(BaseStrategy):

    def __init__(self, model, optimizer, criterion, train_mb_size, train_epochs, eval_mb_size, rm_size, latent_layer_num, freeze_below_layer, lr = 0.01, weight_decay = 0, device="cpu"):
        """
        Init.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param train_mb_size: mini-batch size for training.
        :param train_epochs: number of training epochs.
        :param eval_mb_size: mini-batch size for eval..
        :param rm_size: size of the replay memory.
        :param latent_layer_num: number of layers to consider as latent layers.
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

        self.rm_size = rm_size
        """ Size of the replay memory. """
        self.latent_layer_num = latent_layer_num
        """ Number of layers to consider as latent layers. """

        self.freeze_below_layer = freeze_below_layer
        """ Layer below which the model should be frozen. """

        self.train_exp_counter = 0
        """ Number of training experiences so far. """

        self.lr = lr
        """ Learning rate. """

        self.weight_decay = weight_decay
        """ Weight decay. """

    def before_training_exp(self):
        self.model.eval()
        self.model.end_features.train()
        self.model.output.train()

        for param in self.model.lat_features.parameters():
            param.requires_grad = False

        # Set requires_grad=True for layers you want to train
        for param in self.model.end_features.parameters():
            param.requires_grad = True

        for param in self.model.output.parameters():
            param.requires_grad = True

    def make_train_dataloader(self, dataset, shuffle=True, **kwargs):
        """
        Called after the dataset instantiation. Initialize the data loader.

        For AR1 a "custom" dataloader is used: instead of using
        `self.train_mb_size` as the batch size, the data loader batch size will
        be computed ad `self.train_mb_size - latent_mb_size`. `latent_mb_size`
        is in turn computed as:

        `
        len(train_dataset) // ((len(train_dataset) + len(replay_buffer))
        // self.train_mb_size)
        `

        so that the number of iterations required to run an epoch on the current
        batch is equal to the number of iterations required to run an epoch
        on the replay buffer.

        :param num_workers: number of thread workers for the data loading.
        :param shuffle: True if the data should be shuffled, False otherwise.
        """

        current_batch_mb_size = self.train_mb_size

        if self.train_exp_counter > 0:
            train_patterns = len(dataset)
            current_batch_mb_size = train_patterns // (
                (train_patterns + self.rm_size) // self.train_mb_size
            )

        current_batch_mb_size = max(1, current_batch_mb_size)
        print("current_batch_mb_size: ", current_batch_mb_size)
        self.replay_mb_size = max(0, self.train_mb_size - current_batch_mb_size)
        print("replay_mb_size: ", self.replay_mb_size)

        # AR1 only supports SIT scenarios (no task labels).
        dataloader = DataLoader(
            dataset,
            batch_size=current_batch_mb_size,
            shuffle=shuffle,
        )
        return dataloader

    def train(self, dataset):
        """
        Training loop.

        :param dataset: dataset to train the model.
        """
        
        # freeze the model up to the latent layers
        self.before_training_exp()

        # set the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.weight_decay)

        print("Start of the training process...")
        for exp in dataset:
            print("Training of the experience with class: ", exp.classes_in_this_experience)

            # Test
            #subset_indices = torch.randperm(len(exp.dataset))[:300]
            #train_dataset_subset = torch.utils.data.Subset(exp.dataset, subset_indices)
            #train_loader = self.make_train_dataloader(train_dataset_subset, shuffle=True)

            train_loader = self.make_train_dataloader(exp.dataset, shuffle=True)
            
            for epoch in range(self.train_epochs):
                self.train_exp_epochs = epoch
                self.training_epoch(train_loader)
            self._after_training_exp(exp)
            self.train_exp_counter += 1

            print("")
            print("Test after the training of the experience with class: ", exp.classes_in_this_experience)
            #self.test(dataset)
            print("-----------------------------------------------------------------------------------")

    def _unpack_minibatch(self):
        """Move to device"""
        if isinstance(self.mbatch, tuple):
            self.mbatch = list(self.mbatch)
        for i in range(len(self.mbatch)):
            self.mbatch[i] = self.mbatch[i].to(self.device)
      
    def training_epoch(self, dataloader):
        for mb_it, self.mbatch in enumerate(dataloader):
            self._unpack_minibatch()
            self.optimizer.zero_grad()
            if self.train_exp_counter > 0:
                lat_mb_x = self.rm[0][
                    mb_it
                    * self.replay_mb_size : (mb_it + 1)
                    * self.replay_mb_size
                ]
                lat_mb_x = lat_mb_x.to(self.device)
                lat_mb_y = self.rm[1][
                    mb_it
                    * self.replay_mb_size : (mb_it + 1)
                    * self.replay_mb_size
                ]
                lat_mb_y = lat_mb_y.to(self.device)

                lat_task_id = torch.zeros(lat_mb_y.shape[0]).to(self.device)
                
                self.mbatch[1] = torch.cat((self.mbatch[1], lat_mb_y), 0)
                self.mbatch[2] = torch.cat((self.mbatch[2], lat_task_id), 0)
            else:
                lat_mb_x = None

            # Forward pass. Here we are injecting latent patterns lat_mb_x.
            # lat_mb_x will be None for the very first batch (batch 0), which
            # means that lat_acts.shape[0] == self.mb_x[0].
            self.mb_output, lat_acts = self.model(
                self.mbatch[0], latent_input=lat_mb_x, return_lat_acts=True
            )

            if self.train_exp_epochs == self.train_epochs-1:
                # On the first epoch only: store latent activations. Those
                # activations will be used to update the replay buffer.
                lat_acts = lat_acts.detach().clone().cpu()
                if mb_it == 0:
                    self.cur_acts = lat_acts
                else:
                    self.cur_acts = torch.cat((self.cur_acts, lat_acts), 0)
            # Loss & Backward
            # We don't need to handle latent replay, as self.mb_y already
            # contains both current and replay labels.
            self.loss = self.criterion(self.mb_output, self.mbatch[1])
            self.loss.backward()

            # Optimization step
            self.optimizer.step()

    def _after_training_exp(self, exp):
        h = min(
            self.rm_size // (self.train_exp_counter + 1),
            self.cur_acts.size(0),
        )

        curr_data = exp.dataset
        idxs_cur = torch.randperm(self.cur_acts.size(0))[:h]
        rm_add_y = torch.tensor(
            [curr_data.targets[idx_cur] for idx_cur in idxs_cur]
        )

        rm_add = [self.cur_acts[idxs_cur], rm_add_y]
        print("rm_add[0].shape: ", rm_add[0].shape)
        # replace patterns in random memory
        if self.train_exp_counter == 0:
            self.rm = rm_add
        else:
            idxs_2_replace = torch.randperm(self.rm[0].size(0))[:h]
            print("HERE")
            for j, idx in enumerate(idxs_2_replace):
                idx = int(idx)
                self.rm[0][idx] = rm_add[0][j]
                self.rm[1][idx] = rm_add[1][j]
        print("self.rm[0].shape: ", self.rm[0].shape)
        self.cur_acts = None

    def test(self, dataset):
        """
        Testing loop.
        """
        super().test(dataset)   