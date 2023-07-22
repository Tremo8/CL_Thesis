import torch
import torch.nn.functional as F
from strategy.base_strategy import BaseStrategy
from torch.utils.data import DataLoader
from memory_computation import total_size
import utils
import sys


class LatentReplay(BaseStrategy):

    def __init__(self, model, optimizer, criterion, train_epochs, train_mb_size = 21, replay_mb_size = 107,  eval_mb_size = 128, rm_size = 1500, manual_mb = True, device = "cpu"):
        """
        Init.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param train_epochs: number of training epochs.
        :param train_mb_size: mini-batch size for training.
        :param replay_mb_size: mini-batch size for replay buffer.
        :param eval_mb_size: mini-batch size for eval.
        :param rm_size: size of the replay memory.
        :param manual_mb: If True the mini-batch size should be manually setted. If False it computes the mini-batch.
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

        self.replay_mb_size = replay_mb_size
        """ Replay mini-batch size. """

        self.manual_mb = manual_mb
        """ If True the mini-batch size should be manually setted. If False it computes the mini-batch size"""

        self.train_exp_counter = 0
        """ Number of training experiences so far. """

    def before_training_exp(self):
        """
        Freezing the latent layers.
        """

        self.model.eval()
        self.model.end_features.train()
        self.model.output.train()

        # Set requires_grad=False for layers you want to freeze
        for param in self.model.lat_features.parameters():
            param.requires_grad = False

        # Set requires_grad=True for layers you want to train
        for param in self.model.end_features.parameters():
            param.requires_grad = True

        for param in self.model.output.parameters():
            param.requires_grad = True

    def make_train_dataloader(self, dataset, manual_mb = True, shuffle = True):
        """
        Called after the dataset instantiation. Initialize the data loader.

        :param dataset: The dataset to instantiate the data loader.
        :param manual_mb: If True the mini-batch size should be manually setted. If False it computes the mini-batch
        size as `len(train_dataset) // ( ( len(train_dataset) + len(replay_buffer) ) // train_mb_size )` and the memory
        mini-batch size as `train_mb_size - replay_mb_size`.
        :param shuffle: True if the data should be shuffled, False otherwise.
        """
        current_batch_mb_size = self.train_mb_size

        if not manual_mb and self.train_exp_counter > 0:
            train_patterns = len(dataset)
            current_batch_mb_size = train_patterns // ( (train_patterns + self.rm_size) // self.train_mb_size )
            
            current_batch_mb_size = max(1, current_batch_mb_size)
            self.replay_mb_size = max(0, self.train_mb_size - current_batch_mb_size)

        # Create the data loader
        dataloader = DataLoader(
            dataset,
            batch_size=current_batch_mb_size,
            shuffle=shuffle,
        )

        return dataloader

    def train(self, dataset, test_data = None, plotting = False):
        """
        Training loop.

        :param dataset: dataset to train the model.
        :param test_data: dataset to test the model. If None, the test phase is skipped.
        """
        
        # Freeze the model up to the latent layers
        self.before_training_exp()

        print("AFTER PhiNet: ")
        print(self.model)

        print("Start of the training process...")
        for exp in dataset:
            print("Training of the experience with class: ", exp.classes_in_this_experience)

            # Test
            subset_indices = torch.randperm(len(exp.dataset))[:1500]
            train_dataset_subset = torch.utils.data.Subset(exp.dataset, subset_indices)
            train_loader = self.make_train_dataloader(train_dataset_subset, manual_mb = self.manual_mb, shuffle=True)
            
            # Data loader initialization
            # Called at the start of each learning experience
            #train_loader = self.make_train_dataloader(exp.dataset, shuffle=True)
            
            # Training loop over the current experience
            for self.epoch in range(self.train_epochs):
                self.training_epoch(train_loader)
                print(f"Epoch: {self.epoch+1}/{self.train_epochs}, Train Loss: {self.avg_loss:.4f}, Train Accuracy: {self.acc:.2f}%")
            
            # Update the memory
            self.update_mem_after_exp(exp)

            # Old method to update the memory
            #self._after_training_exp(exp)
            self.train_exp_counter += 1
                
            if test_data is not None:
                print("")
                print("Test after the training of the experience with class: ", exp.classes_in_this_experience)
                exps_acc, _ = self.test(test_data)
                self.update_tasks_acc(exps_acc)
            print("-----------------------------------------------------------------------------------")
            if plotting:
                utils.plot_task_accuracy(self.tasks_acc, plot_task_acc=True, plot_avg_acc=True, plot_encountered_avg=True)
                print("-----------------------------------------------------------------------------------")
                print("Plot 2")
                utils.plot_task_accuracy_multiple(self.tasks_acc, plot_task_acc=True, plot_avg_acc=True, plot_encountered_avg=True)

    def _unpack_minibatch(self):
        """Move to device"""
        if isinstance(self.mbatch, tuple):
            self.mbatch = list(self.mbatch)
        for i in range(len(self.mbatch)):
            self.mbatch[i] = self.mbatch[i].to(self.device)
      
    def training_epoch(self, dataloader):
        self.train_loss = 0
        self.correct = 0
        self.total = 0
        it = 0

        for mb_it, self.mbatch in enumerate(dataloader):
            # Move to device the minibatch
            self._unpack_minibatch()

            # Set the gradients to zero
            self.optimizer.zero_grad()

            # After the first experience, we use the repley buffer
            if self.train_exp_counter > 0:
                lat_mb_x = None
                # Sample different minibatch from the replay buffer at every iteration
                for key, value in self.rm.items():

                    # If the number of sample in the memory are inssuficient, use again already used data
                    if value[0].size(0) < ((it + 1 ) * self.replay_mb_size//(len(self.rm))):
                        it = 0

                    # Sample the minibatch from the replay buffer
                    temp_lat_mb_x = value[0][it * self.replay_mb_size//(len(self.rm)) : (it + 1) * self.replay_mb_size//(len(self.rm))]
                    temp_lat_mb_x = temp_lat_mb_x.to(self.device)
                    
                    # Sample the minibatch from the replay buffer
                    lat_mb_y = value[1][it * self.replay_mb_size//(len(self.rm)) : (it + 1) * self.replay_mb_size//(len(self.rm))]
                    lat_mb_y = lat_mb_y.to(self.device)
                    
                    # Create the task id for the minibatch sampled from the replay buffer
                    lat_task_id = torch.zeros(lat_mb_y.shape[0]).to(self.device)

                    if lat_mb_x is None:
                        lat_mb_x = temp_lat_mb_x
                    else:
                        lat_mb_x = torch.cat((lat_mb_x, temp_lat_mb_x), 0)

                    # Concatenate the y minibatch from the replay buffer with the y minibatch from the current experience
                    self.mbatch[1] = torch.cat((self.mbatch[1], lat_mb_y), 0)
                    # Concatenate the task id minibatch from the replay buffer with the task id minibatch from the current experience
                    self.mbatch[2] = torch.cat((self.mbatch[2], lat_task_id), 0)
            else:
                lat_mb_x = None

            # Forward pass. Here we are injecting latent patterns lat_mb_x.
            # lat_mb_x will be None for the very first batch (batch 0)
            self.mb_output, lat_acts = self.model(self.mbatch[0], latent_input=lat_mb_x, return_lat_acts=True)

            if self.epoch == 0:
                # On the first epoch only: store latent activations. Those
                # activations will be used to update the replay buffer.
                lat_acts = lat_acts.detach().clone().cpu()
                self.mbatch_y = self.mbatch[1][:len(self.mbatch[0])].detach().clone().cpu()
                if mb_it == 0:
                    self.cur_acts = lat_acts
                    self.cur_acts_y = self.mbatch_y
                else:
                    self.cur_acts = torch.cat((self.cur_acts, lat_acts), 0)
                    self.cur_acts_y = torch.cat((self.cur_acts_y, self.mbatch_y), 0)

            # Loss & Backward
            self.loss = self.criterion(self.mb_output, self.mbatch[1])
            self.loss.backward()

            # Optimization step
            self.optimizer.step()

            # Compute avg loss and accuracy
            self.loss_accuracy(mb_it, dataloader)

            it += 1

    def loss_accuracy(self, mb_it, dataloader):

        self.train_loss += self.loss.item()
        self.prob = F.softmax(self.mb_output, dim=1)
        _, predicted = self.prob.max(1)
        self.total += self.mbatch[1].size(0)
        self.correct += predicted.eq(self.mbatch[1]).sum().item()
 
        if mb_it == len(dataloader)-1:
            self.acc = 100.0 * self.correct / self.total
            self.avg_loss = self.train_loss / len(self.mbatch)

    def update_mem_after_exp(self, exp):
        # Number of patterns to add to the random memory
        h = min(
            self.rm_size // (self.train_exp_counter + 1),
            self.cur_acts.size(0),
        )

        # Sample h random patterns from the current experience
        idxs_cur = torch.randperm(self.cur_acts.size(0))[:h]

        # Get the corresponding labels
        rm_add_y = torch.tensor([self.cur_acts_y[idx_cur].item() for idx_cur in idxs_cur])

        # Concatenate the latent activations and the labels
        rm_add = [self.cur_acts[idxs_cur], rm_add_y]
        """
        # replace patterns in random memory
        if self.train_exp_counter == 0:
            self.rm = dict()
            self.rm[exp.current_experience] = rm_add
        else:
            sum_length = 0
            for rm_exp in self.rm.values():
                # Compute the length of each list and add it to the sum_length
                sum_length += len(rm_exp[0])

            if (sum_length + len(rm_add[0])) <= self.rm_size:
                self.rm[exp.current_experience] = rm_add
            else:
                # Iterate over the dictionary items
                for key, value in self.rm.items():
                    perm = torch.randperm(value[0].size(0))
                    temp = value[0][perm]
                    value[0] = temp[0:h]
                    temp = value[1][perm]
                    value[1] = temp[0:h]
                self.rm[exp.current_experience] = rm_add
        """
        if self.train_exp_counter == 0:
            self.rm = {exp.current_experience: rm_add}
        else:
            sum_length = sum(len(rm_exp[0]) for rm_exp in self.rm.values())
            if sum_length + len(rm_add[0]) <= self.rm_size:
                self.rm[exp.current_experience] = rm_add
            else:
                for value in self.rm.values():
                    perm = torch.randperm(value[0].size(0))
                    value[0] = value[0][perm][0:h]
                    value[1] = value[1][perm][0:h]
                self.rm[exp.current_experience] = rm_add


        print("Size of the replay memory: ", self.get_dict_size())
        print("Size of the replay memory 2: ", total_size(self.rm), " MB")

        self.cur_acts = None
        self.cur_acts_y = None

    def get_dict_size(self):
        total_size = sys.getsizeof(self.rm)
        for value in self.rm.values():
            total_size += sys.getsizeof(value[0])
            total_size += sys.getsizeof(value[1])
        return total_size

    def test(self, dataset):
        """
        Testing loop.

        :param dataset: dataset to test on.

        """
        exps_acc, avg_acc = super().test(dataset)
            
        return exps_acc, avg_acc

    def _after_training_exp(self, exp):
        # Number of patterns to add to the random memory
        h = min(
            self.rm_size // (self.train_exp_counter + 1),
            self.cur_acts.size(0),
        )
        # Get the current experience
        curr_data = exp.dataset

        # Sample h random patterns from the current experience
        idxs_cur = torch.randperm(self.cur_acts.size(0))[:h]
        
        # Get the corresponding labels
        rm_add_y = torch.tensor(
            [curr_data.targets[idx_cur] for idx_cur in idxs_cur]
        )
        # Concatenate the latent activations and the labels
        rm_add = [self.cur_acts[idxs_cur], rm_add_y]
        
        #print("rm_add[0].shape: ", rm_add[0].shape)

        # replace patterns in random memory
        if self.train_exp_counter == 0:
            self.rm = rm_add
        else:
            # Sample h random patterns from memory to be removed
            idxs_2_replace = torch.randperm(self.rm[0].size(0))[:h]
            for j, idx in enumerate(idxs_2_replace):
                idx = int(idx)
                self.rm[0][idx] = rm_add[0][j]
                self.rm[1][idx] = rm_add[1][j]

        self.cur_acts = None