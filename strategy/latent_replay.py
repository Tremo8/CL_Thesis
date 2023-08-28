import torch
import torch.nn.functional as F
from strategy.base_strategy import BaseStrategy
from torch.utils.data import DataLoader
from utility.memory_computation import total_size
from utility.CSVsave import save_results_to_csv
import utility.utils as utils
import sys
from torchinfo import summary
import numpy as np

class LatentReplay(BaseStrategy):
    """ Latent replay strategy. """
    def __init__(self, model, optimizer, criterion, train_epochs, train_mb_size = 21, replay_mb_size = 107,  eval_mb_size = 128, rm_size_MB = None, rm_size = None, manual_mb = True, split_ratio = 0, patience = 5, device = "cpu", file_name = None, path = None):
        """Init.

        It is necessary to specify one between rm_size and rm_size_MB, the other must be None. If both are None or both are not None, an error is raised.
        When rm_size_MB is specified, the replay memory is filled up to the specified size in MB. When rm_size is specified, the replay memory is filled up to 
        the specified size in number of elements. The function manages the two options automatically based on the specified parameter.

        Args:
            model: PyTorch model.
            optimizer: PyTorch optimizer.
            criterion: PyTorch criterion.
            train_mb_size: training mini-batch size.
            train_epochs: number of training epochs.
            replay_mb_size: replay mini-batch size.
            eval_mb_size: evaluation mini-batch size.
            rm_size_MB: size of the replay memory in MBytes.
            rm_size: size of the replay memory in number of elements.
            manual_mb: If True the mini-batch size should be manually setted. If False it computes the mini-batch size
            as `len(train_dataset) // ( ( len(train_dataset) + len(replay_buffer) ) // train_mb_size )` and the memory
            mini-batch size as `train_mb_size - replay_mb_size`.
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
            split_ratio=split_ratio,
            patience = patience,
            device=device,
            file_name = file_name,
            path = path
        )

        self.rm_size = rm_size
        """ Size of the replay memory in number of elements. """

        self.rm_size_MB = rm_size_MB
        """ Size of the replay memory in MB. """

        if rm_size_MB is None and rm_size is None:
            raise ValueError("One between rm_size and rm_size_MB must be not None")
        
        if rm_size_MB is not None and rm_size is not None:
            raise ValueError("Only one between rm_size and rm_size_MB must be not None")

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

        Args:
            dataset: dataset to train the model.
            manual_mb: If True the mini-batch size should be manually setted. If False it computes the mini-batch size
            as `len(train_dataset) // ( ( len(train_dataset) + len(replay_buffer) ) // train_mb_size )` and the memory
            mini-batch size as `train_mb_size - replay_mb_size`.
            shuffle: If True the data loader has to shuffle the dataset at each epoch.

        Returns:
            The data loader.
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
        Training loop over the experiences.

        Args:
            dataset: dataset containing the experiences.
            test_data: test dataset.
            plotting: If True, the training history is plotted at the end of the training process.
        """
        
        # Freeze the model up to the latent layers
        self.before_training_exp()
        timings = []
        print("Start of the training process...")
        for exp in dataset:
            print("Training of the experience with class: ", exp.classes_in_this_experience)

            # Test
            # Create the dataloader
            if self.split_ratio != 0:
                train_dataset, val_dataset = utils.split_dataset(exp.dataset, split_ratio=self.split_ratio)
                val_loader = DataLoader(val_dataset, batch_size=self.eval_mb_size, shuffle=True)
            else:
                train_dataset = exp.dataset
                val_loader = None
            
            # Data loader initialization
            # Called at the start of each learning experience
            train_loader = self.make_train_dataloader(train_dataset, manual_mb = self.manual_mb, shuffle=True)
           
            tot_exp_time = 0
            train_losses = []
            train_accuracies = []
            # Training loop over the current experience
            for self.epoch in range(self.train_epochs):

                # Initialize the timer
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                # Start Recording the time
                starter.record()

                # Training
                self.training_epoch(train_loader)

                # Stop Recording the time and compute the elapsed time
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)

                # Save some statics the be saved in output
                tot_exp_time += curr_time
                timings.append(curr_time)
                train_losses.append(self.avg_loss)
                train_accuracies.append(self.acc)

                print(f"Epoch: {self.epoch+1}/{self.train_epochs}, Train Loss: {self.avg_loss:.4f}, Train Accuracy: {self.acc:.2f}%, Training Time: {timings[-1]/1000:.3f} s")

                # If there is the validation dataset, early stopping is active
                if val_loader is not None:
                    early_stopped = super().validate_and_early_stop(val_loader)
                    if early_stopped:
                        if self.file_name is not None:
                            save_results_to_csv([["Stop Epoch"],[self.epoch+1]], self.file_name)
                        print("Early stopping")
                        break

            if self.file_name is not None:
                save_results_to_csv([["Trained Task "],[exp.task_label]], self.file_name)
                # Calculate the average train loss and accuracy
                avg_train_loss = sum(train_losses) / len(train_losses)
                avg_train_acc = sum(train_accuracies) / len(train_accuracies)
                save_results_to_csv([[f"Training Time", "Avg Training Loss", "Avg Training Acc"],[tot_exp_time/1000, avg_train_loss, avg_train_acc/100]], self.file_name)   

            # Reset the early stopping counter after each experience training loop
            if val_loader is not None:
                self.early_stopping.reset_counter()
            
            # Update the memory based on the MB size
            if self.rm_size_MB is not None:
                self.update_mem_after_exp_in_MB(exp)

            # Update the memory based on the number of elements
            if self.rm_size is not None:
                self.update_mem_after_exp(exp)
            
            self.train_exp_counter += 1
                
            if test_data is not None:
                print("")
                print("Test after the training of the experience with class: ", exp.classes_in_this_experience)

                exps_acc, _ = self.test(test_data)
                # Get the first self.train_exp_counter keys
                first_keys = list(exps_acc.keys())[:self.train_exp_counter]

                # Calculate the sum of the values corresponding to the first keys
                sum_of_first_values = sum(exps_acc[key] for key in first_keys)

                # Calculate the average
                average = sum_of_first_values / self.train_exp_counter
                print(f"Average accuracy of the encutered tasks: {average:.2f}%")

                self.update_tasks_acc(exps_acc)
            print("-----------------------------------------------------------------------------------")

        if self.file_name is not None:
            save_results_to_csv([["Total Training Time"],[sum(timings)/1000]], self.file_name)   

        if self.path is not None:
            torch.save(self.model.state_dict(), self.path)
            
        if plotting:
            #utils.plot_task_accuracy(self.tasks_acc, plot_task_acc=True, plot_avg_acc=True, plot_encountered_avg=True)
            plotter = utils.TaskAccuracyPlotter()
            _ = plotter.plot_task_accuracy(self.tasks_acc, plot_task_acc=True, plot_avg_acc=True, plot_encountered_avg=True)
            plotter.show_figures()
            
        print(f"Size of the replay memory: {self.get_dict_size() / 1048576 :.2f} MB")
        # Total element in the replay memory
        total_size =  sum([value[0].size(0) for value in self.rm.values()])
        print(f"Number of element in the replay memory: {total_size}")

        print("End of the training process.")
        print("")

        if self.file_name is not None:
            save_results_to_csv([["Memory MB", "Memory Elements"],[self.get_dict_size() / 1048576, total_size]], self.file_name)

    def _unpack_minibatch(self):
        """Move to device"""
        if isinstance(self.mbatch, tuple):
            self.mbatch = list(self.mbatch)
        for i in range(len(self.mbatch)):
            self.mbatch[i] = self.mbatch[i].to(self.device)
      
    def training_epoch(self, dataloader):
        """
        Training epoch over the current experience.

        Args:
            dataloader: data loader of the current experience.
        """
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
        """Compute loss and accuracy
        
        Args:
            mb_it: minibatch iteration
            dataloader: data loader of the current experience.
        """

        self.train_loss += self.loss.item()
        self.prob = F.softmax(self.mb_output, dim=1)
        _, predicted = self.prob.max(1)
        self.total += self.mbatch[1].size(0)
        self.correct += predicted.eq(self.mbatch[1]).sum().item()
 
        if mb_it == len(dataloader)-1:
            self.acc = 100.0 * self.correct / self.total
            self.avg_loss = self.train_loss / len(self.mbatch)

    def update_mem_after_exp(self, exp):
        """Update the random memory after the end of the current experience to keep a fixed number of element in the memory.

        Args:
            exp: current experience.
        """
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

        self.cur_acts = None
        self.cur_acts_y = None

    def update_mem_after_exp_in_MB(self, exp):
        """Update the random memory after the end of the current experience to keep a fixed number of MB in the memory.

        Args:
            exp: current experience.
        """
        # Size of an element of the random memory
        element_size = self.get_tensor_size(self.cur_acts[0]) + self.get_tensor_size(self.cur_acts_y[0])

        # Size of the random memory in bytes
        rm_size_B = self.rm_size_MB * 1024 * 1024

        b = min(
            rm_size_B // (self.train_exp_counter + 1), # Space available for each experience encountered so far
            self.cur_acts.size(0) * element_size, # Space needed for the current experience
        )
        
        if self.train_exp_counter == 0:
            # In the first experience the replay memory is empty. We add the current experience to the replay memory
            # up to the maximum size of the replay memory or the maximum size of the current experience if it is smaller
            # than the maximum size of the replay memory.

            # Number of elements that can be added to the replay memory
            e = b // element_size
        else:
            dict_B = self.get_dict_size()
            if dict_B + b <= rm_size_B:
                # If the replay memory is not full, we add the current experience to the replay memory without removing any element
                e = b // element_size
                remove_elements = False
            else:
                # If the replay memory is full, we add the current experience to the replay memory 
                # and remove some element from the replay memory to make room for the new elements

                # Number of elements for each experience in the replay memory
                e = (rm_size_B // (self.train_exp_counter + 1)) // element_size
                remove_elements = True

        # Sample e random patterns from the current experience
        idxs_cur = torch.randperm(self.cur_acts.size(0))[:e]

        # Get the corresponding labels
        rm_add_y = torch.tensor([self.cur_acts_y[idx_cur].item() for idx_cur in idxs_cur])

        # Concatenate the latent activations and the labels
        rm_add = [self.cur_acts[idxs_cur], rm_add_y]

        if self.train_exp_counter == 0:
            self.rm = {exp.current_experience: rm_add}
        elif not remove_elements:
                self.rm[exp.current_experience] = rm_add
        else:
            for value in self.rm.values():
                perm = torch.randperm(value[0].size(0))
                value[0] = value[0][perm][0:e]
                value[1] = value[1][perm][0:e]
            self.rm[exp.current_experience] = rm_add

        self.cur_acts = None
        self.cur_acts_y = None

    def get_tensor_size(self, tensor):
        """ Return the size of a tensor in bytes."""
        if isinstance(tensor, np.ndarray):
            return tensor.nbytes
        elif isinstance(tensor, torch.Tensor):
            return tensor.element_size() * tensor.numel()
        else:
            return sys.getsizeof(tensor)

    def get_dict_size(self):
        """ Return the size of the replay memory in bytes."""
        tot_size = total_size(self.rm)
        for value in self.rm.values():
            tot_size += self.get_tensor_size(value[0])
            tot_size += self.get_tensor_size(value[1])
        return tot_size

    def test(self, dataset):
        """Test the model on the given dataset.

        Args:
            dataset: dataset containing the test experiences.

        Returns:
            exps_acc: list of accuracies, one for each experience.
            avg_acc: average accuracy over all the experiences.
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