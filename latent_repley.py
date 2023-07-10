import torch
from torch.optim import Adam, SGD
import torch.nn.functional as F
from base_strategy import BaseStrategy
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
import numpy as np
from avalanche.models.batch_renorm import BatchRenorm2D
from avalanche.training.utils import replace_bn_with_brn


class LatentReplay(BaseStrategy):

    def __init__(self, model, optimizer, criterion, train_mb_size, replay_mb_size, train_epochs, eval_mb_size, rm_size, lr = 0.01, weight_decay = 0, device = "cpu"):
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

        self.train_exp_counter = 0
        """ Number of training experiences so far. """

        self.lr = lr
        """ Learning rate. """

        self.weight_decay = weight_decay
        """ Weight decay. """

        self.replay_mb_size = replay_mb_size

    def before_training_exp(self):
        replace_bn_with_brn(
            self.model
        )
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

    def make_train_dataloader(self, dataset, shuffle = True):
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

        :param shuffle: True if the data should be shuffled, False otherwise.
        """

        current_batch_mb_size = self.train_mb_size

        """
        if self.train_exp_counter > 0:
            train_patterns = len(dataset)
            current_batch_mb_size = train_patterns // (
                (train_patterns + self.rm_size) // self.train_mb_size
            )

        current_batch_mb_size = max(1, current_batch_mb_size)
        print("current_batch_mb_size: ", current_batch_mb_size)
        self.replay_mb_size = max(0, self.train_mb_size - current_batch_mb_size)
        print("replay_mb_size: ", self.replay_mb_size)
        """

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

        print("AFTER PhiNet: ")
        print(self.model)
        # set the optimizer
        #self.optimizer = SGD(self.model.parameters(), lr = self.lr, momentum = 0.9)
        self.optimizer = Adam(self.model.parameters(), lr = self.lr, weight_decay = self.weight_decay)

        print("Start of the training process...")
        for exp in dataset:
            print("Training of the experience with class: ", exp.classes_in_this_experience)

            # Test
            subset_indices = torch.randperm(len(exp.dataset))[:300]
            train_dataset_subset = torch.utils.data.Subset(exp.dataset, subset_indices)
            train_loader = self.make_train_dataloader(train_dataset_subset, shuffle=False)
            
            # Data loader initialization
            # Called at the start of each learning experience
            #train_loader = self.make_train_dataloader(exp.dataset, shuffle=True)
            
            for self.epoch in range(self.train_epochs):
                self.training_epoch(train_loader)
                print(f"Epoch: {self.epoch+1}/{self.train_epochs}, Train Loss: {self.avg_loss:.4f}, Train Accuracy: {self.acc:.2f}%")
            #self.test_on_example(dataset)
            self.update_mem_after_exp(exp)
            #self._after_training_exp(exp)
            self.train_exp_counter += 1
                
            print("")
            print("Test after the training of the experience with class: ", exp.classes_in_this_experience)
            self.test(dataset)
            print("-----------------------------------------------------------------------------------")

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
            #print("BEFORE mbatch[1]: ", self.mbatch[1])
            # After the first experience, we use the repley buffer
            if self.train_exp_counter > 0:
                lat_mb_x = None
                for key, value in self.rm.items():
                    #print("Start: ", mb_it * self.replay_mb_size//(len(self.rm)+1))
                    #print("End: ", (mb_it + 1) * self.replay_mb_size//(len(self.rm)+1))
                    # Sample different minibatch from the replay buffer at every iteration
                    if value[0].size(0) < ((it + 1 ) * self.replay_mb_size//(len(self.rm))):
                        it = 0

                    temp_lat_mb_x = value[0][it * self.replay_mb_size//(len(self.rm)) : (it + 1) * self.replay_mb_size//(len(self.rm))]
                    temp_lat_mb_x = temp_lat_mb_x.to(self.device)
                    
                    lat_mb_y = value[1][it * self.replay_mb_size//(len(self.rm)) : (it + 1) * self.replay_mb_size//(len(self.rm))]
                    lat_mb_y = lat_mb_y.to(self.device)
                    

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

            #print("AFTER mbatch[1]: ", self.mbatch[1])

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

            #print("self.output: ", self.mb_output.shape)
            #print("mbatch[1] out: ", len(self.mbatch[1]))

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
        #print("self.rm[0].shape: ", self.rm[0].shape)

        """
        if self.train_exp_counter == 0:
            self.rm = rm_add
        else:
            if (len(self.rm[0]) + len(rm_add[0])) > 1500:
                idxs_2_replace = torch.randperm(self.rm[0].size(0))[:h]
                for j, idx in enumerate(idxs_2_replace):
                    idx = int(idx)
                    self.rm[0][idx] = rm_add[0][j]
                    self.rm[1][idx] = rm_add[1][j]
            else:
                self.rm = [torch.cat((self.rm[0], rm_add[0]), dim=0),
                       torch.cat((self.rm[1], rm_add[1]), dim=0)] 
        permutation = torch.randperm(len(self.rm[0]))

        # Apply the permutation to both tensors
        self.rm[0] = self.rm[0][permutation]
        self.rm[1] = self.rm[1][permutation]  
        """
        self.cur_acts = None


    def update_mem_after_exp(self, exp):
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
        rm_add_y = torch.tensor([self.cur_acts_y[idx_cur].item() for idx_cur in idxs_cur])

        # Concatenate the latent activations and the labels
        rm_add = [self.cur_acts[idxs_cur], rm_add_y]

        # replace patterns in random memory
        if self.train_exp_counter == 0:
            self.rm = dict()
            self.rm[exp.current_experience] = rm_add
            #print("Dictionary after first: ", self.rm.keys())
        else:
            sum_length = 0
            for rm_exp in self.rm.values():
                # Compute the length of each list and add it to the sum_length
                sum_length += len(rm_exp[0])

            if (sum_length + len(rm_add[0])) <= self.rm_size:
                self.rm[exp.current_experience] = rm_add
                # Sample h random patterns from memory to be removed
            else:
                # Iterate over the dictionary items
                for key, value in self.rm.items():
                    perm = torch.randperm(value[0].size(0))
                    temp = value[0][perm]
                    value[0] = temp[0:h]
                    temp = value[1][perm]
                    value[1] = temp[0:h]
                self.rm[exp.current_experience] = rm_add

        self.cur_acts = None
        self.cur_acts_y = None

    def test(self, dataset):
        """
        Testing loop.
        """
        super().test(dataset)   


    def test_on_example(self, dataset):
        for exp in dataset:
            print("Start of task ", exp.task_label)
            print('Classes in this task:', exp.classes_in_this_experience)
            dataloader = torch.utils.data.DataLoader(exp.dataset, batch_size=1, shuffle=False)
            for index in range(0, 5):
                for i, (image,label,_) in enumerate(dataloader):
                    if i == index:
                        img, lab = image, label
                        break

                img = img.to(self.device)
                self.model.eval()
                with torch.no_grad():
                    output = self.model(img)
                #apply softmax
                output = torch.softmax(output, dim=1)
                #get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                prob = output[0][pred[0]].item()
                # plot images of the experiences in two rows
                fig = plt.figure(figsize=(10, 3))

                plt.imshow(np.asarray(img.squeeze()), cmap='gray')
                plt.title(f"Label: {lab.item()}\nPrediction: {pred.item()}\nProbability: {prob:.4f}")
                plt.axis('off')
                plt.xticks([])
                plt.show()
                plt.show()