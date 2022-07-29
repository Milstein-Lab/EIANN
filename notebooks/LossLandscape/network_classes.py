import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display


class MNIST_net(nn.Module):
    def __init__(self, batch_size, dales_law=False):
        super().__init__()
        self.dales_law = dales_law
        self.batch_size = batch_size
        self.in_size = 28*28
        self.l1_size = 50
        self.l2_size = 500
        self.out_size = 10

        self.layer_1 = nn.Linear(self.in_size,self.l1_size,  bias=True)
        self.layer_2 = nn.Linear(self.l1_size,self.l2_size,  bias=True)
        self.layer_3 = nn.Linear(self.l2_size,self.out_size, bias=True)

        if self.dales_law==True:
            self.layer1.weight.detach().uniform_(0.01, 0.1)
            self.layer2.weight.detach().uniform_(0.01, 0.1)
            self.layer3.weight.detach().uniform_(0.01, 0.1)

    def forward(self, images):
        h1 = self.layer_1(images)
        h1 = F.sigmoid(h1)
        h2 = self.layer_2(h1)
        h2 = F.sigmoid(h2)
        out = self.layer_3(h2)
        return out

    def train(self, train_dataloader, test_dataloader, num_epochs, max_batch, lr, plot_dynamic_loss=False):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        num_batches = len(train_dataloader)
        self.loss_history = []
        self.w1 = []
        self.mean_weight_history1 = []
        self.mean_weight_history2 = []
        self.mean_weight_history3 = []
        self.weight_history1 = torch.zeros(num_epochs,num_batches+1,self.l1_size,self.in_size)
        self.weight_history2 = torch.zeros(num_epochs,num_batches+1,self.l2_size,self.l1_size)
        self.weight_history3 = torch.zeros(num_epochs,num_batches+1,self.out_size,self.l2_size)

        if plot_dynamic_loss:
            fig, ax = plt.subplots(1, 3, figsize=(14, 4))

        for epoch in range(num_epochs):

            # Save initial weights for this epoch
            self.mean_weight_history1.append(torch.mean(self.layer_1.weight.detach()))
            self.weight_history1[epoch, 0, :, :] = self.layer_1.weight.detach()
            self.mean_weight_history2.append(torch.mean(self.layer_2.weight.detach()))
            self.weight_history2[epoch, 0, :, :] = self.layer_2.weight.detach()
            self.mean_weight_history3.append(torch.mean(self.layer_3.weight.detach()))
            self.weight_history3[epoch, 0, :, :] = self.layer_3.weight.detach()

            for i,batch in enumerate(train_dataloader):
                images, labels = batch
                images = images.view(self.batch_size,self.in_size) # remove color channel + flatten image
                output = self.forward(images)
                loss = criterion(output,labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.dales_law == True:
                    self.layer1.weight.detach = self.in2out.weight.detach().clamp(min=0, max=None)
                    self.layer2.weight.detach = self.out2fbi.weight.detach().clamp(min=0, max=None)
                    self.layer3.weight.detach = self.fbi2out.weight.detach().clamp(min=None, max=0)

                self.loss_history.append(loss.detach())

                self.mean_weight_history1.append(torch.mean(self.layer_1.weight.detach()))
                self.weight_history1[epoch,i+1,:,:] = self.layer_1.weight.detach()
                self.mean_weight_history2.append(torch.mean(self.layer_2.weight.detach()))
                self.weight_history2[epoch,i+1,:,:] = self.layer_2.weight.detach()
                self.mean_weight_history3.append(torch.mean(self.layer_3.weight.detach()))
                self.weight_history3[epoch,i+1,:,:] = self.layer_3.weight.detach()

                if i==max_batch:
                    break

                if plot_dynamic_loss:
                    if i>0:
                        ax[0].plot([i-1,i],self.loss_history[i-1:i+1],color='k')
                        ax[0].set_xlabel('training steps')
                        ax[0].set_ylabel('loss')
                        ax[1].plot([i-1,i],self.mean_weight_history1[i-1:i+1],color='r')
                        ax[1].set_ylabel('mean weight l1')
                        ax[1].set_xlabel('training steps')
                        plt.tight_layout()
                        if i%20==0:
                            im = ax[2].imshow(self.layer_1.weight.detach(),aspect='auto')
                        display.clear_output(wait=True)
                        display.display(plt.gcf())

        if max_batch < num_batches: # Crop weight history if training was stopped early
            self.loss_history = torch.tensor(self.loss_history)
            self.weight_history1 = self.weight_history1[epoch,:i+1,:,:]
            self.weight_history2 = self.weight_history2[epoch,:i+1,:,:]
            self.weight_history3 = self.weight_history3[epoch,:i+1,:,:]

        # Compute accuracy on the test set
        images, classes = next(iter(test_dataloader))
        images = images.view(images.shape[0], self.in_size)  # remove color channel + flatten image
        output = F.softmax(self.forward(images).detach(),dim=1)
        self.test_accuracy = torch.sum(torch.argmax(output,dim=1) == classes) / images.shape[0]
        print(f'Final accuracy = {self.test_accuracy*100}%')

        return
