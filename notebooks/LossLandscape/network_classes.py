import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display


class MNIST_net(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.in_size = 28*28
        self.l1_size = 500
        self.l2_size = 50
        self.out_size = 10

        self.layer_1 = nn.Linear(self.in_size,self.l1_size,  bias=True)
        self.layer_2 = nn.Linear(self.l1_size,self.l2_size,  bias=True)
        self.layer_3 = nn.Linear(self.l2_size,self.out_size, bias=True)

    def forward(self, images):
        h1 = self.layer_1(images)
        h1 = F.relu(h1)
        h2 = self.layer_2(h1)
        h2 = F.relu(h2)
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
        self.weight_history1 = torch.zeros(num_epochs,num_batches,self.l1_size,self.in_size)
        self.weight_history2 = torch.zeros(num_epochs,num_batches,self.l2_size,self.l1_size)
        self.weight_history3 = torch.zeros(num_epochs,num_batches,self.out_size,self.l2_size)

        if plot_dynamic_loss:
            fig, ax = plt.subplots(1, 3, figsize=(14, 4))

        for epoch in range(num_epochs):
            for i,batch in enumerate(train_dataloader):
                images, labels = batch
                images = images.view(self.batch_size,self.in_size) # remove color channel + flatten image
                output = self.forward(images)
                loss = criterion(output,labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.loss_history.append(loss.detach())

                self.mean_weight_history1.append(torch.mean(self.layer_1.weight.detach()))
                self.weight_history1[epoch,i,:,:] = self.layer_1.weight.detach()
                self.mean_weight_history2.append(torch.mean(self.layer_2.weight.detach()))
                self.weight_history2[epoch,i,:,:] = self.layer_2.weight.detach()
                self.mean_weight_history3.append(torch.mean(self.layer_3.weight.detach()))
                self.weight_history3[epoch,i,:,:] = self.layer_3.weight.detach()

                if i==max_batch:
                    break

                if plot_dynamic_loss:
                    if i>0:
                        ax[0].plot([i-1,i],self.loss_history[i-1:i+1],color='k')
                        ax[0].set_xlabel('training steps')
                        ax[0].set_ylabel('loss')
                        ax[1].plot([i-1,i],self.mean_weight_history1[i-1:i+1],color='r')
                        ax[1].set_ylabel('mean weight l1')
                        sns.despine()
                        if i%20==0:
                            im = ax[2].imshow(self.layer_1.weight.detach(),aspect='auto')
                        display.clear_output(wait=True)
                        display.display(plt.gcf())

        self.loss_history = torch.tensor(self.loss_history)
        self.weight_history1 = self.weight_history1[epoch,0:i,:,:]
        self.weight_history2 = self.weight_history2[epoch,0:i,:,:]
        self.weight_history3 = self.weight_history3[epoch,0:i,:,:]


        images, classes = next(iter(test_dataloader))
        images = images.view(images.shape[0], self.in_size)  # remove color channel + flatten image
        output = F.softmax(self.forward(images).detach(),dim=1)
        self.test_accuracy = torch.sum(torch.argmax(output,dim=1) == classes) / images.shape[0]
        print(f'Final accuracy = {self.test_accuracy*100}%')


        return
