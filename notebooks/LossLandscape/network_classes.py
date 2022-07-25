import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

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
        h1 = nn.ReLU(h1)
        h2 = self.layer_2(h1)
        h2 = nn.ReLU(h2)
        out = self.layer_3(h2)
        return out


    def train(self, dataloader, num_epochs, lr):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        num_batches = len(dataloader)
        loss_history = []
        for epoch in range(num_epochs):
            for batch in tqdm(dataloader):
                images, labels = batch
                images = images.view(self.batch_size,self.in_size) # remove color channel
                print(images[0])
                output = self.forward(images)
                loss = criterion(output,labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_history.append(loss.detach())

        loss_history = torch.tensor(loss_history)
        return loss_history
