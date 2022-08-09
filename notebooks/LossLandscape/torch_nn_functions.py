#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display
from sklearn.decomposition import PCA
import sklearn.preprocessing as preprocessing


class MNIST_net(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.in_size = 28*28
        self.l1_size = 25
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
        self.criterion = nn.CrossEntropyLoss()

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
                loss = self.criterion(output,labels)

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

def get_line(v1, v2, num_p, num_extrap_p, plot_show=True):
    '''
    Takes two points in weight space and interpolates between them

    Parameters
    vec_1: 1D numpy array of weights (e.g. initial weights)
    vec_2: 1D numpy array of weights (e.g. final weights)
    num_p: (int) number of points to interpolate (including start/end points)
    num_extrap_p: (int) number of points to extrapolate on either side of end points

    Returns
    steps_arr: 2D numpy array (num points x weight vector size) of interpolated points
    '''

    total_points = num_p + num_extrap_p*2
    step = np.negative(np.divide(((v1-v2)),num_p))
    beginning = v1 - (step * num_extrap_p)

#   creating steps_arr of D2, size total number of points x dimention of given vectors
    steps_arr = np.zeros((total_points+1, v1.shape[0]))
        
    for i in range(total_points+1):
        if i==0:
            steps_arr[i] = beginning
        else:
            steps_arr[i,:] = steps_arr[i-1] + step
            
    steps_arr[num_extrap_p] = v1 
    steps_arr[num_extrap_p + num_p] = v2

    steps_arr_x = [x[0] for x in steps_arr]
    steps_arr_y = [x[1] for x in steps_arr]

    
    if plot_show==True:
        if v1.size == 2:
            #plotting points in 2D
            plt.scatter(steps_arr_x, steps_arr_y)
            plt.scatter(v1[0], v1[1], color='red')
            plt.scatter(v2[0], v2[1], color='red')
        else:
            #plotting points (first 3 coords) in 3D
            steps_arr_z = [x[2] for x in steps_arr]
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(projection='3d')
            ax.tick_params(axis='x', pad=5, which='major')
            ax.set_xlabel(r'x',  labelpad=10)
            ax.set_ylabel(r'y', labelpad=10)
            ax.set_zlabel(r'z', labelpad=10)
            ax.tick_params(axis='x', pad=5, which='major')
            ax.tick_params(axis='y', pad=5, which='major')
            ax.tick_params(axis='z', pad=5, which='major')
            ax.scatter(v1[0], v1[1], v1[2], color='purple')
            ax.scatter(v2[0], v2[1], v2[2], color='red')
            ax.scatter(beginning[0], beginning[1], beginning[2], color='green', marker='X')
            ax.scatter(steps_arr_x, steps_arr_y, steps_arr_z, marker='x')


            plt.show()

    return steps_arr

def get_distance(extrapolated_line):
    '''
    Function takes in points on the line in an N-sim spece and finds out the distance between the points

    Parameters
    extrapolated_line: np.array
        an array, where each entry is a coordinated to point

    Returns
    distance: list
        list of distances between two consecutive points in a given line
    '''
        # verify that difference between two adjacent points in extrapolated_line is always the same

    distance =  []
    for i in range(len(extrapolated_line)-1):
        point1 = extrapolated_line[i]
        point2 = extrapolated_line[i+1]
        difference = point1 - point2
        distance.append(np.linalg.norm(difference))
        
    return distance

def calculate_loss(model, weights_vec, dataloader_test):
    '''
    Function calculates the loss on the full MNIST dataset for one instance of the network
    In each instance, the full set of weights needs to be specified

    Parameters
    model: MNIST_net
    weights_vec: torch tensor, size: [1 x number of weights]
        flattened weights of a network (over all layers)
    dataloader_test: dataloader
        test samples from the data set used to calculate loss

    Returns
    loss: torch tensor (?)
        value of a loss given set of weights
    '''

    weights = torch.tensor(weights_vec).float()
    input_h1_list, h2_h1_list, output_h2_list = torch.split(weights, [model.in_size*model.l1_size,
                                                                      model.l1_size*model.l2_size,
                                                                      model.l2_size*model.out_size])

    # Specify new value for weights
    with torch.no_grad():
        model.layer_1.weight.data = torch.reshape(input_h1_list, (model.l1_size, model.in_size))
        model.layer_2.weight.data = torch.reshape(h2_h1_list, (model.l2_size, model.l1_size))
        model.layer_3.weight.data = torch.reshape(output_h2_list, (model.out_size, model.l2_size))

    # Compute loss on the test set
    images, labels = next(iter(dataloader_test))
    images = images.view(images.shape[0], model.in_size)  # remove color channel + flatten image
    output = model.forward(images).detach()
    loss = model.criterion(output, labels)

    return loss

def plot_w_PCs(weight_history):
    '''
    Function performs PCA on a given set of weights and
        1. plots the explained variance
        2. the trajectory of the weights in the PC space during the course of learning
        3. the loading scores of the weights

    Parameters
    weight_history: torch tensor, size: [time_steps x total number of weights]

    Returns
    '''

    w = weight_history
    w = preprocessing.scale(w) # center the data (mean=0, std=1)
    pca = PCA(n_components=5)
    pca.fit(w)
    w_pca = pca.transform(w)

    # Plot explained variance
    fig, ax = plt.subplots(1,3)
    explained_variance = pca.explained_variance_ratio_
    percent_exp_var = np.round(explained_variance*100,decimals=2)
    labels = ['PC' + str(x) for x in range(1,len(percent_exp_var)+1)]
    ax[0].bar(x=range(1,len(percent_exp_var)+1), height=percent_exp_var, tick_label=labels)
    ax[0].set_ylabel('Percentage of variance explained')
    ax[0].set_xlabel('Principal Component')
    ax[0].set_title('Scree Plot')

    # Plot weights in PC space
    PC1 = w_pca[:,0]
    PC2 = w_pca[:,1]
    ax[1].scatter(PC1,PC2)
    ax[1].scatter(PC1[0],PC2[0],color='blue',label='before training')
    ax[1].scatter(PC1[-1],PC2[-1],color='red',label='after training')
    ax[1].set_xlabel(f'PC1 - {percent_exp_var[0]}%')
    ax[1].set_ylabel(f'PC2 - {percent_exp_var[1]}%')
    ax[1].legend()
    ax[1].set_ylim([-250,280])
    ax[1].set_xlim([-250,280])


    # Plot loading scores for PC1 to determine which/how many weights are important for variance along PC1
    sorted_loadings = -np.sort(-np.abs(pca.components_[0])) # Loadings sorted in descending order of abs magnitude
    sorted_idx = np.argsort(-np.abs(pca.components_[0]))

    most_important_weights_flat = sorted_idx[0:10] #
    most_important_weights_idx = [] # index of important weights in original weight matrix

    ax[2].plot(sorted_loadings)
    ax[2].set_xlabel('sorted weights')
    ax[2].set_ylabel('Loading \n(alignment with weight)')
    ax[2].set_title('PC1 weight components')

    plt.tight_layout()
    plt.show()

def flatten_weight_hist(model):
    '''
    Function gets full set of weights from all leyers as a flatten array for each time point

    Parameters
    model: MNIST_net

    Returns
    w: torch tensor, size: [time points x number of weights in the whole network]
    '''

    w1_h = model.weight_history1.flatten(start_dim=1)
    w2_h = model.weight_history2.flatten(start_dim=1)
    w3_h = model.weight_history3.flatten(start_dim=1)
    w = torch.cat((w1_h, w2_h, w3_h), dim=1)
    return w

def get_loss_landscape(model, dataloader_test, num_points, returning=True):
    '''
    Function created loss grid and plots loss landscape

    Parameters
    model: MNIST_net
    dataloader_test: dataloader
        test set from MNIST data set used to calculate loss
    num_points: int
        one of the dimentions of the grid

    Returns
    loss_grid: torch tensor, size: [num_points x num_points]
        values of loss for the given model at a given set of weights
    '''

    w = flatten_weight_hist(model)

    w = preprocessing.scale(w) # center the data (mean=0, std=1)
    pca = PCA(n_components=2)
    pca.fit(w)
    w_pca = pca.transform(w)

    # create grid
    PC1 = w_pca[:,0]
    PC2 = w_pca[:,1]
    delta_PC1 = np.max(PC1) - np.min(PC1)
    delta_PC2 = np.max(PC2) - np.min(PC2)
    PC1_range = np.linspace(np.min(PC1) - delta_PC1, np.max(PC1) + delta_PC1, num_points)
    PC2_range = np.linspace(np.min(PC2) - delta_PC2, np.max(PC2) + delta_PC2, num_points)

    PC1_mesh, PC2_mesh = np.meshgrid(PC1_range, PC2_range)

    # Convert PC coordinates into full weight vectors
    flat_PC1_vals = PC1_mesh.reshape(1, num_points ** 2)
    flat_PC2_vals = PC2_mesh.reshape(1, num_points ** 2)
    meshgrid_points = np.concatenate([flat_PC1_vals, flat_PC2_vals]).T

    gridpoints_weightspace = pca.inverse_transform(meshgrid_points)

    #
    loss_list = []
    for i, point in enumerate(tqdm(gridpoints_weightspace)):
        loss_list.append(calculate_loss(model, point, dataloader_test))

    loss_grid = torch.tensor(loss_list)
    loss_grid = torch.reshape(loss_grid, PC1_mesh.shape)
    #
    plot_loss_surface(loss_grid, PC1_mesh, PC2_mesh)
    if returning == True:
        return loss_grid, PC1_mesh, PC2_mesh

def plot_loss_surface(loss_grid, PC1_mesh, PC2_mesh):
    '''
    Function plots loss surface from the grid based on PCs

    Parameters
    loss_grid: torch tensor, size: [num_points x num_points]
        values of loss for the given model at a given set of weights
    PC1_mesh: torch tensor, size: [1 x num_points(specified in get_loss_landscape] (?)
    PC2_mesh: torch tensor, size: [1 x num_points(specified in get_loss_landscape]

    Returns
    '''
    loss_grid_np = loss_grid.numpy()
    # plotting loss landscapes
    fig = plt.figure(figsize=(10, 7.5))
    ax0 = fig.add_subplot(projection='3d')

    fontsize_ = 20
    labelsize_ = 12
    ax0.view_init(elev=30, azim=-50)
    ax0.set_xlabel(r'PC1', fontsize=fontsize_, labelpad=9)
    ax0.set_ylabel(r'PC2', fontsize=fontsize_, labelpad=10)
    ax0.set_zlabel(r'costs', fontsize=fontsize_, labelpad=-30)
    ax0.tick_params(axis='x', pad=1, which='major', labelsize=labelsize_)
    ax0.tick_params(axis='y', pad=1, which='major', labelsize=labelsize_)
    ax0.tick_params(axis='z', pad=10, which='major', labelsize=labelsize_)
    ax0.set_title('')

    # plotting the surface of loss landscape
    ax0.plot_surface(PC1_mesh, PC2_mesh, loss_grid_np, cmap='terrain', alpha=0.75)

    plt.tight_layout()
    plt.show()

