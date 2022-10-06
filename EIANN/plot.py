import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from sklearn.decomposition import PCA
import math
from tqdm import tqdm
from copy import copy
from . import utils as utils


# Network summary functions
def plot_EIANN_activity(network, num_samples, supervised=True, label=None):

    reversed_layers = list(network)
    reversed_layers.reverse()
    output_pop = next(iter(reversed_layers[0]))

    if len(network.sorted_sample_indexes) > 0:
        sorted_sample_indexes = network.sorted_sample_indexes
    else:
        sorted_sample_indexes = torch.arange(0, output_pop.activity_history.shape[0])

    output = output_pop.activity_history[sorted_sample_indexes, -1, :][-num_samples:, :].T
    if not supervised and output.shape[0] == output.shape[1]:
        sorted_idx = utils.get_diag_argmax_row_indexes(output)
    else:
        sorted_idx = torch.arange(output.shape[0])

    if label is None:
        label_str = ''
    else:
        label_str = '%s ' % label

    max_rows = 1
    cols = len(network.layers) - 1
    for layer in network:
        projection_count = 0
        for population in layer:
            projection_count += len(list(population))
        max_rows = max(max_rows, projection_count)

    fig1, axes = plt.subplots(max_rows, cols, figsize=(3.*cols, 3.*max_rows))
    for i, layer in enumerate(network):
        if i > 0:
            col = i - 1
            row = 0
            for population in layer:
                for projection in population:
                    if cols == 1:
                        if max_rows == 1:
                            this_axis = axes
                        else:
                            this_axis = axes[row]
                    elif max_rows == 1:
                        this_axis = axes[col]
                    else:
                        this_axis = axes[row, col]
                    if projection.post == output_pop:
                        im = this_axis.imshow(projection.weight.data[sorted_idx, :], aspect='auto')
                    else:
                        im = this_axis.imshow(projection.weight.data, aspect='auto')
                    fig1.colorbar(im, ax=this_axis)
                    this_axis.set_xlabel('Pre unit ID')
                    this_axis.set_ylabel('Post unit ID')
                    this_axis.set_title('%s.%s <- %s.%s' %
                              (projection.post.layer.name, projection.post.name,
                               projection.pre.layer.name, projection.pre.name))
                    row += 1
            while row < max_rows:
                if cols == 1:
                    this_axis = axes[row]
                else:
                    this_axis = axes[row, col]
                this_axis.set_visible(False)
                row += 1
    fig1.suptitle('%sweights' % label_str)
    fig1.tight_layout()

    max_rows = 1
    cols = len(network.layers)
    for layer in network:
        max_rows = max(max_rows, len(layer.populations))

    fig2, axes = plt.subplots(max_rows, cols, figsize=(3. * cols, 3. * max_rows))
    for col, layer in enumerate(network):
        for row, population in enumerate(layer):
            if cols == 1:
                if max_rows == 1:
                    this_axis = axes
                else:
                    this_axis = axes[row]
            elif max_rows == 1:
                this_axis = axes[col]
            else:
                this_axis = axes[row, col]
            if population == output_pop:
                im = this_axis.imshow(population.activity_history[sorted_sample_indexes, -1, :][
                                      -num_samples:, sorted_idx].T, aspect='auto')
            else:
                im = this_axis.imshow(population.activity_history[sorted_sample_indexes, -1, :][
                                      -num_samples:, :].T, aspect='auto')
            fig2.colorbar(im, ax=this_axis)
            this_axis.set_xlabel('Input pattern ID')
            this_axis.set_ylabel('Unit ID')
            this_axis.set_title('%s.%s' % (layer.name, population.name))
        row += 1
        while row < max_rows:
            if cols == 1:
                this_axis = axes[row]
            else:
                this_axis = axes[row, col]
            this_axis.set_visible(False)
            row += 1
    fig2.suptitle('%sactivity' % label_str)
    fig2.tight_layout()

    cols = len(network.layers) - 1
    fig3, axes = plt.subplots(max_rows, cols, figsize=(3. * cols, 3. * max_rows))
    for i, layer in enumerate(network):
        if i > 0:
            col = i - 1
            for row, population in enumerate(layer):
                if cols == 1:
                    if max_rows == 1:
                        this_axis = axes
                    else:
                        this_axis = axes[row]
                elif max_rows == 1:
                    this_axis = axes[col]
                else:
                    this_axis = axes[row, col]
                for i in range(population.size):
                    this_axis.plot(torch.mean(population.activity_history[-num_samples:, :, i], axis=0))
                this_axis.set_xlabel('Equilibration time steps')
                this_axis.set_ylabel('Unit ID')
                this_axis.set_title('%s.%s' % (layer.name, population.name))
            row += 1
            while row < max_rows:
                if cols == 1:
                    this_axis = axes[row]
                else:
                    this_axis = axes[row, col]
                this_axis.set_visible(False)
                row += 1
    fig3.suptitle('%sactivity dynamics' % label_str)
    fig3.tight_layout()

    plt.show()

    print('%spopulation biases:' % label_str)
    for i, layer in enumerate(network):
        if i > 0:
            for population in layer:
                print(layer.name, population.name, population.bias)


def plot_performance(network):
    '''
    Plot loss and accuracy history from training
    '''
    fig = plt.figure()
    axes = gs.GridSpec(nrows=2, ncols=2,
                       left=0.05, right=0.98,
                       top=0.83, bottom=0.1,
                       wspace=0.3, hspace=0.5)

    ax = fig.add_subplot(axes[0, 0])
    ax.plot(network.loss_history)
    ax.set_ylabel('Loss')
    ax.set_xlabel('training steps')

    # ax = fig.add_subplot(axes[0, 1])
    # ax.plot(network.accuracy)
    # ax.set_ylabel('% correct argmax')
    # ax.set_xlabel('training steps /10')

    plt.show()


def plot_MNIST_examples(network, dataloader):
    """
    Test network performance on one example image from each MNIST class
    :param network:
    :param dataloader:
    """
    fig = plt.figure()
    axes = gs.GridSpec(nrows=1, ncols=10,
                       left=0.05, right=0.98,
                       wspace=0.5, hspace=0.5)

    # Compute accuracy on the test set
    indexes, images, targets = next(iter(dataloader))
    image_dim = int(math.sqrt(images.shape[1]))
    labels = torch.argmax(targets, axis=1)

    for i in range(10):
        ax = fig.add_subplot(axes[0, i])
        idx = torch.where(labels == i)[0][0]
        im = ax.imshow(images[idx].reshape((image_dim, image_dim)), cmap='Greys')
        ax.axis('off')
        output = network.forward(images[idx])
        if labels[idx] == torch.argmax(output):
            color = 'k'
        else:
            color = 'red'
        ax.text(0, 35, f'pred.={torch.argmax(output)}', color=color)
    plt.suptitle('Example images',y=0.7)
    plt.show()


# Loss landscape functions
def plot_weight_history_PCs(network):
    '''
    Function performs PCA on a given set of weights and
        1. plots the explained variance
        2. the trajectory of the weights in the PC space during the course of learning
        3. the loading scores of the weights

    Parameters
    weight_history: torch tensor, size: [time_steps x total number of weights]

    Returns
    '''
    flat_weight_history,_ = get_flat_weight_history(network)

    # Center the data (mean=0, std=1)
    w_mean = torch.mean(flat_weight_history, axis=0)
    w_std = torch.std(flat_weight_history, axis=0)
    flat_weight_history = (flat_weight_history - w_mean) / (w_std + 1e-10) # add epsilon to avoid NaNs

    pca = PCA(n_components=5)
    pca.fit(flat_weight_history)
    weights_pca_space = pca.transform(flat_weight_history)

    # Plot explained variance
    fig, ax = plt.subplots(1, 3)
    explained_variance = pca.explained_variance_ratio_
    percent_exp_var = np.round(explained_variance * 100, decimals=2)
    labels = ['PC' + str(x) for x in range(1, len(percent_exp_var) + 1)]
    ax[0].bar(x=range(1, len(percent_exp_var) + 1), height=percent_exp_var, tick_label=labels)
    ax[0].set_ylabel('Percentage of variance explained')
    ax[0].set_xlabel('Principal Component')
    ax[0].set_title('Scree Plot')

    # Plot weights in PC space
    PC1 = weights_pca_space[:, 0]
    PC2 = weights_pca_space[:, 1]
    ax[1].scatter(PC1, PC2)
    ax[1].scatter(PC1[0], PC2[0], color='blue', label='before training')
    ax[1].scatter(PC1[-1], PC2[-1], color='red', label='after training')
    ax[1].set_xlabel(f'PC1 - {percent_exp_var[0]}%')
    ax[1].set_ylabel(f'PC2 - {percent_exp_var[1]}%')
    ax[1].legend()

    # Plot loading scores for PC1 to determine which/how many weights are important for variance along PC1
    sorted_loadings = -np.sort(-np.abs(pca.components_[0]))  # Loadings sorted in descending order of abs magnitude
    sorted_idx = np.argsort(-np.abs(pca.components_[0]))

    most_important_weights_flat = sorted_idx[0:10]  #
    most_important_weights_idx = []  # index of important weights in original weight matrix

    ax[2].plot(sorted_loadings)
    ax[2].set_xlabel('sorted weights')
    ax[2].set_ylabel('Loading \n(alignment with weight)')
    ax[2].set_title('PC1 weight components')

    plt.tight_layout()
    plt.show()

def get_flat_weight_history(network):
    """
    Flattens all weights into a single vector for every point in the training history

    :param: network: EIANN_archive network with weights stored as Projection attributes
    :return: matrix of flattened weight vector for every point in the training hisotry
    :return: list containing tuples of (num weights, weight matrix shape)
    """
    flat_weight_history = []
    weight_sizes = []
    for layer in network:
        for population in layer:
            for projection in population:
                W_hist = projection.weight_history
                flat_weight_history.append(W_hist.flatten(1))
                weight_sizes.append([W_hist[0].numel(), W_hist[0].shape])

    flat_weight_history = torch.cat(flat_weight_history,dim=1)

    return flat_weight_history, weight_sizes

def flatten_weights(network):
    """
    Flattens all weights into a single vector.

    :param: network: EIANN_archive network with weights stored as Projection attributes
    :return: flattened weight vector (containing all network weights)
    :return: list containing tuples of (num weights, weight matrix shape)
    """

    flat_weights_ls = []
    weight_sizes = []

    for layer in network:
        for population in layer:
            for projection in population:
                W = projection.weight.detach()
                flat_weights_ls.append(W.flatten())
                weight_sizes.append([W.numel(), W.shape])

    flat_weights = torch.cat(flat_weights_ls)

    return flat_weights, weight_sizes

def unflatten_weights(flat_weights, weight_sizes):
    '''
    Convert flat weight vector to list of tensors (with correct dimensions)

    :param: flat_weights: flat vector of weights
    :param: weight_sizes: list containing tuples of (num weights, weight matrix shape)
    :return: list containing reshaped weight matrices
    '''
    weight_mat_ls = []
    idx_start = 0
    for length, shape in weight_sizes:
        w = flat_weights[idx_start:idx_start + length]
        weight_mat_ls.append(w.reshape(shape).type(torch.float32))
        idx_start += length

    return weight_mat_ls

def compute_loss(network, weight_mat_ls, test_dataloader):
    '''
    Calculate the loss for the network given a specified set of weights and test dataset

    :param: network: EIANN_archive network with weights stored as Projection attributes
    :param: weight_mat_ls: list containing desired weight matrices for the network
    :param: test_dataloader: dataloader with (data,target) to use for computing loss.
    :return: loss
    '''
    # Insert weight matrices into network
    i = 0
    for layer in network:
        for population in layer:
            for projection in population:
                with torch.no_grad():
                    projection.weight.data = weight_mat_ls[i]
                i += 1

    # Compute loss on dataset
    loss = 0
    for batch in test_dataloader:
        batch_data, batch_target = batch
        output = network.forward(batch_data).detach()
        loss += network.criterion(output, batch_target)
    loss /= len(test_dataloader)

    return loss

def plot_loss_landscape(network, test_dataloader, num_points=20, plot_line_loss=False):
    '''
    :param network:
    :return:
    '''
    flat_weight_history, weight_sizes = get_flat_weight_history(network)

    # Center the data (mean=0, std=1)
    w_mean = torch.mean(flat_weight_history, axis=0)
    w_std = torch.std(flat_weight_history, axis=0)
    centered_weight_history = (flat_weight_history - w_mean) / (w_std + 1e-10) # add epsilon to avoid NaNs

    # Get weights in gridplane defined by PC dimensions
    pca = PCA(n_components=2)
    pca.fit(centered_weight_history)
    weight_hist_pca_space = pca.transform(centered_weight_history)

    PC1 = weight_hist_pca_space[:, 0]
    PC2 = weight_hist_pca_space[:, 1]
    range_extension = 0.3 #proportion to sample further on each side of the grid
    PC1_extension = (np.max(PC1) - np.min(PC1)) * range_extension
    PC2_extension = (np.max(PC2) - np.min(PC2)) * range_extension
    PC1_range = np.linspace(np.min(PC1) - PC1_extension, np.max(PC1) + PC1_extension, num_points)
    PC2_range = np.linspace(np.min(PC2) - PC2_extension, np.max(PC2) + PC2_extension, num_points)
    PC1_mesh, PC2_mesh = np.meshgrid(PC1_range, PC2_range)

    # Convert PC coordinates into full weight vectors
    flat_PC1_vals = PC1_mesh.reshape(1, num_points**2)
    flat_PC2_vals = PC2_mesh.reshape(1, num_points**2)
    meshgrid_points = np.concatenate([flat_PC1_vals, flat_PC2_vals]).T

    gridpoints_weightspace = pca.inverse_transform(meshgrid_points)
    gridpoints_weightspace = torch.tensor(gridpoints_weightspace) * w_std + w_mean

    # Compute loss for points in grid
    losses = torch.zeros(num_points**2)
    for i,gridpoint_flat in enumerate(tqdm(gridpoints_weightspace)):
        weight_mat_ls = unflatten_weights(gridpoint_flat, weight_sizes)
        losses[i] = compute_loss(network, weight_mat_ls, test_dataloader)
    loss_grid = losses.reshape([PC1_range.size, PC2_range.size])

    plot_loss_surface(loss_grid, PC1_mesh, PC2_mesh)

    if plot_line_loss:
        # Compute loss for points in weight history
        loss_history = torch.zeros(flat_weight_history.shape[0])
        for i, flat_weights in enumerate(tqdm(flat_weight_history)):
            weight_mat_ls = unflatten_weights(flat_weights, weight_sizes)
            loss_history[i] = compute_loss(network, weight_mat_ls, test_dataloader)

        # Plot loss heatmap
        vmax = torch.max(torch.cat([loss_history, loss_grid.flatten()]))
        vmin = torch.min(torch.cat([loss_history, loss_grid.flatten()]))

        fig = plt.figure()
        im = plt.imshow(loss_grid, cmap='Reds', vmax=vmax, vmin=vmin,
                        extent=[np.min(PC1_range), np.max(PC1_range),
                                np.max(PC2_range), np.min(PC2_range)])
        plt.colorbar(im)
        plt.scatter(PC1, PC2, c=loss_history, cmap='Reds', edgecolors='k', linewidths=0., vmax=vmax, vmin=vmin)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.show()
    else:
        fig = plt.figure()
        im = plt.imshow(loss_grid, cmap='Reds',
                        extent=[np.min(PC1_range), np.max(PC1_range),
                                np.max(PC2_range), np.min(PC2_range)])
        plt.colorbar(im)
        plt.scatter(PC1, PC2, s=0.1, color='k')
        plt.scatter(PC1[0], PC2[0], s=80, color='b',edgecolor='k',label='Start')
        plt.scatter(PC1[-1], PC2[-1], s=80, color='orange',edgecolor='k',label='End')
        plt.legend()
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.show()

def plot_combined_loss_landscape(network1, network2, test_dataloader, num_points=20, plot_line_loss=False):

    # Combine weights to compute combined PCA
    flat_weight_history1, weight_sizes = get_flat_weight_history(network1)
    flat_weight_history2, weight_sizes = get_flat_weight_history(network2)
    flat_weight_history_combined = torch.cat([flat_weight_history1, flat_weight_history2], dim=0)

    # Center the data (mean=0, std=1)
    w_mean = torch.mean(flat_weight_history_combined, axis=0)
    w_std = torch.std(flat_weight_history_combined, axis=0)
    centered_weight_history = (flat_weight_history_combined - w_mean) / (w_std + 1e-10)  # add epsilon to avoid NaNs

    # Get weights in gridplane defined by PC dimensions
    pca = PCA(n_components=2)
    pca.fit(centered_weight_history)
    weight_hist_pca_space = pca.transform(centered_weight_history)

    PC1 = weight_hist_pca_space[:, 0]
    PC2 = weight_hist_pca_space[:, 1]
    range_extension = 0.3  # proportion to sample further on each side of the grid
    PC1_extension = (np.max(PC1) - np.min(PC1)) * range_extension
    PC2_extension = (np.max(PC2) - np.min(PC2)) * range_extension
    PC1_range = np.linspace(np.min(PC1) - PC1_extension, np.max(PC1) + PC1_extension, num_points)
    PC2_range = np.linspace(np.min(PC2) - PC2_extension, np.max(PC2) + PC2_extension, num_points)
    PC1_mesh, PC2_mesh = np.meshgrid(PC1_range, PC2_range)

    # Convert PC coordinates into full weight vectors
    flat_PC1_vals = PC1_mesh.reshape(1, num_points ** 2)
    flat_PC2_vals = PC2_mesh.reshape(1, num_points ** 2)
    meshgrid_points = np.concatenate([flat_PC1_vals, flat_PC2_vals]).T

    gridpoints_weightspace = pca.inverse_transform(meshgrid_points)
    gridpoints_weightspace = torch.tensor(gridpoints_weightspace) * w_std + w_mean

    # Compute loss for points in grid
    test_network = copy(network1) # create copy to avoid modifying original networks
    losses = torch.zeros(num_points ** 2)
    for i, gridpoint_flat in enumerate(tqdm(gridpoints_weightspace)):
        weight_mat_ls = unflatten_weights(gridpoint_flat, weight_sizes)
        losses[i] = compute_loss(test_network, weight_mat_ls, test_dataloader)
    loss_grid = losses.reshape([PC1_range.size, PC2_range.size])

    plot_loss_surface(loss_grid, PC1_mesh, PC2_mesh)

    PC1_network1 = PC1[0:flat_weight_history1.shape[0]]
    PC2_network1 = PC2[0:flat_weight_history1.shape[0]]

    PC1_network2 = PC1[flat_weight_history1.shape[0]:]
    PC2_network2 = PC2[flat_weight_history1.shape[0]:]

    if plot_line_loss:
        # Compute loss for points in weight history
        loss_history = torch.zeros(flat_weight_history.shape[0])
        for i, flat_weights in enumerate(tqdm(flat_weight_history)):
            weight_mat_ls = unflatten_weights(flat_weights, weight_sizes)
            loss_history[i] = compute_loss(test_network, weight_mat_ls, test_dataloader)

        # Plot loss heatmap
        vmax = torch.max(torch.cat([loss_history, loss_grid.flatten()]))
        vmin = torch.min(torch.cat([loss_history, loss_grid.flatten()]))

        fig = plt.figure()
        im = plt.imshow(loss_grid, cmap='Reds', vmax=vmax, vmin=vmin,
                        extent=[np.min(PC1_range), np.max(PC1_range),
                                np.max(PC2_range), np.min(PC2_range)])
        plt.colorbar(im)
        plt.scatter(PC1, PC2, c=loss_history, cmap='Reds', edgecolors='k', linewidths=0., vmax=vmax, vmin=vmin)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.show()
    else:
        fig = plt.figure()
        im = plt.imshow(loss_grid, cmap='Reds',
                        extent=[np.min(PC1_range), np.max(PC1_range),
                                np.max(PC2_range), np.min(PC2_range)])
        plt.colorbar(im)
        plt.scatter(PC1, PC2, s=0.1, color='k')

        plt.scatter(PC1_network1[0], PC2_network1[0], s=80, color='b', edgecolor='k', label='Start')
        plt.scatter(PC1_network2[0], PC2_network2[0], s=80, color='b', edgecolor='k')

        plt.scatter(PC1_network1[-1], PC2_network1[-1], s=80, color='orange', edgecolor='k', label='End')
        plt.scatter(PC1_network2[-1], PC2_network2[-1], s=80, color='orange', edgecolor='k')

        plt.legend()
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.show()

def plot_loss_surface(loss_grid, PC1_mesh, PC2_mesh):
    '''
    Function plots loss surface from the grid based on PCs

    Parameters
    loss_grid: torch tensor, size: [num_points x num_points]
        values of loss for the given model at a given set of weights
    PC1_mesh: torch tensor, size: [1 x num_points(specified in get_loss_landscape] (?)
    PC2_mesh: torch tensor, size: [1 x num_points(specified in get_loss_landscape]
    '''
    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(projection='3d')

    ax.plot_surface(PC1_mesh, PC2_mesh, loss_grid.numpy(), cmap='terrain', alpha=0.9)

    fontsize = 20
    ax.view_init(elev=30, azim=-50)
    ax.set_xlabel(r'PC1', fontsize=fontsize, labelpad=9)
    ax.set_ylabel(r'PC2', fontsize=fontsize, labelpad=10)
    ax.set_zlabel(r'Loss', fontsize=fontsize, labelpad=25)
    ax.tick_params(axis='x', pad=0)
    ax.tick_params(axis='y', pad=0)
    ax.tick_params(axis='z', pad=10)

    plt.tight_layout()
    plt.show()

def update_plot_defaults():
    plt.rcParams.update({'font.size': 15,
                     'axes.spines.right': False,
                     'axes.spines.top': False,
                     'axes.linewidth':1.2,
                     'xtick.major.size': 6,
                     'xtick.major.width': 1.2,
                     'ytick.major.size': 6,
                     'ytick.major.width': 1.2,
                     'legend.frameon': False,
                     'legend.handletextpad': 0.1,
                     'figure.figsize': [14.0, 4.0],})
