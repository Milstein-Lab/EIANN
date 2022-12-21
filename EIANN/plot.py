import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from sklearn.decomposition import PCA
import math
from tqdm.autonotebook import tqdm
from copy import copy
from . import utils as utils


def update_plot_defaults():
    plt.rcParams.update({'font.size': 12,
                     'axes.spines.right': False,
                     'axes.spines.top': False,
                     'axes.linewidth':1.2,
                     'xtick.major.size': 6,
                     'xtick.major.width': 1.2,
                     'ytick.major.size': 6,
                     'ytick.major.width': 1.2,
                     'legend.frameon': False,
                     'legend.handletextpad': 0.1,
                     'figure.figsize': [10.0, 3.0],
                     'svg.fonttype': 'none',
                     'text.usetex': False})


# *******************************************************************
# Network summary functions
# *******************************************************************
def plot_simple_EIANN_config_summary(network, num_samples, start_index=None, sorted_output_idx=None, label=None):
    """

    :param network:
    :param num_samples: int
    :param start_index: int
    :parm sorted_output_idx: tensor of int
    :param label: str
    """
    output_layer = list(network)[-1]
    output_pop = next(iter(output_layer))

    if sorted_output_idx is None:
        sorted_output_idx = torch.arange(0, output_pop.size)

    if len(network.sorted_sample_indexes) > 0:
        sorted_sample_indexes = network.sorted_sample_indexes
    else:
        sorted_sample_indexes = torch.arange(0, output_pop.activity_history.shape[0])

    if start_index is None:
        end_index = len(sorted_sample_indexes)
        start_index = end_index - num_samples
    else:
        end_index = start_index + num_samples

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

    fig1, axes = plt.subplots(max_rows, cols, figsize=(3.2*cols, 3.*max_rows))
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
                        im = this_axis.imshow(projection.weight.data[sorted_output_idx, :], aspect='auto')
                    elif projection.pre == output_pop:
                        im = this_axis.imshow(projection.weight.data[:, sorted_output_idx], aspect='auto')
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
    fig1.show()

    max_rows = 1
    cols = len(network.layers)
    for layer in network:
        max_rows = max(max_rows, len(layer.populations))

    fig2, axes = plt.subplots(max_rows, cols, figsize=(3.2 * cols, 3. * max_rows))
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
                                      start_index:end_index, sorted_output_idx].T, aspect='auto')
            else:
                im = this_axis.imshow(population.activity_history[sorted_sample_indexes, -1, :][
                                      start_index:end_index, :].T, aspect='auto')
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
    fig2.show()

    cols = len(network.layers) - 1
    fig3, axes = plt.subplots(max_rows, cols, figsize=(3.2 * cols, 3. * max_rows))
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
                    this_axis.plot(torch.mean(population.activity_history[start_index:end_index, :, i], axis=0))
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
    fig3.show()

    print('%spopulation biases:' % label_str)
    for i, layer in enumerate(network):
        if i > 0:
            for population in layer:
                print(layer.name, population.name, population.bias)


def plot_train_loss_history(network):
    """
    Plot loss and accuracy history from training
    :param network:
    """
    # network.loss_history = network.loss_history.cpu()
    fig = plt.figure()
    axes = gs.GridSpec(nrows=1, ncols=1,
                       left=0.05, right=0.98,
                       top=0.83, bottom=0.1,
                       wspace=0.3, hspace=0.5)

    ax = fig.add_subplot(axes[0, 0])
    ax.plot(network.loss_history)
    ax.set_ylabel('Train loss')
    ax.set_xlabel('Training steps')
    fig.suptitle('Train loss')
    fig.tight_layout()
    fig.show()


def plot_validate_loss_history(network):
    """
    Assumes network has been trained and a val_loss_history has been stored.
    :param network:
    """
    assert len(network.val_loss_history) > 0, 'Network must contain a stored val_loss_history'
    fig = plt.figure()
    plt.plot(network.val_history_train_steps, network.val_loss_history)
    plt.xlabel('Training steps')
    plt.ylabel('Validation loss')
    fig.suptitle('Validation loss')
    fig.tight_layout()
    fig.show()


def evaluate_test_loss_history(network, test_dataloader, store_history=False, plot=False):
    """
    Assumes network has been trained with store_weights=True. Evaluates test_loss at each train step in the
    param_history.
    :param network:
    :param test_dataloader:
    :param store_history:
    :param plot: bool
    """
    assert len(test_dataloader)==1, 'Dataloader must have a single large batch'
    assert len(network.param_history) > 0, 'Network must contain a stored param_history'

    idx, test_data, test_target = next(iter(test_dataloader))
    test_data = test_data.to(network.device)
    test_target = test_target.to(network.device)
    test_loss_history = []

    for state_dict in network.param_history:
        network.load_state_dict(state_dict)
        output = network.forward(test_data, store_history=store_history)

        test_loss_history.append(network.criterion(output, test_target).detach())

    network.test_loss_history = torch.stack(test_loss_history).cpu()

    fig = plt.figure()
    plt.plot(network.param_history_train_steps, network.test_loss_history)
    plt.xlabel('Training steps')
    plt.ylabel('Test loss')
    fig.suptitle('Test loss')
    fig.tight_layout()
    fig.show()


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

    for label in range(10):
        ax = fig.add_subplot(axes[0, label])
        idx = torch.where(labels == label)[0][0]
        im = ax.imshow(images[idx].reshape((image_dim, image_dim)), cmap='Greys')
        ax.axis('off')
        output = network.forward(images[idx])
        if labels[idx] == torch.argmax(output):
            color = 'k'
        else:
            color = 'red'
        ax.text(0, 35, f'pred.={torch.argmax(output)}', color=color)
    plt.suptitle('Example images',y=0.7)
    fig.show()


def plot_network_dynamics(network):
    '''
    Plots activity dynamics for every population in the network
    '''
    rows = len(network.layers)
    cols = np.max([len(layer.populations) for layer in network])

    fig = plt.figure(figsize=(8, 6))
    axes = gs.GridSpec(nrows=rows, ncols=cols,
                       left=0.05, right=0.98,
                       top=0.83, bottom=0.1,
                       wspace=0.3, hspace=0.8)

    for row, layer in enumerate(network):
        for col, population in enumerate(layer):
            ax = fig.add_subplot(axes[row, col])
            # average_activity_dynamics = torch.mean(population.activity_history, dim=0)
            average_activity_dynamics = torch.mean(population.activity_history[-21:], dim=(0,2))
            ax.plot(average_activity_dynamics)
            ax.set_title(f'{population.fullname} dynamics')


def plot_sparsity_history(network):
    rows = len(network.layers)
    cols = np.max([len(layer.populations) for layer in network])

    fig = plt.figure(figsize=(8, 6))
    axes = gs.GridSpec(nrows=rows, ncols=cols,
                       left=0.05, right=0.98,
                       top=0.83, bottom=0.1,
                       wspace=0.3, hspace=0.8)

    for row, layer in enumerate(network):
        for col, population in enumerate(layer):
            if 'Dend' not in population.name:
                ax = fig.add_subplot(axes[row, col])
                sparsity = torch.mean(population.sparsity_history[:,-1,:],dim=1)
                ax.plot(sparsity)
                ax.set_ylim(bottom=0.4,top=1)
                ax.set_title(f'{population.fullname} sparsity during training')


def plot_simple_EIANN_weight_history_diagnostic(network):
    fig, axes = plt.subplots(5, sharex=True)
    axes[0].plot(network.loss_history)
    axes[0].set_title('Loss')
    axes[1].plot(torch.mean(network.Output.E.H1.E.weight_history[:, :, 0], axis=1))
    axes[1].set_title('Output.E.H1.E')
    axes[2].plot(torch.mean(network.H1.E.Output.E.weight_history[:, :, 0], axis=1))
    axes[2].set_title('H1.E.Output.E')
    axes[3].plot(torch.mean(network.H1.E.Input.E.weight_history[:, :, 0], axis=1))
    axes[3].set_title('H1.E.Input.E')
    axes[4].plot(torch.mean(network.H1.E.H1.Dend_I.weight_history[:, :, 0], axis=1))
    axes[4].set_title('H1.E.H1.Dend_I')
    fig.show()


def plot_hidden_weights(weights):
    num_rows = weights.shape[0]
    num_cols = int(num_rows ** 0.5)  # make the number of rows and columns approximately equal

    axes = gs.GridSpec(num_rows, num_cols)
    fig = plt.figure(figsize=(12, 12 * num_rows / num_cols))

    for i, unit_weight_vec in enumerate(weights):
        # Calculate the row and column indices for the current subplot
        row_idx = i // num_cols
        col_idx = i % num_cols

        img_dim = int(unit_weight_vec.shape[0] ** 0.5)
        img = unit_weight_vec.view(img_dim, img_dim)

        # Add a subplot to the figure at the specified row and column
        ax = fig.add_subplot(axes[row_idx, col_idx])
        ax.imshow(img)
        ax.axis('off')

    fig.tight_layout(pad=0.2)


# *******************************************************************
# Loss landscape functions
# *******************************************************************
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
    # fig.show()


def plot_param_history_PCs(flat_param_history):
    '''
    Function performs PCA on a given set of parameters (drawn from the network state_dict()) and
        1. plots the explained variance
        2. the trajectory of the weights in the PC space during the course of learning
        3. the loading scores of the weights

    Parameters
    flat_param_history: torch tensor, size: [time_steps x total number of parameters]

    Returns
    '''
    # Center the data (mean=0, std=1)
    p_mean = torch.mean(flat_param_history, axis=0)
    p_std = torch.std(flat_param_history, axis=0)
    flat_param_history = (flat_param_history - p_mean) / (p_std + 1e-10)  # add epsilon to avoid NaNs

    pca = PCA(n_components=5)
    pca.fit(flat_param_history)
    params_pca_space = pca.transform(flat_param_history)

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
    PC1 = params_pca_space[:, 0]
    PC2 = params_pca_space[:, 1]
    ax[1].scatter(PC1, PC2)
    ax[1].scatter(PC1[0], PC2[0], color='blue', label='before training')
    ax[1].scatter(PC1[-1], PC2[-1], color='red', label='after training')
    ax[1].set_xlabel(f'PC1 - {percent_exp_var[0]}%')
    ax[1].set_ylabel(f'PC2 - {percent_exp_var[1]}%')
    ax[1].legend()

    # Plot loading scores for PC1 to determine which/how many weights are important for variance along PC1
    sorted_loadings1 = -np.sort(-np.abs(pca.components_[0]))  # Loadings sorted in descending order of abs magnitude
    sorted_idx = np.argsort(-np.abs(pca.components_[0]))

    most_important_weights_flat = sorted_idx[0:10]  #
    most_important_weights_idx = []  # index of important weights in original weight matrix

    ax[2].plot(sorted_loadings1)
    ax[2].set_xlabel('sorted weights')
    ax[2].set_ylabel('Loading \n(alignment with weight)')
    ax[2].set_title('PC1 weight components')

    plt.tight_layout()

    return fig, sorted_loadings1


def get_flat_param_history(param_history):
    """
    Flattens all parameters (weights, biases, and other registered paramters)
    into a single vector for every point in the training history

    :param: network: EIANN network
    :return: matrix of flattened parameter vectors for every point in the training hisotry
    :return: list containing tuples of (num weights, weight matrix shape)
    """

    flat_param_history = []
    param_metadata = {}

    for i, state_dict in enumerate(param_history):
        param_vector = []
        for name, param in state_dict.items():
            param_vector.append(param.flatten())
            if i == 0:
                param_metadata[name] = (param.numel(), param.shape)
        flat_param_history.append(torch.cat(param_vector))
    flat_param_history = torch.stack(flat_param_history)

    return flat_param_history, param_metadata


def get_flat_weight_history(network):
    """
    Flattens all weights into a single vector for every point in the training history

    :param: network: EIANN network with weights stored as Projection attributes
    :return: tuple: matrix of flattened weight vectors for every point in the training hisotry
                    list containing tuples of (num weights, weight matrix shape)
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


def unflatten_params(flat_params, param_metadata):
    '''
    Convert flat vector of parameters to list of tensors (with correct dimensions)

    :param: flat_params: flat vector of parameters
    :param: param_metadata: list containing tuples of (num params, param dimensions)
    :return: list containing reshaped param matrices
    '''
    state_dict = {}
    idx_start = 0
    for param_name in param_metadata:
        length, shape = param_metadata[param_name]
        p = flat_params[idx_start:idx_start + length]
        state_dict[param_name] = p.reshape(shape).type(torch.float32)
        idx_start += length

    return state_dict


def compute_loss(network, state_dict, test_dataloader):
    '''
    Calculate the loss for the network given a specified set of parameters and test dataset

    :param: network: EIANN_archive network
    :param: state_dict: network state dict containing desired parameters to test
    :param: test_dataloader: dataloader with (data,target) to use for computing loss.
    :return: loss
    '''
    # Insert weight matrices into network
    network.load_state_dict(state_dict)

    # Compute loss on dataset
    loss = 0
    for batch in test_dataloader:
        idx, batch_data, batch_target = batch
        output = network.forward(batch_data).detach()
        loss += network.criterion(output, batch_target)
    loss /= len(test_dataloader)

    return loss


def plot_loss_landscape(test_dataloader, network1, network2=None, num_points=20, extension=0.2, vmax=1.2, plot_line_loss=False):
    '''
    :param test_dataloader:
    :param network1:
    :param network2:

    :return:
    '''
    flat_param_history, param_metadata = get_flat_param_history(network1.param_history)
    history_len1 = flat_param_history.shape[0]

    if network2 is not None:
        flat_param_history2, param_metadata2 = get_flat_param_history(network2.param_history)
        flat_param_history = torch.cat([flat_param_history, flat_param_history2])

        assert param_metadata == param_metadata2, "Network architecture must match"

    # Center the data (mean=0, std=1)
    p_mean = torch.mean(flat_param_history, axis=0)
    p_std = torch.std(flat_param_history, axis=0)
    centered_param_history = (flat_param_history - p_mean) / (p_std + 1e-10)  # add epsilon to avoid NaNs

    # Get weights in gridplane defined by PC dimensions
    pca = PCA(n_components=2)
    pca.fit(centered_param_history)
    param_hist_pca_space = pca.transform(centered_param_history)

    explained_variance = pca.explained_variance_ratio_
    percent_exp_var = np.round(explained_variance * 100, decimals=2)

    PC1 = param_hist_pca_space[:, 0]
    PC2 = param_hist_pca_space[:, 1]
    PC1_extension = (np.max(PC1) - np.min(PC1)) * extension
    PC2_extension = (np.max(PC2) - np.min(PC2)) * extension
    PC1_range = np.linspace(np.min(PC1) - PC1_extension, np.max(PC1) + PC1_extension, num_points)
    PC2_range = np.linspace(np.min(PC2) - PC2_extension, np.max(PC2) + PC2_extension, num_points)
    PC1_mesh, PC2_mesh = np.meshgrid(PC1_range, PC2_range)

    # Convert PC coordinates into full weight vectors
    flat_PC1_vals = PC1_mesh.reshape(1, num_points ** 2)
    flat_PC2_vals = PC2_mesh.reshape(1, num_points ** 2)
    meshgrid_points = np.concatenate([flat_PC1_vals, flat_PC2_vals]).T

    gridpoints_paramspace = pca.inverse_transform(meshgrid_points)
    gridpoints_paramspace = torch.tensor(gridpoints_paramspace) * p_std + p_mean

    # Compute loss for points in grid
    test_network = copy(network1)  # create copy to avoid modifying original networks
    losses = torch.zeros(num_points ** 2)
    for i, gridpoint_flat in enumerate(tqdm(gridpoints_paramspace)):
        state_dict = unflatten_params(gridpoint_flat, param_metadata)
        losses[i] = compute_loss(test_network, state_dict, test_dataloader)
    loss_grid = losses.reshape([PC1_range.size, PC2_range.size])

    plot_loss_surface(loss_grid, PC1_mesh, PC2_mesh)

    if network2 is not None:
        PC1_network1 = PC1[0:history_len1]
        PC2_network1 = PC2[0:history_len1]

        PC1_network2 = PC1[history_len1:]
        PC2_network2 = PC2[history_len1:]

    if plot_line_loss:
        # Compute loss for points in weight history
        loss_history = torch.zeros(flat_weight_history.shape[0])
        for i, flat_params in enumerate(tqdm(flat_param_history)):
            state_dict = unflatten_params(flat_params, param_metadata)
            loss_history[i] = compute_loss(network, state_dict, test_dataloader)

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
    else:
        fig = plt.figure()

        if network2 is not None:
            vmax_net = vmax*torch.max(torch.cat([network1.test_loss_history, network2.test_loss_history]))
        else:
            vmax_net = vmax*torch.max(network1.test_loss_history)
        vmax_grid = torch.max(loss_grid)
        vmax = torch.min(vmax_grid,vmax_net)

        im = plt.imshow(loss_grid, cmap='Reds', vmax=vmax,
                        extent=[np.min(PC1_range), np.max(PC1_range),
                                np.max(PC2_range), np.min(PC2_range)])

        plt.colorbar(im)
        plt.scatter(PC1, PC2, s=0.1, color='k')

        if network2 is not None:
            plt.scatter(PC1_network1[0], PC2_network1[0], s=80, color='b', edgecolor='k', label='Start')
            plt.scatter(PC1_network2[0], PC2_network2[0], s=80, color='b', edgecolor='k')

            if not hasattr(network1, 'name'):
                network1.name = '1'
            if not hasattr(network2, 'name'):
                network2.name = '2'
            plt.scatter(PC1_network1[-1], PC2_network1[-1], s=80, color='orange', edgecolor='k',
                        label=f'End {network1.name}')
            plt.scatter(PC1_network2[-1], PC2_network2[-1], s=80, color='cyan', edgecolor='k',
                        label=f'End {network2.name}')
        else:
            plt.scatter(PC1[0], PC2[0], s=80, color='b', edgecolor='k', label='Start')
            plt.scatter(PC1[-1], PC2[-1], s=80, color='orange', edgecolor='k', label='End')
        plt.legend()
        plt.xlabel(f'PC1 \n({percent_exp_var[0]}% var. explained)')
        plt.ylabel(f'PC2 \n({percent_exp_var[1]}% var. explained)')

    return fig


def plot_loss_landscape_multiple(test_network, param_history_dict, test_dataloader, networksnum_points=20, extension=0.2):
    flat_param_history_all = []

    for network_name in param_history_dict:
        for i in param_history_dict[network_name]:
            flat_param_history, param_metadata = get_flat_param_history(param_history_dict[network_name][i])
            flat_param_history_all.append(flat_param_history)
    flat_param_history_all = torch.cat(flat_param_history_all)

    history_len = flat_param_history.shape[0]
    num_networks = ut.count_dict_elements(param_history_dict)
    flat_param_history = flat_param_history_all


    # Center the data (mean=0, std=1)
    p_mean = torch.mean(flat_param_history, axis=0)
    p_std = torch.std(flat_param_history, axis=0)
    centered_param_history = (flat_param_history - p_mean) / (p_std + 1e-10)  # add epsilon to avoid NaNs


    # Get weights in gridplane defined by PC dimensions
    pca = PCA(n_components=2)
    pca.fit(centered_param_history)
    param_hist_pca_space = pca.transform(centered_param_history)

    PC1 = param_hist_pca_space[:, 0]
    PC2 = param_hist_pca_space[:, 1]
    PC1_extension = (np.max(PC1) - np.min(PC1)) * extension
    PC2_extension = (np.max(PC2) - np.min(PC2)) * extension
    PC1_range = np.linspace(np.min(PC1) - PC1_extension, np.max(PC1) + PC1_extension, num_points)
    PC2_range = np.linspace(np.min(PC2) - PC2_extension, np.max(PC2) + PC2_extension, num_points)
    PC1_mesh, PC2_mesh = np.meshgrid(PC1_range, PC2_range)


    # Convert PC coordinates into full weight vectors
    flat_PC1_vals = PC1_mesh.reshape(1, num_points ** 2)
    flat_PC2_vals = PC2_mesh.reshape(1, num_points ** 2)
    meshgrid_points = np.concatenate([flat_PC1_vals, flat_PC2_vals]).T

    gridpoints_paramspace = pca.inverse_transform(meshgrid_points)
    gridpoints_paramspace = torch.tensor(gridpoints_paramspace) * p_std + p_mean


    # Compute loss for points in grid
    test_network = copy(test_network)  # create copy to avoid modifying original networks
    losses = torch.zeros(num_points ** 2)
    for i, gridpoint_flat in enumerate(tqdm(gridpoints_paramspace)):
        state_dict = unflatten_params(gridpoint_flat, param_metadata)
        losses[i] = compute_loss(test_network, state_dict, test_dataloader)
    loss_grid = losses.reshape([PC1_range.size, PC2_range.size])

    plot_loss_surface(loss_grid, PC1_mesh, PC2_mesh)

    lines_PC1 = []
    lines_PC2 = []
    for i in range(0, len(PC1), history_len):
        lines_PC1.append(PC1[i:i + history_len])
        lines_PC2.append(PC2[i:i + history_len])

    fig = plt.figure()
    im = plt.imshow(loss_grid, cmap='Reds',
                    extent=[np.min(PC1_range), np.max(PC1_range),
                            np.max(PC2_range), np.min(PC2_range)])
    plt.colorbar(im)

    plt.scatter(PC1, PC2, s=0.1, color='k')

    # plt.scatter(PC1_network1[0], PC2_network1[0], s=80, color='b', edgecolor='k', label='Start')
    # plt.scatter(PC1_network2[0], PC2_network2[0], s=80, color='b', edgecolor='k')

    # plt.scatter(PC1_network1[-1], PC2_network1[-1], s=80, color='orange', edgecolor='k',
    #             label=f'End {network1.name}')
    # plt.scatter(PC1_network2[-1], PC2_network2[-1], s=80, color='cyan', edgecolor='k',
    #             label=f'End {network2.name}')

    plt.legend()
    plt.xlabel('PC1')
    plt.ylabel('PC2')


def plot_loss_surface(loss_grid, PC1_mesh, PC2_mesh):
    '''
    Function plots loss surface from the grid based on PCs

    Parameters
    loss_grid: torch tensor, size: [num_points x num_points]
        values of loss for the given model at a given set of weights
    PC1_mesh: torch tensor, size: [1 x num_points(specified in get_loss_landscape] (?)
    PC2_mesh: torch tensor, size: [1 x num_points(specified in get_loss_landscape]
    '''
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(projection='3d')

    ax.plot_surface(PC1_mesh, PC2_mesh, loss_grid.numpy(), cmap='terrain', alpha=0.9)

    ax.view_init(elev=30, azim=-50)
    ax.set_xlabel(r'PC1', labelpad=9)
    ax.set_ylabel(r'PC2', labelpad=10)
    ax.set_zlabel(r'Loss', labelpad=25)
    ax.tick_params(axis='x', pad=0)
    ax.tick_params(axis='y', pad=0)
    ax.tick_params(axis='z', pad=10)

    plt.tight_layout()
    # fig.show()


def plot_binary_decision_boundary(network, test_dataloader, hard_boundary=False, num_points = 1000):
    assert len(test_dataloader)==1, "Dataloader must have a single large batch"

    # Compute accuracy on test data
    idx, data, target = next(iter(test_dataloader))
    output = network(data)
    prediction = torch.heaviside(output - 0.5, torch.tensor(0.))
    test_accuracy = 100 * torch.sum(prediction == target) / len(target)
    print(f"Test Accuracy {test_accuracy} %")

    # Plot decision boundary (test a grid of X values)
    extension = 2.
    x1 = data[:, 0].numpy()
    x2 = data[:, 1].numpy()
    # x1_extension = (np.max(x1) - np.min(x1)) * extension
    # x2_extension = (np.max(x2) - np.min(x2)) * extension
    x1_extension = 1.5
    x2_extension = 1.5
    x1_range = np.linspace(np.min(x1) - x1_extension, np.max(x1) + x1_extension, num_points)
    x2_range = np.linspace(np.min(x2) - x2_extension, np.max(x2) + x2_extension, num_points)
    x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)

    flat_x1_vals = x1_mesh.reshape(1, num_points ** 2)
    flat_x2_vals = x2_mesh.reshape(1, num_points ** 2)
    meshgrid_points = np.concatenate([flat_x1_vals, flat_x2_vals]).T

    meshgrid_points = torch.tensor(meshgrid_points).type(torch.float32)
    outputs = network(meshgrid_points).detach()
    output_grid = outputs.reshape([x1_range.size, x2_range.size]).flipud()

    if hard_boundary:
        output_grid = np.heaviside(output_grid-0.5,0)

    fig, ax = plt.subplots(figsize=(4, 4))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["orange", "white", "blue"])
    ax.scatter(data[:, 0], data[:, 1], c=target, edgecolors='w', cmap=cmap)
    im = plt.imshow(output_grid, cmap=cmap, vmin=0, vmax=1, alpha=0.6, interpolation='nearest',
                    extent=[np.min(x1_range), np.max(x1_range), np.min(x2_range), np.max(x2_range)])
    cax = fig.add_axes([ax.get_position().x1 + 0.04, ax.get_position().y0, 0.03, ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax)
    cbar.outline.set_visible(False)
    fig.show()


def plot_batch_accuracy(network, test_dataloader, title=None):
    """
    Compute total accuracy (% correct) on given dataset
    :param network:
    :param test_dataloader:
    :param title: str
    """
    assert len(test_dataloader)==1, 'Dataloader must have a single large batch'

    indexes, data, targets = next(iter(test_dataloader))
    data = data.to(network.device)
    targets = targets.to(network.device)
    labels = torch.argmax(targets, axis=1)
    output = network.forward(data).detach()
    percent_correct = 100 * torch.sum(torch.argmax(output, dim=1) == labels) / data.shape[0]
    percent_correct = torch.round(percent_correct, decimals=2)
    print(f'Batch accuracy = {percent_correct}%')

    # Plot average output for each label class
    num_units = targets.shape[1]
    num_labels = num_units
    avg_output = torch.zeros(num_units, num_labels)
    targets = torch.argmax(targets, dim=1)  # convert from 1-hot vector to int label
    for label in range(num_labels):
        label_idx = torch.where(targets == label)  # find all instances of given label
        avg_output[:, label] = torch.mean(output[label_idx], dim=0)

    fig, ax = plt.subplots()
    im = ax.imshow(avg_output, interpolation='none')
    cbar = plt.colorbar(im)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel('Labels')
    ax.set_ylabel('Output unit')
    if title is not None:
        ax.set_title('Average output activity - %s' % title)
    else:
        ax.set_title('Average output activity')
    fig.show()
