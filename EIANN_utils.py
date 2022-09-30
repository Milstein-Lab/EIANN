import torch
import itertools
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def n_choose_k(n, k):
    '''
    Calculates number of ways to choose k things out of n, using binomial coefficients

    :param n: number of things to choose from
    :type n: int
    :param k: number of things chosen
    :type k: int
    :return: int
    '''
    assert n>k, "k must be smaller than n"
    num_permutations = np.math.factorial(n) / (np.math.factorial(k)*np.math.factorial(n-k))
    return int(num_permutations)


def n_hot_patterns(n, length):
    '''
    Generates all possible binary n-hot patterns of given length

    :param n: number of bits set to 1
    :type n: int
    :param length: size of pattern (number of bits)
    :type length: int
    :return: torch.tensor
    '''
    all_permutations = torch.tensor(list(itertools.product([0., 1.], repeat=length)))
    pattern_hotness = torch.sum(all_permutations,axis=1)
    idx = torch.where(pattern_hotness == n)[0]
    n_hot_patterns = all_permutations[idx]
    return n_hot_patterns


def get_diag_argmax_row_indexes(data):
    """
    Sort the rows of a square matrix such that whenever row argmax and col argmax are equal, that value appears
    on the diagonal. Returns row indexes.

    :param data: 2d array; square matrix
    :return: array of int
    """
    data = np.array(data)
    if data.shape[0] != data.shape[1]:
        raise Exception('get_diag_argmax_row_indexes: data must be a square matrix')
    dim = data.shape[0]
    avail_row_indexes = list(range(dim))
    avail_col_indexes = list(range(dim))
    final_row_indexes = np.empty_like(avail_row_indexes)
    while(len(avail_col_indexes) > 0):
        row_selectivity = np.zeros_like(avail_row_indexes)
        row_max = np.max(data[avail_row_indexes, :][:, avail_col_indexes], axis=1)
        row_mean = np.mean(data[avail_row_indexes,:][:,avail_col_indexes], axis=1)
        nonzero_indexes = np.where(row_mean > 0)
        row_selectivity[nonzero_indexes] = row_max[nonzero_indexes] / row_mean[nonzero_indexes]

        row_index = avail_row_indexes[np.argsort(row_selectivity)[-1]]
        col_index = avail_col_indexes[np.argmax(data[row_index,avail_col_indexes])]
        final_row_indexes[col_index] = row_index
        avail_row_indexes.remove(row_index)
        avail_col_indexes.remove(col_index)
    return final_row_indexes


def scaled_kaining_init(data, fan_in, scale=1):
    kaining_bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    data.uniform_(-scale * kaining_bound, scale * kaining_bound)


def half_kaining_init(data, fan_in, scale=1, bounds=None):
    kaining_bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    if bounds is None or (bounds[0] is not None and bounds[0] >= 0):
        data.uniform_(bounds[0], scale * kaining_bound)
    elif bounds[1] is not None and bounds[1] <= 0:
        data.uniform_(-scale * kaining_bound, bounds[1])
    else:
        raise RuntimeError('half_kaining_init: bounds should be either >=0 or <=0: %s' % str(bounds))


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
        sorted_idx = get_diag_argmax_row_indexes(output)
    else:
        sorted_idx = np.arange(output.shape[0])

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

    fig, axes = plt.subplots(max_rows, cols, figsize=(3.*cols, 3.*max_rows))
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
                    fig.colorbar(im, ax=this_axis)
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
    fig.suptitle('%sweights' % label_str)
    fig.tight_layout()
    fig.show()

    max_rows = 1
    cols = len(network.layers)
    for layer in network:
        max_rows = max(max_rows, len(layer.populations))

    fig, axes = plt.subplots(max_rows, cols, figsize=(3. * cols, 3. * max_rows))
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
            fig.colorbar(im, ax=this_axis)
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
    fig.suptitle('%sactivity' % label_str)
    fig.tight_layout()
    fig.show()

    cols = len(network.layers) - 1
    fig, axes = plt.subplots(max_rows, cols, figsize=(3. * cols, 3. * max_rows))
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
    fig.suptitle('%sactivity dynamics' % label_str)
    fig.tight_layout()
    fig.show()

    print('%spopulation biases:' % label_str)
    for i, layer in enumerate(network):
        if i > 0:
            for population in layer:
                print(layer.name, population.name, population.bias)


def analyze_EIANN_loss(network, target, supervised=True, plot=False):

    reversed_layers = list(network)
    reversed_layers.reverse()
    output_pop = next(iter(reversed_layers[0]))

    argmax_correct = []
    if supervised or target.shape[0] != target.shape[1]:
        loss_history = network.loss_history
        for i in range(output_pop.activity_history.shape[0]):
            sample_idx = network.sample_order[i]
            sample_target = target[sample_idx, :]
            output = output_pop.activity_history[i, -1, :]
            argmax_correct.append(torch.argmax(output) == torch.argmax(sample_target))
    else:
        final_output = output_pop.activity_history[network.sorted_sample_indexes, -1, :][-target.shape[0]:, :].T
        sorted_idx = get_diag_argmax_row_indexes(final_output)
        loss_history = []
        for i in range(output_pop.activity_history.shape[0]):
            sample_idx = network.sample_order[i]
            sample_target = target[sample_idx, :]
            output = output_pop.activity_history[i, -1, sorted_idx]
            loss = network.criterion(output, sample_target)
            loss_history.append(loss)
            argmax_correct.append(torch.argmax(output) == torch.argmax(sample_target))
        loss_history = torch.tensor(loss_history)
    argmax_correct = torch.tensor(argmax_correct)

    epoch_argmax_accuracy = []
    start = 0
    while start < len(argmax_correct):
        epoch_argmax_accuracy.append(torch.sum(argmax_correct[start:start+target.shape[0]]) / target.shape[0] * 100.)
        start += target.shape[0]

    epoch_argmax_accuracy = torch.tensor(epoch_argmax_accuracy)

    if plot:
        fig = plt.figure()
        plt.plot(loss_history)
        plt.xlabel('Training steps')
        plt.ylabel('MSE loss')
        plt.title('Training loss')
        fig.tight_layout()
        fig.show()

        fig = plt.figure()
        plt.plot(epoch_argmax_accuracy)
        plt.xlabel('Training epochs')
        plt.ylabel('% correct argmax')
        plt.title('Argmax accuracy')
        fig.tight_layout()
        fig.show()

    return loss_history, epoch_argmax_accuracy


def test_EIANN_config(network, dataset, target, epochs, supervised=True):

    for sample in dataset:
        network.forward(sample, store_history=True)
    plot_EIANN_activity(network, num_samples=dataset.shape[0], supervised=supervised, label='Initial')
    network.reset_history()

    network.train(dataset, target, epochs, store_history=True, shuffle=True, status_bar=True)
    loss_history, epoch_argmax_accuracy = analyze_EIANN_loss(network, target, supervised=supervised, plot=True)
    plot_EIANN_activity(network, num_samples=dataset.shape[0], supervised=supervised, label='Final')


def test_EIANN_CL_config(network, dataset, target, epochs, split=0.75, supervised=True):

    for sample in dataset:
        network.forward(sample, store_history=True)
    plot_EIANN_activity(network, num_samples=dataset.shape[0], supervised=supervised, label='Initial')
    network.reset_history()

    phase1_num_samples = round(dataset.shape[0] * split)

    network.train(dataset[:phase1_num_samples], target[:phase1_num_samples], epochs, store_history=True, shuffle=True,
                  status_bar=True)
    loss_history, epoch_argmax_accuracy = analyze_EIANN_loss(network, target[:phase1_num_samples],
                                                             supervised=supervised, plot=True)
    network.reset_history()

    for sample in dataset:
        network.forward(sample, store_history=True)
    plot_EIANN_activity(network, num_samples=dataset.shape[0], supervised=supervised, label='After Phase 1')
    network.reset_history()

    network.train(dataset[phase1_num_samples:], target[phase1_num_samples:], epochs, store_history=True, shuffle=True,
                  status_bar=True)
    loss_history, epoch_argmax_accuracy = analyze_EIANN_loss(network, target[phase1_num_samples:],
                                                             supervised=supervised, plot=True)
    network.reset_history()

    for sample in dataset:
        network.forward(sample, store_history=True)
    plot_EIANN_activity(network, num_samples=dataset.shape[0], supervised=supervised, label='After Phase 2')


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

    :param: network: EIANN network with weights stored as Projection attributes
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

    :param: network: EIANN network with weights stored as Projection attributes
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

    :param: network: EIANN network with weights stored as Projection attributes
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

    return loss

def plot_loss_landscape(network, test_dataloader, num_points=20):
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
    range_extension = 0.1 #proportion to sample further on each side of the grid
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
    for i,gridpoint_flat in enumerate(gridpoints_weightspace):
        weight_mat_ls = unflatten_weights(gridpoint_flat, weight_sizes)
        losses[i] = compute_loss(network, weight_mat_ls, test_dataloader)
    loss_grid = losses.reshape([PC1_range.size, PC2_range.size])

    plot_loss_surface(loss_grid, PC1_mesh, PC2_mesh)

    # Compute loss for points in weight history
    loss_history = torch.zeros(flat_weight_history.shape[0])
    for i,flat_weights in enumerate(flat_weight_history):
        weight_mat_ls = unflatten_weights(flat_weights, weight_sizes)
        loss_history[i] = compute_loss(network, weight_mat_ls, test_dataloader)

    # # Compute loss for points in weight history
    # weight_history = torch.tensor(pca.inverse_transform(weight_hist_pca_space)) * w_std + w_mean
    # loss_history = torch.zeros(weight_history.shape[0])
    # for i,flat_weights in enumerate(weight_history):
    #     weight_mat_ls = unflatten_weights(flat_weights, weight_sizes)
    #     loss_history[i] = compute_loss(network, weight_mat_ls, test_dataloader)

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

    ax.plot_surface(PC1_mesh, PC2_mesh, loss_grid.numpy(), cmap='terrain', alpha=0.75)

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

