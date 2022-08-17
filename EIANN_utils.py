import torch
import itertools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def n_choose_k(n,k):
    num_permutations = np.math.factorial(n) / (np.math.factorial(k)*np.math.factorial(n-k))
    return int(num_permutations)


def n_hot_patterns(n,length):
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


def test_EIANN_config(network, dataset, target, epochs):

    for sample in dataset:
        network.forward(sample, store_history=True)

    plt.figure()
    plt.imshow(network.Output.E.Input.E.weight.data)
    plt.colorbar()
    plt.xlabel('Input unit ID')
    plt.ylabel('Output unit ID')
    plt.title('Initial weights\nOutput_E <- Input_E')

    fig, axes = plt.subplots(1, 2)
    this_axis = axes[0]
    im = this_axis.imshow(network.Output.E.activity_history[-dataset.shape[0]:, -1, :].T)
    plt.colorbar(im, ax=this_axis)
    this_axis.set_xlabel('Input pattern ID')
    this_axis.set_ylabel('Output unit ID')
    this_axis.set_title('Initial activity\nOutput_E')
    this_axis = axes[1]
    im = this_axis.imshow(network.Output.FBI.activity_history[-dataset.shape[0]:, -1, :].T)
    plt.colorbar(im, ax=this_axis)
    this_axis.set_xlabel('Input pattern ID')
    this_axis.set_ylabel('Output unit ID')
    this_axis.set_title('Initial activity\nOutput_FBI')
    fig.tight_layout()
    fig.show()

    fig, axes = plt.subplots(1, 2)
    this_axis = axes[0]
    for i in range(network.Output.E.size):
        this_axis.plot(torch.mean(network.Output.E.activity_history[-dataset.shape[0]:, :, i], axis=0))
    this_axis.set_xlabel('Equilibration time steps')
    this_axis.set_ylabel('Output unit ID')
    this_axis.set_title('Mean activity dynamics\nOutput_E')
    this_axis = axes[1]
    for i in range(network.Output.FBI.size):
        this_axis.plot(torch.mean(network.Output.FBI.activity_history[-dataset.shape[0]:, :, i], axis=0))
    this_axis.set_xlabel('Equilibration time steps')
    this_axis.set_ylabel('Output unit ID')
    this_axis.set_title('Mean activity dynamics\nOutput_FBI')
    fig.tight_layout()
    fig.show()

    for i, layer in enumerate(network):
        if i > 0:
            for population in layer:
                print(layer.name, population.name, population.bias)

    network.reset_history()

    network.train(dataset, target, epochs, store_history=True, shuffle=True, status_bar=True)

    final_output = network.Output.E.activity_history[network.sorted_sample_indexes, -1, :][-dataset.shape[0]:, :].T
    if final_output.shape[0] == final_output.shape[1]:
        sorted_idx = get_diag_argmax_row_indexes(final_output)
    else:
        sorted_idx = np.arange(final_output.shape[0])

    sorted_loss_history = []
    for i in range(network.Output.E.activity_history.shape[0]):
        sample_idx = network.sample_order[i]
        sample_target = target[sample_idx,:]
        output = network.Output.E.activity_history[i,-1,sorted_idx]
        loss = network.criterion(output, sample_target)
        sorted_loss_history.append(loss)
    sorted_loss_history = torch.tensor(sorted_loss_history)

    fig = plt.figure()
    plt.plot(network.loss_history, label='Unsorted')
    plt.plot(sorted_loss_history, label='Sorted')
    plt.legend(bbox_to_anchor=(1.05, 1.), loc='upper left', frameon=False)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Training loss')
    fig.tight_layout()
    fig.show()

    fig = plt.figure()
    plt.imshow(network.Output.E.Input.E.weight.data[sorted_idx, :])
    plt.colorbar()
    plt.xlabel('Input unit ID')
    plt.ylabel('Output unit ID')
    plt.title('Final weights\nOutput_E <- Input_E')
    fig.show()

    fig, axes = plt.subplots(1, 2)
    this_axis = axes[0]
    im = this_axis.imshow(final_output[sorted_idx, :])
    plt.colorbar(im, ax=this_axis)
    this_axis.set_xlabel('Input pattern ID')
    this_axis.set_ylabel('Output unit ID')
    this_axis.set_title('Final activity\nOutput_E')
    this_axis = axes[1]
    im = this_axis.imshow(network.Output.FBI.activity_history[network.sorted_sample_indexes, -1, :][
                   -dataset.shape[0]:, :].T)
    plt.colorbar(im, ax=this_axis)
    this_axis.set_xlabel('Input pattern ID')
    this_axis.set_ylabel('Output unit ID')
    this_axis.set_title('Final activity\nOutput_FBI')
    fig.tight_layout()
    fig.show()

    fig, axes = plt.subplots(1, 2)
    this_axis = axes[0]
    for i in range(network.Output.E.size):
        this_axis.plot(torch.mean(network.Output.E.activity_history[-dataset.shape[0]:, :, i], axis=0))
    this_axis.set_xlabel('Equilibration time steps')
    this_axis.set_ylabel('Output unit ID')
    this_axis.set_title('Mean activity dynamics\nOutput_E')
    this_axis = axes[1]
    for i in range(network.Output.FBI.size):
        this_axis.plot(torch.mean(network.Output.FBI.activity_history[-dataset.shape[0]:, :, i], axis=0))
    this_axis.set_xlabel('Equilibration time steps')
    this_axis.set_ylabel('Output unit ID')
    this_axis.set_title('Mean activity dynamics\nOutput_FBI')
    fig.tight_layout()
    fig.show()

    for i, layer in enumerate(network):
        if i > 0:
            for population in layer:
                print(layer.name, population.name, 'bias:', population.bias)