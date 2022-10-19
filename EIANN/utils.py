import torch
import yaml
import itertools
import os
import numpy as np
import math
from collections import Iterable
from . import plot as plot
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10,
                     #'axes.spines.right': False,
                     #'axes.spines.top': False,
                     #'axes.linewidth': 1,
                     'axes.labelpad': 0,
                     'xtick.major.size': 3.5,
                     'xtick.major.width': 1,
                     'ytick.major.size': 3.5,
                     'ytick.major.width': 1,
                     'legend.frameon': False,
                     'legend.handletextpad': 0.1,
                     #'figure.figsize': [14.0, 4.0],
                    })


def nested_convert_scalars(data):
    """
    Crawls a nested dictionary, and converts any scalar objects from numpy types to python types.
    :param data: dict
    :return: dict
    """
    if isinstance(data, dict):
        converted_data = dict()
        for key in data:
            converted_key = nested_convert_scalars(key)
            converted_data[converted_key] = nested_convert_scalars(data[key])
        data = converted_data
    elif isinstance(data, Iterable) and not isinstance(data, str):
        data_as_list = list(data)
        for i in range(len(data)):
            data_as_list[i] = nested_convert_scalars(data[i])
        if isinstance(data, tuple):
            data = tuple(data_as_list)
        else:
            data = data_as_list
    elif hasattr(data, 'item'):
        data = data.item()
    return data


def write_to_yaml(file_path, data, convert_scalars=True):
    """

    :param file_path: str (should end in '.yaml')
    :param data: dict
    :param convert_scalars: bool
    :return:
    """
    import yaml
    with open(file_path, 'w') as outfile:
        if convert_scalars:
            data = nested_convert_scalars(data)
        yaml.dump(data, outfile, default_flow_style=False, sort_keys=False)


def read_from_yaml(file_path, Loader=None):
    """
    Import a python dict from .yaml
    :param file_path: str (should end in '.yaml')
    :param Loader: :class:'yaml.Loader'
    :return: dict
    """
    if Loader is None:
        Loader = yaml.FullLoader
    if os.path.isfile(file_path):
        with open(file_path, 'r') as stream:
            data = yaml.load(stream, Loader=Loader)
        return data
    else:
        raise Exception('File: {} does not exist.'.format(file_path))


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


def get_scaled_rectified_sigmoid(th, peak, x=None, ylim=None):
    """
    Transform a sigmoid to intersect x and y range limits.
    :param th: float
    :param peak: float
    :param x: array
    :param ylim: pair of float
    :return: callable
    """
    if x is None:
        x = (0., 1.)
    if ylim is None:
        ylim = (0., 1.)
    if th < x[0] or th > x[-1]:
        raise ValueError('scaled_single_sigmoid: th: %.2E is out of range for xlim: [%.2E, %.2E]' % (th, x[0], x[-1]))
    if peak == th:
        raise ValueError('scaled_single_sigmoid: peak and th: %.2E cannot be equal' % th)
    slope = 2. / (peak - th)
    y = lambda x: 1. / (1. + np.exp(-slope * (x - th)))
    start_val = y(x[0])
    end_val = y(x[-1])
    amp = end_val - start_val
    target_amp = ylim[1] - ylim[0]
    return np.vectorize(
        lambda xi:
        (target_amp / amp) * (1. / (1. + np.exp(-slope * (max(min(xi, x[-1]), x[0]) - th))) - start_val) + ylim[0])


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


def sort_unsupervised_by_best_epoch(network, target, plot=False):

    output_layer = list(network)[-1]
    output_pop = next(iter(output_layer))

    dynamic_epoch_loss_history = []
    sorted_idx_history = []

    output_history = output_pop.activity_history[network.sorted_sample_indexes, -1, :]
    start = 0
    while start < output_history.shape[0]:
        end = start + target.shape[0]
        epoch_output = output_history[start:end, :]
        sorted_idx = get_diag_argmax_row_indexes(epoch_output.T)
        loss = network.criterion(epoch_output[:, sorted_idx], target)
        dynamic_epoch_loss_history.append(loss)
        sorted_idx_history.append(sorted_idx)
        start += target.shape[0]

    best_index = np.where(dynamic_epoch_loss_history == np.min(dynamic_epoch_loss_history))[0][0]
    sorted_idx = sorted_idx_history[best_index]
    epoch_loss_history = []
    start = 0
    while start < output_pop.activity_history.shape[0]:
        end = start + target.shape[0]
        epoch_output = output_history[start:end, :]
        loss = network.criterion(epoch_output[:, sorted_idx], target)
        epoch_loss_history.append(loss)
        start += target.shape[0]

    if plot:
        fig = plt.figure()
        plt.plot(dynamic_epoch_loss_history, label='Dynamic')
        plt.plot(epoch_loss_history, label='Sorted by peak')
        plt.xlabel('Training epochs')
        plt.ylabel('MSE loss')
        plt.title('Epoch training loss')
        plt.legend(loc='best', frameon=False)
        fig.tight_layout()
        fig.show()

    return sorted_idx


def analyze_simple_EIANN_epoch_loss_and_accuracy(network, target, sorted_output_idx=None, plot=False):
    """

    :param network:
    :param target:
    :param sorted_output_idx:
    :param plot:
    :return: tuple: (int, tensor of float, tensor of float)
    """

    output_layer = list(network)[-1]
    output_pop = next(iter(output_layer))

    epoch_loss = []
    epoch_argmax_accuracy = []

    if sorted_output_idx is None:
        sorted_output_idx = torch.arange(0, output_pop.size)

    output_history = output_pop.activity_history[network.sorted_sample_indexes, -1, :][:, sorted_output_idx]
    start = 0
    while start < output_history.shape[0]:
        end = start + target.shape[0]
        epoch_output = output_history[start:end, :]
        loss = network.criterion(epoch_output, target)
        epoch_loss.append(loss)
        start += target.shape[0]
        accuracy = torch.sum(torch.argmax(epoch_output, axis=1) == torch.argmax(target, axis=1))
        epoch_argmax_accuracy.append(accuracy / target.shape[0] * 100.)

    epoch_loss = torch.tensor(epoch_loss)
    epoch_argmax_accuracy = torch.tensor(epoch_argmax_accuracy)
    best_epoch_index = torch.where(epoch_loss == torch.min(epoch_loss))[0][0]

    if plot:
        fig = plt.figure()
        plt.plot(epoch_loss)
        plt.xlabel('Training epochs')
        plt.ylabel('MSE loss')
        plt.title('Epoch training loss')
        fig.tight_layout()
        fig.show()

        fig = plt.figure()
        plt.plot(epoch_argmax_accuracy)
        plt.xlabel('Training epochs')
        plt.ylabel('% correct argmax')
        plt.title('Argmax accuracy')
        fig.tight_layout()
        fig.show()

    return best_epoch_index, epoch_loss, epoch_argmax_accuracy


def test_simple_EIANN_config(network, dataloader, epochs, supervised=True):

    num_samples = len(dataloader)
    network.test(dataloader, store_history=True, status_bar=True)
    plot.plot_simple_EIANN_config_summary(network, num_samples=num_samples, label='Initial')
    network.reset_history()

    network.train(dataloader, epochs, store_history=True, status_bar=True)

    target = torch.stack([sample_target for _, _, sample_target in dataloader.dataset])
    if not supervised:
        sorted_output_idx = sort_unsupervised_by_best_epoch(network, target, plot=plot)
    else:
        sorted_output_idx = None
    best_epoch_index, loss_history, epoch_argmax_accuracy = \
        analyze_simple_EIANN_epoch_loss_and_accuracy(network, target, sorted_output_idx=sorted_output_idx, plot=True)
    start_index = best_epoch_index * num_samples
    plot.plot_simple_EIANN_config_summary(network, start_index=start_index, num_samples=num_samples,
                                           sorted_output_idx=sorted_output_idx, label='Final')


def test_EIANN_CL_config(network, dataloader, epochs, split=0.75, supervised=True, generator=None):

    num_samples = len(dataloader)
    network.test(dataloader, store_history=True, status_bar=True)
    plot.plot_simple_EIANN_config_summary(network, num_samples=num_samples, label='Initial')
    network.reset_history()

    target = torch.stack([sample_target for _, _, sample_target in dataloader.dataset])
    phase1_num_samples = round(len(dataloader) * split)
    phase1_sample_indexes = torch.arange(phase1_num_samples)
    _, phase1_dataset, phase1_target = map(torch.stack, zip(*dataloader.dataset[:phase1_num_samples]))
    phase1_dataloader = DataLoader(list(zip(phase1_sample_indexes, phase1_dataset, phase1_target)), shuffle=True,
                                   generator=generator)
    network.train(phase1_dataloader, epochs, store_history=True, status_bar=True)
    best_epoch_index, loss_history, epoch_argmax_accuracy = \
        analyze_simple_EIANN_epoch_loss_and_accuracy(network, phase1_target, plot=True)
    network.reset_history()

    network.test(dataloader, store_history=True, status_bar=True)
    if not supervised:
        sorted_output_idx = sort_unsupervised_by_best_epoch(network, target, plot=plot)
    else:
        sorted_output_idx = None
    plot.plot_simple_EIANN_config_summary(network, num_samples=num_samples, sorted_output_idx=sorted_output_idx,
                                          label='After Phase 1')
    network.reset_history()

    phase2_num_samples = len(dataloader) - phase1_num_samples
    phase2_sample_indexes = torch.arange(phase2_num_samples)
    _, phase2_dataset, phase2_target = map(torch.stack, zip(*dataloader.dataset[phase1_num_samples:]))
    phase2_dataloader = DataLoader(list(zip(phase2_sample_indexes, phase2_dataset, phase2_target)), shuffle=True,
                                   generator=generator)
    network.train(phase2_dataloader, epochs, store_history=True, status_bar=True)
    best_epoch_index, loss_history, epoch_argmax_accuracy = \
        analyze_simple_EIANN_epoch_loss_and_accuracy(network, phase2_target, plot=True)
    network.reset_history()

    network.test(dataloader, store_history=True, status_bar=True)
    if not supervised:
        sorted_output_idx = sort_unsupervised_by_best_epoch(network, target, plot=plot)
    else:
        sorted_output_idx = None
    plot.plot_simple_EIANN_config_summary(network, num_samples=num_samples, sorted_output_idx=sorted_output_idx,
                                          label='After Phase 2')


def compute_batch_accuracy(network, test_dataloader):
    """
    Compute total accuracy (% correct) on given dataset
    :param network:
    :param test_dataloader:
    """
    indexes, data, targets = next(iter(test_dataloader))
    labels = torch.argmax(targets, axis=1)
    output = network.forward(data).detach()
    percent_correct = 100 * torch.sum(torch.argmax(output, dim=1) == labels) / data.shape[0]
    percent_correct = torch.round(percent_correct, decimals=2)
    print(f'Batch accuracy = {percent_correct}%')

