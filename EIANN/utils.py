import torch
import yaml
import itertools
import os
import numpy as np
from . import plot as plot
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8,
                     'axes.spines.right': False,
                     'axes.spines.top': False,
                     'axes.linewidth': 1,
                     'axes.labelpad': 0,
                     'xtick.major.size': 3.5,
                     'xtick.major.width': 1,
                     'ytick.major.size': 3.5,
                     'ytick.major.width': 1,
                     'legend.frameon': False,
                     'legend.handletextpad': 0.1,
                     'figure.figsize': [14.0, 4.0],})


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


def normalize_weight(projection, scale, autapses=False, axis=1):
    projection.weight.data /= torch.sum(torch.abs(projection.weight.data), axis=axis).unsqueeze(1)
    projection.weight.data *= scale
    if not autapses and projection.pre == projection.post:
        for i in range(projection.post.size):
            projection.weight.data[i, i] = 0.


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


def analyze_EIANN_loss(network, target, supervised=True, plot=False):

    output_layer = list(network)[-1]
    output_pop = output_layer.E

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


def test_EIANN_config(network, dataloader, epochs, supervised=True):

    num_samples = len(dataloader)
    sample_order = []
    for sample_idx, sample_data, sample_target in dataloader:
        sample_order.append(sample_idx)
        network.forward(sample_data, store_history=True)
    network.sorted_sample_indexes = torch.argsort(torch.tensor(sample_order))
    plot.plot_EIANN_activity(network, num_samples=num_samples, supervised=supervised, label='Initial')
    network.reset_history()

    network.train(dataloader, epochs, store_history=True, status_bar=True)
    target = torch.stack([sample_target for _, _, sample_target in dataloader.dataset])
    loss_history, epoch_argmax_accuracy = analyze_EIANN_loss(network, target, supervised=supervised, plot=True)
    plot.plot_EIANN_activity(network, num_samples=num_samples, supervised=supervised, label='Final')


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
