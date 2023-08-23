import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softplus, relu, sigmoid, elu
import numpy as np
from scipy import signal, stats
from skimage import metrics
from sklearn.metrics.pairwise import cosine_similarity
import math
import yaml
import h5py
import itertools
import os
import copy
from . import plot as plot
from . import external as external
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

try:
    from collections import Iterable
except:
    from collections.abc import Iterable



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


# *******************************************************************
# Functions to import and export data
# *******************************************************************

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


def export_metrics_data(metrics_dict, model_name, path):
    """
    Exports data from metrics_dict to hdf5 file.
    :param metrics_dict: dictionary of metrics computed on EIANN network
    :param model_name: string of model name for top_level hdf5 group
    :param file_name: string name of file to save to
    """

    if '.hdf5' not in path:
        path = path + '.hdf5'
    with h5py.File(path, mode='a') as file:

        if model_name in file:
            overwrite = input('File already contains metrics for this model. Overwrite? (y/n)')
            if overwrite == 'y':
                del file[model_name]
            else:
                print('Model metrics not saved')
                return

        file.create_group(model_name)

        for metric in metrics_dict.keys():
            file[model_name].create_dataset(metric, data=metrics_dict[metric])

 
def import_metrics_data(filename):
    """
    Imports metrics data from hdf5 file
    :param file_name: string name of hdf5 file
    :return sim_dict: dictionary of values
    """
    metrics_dict = {}
    with h5py.File(filename, 'r') as file:
        for model_name in file:
            metrics_dict[model_name] = {}
            for metric in file[model_name]:
                metrics_dict[model_name][metric] = file[model_name][metric][:]

    return metrics_dict


def hdf5_to_dict(file_path):
    """
    Load an HDF5 file and convert it to a nested Python dictionary.

    :param file_path (str): Path to the HDF5 file.
    :return dict: nested Python dictionary with identical structure as the HDF5 file.
    """
    with h5py.File(file_path, 'r') as f:
        data_dict = {}
        # Loop over the top-level keys in the HDF5 file
        for key in f.keys():
            if isinstance(f[key], h5py.Group):
                # Recursively convert the group to a nested dictionary
                data_dict[key] = hdf5_to_dict_helper(f[key])
            else:
                # If the key corresponds to a dataset, add it to the dictionary
                data_dict[key] = f[key][()]
    return data_dict


def hdf5_to_dict_helper(group):
    """
    Helper function to recursively convert an HDF5 group to a nested Python dictionary.

    :param group (h5py.Group): The HDF5 group to convert.
    :return dict: Nested Python dictionary with identical structure as the HDF5 group.
    """
    data_dict = {}
    # Loop over the keys in the HDF5 group
    for key in group.keys():
        if isinstance(group[key], h5py.Group):
            # Recursively convert the group to a nested dictionary
            data_dict[key] = hdf5_to_dict_helper(group[key])
        else:
            # If the key corresponds to a dataset, add it to the dictionary
            data_dict[key] = group[key][()]

    return data_dict


# *******************************************************************
# Functions to generate and process data
# *******************************************************************

def n_choose_k(n, k):
    """
    Calculates number of ways to choose k things out of n, using binomial coefficients

    :param n: number of things to choose from
    :type n: int
    :param k: number of things chosen
    :type k: int
    :return: int
    """
    assert n>k, "k must be smaller than n"
    num_permutations = np.math.factorial(n) / (np.math.factorial(k)*np.math.factorial(n-k))
    return int(num_permutations)


def n_hot_patterns(n, length):
    """
    Generates all possible binary n-hot patterns of given length

    :param n: number of bits set to 1
    :type n: int
    :param length: size of pattern (number of bits)
    :type length: int
    :return: torch.tensor
    """
    all_permutations = torch.tensor(list(itertools.product([0., 1.], repeat=length)))
    pattern_hotness = torch.sum(all_permutations,axis=1)
    idx = torch.where(pattern_hotness == n)[0]
    n_hot_patterns = all_permutations[idx]
    return n_hot_patterns


def get_scaled_rectified_sigmoid_orig(th, peak, x=None, ylim=None):
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
    return lambda xi: (target_amp / amp) * (1. / (1. + torch.exp(-slope * (torch.clamp(xi, x[0], x[-1]) - th))) -
                                            start_val) + ylim[0]


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


def sort_by_val_history(network, plot=False):
    """
    Find the sorting giving the best argmax across the full validation history

    :param network:
    :param plot:
    :return: min_loss_idx: index of the point with lowest loss (index relative only to the validation points, not the full training)
    :return: min_loss_sorting: optimal sorting indices for the point with lowest loss
    """
    output_pop = network.output_pop

    num_units = network.val_target.shape[1]
    num_labels = num_units
    num_patterns = network.val_target.shape[0]

    sorting_history = []
    optimal_loss_history = []
    optimal_accuracy_history = []
    sorted_idx_history = []

    for output in network.val_output_history:
        # Get average output for each label class
        avg_output = torch.zeros(num_labels, num_units)
        targets = torch.argmax(network.val_target, dim=1)  # convert from 1-hot vector to int label
        for label in range(num_labels):
            label_idx = torch.where(targets == label)  # find all instances of given label
            avg_output[label, :] = torch.mean(output[label_idx], dim=0)

        # Find optimal output unit (column) sorting given average responses
        optimal_sorting = get_diag_argmax_row_indexes(avg_output.T)
        sorted_activity = output[:, optimal_sorting]
        optimal_loss = network.criterion(sorted_activity, network.val_target)
        optimal_loss_history.append(optimal_loss.item())
        optimal_accuracy = 100 * torch.sum(
            torch.argmax(sorted_activity, dim=1) == torch.argmax(network.val_target, dim=1)) / num_patterns
        optimal_accuracy_history.append(optimal_accuracy.item())
        sorting_history.append(optimal_sorting)

    # Pick timepoint with lowest sorted loss
    optimal_loss_history = torch.tensor(optimal_loss_history)
    min_loss_idx = torch.argmin(optimal_loss_history)
    min_loss_sorting = sorting_history[min_loss_idx]

    if plot:
        fig = plt.figure()
        plt.scatter(min_loss_idx, torch.min(optimal_loss_history), color='red')
        plt.plot(optimal_loss_history)
        plt.title('optimal loss history (re-sorted for each point)')
        fig.show()

    return min_loss_idx, min_loss_sorting


def sort_unsupervised_by_test_batch_autoenc(network, test_dataloader):
    """
    Run a test batch and return the output unit sorting that best matches the output activity to the test labels.
    :param network:
    :param test_dataloader:
    :return: tensor of int
    """
    assert len(test_dataloader) == 1, 'Dataloader must have a single large batch'

    network.test(test_dataloader)

    # Find optimal output unit (column) sorting
    optimal_sorting = get_diag_argmax_row_indexes(network.output_pop.activity.T)

    return optimal_sorting


def sort_unsupervised_by_best_epoch(network, target, plot=False):

    output_pop = network.output_pop

    dynamic_epoch_loss_history = []
    sorted_idx_history = []

    if output_pop.activity_history.dim() > 2:
        output_history = output_pop.activity_history[network.sorted_sample_indexes, -1, :]
    else:
        output_history = output_pop.activity_history[network.sorted_sample_indexes, :]
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


def compute_test_loss_and_accuracy_single_batch(network, test_dataloader, sorted_output_idx=None):
    """

    :param network:
    :param test_dataloader:
    :param sorted_output_idx: tensor of int
    """
    assert len(test_dataloader)==1, 'Dataloader must have a single large batch'

    idx, test_data, test_target = next(iter(test_dataloader))
    test_data = test_data.to(network.device)
    test_target = test_target.to(network.device)
    num_patterns = test_data.shape[0]

    output = network.forward(test_data, no_grad=True)
    if sorted_output_idx is not None:
        output = output[:, sorted_output_idx]
    test_loss = network.criterion(output, test_target).item()
    test_accuracy = 100 * torch.sum(torch.argmax(output, dim=1) ==
                               torch.argmax(test_target, dim=1)) / num_patterns
    test_accuracy = test_accuracy.item()

    return test_loss, test_accuracy


def compute_test_loss_and_accuracy_history(network, test_dataloader, sorted_output_idx=None, store_history=False,
                                           plot=False, status_bar=False, title=None):
    """
    Assumes network has been trained with store_params=True. Evaluates test_loss at each train step in the
    param_history.
    :param network:
    :param test_dataloader:
    :param sorted_output_idx: tensor of int
    :param store_history: bool
    :param plot: bool
    :param status_bar: bool
    :param title: str
    """
    assert len(test_dataloader)==1, 'Dataloader must have a single large batch'
    assert len(network.param_history) > 0, 'Network must contain a stored param_history'

    idx, test_data, test_target = next(iter(test_dataloader))
    test_data = test_data.to(network.device)
    test_target = test_target.to(network.device)
    test_loss_history = []
    test_accuracy_history = []
    num_patterns = test_data.shape[0]

    if store_history:
        network.reset_history()

    if status_bar:
        iter_param_history = tqdm(network.param_history, desc='Test history')
    else:
        iter_param_history = network.param_history
    for state_dict in iter_param_history:
        network.load_state_dict(state_dict)
        output = network.forward(test_data, store_history=store_history, no_grad=True)
        if sorted_output_idx is not None:
            output = output[:, sorted_output_idx]
        test_loss_history.append(network.criterion(output, test_target).item())
        accuracy = 100 * torch.sum(torch.argmax(output, dim=1) ==
                                   torch.argmax(test_target, dim=1)) / num_patterns
        test_accuracy_history.append(accuracy.item())

    network.test_loss_history = torch.tensor(test_loss_history).cpu()
    network.test_accuracy_history = torch.tensor(test_accuracy_history).cpu()

    if title is None:
        title_str = ''
    else:
        title_str = ': %s' % str(title)

    if plot:
        fig = plt.figure()
        plt.plot(network.param_history_steps, network.test_loss_history)
        plt.xlabel('Training steps')
        plt.ylabel('Test loss')
        fig.suptitle('Test loss%s' % title_str)
        fig.tight_layout()
        fig.show()

        fig = plt.figure()
        plt.plot(network.param_history_steps, network.test_accuracy_history)
        plt.xlabel('Training steps')
        plt.ylabel('Test accuracy')
        fig.suptitle('Test accuracy%s' % title_str)
        fig.tight_layout()
        fig.show()

    return network.test_loss_history, network.test_accuracy_history


def recompute_validation_loss_and_accuracy(network, sorted_output_idx, store=False):
    """

    :param network:
    :param sorted_output_idx:
    :param store:
    :param plot:
    :return:
    """

    # Sort output history
    val_output_history = network.val_output_history[:, :, sorted_output_idx]

    # Recompute loss
    sorted_val_loss_history = []
    sorted_val_accuracy_history = []
    num_patterns = val_output_history.shape[1]
    for batch_output in val_output_history:
        loss = network.criterion(batch_output, network.val_target).item()
        accuracy = 100 * torch.sum(torch.argmax(batch_output, dim=1) ==
                                   torch.argmax(network.val_target, dim=1)) / num_patterns

        sorted_val_loss_history.append(loss)
        sorted_val_accuracy_history.append(accuracy.item())

    sorted_val_loss_history = torch.tensor(sorted_val_loss_history)
    sorted_val_accuracy_history = torch.tensor(sorted_val_accuracy_history)

    if store:
        network.val_output_history = val_output_history
        network.val_loss_history = sorted_val_loss_history
        network.val_accuracy_history = sorted_val_accuracy_history

    return sorted_val_loss_history, sorted_val_accuracy_history


def recompute_train_loss_and_accuracy(network, sorted_output_idx=None, bin_size=100, plot=False, title=None):
    """

    :param network:
    :param sorted_output_idx:
    :param bin_size: int
    :param plot: bool
    :param title: str
    :return: tuple of tensor
    """

    # Sort output history
    if network.Output.E.activity_history.dim() > 2:
        output_history = network.Output.E.activity_history[:, -1, :]
    else:
        output_history = network.Output.E.activity_history
    if sorted_output_idx is not None:
        output_history = output_history[:, sorted_output_idx]
    target_history = network.target_history
    num_units = output_history.shape[1]
    num_patterns = output_history.shape[0]

    # Bin output history to compute average loss & accuracy over training
    num_bins = num_patterns // bin_size
    excess = num_patterns % bin_size
    if excess > 0:
        output_history = output_history[:-excess]
        target_history = target_history[:-excess]

    binned_output_history = output_history.reshape(num_bins, bin_size, num_units)
    binned_target_history = target_history.reshape(num_bins, bin_size, num_units)
    binned_train_loss_steps = torch.arange(bin_size, bin_size * (num_bins + 1), bin_size)

    # Recompute loss
    sorted_loss_history = []
    sorted_accuracy_history = []

    for (batch_output, batch_target) in zip(binned_output_history, binned_target_history):
        loss = network.criterion(batch_output, batch_target).item()
        predictions = torch.argmax(batch_output, dim=1)
        labels = torch.argmax(batch_target, dim=1)
        accuracy = 100 * torch.sum(predictions == labels) / bin_size

        sorted_loss_history.append(loss)
        sorted_accuracy_history.append(accuracy.item())

    sorted_loss_history = torch.tensor(sorted_loss_history)
    sorted_accuracy_history = torch.tensor(sorted_accuracy_history)

    if title is None:
        title_str = ''
    else:
        title_str = ': %s' % str(title)
    if plot:
        fig = plt.figure()
        plt.plot(binned_train_loss_steps, sorted_loss_history)

        plt.title('Train Loss%s' % title_str)
        plt.ylabel('Loss')
        plt.xlabel('Train steps')
        plt.ylim((0, plt.ylim()[1]))
        fig.show()

        fig = plt.figure()
        plt.plot(binned_train_loss_steps, sorted_accuracy_history)
        plt.title('Train accuracy%s' % title_str)
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Train steps')
        plt.ylim((0, max(100, plt.ylim()[1])))
        fig.show()

    return binned_train_loss_steps, sorted_loss_history, sorted_accuracy_history


def get_optimal_sorting(network, test_dataloader, plot=False):
    """
    Measure test loss on re-sorted activity at every point in the training history
    :param network:
    :param test_dataloader:
    :param plot:
    :return:
    """
    assert len(test_dataloader)==1, 'Dataloader must have a single large batch'
    output_pop = network.output_pop

    optimal_loss_history = []
    sorting_history = []
    history_len = output_pop.activity_history.shape[0]
    idx, test_data, test_target = next(iter(test_dataloader))

    from tqdm.autonotebook import tqdm
    for t in tqdm(range(history_len)):
        network.load_state_dict(network.param_history[t])
        output = network.forward(test_data, no_grad=True)  # row=patterns, col=units

        # Get average output for each label class
        num_units = network.val_target.shape[1]
        num_labels = num_units
        avg_output = torch.zeros(num_labels, num_units)
        targets = torch.argmax(network.val_target, dim=1)  # convert from 1-hot vector to int label
        for label in range(num_labels):
            label_idx = torch.where(targets == label)  # find all instances of given label
            avg_output[label, :] = torch.mean(output[label_idx], dim=0)

        # Find optimal output unit (column) sorting given average responses
        optimal_sorting = get_diag_argmax_row_indexes(avg_output.T)
        sorted_activity = avg_output[:, optimal_sorting]
        optimal_loss = network.criterion(sorted_activity, torch.eye(num_units))
        optimal_loss_history.append(optimal_loss)
        sorting_history.append(optimal_sorting)

        # Pick timepoint with lowest sorted loss
    optimal_loss_history = torch.stack(optimal_loss_history)
    min_loss_idx = torch.argmin(optimal_loss_history)
    min_loss_sorting = sorting_history[min_loss_idx]

    if plot:
        plt.scatter(min_loss_idx,torch.min(optimal_loss_history),color='red')
        plt.plot(optimal_loss_history)
        plt.title('optimal loss history (re-sorted for each point)')
        plt.show()

    return min_loss_sorting


def recompute_history(network, output_sorting):
    """
    Re-compute activity history, loss history, and weight+bias history
    with new sorting of the output units
    """
    output_pop = network.output_pop

    # Sort activity history
    if output_pop.activity_history.dim() > 2:
        output_pop.activity_history.data = output_pop.activity_history[:, :, output_sorting]
    else:
        output_pop.activity_history.data = output_pop.activity_history[:, output_sorting]

    for t in range(len(network.param_history)):
        # TODO: why is this starting with index == -1?
        # Recompute loss history
        if output_pop.activity_history.dim() > 2:
            output = output_pop.activity_history[t-1, -1, :]
        else:
            output = output_pop.activity_history[t - 1, :]
        target = network.target_history[t-1]
        network.loss_history[t-1] = network.criterion(output, target)

        # Sort weights going to and from the output population
        for proj in output_pop.incoming_projections.values():
            sorted_weights = network.param_history[t][f'module_dict.{proj.name}.weight'][output_sorting,:]
            network.param_history[t][f'module_dict.{proj.name}.weight'] = sorted_weights

        for proj in output_pop.outgoing_projections.values():
            sorted_weights = network.param_history[t][f'module_dict.{proj.name}.weight'][:,output_sorting]
            network.param_history[t][f'module_dict.{proj.name}.weight'] = sorted_weights

        # Sort output bias
        sorted_bias = network.param_history[t][f'parameter_dict.{output_pop.fullname}_bias'][output_sorting]
        network.param_history[t][f'parameter_dict.{output_pop.fullname}_bias'] = sorted_bias

    # Update network with re-sorted weights from final state
    network.load_state_dict(network.param_history[-1])


def analyze_simple_EIANN_epoch_loss_and_accuracy(network, target, sorted_output_idx=None, plot=False):
    """
    Split output activity history into "epoch" blocks (e.g. containint all 21 patterns) and compute accuracy for each
    block (with a given sorting of output units) to find the epoch with lowest loss.
    TODO: this method is meant to be used after train, needs a separate method to compute test loss and accuracy
    :param network:
    :param target:
    :param sorted_output_idx:
    :param plot:
    :return: tuple: (int, tensor of float, tensor of float)
    """
    output_pop = network.output_pop

    epoch_loss = []
    epoch_argmax_accuracy = []

    if output_pop.activity_history.dim() > 2:
        output_history = output_pop.activity_history[network.sorted_sample_indexes, -1, :]
    else:
        output_history = output_pop.activity_history[network.sorted_sample_indexes, :]

    if sorted_output_idx is not None:
        output_history = output_history[:, sorted_output_idx]
    start = 0
    while start < output_history.shape[0]:
        end = start + target.shape[0]
        epoch_output = output_history[start:end, :]
        loss = network.criterion(epoch_output, target)
        epoch_loss.append(loss.item())
        start += target.shape[0]
        accuracy = torch.sum(torch.argmax(epoch_output, axis=1) == torch.argmax(target, axis=1))
        epoch_argmax_accuracy.append(accuracy.item() / target.shape[0] * 100.)

    epoch_loss = torch.tensor(epoch_loss)
    epoch_argmax_accuracy = torch.tensor(epoch_argmax_accuracy)
    print(epoch_loss.shape)
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


def test_EIANN_autoenc_config(network, train_dataloader, test_dataloader, epochs, supervised=True,
                              store_dynamics=False):
    """

    :param network:
    :param train_dataloader:
    :param test_dataloader:
    :param epochs:
    :param supervised:
    :param store_dynamics:
    """

    plot.plot_EIANN_1_hidden_autoenc_config_summary(network, test_dataloader, title='Initial')

    network.train(train_dataloader, val_dataloader=test_dataloader, epochs=epochs, store_history=True,
                  store_dynamics=store_dynamics, status_bar=True)

    if not supervised:
        min_loss_idx, sorted_output_idx = sort_by_val_history(network, plot=plot)
        sorted_val_loss_history, sorted_val_accuracy_history = \
            recompute_validation_loss_and_accuracy(network, sorted_output_idx=sorted_output_idx, store=True)
    else:
        min_loss_idx = torch.argmin(network.val_loss_history)
        sorted_output_idx = None
        sorted_val_loss_history = network.val_loss_history
        sorted_val_accuracy_history = network.val_accuracy_history

    binned_train_loss_steps, sorted_train_loss_history, sorted_train_accuracy_history = \
        recompute_train_loss_and_accuracy(network, sorted_output_idx=sorted_output_idx, plot=True)

    plot.plot_train_loss_history(network)
    plot.plot_validate_loss_history(network)

    plot.plot_EIANN_1_hidden_autoenc_config_summary(network, test_dataloader, sorted_output_idx=sorted_output_idx,
                                               title='Final')


def test_EIANN_CL_config(network, dataloader, epochs, split=0.75, supervised=True, generator=None):

    num_samples = len(dataloader)
    network.test(dataloader, store_history=True, store_dynamics=True, status_bar=True)
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

    network.test(dataloader, store_history=True, store_dynamics=True, status_bar=True)
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

    network.test(dataloader, store_history=True, store_dynamics=True, status_bar=True)
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
    assert len(test_dataloader)==1, 'Dataloader must have a single large batch'

    indexes, data, targets = next(iter(test_dataloader))
    data = data.to(network.device)
    targets = targets.to(network.device)
    labels = torch.argmax(targets, axis=1)
    output = network.forward(data, no_grad=True)
    percent_correct = 100 * torch.sum(torch.argmax(output, dim=1) == labels) / data.shape[0]
    percent_correct = torch.round(percent_correct, decimals=2)
    print(f'Batch accuracy = {percent_correct}%')
    return percent_correct


def compute_dParam_history(network):
    dParam_history = {name: [] for name in network.state_dict()}

    for i in range(len(network.param_history) - 1):
        state_dict1 = network.param_history[i]
        state_dict2 = network.param_history[i + 1]

        for param_name, param_val1, param_val2 in zip(state_dict1.keys(), state_dict1.values(), state_dict2.values()):
            d_param = param_val2 - param_val1
            dParam_history[param_name].append(d_param)

    for name, value in dParam_history.items():
        dParam_history[name] = torch.stack(value)

    return dParam_history


def compute_sparsity_history(activity_history):
    """
    Sparsity metric from (Vinje & Gallant 2000): https://www.science.org/doi/10.1126/science.287.5456.1273
    TODO: check activity_history dimensions
    """
    population_activity = activity_history #dims: 0=history, 1=dynamics, 2=patterns, 3=units
    n = population_activity.shape[3]
    activity_fraction = (torch.sum(population_activity,dim=3) / n) ** 2 / (torch.sum((population_activity**2 / n),dim=3)+1e-10)
    sparsity_history = (1 - activity_fraction) / (1 - 1 / n)
    return sparsity_history


def compute_selectivity_history(activity_history):
    """
    Sparsity metric from (Vinje & Gallant 2000): https://www.science.org/doi/10.1126/science.287.5456.1273
    TODO: check activity_history dimensions
    """
    population_activity = activity_history #dims: 0=history, 1=dynamics, 2=patterns, 3=units
    n = population_activity.shape[2]
    activity_fraction = (torch.sum(population_activity,dim=2) / n) ** 2 / (torch.sum((population_activity**2 / n),dim=2)+1e-10)
    selectivity_history = (1 - activity_fraction) / (1 - 1 / n)
    return selectivity_history


def count_dict_elements(dict1, leaf=0):
    nodes = dict1.keys()
    for node in nodes:
        subnode = dict1[node]
        if isinstance(subnode, dict):
            leaf = count_dict_elements(subnode, leaf)
        else:
            leaf += 1
    return leaf


def linear(x):
    """
    Linear activation function
    """
    return x


def spatial_structure_similarity_fft(img1, img2):
    '''
    Compute the structural similarity of two images based on the correlation of their 2D spatial frequency distributions
    :param img1: 2D numpy array of pixels
    :param img2: 2D numpy array of pixels
    :return:
    '''
    # Compute the 2D spatial frequency distribution of each image
    freq1 = np.abs(np.fft.fftshift(np.fft.fft2(img1 - np.mean(img1))))
    freq2 = np.abs(np.fft.fftshift(np.fft.fft2(img2 - np.mean(img2))))

    # Compute the frequency correlation
    spatial_structure_similarity =  signal.correlate2d(freq1, freq2, mode='valid')[0][0]

    return spatial_structure_similarity


def compute_rf_structure(receptive_fields, method='fft'):
    structure_sim_ls = []
    for unit_rf in receptive_fields:
        s = 0
        if torch.all(a == 0): # if receptive field is all zeros
            structure_sim_ls.append(np.nan)
        else:
            for i in range(3):  # structural similarity to noise (averaged across 3 random noise images)
                noise = np.random.uniform(min(unit_rf), max(unit_rf), (28, 28))

                if method == 'fft':
                    reference_correlation = spatial_structure_similarity_fft(noise, noise)
                    s += spatial_structure_similarity_fft(unit_rf.view(28, 28).numpy(), noise) / reference_correlation
                elif method == 'ssim':
                    s += metrics.structural_similarity(unit_rf.view(28, 28).numpy(), noise)

            structure_sim_ls.append(s / 3)
    structure = 1 - np.array(structure_sim_ls)
    return structure


def compute_representation_metrics(population, test_dataloader, receptive_fields=None, plot=False):
    """
    Compute representation metrics for a population of neurons
    :param population: Population object
    :param receptive_fields: (optional) receptive fields for each neuron
    :return: dictionary of metrics
    """

    network = population.network
    idx, data, target = next(iter(test_dataloader))
    data.to(network.device)
    network.forward(data, no_grad=True)
    activity = population.activity

    num_patterns = activity.shape[0]
    num_units = activity.shape[1]

    # Compute population sparsity
    activity_fraction = (torch.sum(activity, dim=1) / num_units) ** 2 / torch.sum(activity ** 2 / num_units, dim=1)
    sparsity = (1 - activity_fraction) / (1 - 1 / num_units)
    sparsity[torch.where(torch.sum(activity, dim=1) == 0.)] = 0.
        # fraction_nonzero_units = np.count_nonzero(activity, axis=1) / num_units
        # active_pattern_idx = np.where(fraction_nonzero_units != 0.)[0] #exlcude silent patterns
        # sparsity = 1 - fraction_nonzero_units[active_pattern_idx]

    total_act = torch.sum(population.activity, dim=0)
    active_units_idx = torch.where(total_act > 1e-10)[0]

    # Compute unit selectivity
    activity_fraction = (torch.sum(activity[:,active_units_idx], dim=0) / num_patterns)**2 / \
                        torch.sum(activity[:,active_units_idx]**2 / num_patterns, dim=0)
    selectivity = (1 - activity_fraction) / (1 - 1 / num_patterns)
    selectivity[torch.where(torch.sum(activity[:,active_units_idx], dim=0) == 0.)] = 0.
        # fraction_nonzero_patterns = np.count_nonzero(activity, axis=0) / num_patterns
        # active_unit_idx = np.where(fraction_nonzero_patterns != 0.)[0] #exlcude silent units
        # selectivity = 1 - fraction_nonzero_patterns[active_unit_idx]

    # Compute discriminability
    silent_pattern_idx = np.where(torch.sum(activity, dim=1) == 0.)[0]
    similarity_matrix = cosine_similarity(activity)
    similarity_matrix[silent_pattern_idx,:] = 1
    similarity_matrix[:,silent_pattern_idx] = 1
    similarity_matrix_idx = np.tril_indices_from(similarity_matrix, -1) # select values below diagonal
    similarity = similarity_matrix[similarity_matrix_idx]
    discriminability = 1 - similarity

    # Compute structure
    if receptive_fields is not None:
        receptive_fields = receptive_fields[active_units_idx]
        structure = compute_rf_structure(receptive_fields)
    else:
        structure = None

    if plot:
        fig, ax = plt.subplots(2,2,figsize=[12,5])
        ax[0,0].hist(sparsity,50)
        ax[0,0].set_title('Sparsity distribution')
        ax[0,0].set_ylabel('num patterns')
        ax[0,0].set_xlabel('(1 - fraction active units)')

        ax[0,1].hist(selectivity,50)
        ax[0,1].set_title('Selectivity distribution')
        ax[0,1].set_ylabel('num units')
        ax[0,1].set_xlabel('(1 - fraction active patterns)')

        ax[1,0].set_title('Discriminability distribution')
        ax[1,0].hist(discriminability, 50)
        ax[1,0].set_ylabel('pattern pairs')
        ax[1,0].set_xlabel('(1 - cosine similarity)')

        if receptive_fields is not None:
            ax[1,1].hist(structure, 50)
            ax[1,1].set_title('Structure')
            ax[1,1].set_ylabel('num units')
            ax[1,1].set_xlabel('(1 - similarity to random noise)')
            plt.tight_layout()
        else:
            ax[1,1].axis('off')

    return {'sparsity': sparsity, 'selectivity': selectivity,
            'discriminability': discriminability, 'structure': structure}


def compute_alternate_dParam_history(dataloader, network, network2=None, data_seed=0, save_path=None):
    """
    Iterate through the parameter history and compute both the actual dParam at each training step 
    and the alternate dParam predicted from either a backprop/gradient descent step or from an identical 
    network with a custom learning rule

    :param dataloader:
    :param network: network with param_history
    :param network2: (optional) network with a custom learning rule
    :return: list of angles (in degrees)
    """
    assert len(network.param_history)>0, 'Network must have param_history'

    if len(dataloader) > 1: # if dataloader is split into batches
        if dataloader.generator is not None:
            dataloader.generator.manual_seed(data_seed)
        data_iter = iter(train_sub_dataloader)
    else:
        idx, data, target = next(iter(dataloader)) # if dataloader is a single batch

    if network2 is None:
        # Turn on gradient tracking to compute backprop dW
        test_network = copy.copy(network)
        test_network.backward_steps = 3
        for parameter in network.parameters(): 
            if parameter.is_learned:
                parameter.requires_grad = True
    else:
        test_network = copy.copy(network2)
        if "Backprop" in str(test_network.backward_methods):
            assert test_network.backward_steps > 0, "Backprop network must have backward_steps>0!"
    test_network.reset_history()

    if len(network.prev_param_history)==0: # if interval step is 1
        prev_param_history = network.param_history[:]
        param_history = network.param_history[1:]
    else:
        prev_param_history = network.prev_param_history
        param_history = network.param_history[1:] # exclude initial params


    actual_dParam_history_dict = {name:[] for name,param in test_network.named_parameters() if param.is_learned}
    predicted_dParam_history_dict = {name:[] for name,param in test_network.named_parameters() if param.is_learned}
    actual_dParam_history_all = []
    predicted_dParam_history_all = []

    for t in tqdm(range(len(param_history))):  
        # Load params into network
        state_dict = prev_param_history[t]
        if network2 is not None:
            state_dict = {name:param for name,param in state_dict.items() if name in test_network.state_dict()}
        test_network.load_state_dict(state_dict)
        test_network.param_history.append(copy.deepcopy(state_dict))

        # Compute forward pass
        if len(dataloader) > 1:
            idx, data, target = next(data_iter)
        target = target.squeeze()
        output = test_network.forward(data)
        loss = test_network.criterion(output, target)            

        # Compute backward pass update
        if network2 is None:
            # Regular backprop update
            test_network.zero_grad()
            loss.backward() 
            test_network.optimizer.step()    
        else:
            # Backward update speficied by learning rules in network2
            for backward in test_network.backward_methods:
                backward(test_network, output, target)

            # Step weights and biases
            for layer in test_network:
                for population in layer:
                    if population.include_bias:
                        population.bias_learning_rule.step()
                    for projection in population:
                        projection.learning_rule.step()
            test_network.constrain_weights_and_biases()

        new_state_dict = test_network.state_dict() 
        test_network.param_history.append(copy.deepcopy(new_state_dict))

        # Compute the predicted dParam (from the test network)
        dParam_vec = []
        for key in predicted_dParam_history_dict:
            dParam = (new_state_dict[key]-state_dict[key])
            predicted_dParam_history_dict[key].append(dParam)
            dParam_vec.append(dParam.flatten())
        predicted_dParam_history_all.append(torch.cat(dParam_vec))

        # Compute the actual dParam of the first network
        next_state_dict = param_history[t]
        dParam_vec = []
        for key in actual_dParam_history_dict:
            dParam = (next_state_dict[key]-state_dict[key])
            actual_dParam_history_dict[key].append(dParam)
            dParam_vec.append(dParam.flatten())
        actual_dParam_history_all.append(torch.cat(dParam_vec))

    predicted_dParam_history_dict['all_params'] = predicted_dParam_history_all
    actual_dParam_history_dict['all_params'] = actual_dParam_history_all

    test_network.predicted_dParam_history = predicted_dParam_history_dict
    test_network.actual_dParam_history = actual_dParam_history_dict

    if save_path is not None:
        test_network.params_to_save.append('predicted_dParam_history')
        test_network.params_to_save.append('actual_dParam_history')
        test_network.save(save_path)

    return test_network


def compute_dW_angles(test_network, plot=False):
    '''
    Compute the angle between the actual and predicted parameter updates (dW) for each training step.
    The angle is computed as the arccosine of the dot product between the two vectors, normalized by the product of their norms.    
    '''

    if plot:
        n_params = len(test_network.actual_dParam_history)
        fig, axes = plt.subplots(n_params, 1, figsize=(8,n_params*2))
        ax_top = axes[0]

    angles = {}

    for i, param_name in enumerate(test_network.actual_dParam_history):
        angles[param_name] = []
        for predicted_dParam, actual_dParam in zip(test_network.predicted_dParam_history[param_name], test_network.actual_dParam_history[param_name]):
            # Compute angle between parameter update (dW) vectors
            predicted_dParam = predicted_dParam.flatten()
            actual_dParam = actual_dParam.flatten()
            vector_product = torch.dot(predicted_dParam, actual_dParam) / (torch.norm(predicted_dParam)*torch.norm(actual_dParam)+1e-100)
            angle_rad = np.arccos(torch.round(vector_product,decimals=5))
            angle = angle_rad * 180 / np.pi
            angles[param_name].append(angle)
        
        if plot:
            ax = axes[n_params-(i+1)]
            ax.plot(angles[param_name])
            ax_top.plot(angles[param_name], color='gray', alpha=0.3)
            ax.plot([0, len(angles[param_name])], [90, 90], color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Training step')
            ax.set_ylabel('Angle between \nlearning rules (degrees)')

            max_angle = max(95, np.max(angles[param_name]))
            ax.set_ylim(bottom=0, top=max_angle)
            ax.set_yticks(np.arange(0, max_angle+1, 30))
            if i == n_params-1:
                ax.set_ylim(bottom=0, top=120)
                ax.set_yticks(np.arange(0, 121, 30))

            avg_angle = np.mean(angles[param_name])
            ax.text(0.03, 0.08, f'Avg angle = {avg_angle:.2f} degrees', transform=ax.transAxes)
            if '.' in param_name:
                param_name = param_name.split('.')[1]
            ax.set_title(param_name)

            plt.tight_layout()
            fig.show()

    return angles


# *******************************************************************
# MNIST-specific functions
# *******************************************************************

def compute_act_weighted_avg(population, dataloader):
    """
    Compute activity-weighted average input for every unit in the population

    :param population:
    :param dataloader:
    :return:
    """

    idx, data, target = next(iter(dataloader))
    network = population.network

    network.forward(data, no_grad=True)  # compute unit activities in forward pass
    pop_activity = population.activity
    weighted_avg_input = (data.T @ pop_activity) / (pop_activity.sum(axis=0) + 0.0001) # + epsilon to avoid div-by-0 error
    weighted_avg_input = weighted_avg_input.T

    network.forward(weighted_avg_input, no_grad=True)  # compute unit activities in forward pass
    activity_preferred_inputs = population.activity.detach().clone()

    return weighted_avg_input, activity_preferred_inputs


def compute_maxact_receptive_fields(population, dataloader, num_units=None, sigmoid=False):
    """
    Use the 'activation maximization' method to compute receptive fields for all units in the population

    :param population:
    :param dataloader:
    :param num_units:
    :param sigmoid: if True, use sigmoid activation function for the input images;
                    if False, returns unfiltered receptive fields and activities from act_weighted_avg images
    :return:
    """

    idx, data, target = next(iter(dataloader))
    learning_rate = 0.1
    num_steps = 10000
    network = population.network

    # turn on network gradients
    if network.forward(data[0]).requires_grad == False:
        network.backward_steps = 1
        for param in network.parameters():
            param.requires_grad = True

    weighted_avg_input, activity_preferred_inputs = compute_act_weighted_avg(population, dataloader)

    if num_units is None or num_units>population.size:
        num_units = population.size

    input_images = weighted_avg_input[0:num_units,:]
    input_images.requires_grad = True
    optimizer = torch.optim.SGD([input_images], lr=learning_rate)

    loss_history = []
    print("Optimizing receptive field images...")
    for step in tqdm(range(num_steps)):
        if sigmoid:
            im = torch.sigmoid((input_images-0.5)*10)
            network.forward(im)  # compute unit activities in forward pass
        else:
            network.forward(input_images)  # compute unit activities in forward pass
        pop_activity = population.activity[:,0:num_units]
        loss = torch.sum(-torch.log(torch.diagonal(pop_activity) + 0.001))
        loss_history.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    receptive_fields = input_images.detach().clone()
    if sigmoid:
        receptive_fields = torch.sigmoid((receptive_fields-0.5)*10)
        network.forward(receptive_fields, no_grad=True)  # compute unit activities in forward pass
        activity_preferred_inputs = population.activity[:,0:num_units].detach().clone()

    return receptive_fields, activity_preferred_inputs


def set_activation(network, activation, **kwargs):

    # Set callable activation function
    if isinstance(activation, str):
        if activation in globals():
            activation = globals()[activation]
        elif hasattr(external, activation):
            activation = getattr(external, activation)
    if not callable(activation):
        raise RuntimeError \
            ('Population: callable for activation: %s must be imported' % activation)
    activation_f = lambda x: activation(x, **kwargs)

    for i, layer in enumerate(network):
        if i > 0:
            for population in layer:
                population.activation = activation_f


def compute_unit_receptive_field(population, dataloader, unit):
    """
    Use the 'activation maximization' method to compute receptive fields for all units in the population

    :param population:
    :param dataloader:
    :param num_units:
    :return:
    """

    idx, data, target = next(iter(dataloader))
    learning_rate = 0.1
    num_steps = 10000
    network = population.network

    # turn on network gradients
    if network.forward(data[0]).requires_grad == False:
        network.backward_steps = 1
        for param in network.parameters():
            param.requires_grad = True

    weighted_avg_input = compute_act_weighted_avg(population, dataloader)

    input_image = weighted_avg_input[unit]
    input_image.requires_grad = True
    optimizer = torch.optim.SGD([input_image], lr=learning_rate)

    print("Optimizing receptive field images...")
    for step in tqdm(range(num_steps)):
        network.forward(input_image)  # compute unit activities in forward pass
        unit_activity = population.activity[unit]
        loss = -torch.log(unit_activity + 0.001)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return input_image.detach()


def compute_PSD(receptive_field, plot=False):
    '''
    Compute the power spectral density of a receptive field image
    Function based on https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/

    :param receptive_field: 2D numpy array of pixels
    :param plot: bool
    :return: frequencies, spectral_power, peak_spatial_frequency
    '''

    # Take Fourier transform of the receptive field
    fourier_image = np.fft.fftn(receptive_field)
    fourier_amplitudes = np.abs(fourier_image)**2

    # Get frequencies corresponding to signal PSD
    # (bin the results of the Fourier analysis by contstructing an array of wave vector norms)
    npix = receptive_field.shape[0] # this only works for a square image
    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knorm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knorm = knorm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    # Create the frequency power spectrum
    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knorm, fourier_amplitudes,
                                         statistic = "mean",
                                         bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    peak_spatial_frequency = np.argmax(Abins)
    spectral_power = Abins
    frequencies = kvals

    if plot:
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(receptive_field)
        ax[1].loglog(kvals, Abins)
        ax[1].set_xlabel("Spatial Frequency $k$ [pixels]")
        ax[1].set_ylabel("Power per Spatial Frequency $P(k)$")
        plt.show()

    return frequencies, spectral_power, peak_spatial_frequency


def check_equilibration_dynamics(network, dataloader, equilibration_activity_tolerance, debug=False, disp=False,
                                 plot=False):
    """

    :param network: :class:'Network'
    :param dataloader: :class:'torch.DataLoader'
    :param equilibration_activity_tolerance: float in [0, 1]
    :param debug: bool
    :param disp: bool
    :param: plot: bool
    :return: bool
    """
    idx, data, targets = next(iter(dataloader))
    network.forward(data, store_dynamics=True, no_grad=True)

    if plot:
        max_rows = 1
        for layer in network:
            max_rows = max(max_rows, len(layer.populations))
        cols = len(network.layers) - 1
        fig, axes = plt.subplots(max_rows, cols, figsize=(3.2 * cols, 3. * max_rows))
        if max_rows == 1:
            if cols == 1:
                axes = [[axes]]
            else:
                axes = [axes]
        elif cols == 1:
            axes = [[axis] for axis in axes]

    for i, layer in enumerate(network):
        if i > 0:
            col = i - 1
            for row, population in enumerate(layer):
                pop_activity = torch.stack(population.forward_steps_activity)
                if pop_activity.shape[0] == 1:
                    return True
                average_activity = torch.mean(pop_activity, dim=(1, 2))
                if plot:
                    this_axis = axes[row][col]
                    this_axis.plot(average_activity)
                    this_axis.set_xlabel('Equilibration time steps')
                    this_axis.set_ylabel('Average population activity')
                    this_axis.set_title('%s.%s' % (layer.name, population.name))
                    this_axis.set_ylim((0., this_axis.get_ylim()[1]))
                equil_mean = torch.mean(average_activity[-2:])
                if equil_mean > 0:
                    equil_delta = torch.abs(average_activity[-1] - average_activity[-2])
                    equil_error = equil_delta/equil_mean
                    if equil_error > equilibration_activity_tolerance:
                        if disp:
                            print('population: %s failed check_equilibration_dynamics: %.2f' %
                                  (population.fullname, equil_error))
                        if not debug:
                            return False
    if plot:
        fig.suptitle('Activity dynamics')
        fig.tight_layout()
        fig.show()
    return True
