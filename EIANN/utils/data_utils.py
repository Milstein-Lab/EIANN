
import torch
from torch.utils.data import DataLoader
import numpy as np
import itertools
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import h5py
import os
import yaml
import torchvision

try:
    from collections import Iterable
except:
    from collections.abc import Iterable


# *******************************************************************
# Functions to import and export data
# *******************************************************************


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
        yaml.dump(data, outfile, default_flow_style=False, sort_keys=False, indent=4)


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
    # Initial call to convert the top-level group in the HDF5 file
    # (necessary because the top-level group is not a h5py.Group object)
    with h5py.File(file_path, 'r') as f:
        data_dict = {}
        # Loop over the top-level keys in the HDF5 file
        for key in f.keys():
            if isinstance(f[key], h5py.Group):
                # Recursively convert the group to a nested dictionary
                data_dict[key] = convert_hdf5_group_to_dict(f[key])
            else:
                # If the key corresponds to a dataset, add it to the dictionary
                data_dict[key] = f[key][()]
    return data_dict


def convert_hdf5_group_to_dict(group):
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
            data_dict[key] = convert_hdf5_group_to_dict(group[key])
        else:
            # If the key corresponds to a dataset, add it to the dictionary
            data_dict[key] = group[key][()]

    return data_dict


def dict_to_hdf5(data_dict, file_path):
    """
    Save a nested Python dictionary to an HDF5 file.

    :param data_dict (dict): Nested Python dictionary to save.
    :param file_path (str): Path to the HDF5 file.
    """
    with h5py.File(file_path, 'w') as f:
        # Initial call to save the top-level dictionary to the HDF5 file
        convert_dict_to_hdf5_group(data_dict, f)


def convert_dict_to_hdf5_group(data_dict, group):
    """
    Recursive function to save a nested Python dictionary to an HDF5 group.

    :param data_dict (dict): Nested Python dictionary to save.
    :param group (h5py.Group): The HDF5 group to save the dictionary to.
    """
    for key, value in data_dict.items():
        if isinstance(value, dict):
            # Recursively save nested dictionaries as groups
            subgroup = group.create_group(key, track_order=True)
            convert_dict_to_hdf5_group(value, subgroup)
        else:
            # Save datasets to the HDF5 group
            group.create_dataset(key, data=value, track_order=True)


def load_plot_data(network_name, seed, data_key, file_path=None):
    """
    Load plot data from a hdf5 file
    :param file_path: str
    :param network_name: str
    :param seed: int
    """
    if file_path is None:
        root_dir = get_project_root()
        file_path = root_dir+'/EIANN/data/.plot_data.h5'

    seed = str(seed)
    if os.path.exists(file_path):
        with h5py.File(file_path, 'r') as hdf5_file:            
            if network_name in hdf5_file:
                if seed in hdf5_file[network_name]:
                    if data_key in hdf5_file[network_name][seed]:
                        if isinstance(hdf5_file[network_name][seed][data_key], h5py.Group):
                            data = convert_hdf5_group_to_dict(hdf5_file[network_name][seed][data_key])
                        elif isinstance(hdf5_file[network_name][seed][data_key], h5py.Dataset):
                            data = hdf5_file[network_name][seed][data_key][()]
                        print(f'{data_key} loaded from file: {file_path}')
                        return data
    print(f'{data_key} not found in file: {file_path}')
    return None


def save_plot_data(network_name, seed, data_key, data, file_path=None, overwrite=False):
    """
    Save plot data to an hdf5 file
    :param network_name: str
    :param seed: int
    :param plot_name: str
    :param data: array
    :param file_path: str
    :param overwrite: bool
    """
    if file_path is None:
        root_dir = get_project_root()
        file_path = root_dir + '/EIANN/data/.plot_data.h5'

    seed = str(seed)
    if os.path.exists(file_path):
        with h5py.File(file_path, 'a') as hdf5_file:
            if network_name not in hdf5_file:
                hdf5_file.create_group(network_name, track_order=True)
            if seed not in hdf5_file[network_name]:
                hdf5_file[network_name].create_group(seed, track_order=True)
            if data_key in hdf5_file[network_name][seed] and overwrite:
                del hdf5_file[network_name][seed][data_key]

            if data_key not in hdf5_file[network_name][seed]:
                if isinstance(data, dict):
                    hdf5_file[network_name][seed].create_group(data_key, track_order=True)
                    convert_dict_to_hdf5_group(data, hdf5_file[network_name][seed][data_key])
                else:
                    hdf5_file[network_name][seed].create_dataset(data_key, data=data, track_order=True)
                print(f'{data_key} saved to file: {file_path}')
            else:
                print(f'{data_key} already exists in file: {file_path}')
    else:
        with h5py.File(file_path, 'w') as hdf5_file:
            hdf5_file.create_group(network_name, track_order=True)
            hdf5_file[network_name].create_group(seed, track_order=True)
            hdf5_file[network_name][seed].create_dataset(data_key, data=data, track_order=True)
            print(f'{data_key} saved to file: {file_path}')
    

def get_project_root():
    # Assuming the current script is somewhere within the project directory
    current_path = os.path.abspath(__file__)
    
    # Traverse up the directory tree until the project root is found
    while not os.path.isdir(os.path.join(current_path, 'EIANN')):
        current_path = os.path.dirname(current_path)
        if current_path == os.path.dirname(current_path):
            raise FileNotFoundError("Project root directory 'EIANN' not found")
    
    # return os.path.join(current_path, 'EIANN')
    return current_path


def get_MNIST_dataloaders(sub_dataloader_size=1000, classes=None, batch_size=1):
    """
    Load MNIST dataset into custom dataloaders with sample index
    """

    # Load dataset
    root_dir = get_project_root()
    tensor_flatten = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Lambda(torch.flatten)])
    MNIST_train_dataset = torchvision.datasets.MNIST(root=root_dir+'/EIANN/data/datasets/', train=True, download=True,
                                                     transform=tensor_flatten)
    MNIST_test_dataset = torchvision.datasets.MNIST(root=root_dir+'/EIANN/data/datasets/', train=False, download=True,
                                                    transform=tensor_flatten)

    # Add index to train & test data
    MNIST_train = []
    MNIST_train_CL1 = [] # phase 1 dataset for continual learning
    MNIST_train_CL2 = [] # phase 1 dataset for continual learning
    for idx,(data,label) in enumerate(MNIST_train_dataset):
        target = torch.eye(len(MNIST_train_dataset.classes))[label]
        MNIST_train.append((idx, data, target))

        if classes is not None:
            if label in classes:
                MNIST_train_CL1.append((idx, data, target))
            else:
                MNIST_train_CL2.append((idx, data, target))
        
    MNIST_test = []
    for idx,(data,target) in enumerate(MNIST_test_dataset):
        target = torch.eye(len(MNIST_test_dataset.classes))[target]
        MNIST_test.append((idx, data, target))
        
    # Put data in dataloader
    data_generator = torch.Generator()
    train_dataloader = torch.utils.data.DataLoader(MNIST_train[0:50_000], batch_size=50_000)
    train_sub_dataloader = torch.utils.data.DataLoader(MNIST_train[0:sub_dataloader_size], shuffle=True, generator=data_generator, batch_size=batch_size)
    val_dataloader = torch.utils.data.DataLoader(MNIST_train[-10_000:], batch_size=10_000, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(MNIST_test, batch_size=10_000, shuffle=False)

    if classes is not None:
        train_dataloader_CL1 = torch.utils.data.DataLoader(MNIST_train_CL1[0:sub_dataloader_size], shuffle=True, generator=data_generator, batch_size=1)
        train_dataloader_CL2 = torch.utils.data.DataLoader(MNIST_train_CL2[0:sub_dataloader_size], shuffle=True, generator=data_generator, batch_size=1)
        train_dataloader_CL1_full = torch.utils.data.DataLoader(MNIST_train_CL1[0:sub_dataloader_size], shuffle=True, generator=data_generator, batch_size=sub_dataloader_size)
        train_dataloader_CL2_full = torch.utils.data.DataLoader(MNIST_train_CL2[0:sub_dataloader_size], shuffle=True, generator=data_generator, batch_size=sub_dataloader_size)

        return train_dataloader, train_dataloader_CL1_full, train_dataloader_CL2_full, train_dataloader_CL1, train_dataloader_CL2, train_sub_dataloader, val_dataloader, test_dataloader, data_generator
    else:
        return train_dataloader, train_sub_dataloader, val_dataloader, test_dataloader, data_generator



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


def sort_by_class_averaged_val_output(network, index=-1):
    """
    Find the sorting for the class-averaged output at the given index of the validation history

    :param network:
    :param index: int
    :return: sorted_output_idx: array of int
    """
    num_units = network.val_target.shape[1]
    num_labels = num_units
    
    output = network.val_output_history[index]
    
    # Get average output for each label class
    avg_output = torch.zeros(num_labels, num_units)
    targets = torch.argmax(network.val_target, dim=1)  # convert from 1-hot vector to int label
    for label in range(num_labels):
        label_idx = torch.where(targets == label)  # find all instances of given label
        avg_output[label, :] = torch.mean(output[label_idx], dim=0)
    
    # Find optimal output unit (column) sorting given average responses
    sorted_output_idx = get_diag_argmax_row_indexes(avg_output.T)
    
    return sorted_output_idx


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
