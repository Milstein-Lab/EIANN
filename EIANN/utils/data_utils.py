
import torch
import numpy as np
import itertools
import h5py
import os
import yaml
import torchvision
import subprocess
import tempfile
import shutil


# *******************************************************************
# Functions to import and export data
# *******************************************************************

def get_project_root():
    """
    Find the root directory of the project containing the 'EIANN' folder.

    Returns
    -------
    str
        Absolute path to the project root directory.

    Raises
    ------
    FileNotFoundError
        If the 'EIANN' directory cannot be found.
    """
    # Assuming the current script is somewhere within the project directory
    current_path = os.path.abspath(__file__)
    
    # Traverse up the directory tree until the project root is found
    while not os.path.isdir(os.path.join(current_path, 'EIANN')):
        current_path = os.path.dirname(current_path)
        if current_path == os.path.dirname(current_path):
            raise FileNotFoundError("Project root directory 'EIANN' not found")    
    return current_path


def nested_convert_scalars(data):
    """
    Crawl a nested dictionary and recursively convert all NumPy scalar types to native Python types in the nested structure.

    Parameters
    ----------
    data : any
        A potentially nested structure (dict, list, tuple) containing scalars.

    Returns
    -------
    any
        The same structure with NumPy scalars converted to native Python types.
    """
    try:
        from collections import Iterable
    except:
        from collections.abc import Iterable

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
    Write a dictionary to a YAML file.

    Parameters
    ----------
    file_path : str
        Path to the output YAML file. Should end with '.yaml'.
    data : dict
        Dictionary to write to the file.
    convert_scalars : bool, optional
        Whether to convert NumPy scalar types to native Python types before writing. Default is True.
    """
    import yaml
    with open(file_path, 'w') as outfile:
        if convert_scalars:
            data = nested_convert_scalars(data)
        yaml.dump(data, outfile, default_flow_style=False, sort_keys=False, indent=4)


def read_from_yaml(file_path, Loader=None):
    """
    Read a dictionary from a YAML file.

    Parameters
    ----------
    file_path : str
        Path to the YAML file to read.
    Loader : yaml.Loader or None, optional
        YAML loader to use. Defaults to `yaml.FullLoader`.

    Returns
    -------
    dict
        Dictionary parsed from the YAML file.

    Raises
    ------
    Exception
        If the specified file does not exist.
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
    Export metrics data to an HDF5 file under a specific model group.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary of metrics to export.
    model_name : str
        Name of the model used as the top-level group in the HDF5 file.
    path : str
        Path to the HDF5 file. If missing '.hdf5', it will be appended.
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
    Import metrics data from an HDF5 file into a nested dictionary.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.

    Returns
    -------
    dict
        Nested dictionary of metrics by model and metric name.
    """
    metrics_dict = {}
    with h5py.File(filename, 'r') as file:
        for model_name in file:
            metrics_dict[model_name] = {}
            for metric in file[model_name]:
                metrics_dict[model_name][metric] = file[model_name][metric][:]

    return metrics_dict


def hdf5_to_dict(file_path, variable_name=None):
    """
    Convert the contents of an HDF5 file into a nested dictionary.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file.
    variable_name : str, optional
        Name of a specific group or dataset to load. If None, loads the entire file.
        Can be a path like 'group1/subgroup2' to load nested groups.

    Returns
    -------
    dict
        Nested Python dictionary representing the HDF5 file structure.
        If variable_name is specified and points to a dataset, returns a dict with just that dataset.
        If variable_name is specified and points to a group, returns a dict with that group's contents.

    Notes
    -----
    Example usage:
    # Load entire file
    data = hdf5_to_dict('data.h5')

    # Load only a specific group
    data = hdf5_to_dict('data.h5', variable_name='group1')

    # Load only a nested subgroup
    data = hdf5_to_dict('data.h5', variable_name='group1/subgroup2')

    # Load only a specific dataset
    data = hdf5_to_dict('data.h5', variable_name='group1/dataset_name')
    """
    with h5py.File(file_path, 'r') as f:
        if variable_name is None:
            # Load entire file
            data_dict = {}
            for key in f.keys():
                if isinstance(f[key], h5py.Group):
                    data_dict[key] = convert_hdf5_group_to_dict(f[key])
                else:
                    data_dict[key] = f[key][()]
            return data_dict
        else:
            # Load specific group or dataset
            try:
                item = f[variable_name]
                if isinstance(item, h5py.Group):
                    # If it's a group, return its contents as a dict
                    return convert_hdf5_group_to_dict(item)
                else:
                    # If it's a dataset, return a dict with the dataset name as key
                    dataset_name = variable_name.split('/')[-1]  # Get the last part of the path
                    return {dataset_name: item[()]}
            except KeyError:
                print(f"WARNING: '{variable_name.split('/')[-1]}' not found in HDF5 file")
                return None


def convert_hdf5_group_to_dict(group):
    """
    Helper function to recursively convert an HDF5 group to a nested dictionary.

    Parameters
    ----------
    group : h5py.Group
        The HDF5 group to convert.

    Returns
    -------
    dict
        Dictionary representing the structure and datasets within the group.
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
    Save a nested dictionary to an HDF5 file.

    Parameters
    ----------
    data_dict : dict
        Dictionary to save to the file.
    file_path : str
        Destination path for the HDF5 file.
    """
    with h5py.File(file_path, 'w') as f:
        # Initial call to save the top-level dictionary to the HDF5 file
        convert_dict_to_hdf5_group(data_dict, f)


def convert_dict_to_hdf5_group(data_dict, group):
    """
    Recursively write a nested dictionary to an HDF5 group.

    Parameters
    ----------
    data_dict : dict
        Dictionary to write to the HDF5 group.
    group : h5py.Group
        Target HDF5 group for storing the dictionary data.
    """
    for key, value in data_dict.items():
        if isinstance(value, dict):
            # Recursively save nested dictionaries as groups
            subgroup = group.create_group(key, track_order=True)
            convert_dict_to_hdf5_group(value, subgroup)
        else:
            # Save datasets to the HDF5 group
            group.create_dataset(key, data=value, track_order=True)


def save_plot_data(network_name, seed, data_key, data, file_path=None, overwrite=False):
    """
    Save plot data for a specific network and seed into an HDF5 file.

    Parameters
    ----------
    network_name : str
        Name of the network.
    seed : int
        Seed identifier for the data.
    data_key : str
        Key under which to store the data.
    data : array-like or dict
        Data to be saved.
    file_path : str, optional
        Path to the HDF5 file. If None, a default path is used.
    overwrite : bool, optional
        Whether to overwrite existing data at the specified key.
    """
    if file_path is None:
        root_dir = get_project_root()
        file_path = root_dir + '/EIANN/data/plot_data.h5'

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


def load_plot_data(network_name, seed, data_key, file_path=None):
    """
    Load plot data for a specific network and seed from an HDF5 file.

    Parameters
    ----------
    network_name : str
        Name of the network.
    seed : int
        Seed identifier for the data.
    data_key : str
        Key under which the data is stored.
    file_path : str, optional
        Path to the HDF5 file. If None, a default path is used.

    Returns
    -------
    any or None
        The loaded data if present, otherwise None.
    """
    if file_path is None:
        root_dir = get_project_root()
        file_path = root_dir+'/EIANN/data/plot_data.h5'

    seed = str(seed)
    if os.path.exists(file_path):
        print(f'Loading {data_key} from file: {file_path}')
        with h5py.File(file_path, 'r') as hdf5_file:            
            if network_name in hdf5_file:
                if seed in hdf5_file[network_name]:
                    if data_key in hdf5_file[network_name][seed]:
                        if isinstance(hdf5_file[network_name][seed][data_key], h5py.Group):
                            data = convert_hdf5_group_to_dict(hdf5_file[network_name][seed][data_key])
                        elif isinstance(hdf5_file[network_name][seed][data_key], h5py.Dataset):
                            data = hdf5_file[network_name][seed][data_key][()]
                        return data
                    else:
                        print(f'Data key {data_key} not found in seed {seed} of network {network_name} in file: {file_path}')
                else:
                    print(f'Seed {seed} not found in network {network_name} in file: {file_path}')
            else:
                print(f'Network {network_name} not found in file: {file_path}')
    else:
        print(f'File not found: {file_path}')
    return None


def delete_plot_data(variable_name, file_name, file_path_prefix=None):
    """
    Delete a specific variable from an HDF5 file and repack to reclaim disk space.

    Parameters
    ----------
    variable_name : str
        Name of the variable to delete.
    file_name : str
        Name of the HDF5 file.
    file_path_prefix : str, optional
        Path prefix for the file location.
    """
    if file_path_prefix is None:
        root_dir = get_project_root()
        file_path_prefix = root_dir + "/EIANN/data/model_hdf5_plot_data/"
        
    file_path = file_path_prefix + file_name
    if not os.path.exists(file_path):
        print(f'File not found: {file_path}')
        return
    original_size = os.path.getsize(file_path)

    # First pass: delete the variable(s)
    with h5py.File(file_path, 'a') as hdf5_file:
        for network_name in list(hdf5_file.keys()):
            if variable_name == network_name:
                del hdf5_file[network_name]
                print(f"Deleted entire network group '{network_name}' from {file_name}")
                continue
            for seed in list(hdf5_file[network_name].keys()):
                if variable_name == seed:
                    del hdf5_file[network_name][seed]
                    print(f"Deleted '{variable_name}' from {file_name}, seed: {seed}")
                
                seed_group = hdf5_file[network_name][seed]
                if variable_name in seed_group:
                    del seed_group[variable_name]
                    print(f"Deleted '{variable_name}' from {file_name}, seed: {seed}")
                else:
                    del_counter = 0
                    for subgroup_name in list(seed_group.keys()):
                        obj = seed_group[subgroup_name]
                        if isinstance(obj, h5py.Group) and variable_name in obj:
                            del obj[variable_name]
                            del_counter += 1
                            print(f"Deleted '{variable_name}' from {file_name}, seed: {seed}, subgroup: {subgroup_name}")
                    if del_counter == 0:
                        print(f"Variable '{variable_name}' not found in {file_name}, seed: {seed}")
                        return

    # Second pass: repack file to reclaim disk space
    tmp_fd, tmp_file = tempfile.mkstemp(suffix=".h5")
    os.close(tmp_fd)
    try:
        subprocess.run(["h5repack", file_path, tmp_file], check=True)
        shutil.move(tmp_file, file_path)
        print(f"Repacked file: {file_path.split('/')[-1]}")
        if original_size < 1e9:
            print(f"Original file size: {original_size / 1e6:.4f} MB")
        else:
            print(f"Original file size: {original_size / 1e9:.4f} GB")
        new_size = os.path.getsize(file_path)
        if new_size < 1e9:
            print(f"New file size: {new_size / 1e6:.4f} MB")
        else:
            print(f"New file size: {new_size / 1e9:.4f} GB")
        if new_size < original_size:
            reclaimed = original_size - new_size
            if reclaimed < 1e9:
                print(f"Disk space reclaimed: {reclaimed / 1e6:.4f} MB")
            else:
                print(f"Disk space reclaimed: {reclaimed / 1e9:.4f} GB")
    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


def print_hdf5_dataset_sizes(file_path):
    """
    Print the sizes of datasets in an HDF5 file.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file.
    """
    def print_dataset_sizes(name, obj):
        if isinstance(obj, h5py.Dataset):
            size_bytes = obj.size * obj.dtype.itemsize  # logical size
            storage_bytes = obj.id.get_storage_size()   # actual storage on disk
            if size_bytes/1e6 > 1:
                print(f"{name.split('/')[-2:]}: shape={obj.shape}, "
                    f"storage_size={storage_bytes/1e6:.2f} MB")

    with h5py.File(file_path, "r") as f:
        f.visititems(print_dataset_sizes)


def get_MNIST_dataloaders(sub_dataloader_size=None, batch_size=1, data_dir=None):
    """
    Load MNIST dataset and return custom dataloaders that include sample indices.

    Parameters
    ----------
    sub_dataloader_size : int, optional
        If set, creates a separate dataloader with this many samples.
    batch_size : int, optional
        Batch size for the sub-dataloader.
    data_dir : str, optional
        Path to the dataset directory. If None, a default path is used.

    Returns
    -------
    tuple
        Tuple of DataLoaders: (train, [train_sub], val, test, generator).
    """
    if data_dir is None:
        root_dir = get_project_root()
        data_dir = root_dir + '/EIANN/data/datasets/MNIST'
        
    # Load dataset
    tensor_flatten = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(torch.flatten)])
    MNIST_train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=tensor_flatten)
    MNIST_test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=tensor_flatten)

    # Add index to train & test data
    MNIST_train = []
    for idx,(data,label) in enumerate(MNIST_train_dataset):
        target = torch.eye(len(MNIST_train_dataset.classes))[label]
        MNIST_train.append((idx, data, target))

    MNIST_test = []
    for idx,(data,label) in enumerate(MNIST_test_dataset):
        target = torch.eye(len(MNIST_test_dataset.classes))[label]
        MNIST_test.append((idx, data, target))
        
    # Put data in dataloader
    data_generator = torch.Generator()

    if batch_size in ['all', 'full_dataset']:
        batch_size = 50_000
        train_dataloader = torch.utils.data.DataLoader(MNIST_train[0:50_000], batch_size=batch_size, shuffle=False, generator=data_generator)
    else:
        train_dataloader = torch.utils.data.DataLoader(MNIST_train[0:50_000], batch_size=batch_size, shuffle=True, generator=data_generator)

    val_dataloader = torch.utils.data.DataLoader(MNIST_train[-10_000:], batch_size=10_000, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(MNIST_test, batch_size=10_000, shuffle=False)
    
    if sub_dataloader_size is not None:
        train_sub_dataloader = torch.utils.data.DataLoader(MNIST_train[0:sub_dataloader_size], shuffle=True, generator=data_generator, batch_size=batch_size)
        return train_dataloader, train_sub_dataloader, val_dataloader, test_dataloader, data_generator
    else:
        return train_dataloader, val_dataloader, test_dataloader, data_generator


def get_MNIST_dataloaders_with_noise(sub_dataloader_size=None, batch_size=1, data_dir=None, mean=0.0, std=0.1, seed=42):
    """
    Load MNIST dataset with added Gaussian noise and return custom dataloaders that include sample indices.

    Parameters
    ----------
    sub_dataloader_size : int, optional
        If set, creates a separate dataloader with this many samples.
    batch_size : int, optional
        Batch size for the sub-dataloader.
    data_dir : str, optional
        Path to the dataset directory. If None, a default path is used.
    mean : float, optional
        Mean of the Gaussian noise to add to images. Default is 0.0.
    std : float, optional
        Standard deviation of the Gaussian noise to add to images. Default is 0.1.

    Returns
    -------
    tuple
        Tuple of DataLoaders: (train, [train_sub], val, test, generator).
        
    Notes
    -----
    Gaussian noise is added to both training and test images. Noisy pixel values 
    are clamped to the range [0, 1] to maintain valid image data.
    """

    if data_dir is None:
        root_dir = get_project_root()
        data_dir = root_dir + '/EIANN/data/datasets/MNIST'
        
    torch.manual_seed(seed)

    # Load dataset
    tensor_flatten = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(torch.flatten)])
    MNIST_train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=tensor_flatten)
    MNIST_test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=tensor_flatten)

    def add_gaussian_noise(image, mean=0.0, std=0.1):
        noise = torch.randn(image.size()) * std + mean
        noisy_image = image + noise
        noisy_image = torch.clamp(noisy_image, 0.0, 1.0)  # Ensure pixel values are in [0, 1]
        return noisy_image

    # Add index to train & test data
    MNIST_train = []
    for idx,(data,label) in enumerate(MNIST_train_dataset):
        data = add_gaussian_noise(data, mean, std)
        target = torch.eye(len(MNIST_train_dataset.classes))[label]
        MNIST_train.append((idx, data, target))

    MNIST_test = []
    for idx,(data,label) in enumerate(MNIST_test_dataset):
        data = add_gaussian_noise(data, mean, std)
        target = torch.eye(len(MNIST_test_dataset.classes))[label]
        MNIST_test.append((idx, data, target))
        
    # Put data in dataloader
    data_generator = torch.Generator()

    if batch_size in ['all', 'full_dataset']:
        batch_size = 50_000
        train_dataloader = torch.utils.data.DataLoader(MNIST_train[0:50_000], batch_size=batch_size, shuffle=False, generator=data_generator)
    else:
        train_dataloader = torch.utils.data.DataLoader(MNIST_train[0:50_000], batch_size=batch_size, shuffle=True, generator=data_generator)

    val_dataloader = torch.utils.data.DataLoader(MNIST_train[-10_000:], batch_size=10_000, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(MNIST_test, batch_size=10_000, shuffle=False)

    if sub_dataloader_size is not None:
        train_sub_dataloader = torch.utils.data.DataLoader(MNIST_train[0:sub_dataloader_size], shuffle=True, generator=data_generator, batch_size=batch_size)
        return train_dataloader, train_sub_dataloader, val_dataloader, test_dataloader, data_generator
    else:
        return train_dataloader, val_dataloader, test_dataloader, data_generator


def get_FashionMNIST_dataloaders(sub_dataloader_size=None, batch_size=1, data_dir=None):
    """
    Load FashionMNIST dataset and return custom dataloaders that include sample indices.

    Parameters
    ----------
    sub_dataloader_size : int, optional
        If set, creates a separate dataloader with this many samples.
    batch_size : int, optional
        Batch size for the sub-dataloader.
    data_dir : str, optional
        Path to the dataset directory. If None, a default path is used.

    Returns
    -------
    tuple
        Tuple of DataLoaders: (train, [train_sub], val, test, generator).
    """
    if data_dir is None:
        root_dir = get_project_root()
        data_dir = root_dir + '/EIANN/data/datasets/FashionMNIST'
        
    # Load dataset
    tensor_flatten = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Lambda(torch.flatten)])
    train_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True,
                                                     transform=tensor_flatten)
    test_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True,
                                                    transform=tensor_flatten)

    # Add index to train & test data
    fmnist_train = []
    for idx,(data,label) in enumerate(train_dataset):
        target = torch.eye(len(train_dataset.classes))[label]
        fmnist_train.append((idx, data, target))

    fmnist_test = []
    for idx,(data,target) in enumerate(test_dataset):
        target = torch.eye(len(test_dataset.classes))[target]
        fmnist_test.append((idx, data, target))
        
    # Put data in dataloader
    data_generator = torch.Generator()
    if batch_size in ['all', 'full_dataset']:
        batch_size = 50_000
        train_dataloader = torch.utils.data.DataLoader(fmnist_train[0:50_000], batch_size=batch_size, shuffle=False, generator=data_generator)
    else:
        train_dataloader = torch.utils.data.DataLoader(fmnist_train[0:50_000], batch_size=batch_size, shuffle=True, generator=data_generator)

    val_dataloader = torch.utils.data.DataLoader(fmnist_train[-10_000:], batch_size=10_000, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(fmnist_test, batch_size=10_000, shuffle=False)

    if sub_dataloader_size is not None:
        train_sub_dataloader = torch.utils.data.DataLoader(fmnist_train[0:sub_dataloader_size], shuffle=True, generator=data_generator, batch_size=batch_size)
        return train_dataloader, train_sub_dataloader, val_dataloader, test_dataloader, data_generator
    else:
        return train_dataloader, val_dataloader, test_dataloader, data_generator


def get_cifar10_dataloaders(sub_dataloader_size=None, batch_size=1, data_dir=None):
    if data_dir is None:
        root_dir = get_project_root()
        data_dir = root_dir + '/EIANN/data/datasets/CIFAR10'

    tensor_flatten = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Lambda(torch.flatten)])

    CIFAR10_train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tensor_flatten)
    CIFAR10_test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tensor_flatten)

    # Add index to train & test data
    CIFAR10_train = []
    for idx, (data, target) in enumerate(CIFAR10_train_dataset):
        target = torch.eye(len(CIFAR10_train_dataset.classes))[target]
        CIFAR10_train.append((idx, data, target))
    CIFAR10_train = CIFAR10_train[:-10_000]
    CIFAR10_val = CIFAR10_train[-10_000:]
    
    CIFAR10_test = []
    for idx, (data, target) in enumerate(CIFAR10_test_dataset):
        target = torch.eye(len(CIFAR10_test_dataset.classes))[target]
        CIFAR10_test.append((idx, data, target))

    # Put data in dataloader
    data_generator = torch.Generator()

    if batch_size in ['all', 'full_dataset']:
        batch_size = len(CIFAR10_train)
        train_dataloader = torch.utils.data.DataLoader(CIFAR10_train, batch_size=batch_size, shuffle=False, generator=data_generator)
    else:
        train_dataloader = torch.utils.data.DataLoader(CIFAR10_train, batch_size=batch_size, shuffle=True, generator=data_generator)

    val_dataloader = torch.utils.data.DataLoader(CIFAR10_val, batch_size=10_000, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(CIFAR10_test, batch_size=10_000, shuffle=False)

    if sub_dataloader_size is not None:
        train_sub_dataloader = torch.utils.data.DataLoader(CIFAR10_train[0:sub_dataloader_size], shuffle=True, generator=data_generator, batch_size=batch_size)
        return train_dataloader, train_sub_dataloader, val_dataloader, test_dataloader, data_generator
    else:
        return train_dataloader, val_dataloader, test_dataloader, data_generator



# *******************************************************************
# Functions to generate and process data
# *******************************************************************

def generate_spiral_data(arm_size=500, K=4, sigma=0.16, seed=0, offset=0.):
    """
    Generate a synthetic spiral dataset.

    Parameters
    ----------
    arm_size : int, optional
        Number of points per spiral arm. Default is 500.
    K : int, optional
        Number of spiral arms. Default is 4.
    sigma : float, optional
        Noise level. Default is 0.16.
    seed : int, optional
        Random seed. Default is 0.
    offset : float, optional
        Offset to apply to spiral coordinates. Default is 0.

    Returns
    -------
    list of tuples
        Each element is (index, data_tensor, one_hot_target).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    t = torch.linspace(0, 1, arm_size) # Generate linearly spaced values from 0 to 1, used as the parameter that varies along the length of the spiral
    X = torch.zeros(K*arm_size, 2)
    y = torch.zeros(K*arm_size)

    # arm_index is the offset or phase shift, allowing the function to generate points for each arm in the spiral
    for arm_index in range(K):
        X[arm_index*arm_size:(arm_index+1)*arm_size, 0] = (
                t*(torch.sin(2*np.pi/K*(2*t+arm_index)) + sigma*torch.randn(arm_size))) + offset
        X[arm_index*arm_size:(arm_index+1)*arm_size, 1] = (
                t*(torch.cos(2*np.pi/K*(2*t+arm_index)) + sigma*torch.randn(arm_size))) + offset
        y[arm_index*arm_size:(arm_index+1)*arm_size] = arm_index    

    all_data = []
    for index, (data, label) in enumerate(zip(X, y)):
        target = torch.eye(K)[int(label)]
        all_data.append((index, data, target))

    return all_data

    
def get_spiral_dataloaders(batch_size=1, points_per_spiral_arm=2000, seed=0):
    """
    Generate spiral dataset and return dataloaders.

    Parameters
    ----------
    batch_size : int or str, optional
        Batch size for training. Use 'all' or 'full_dataset' to load all data in one batch.
    points_per_spiral_arm : int, optional
        Number of points per spiral arm. Default is 2000.
    seed : int, optional
        Random seed. Default is 0.

    Returns
    -------
    tuple
        Tuple of DataLoaders: (train, val, test, generator).
    """

    # Split data into train, validation, and test sets
    train_data = generate_spiral_data(arm_size=int(0.7*points_per_spiral_arm), seed=seed)
    val_data = generate_spiral_data(arm_size=int(0.15*points_per_spiral_arm), seed=seed+1)
    test_data = generate_spiral_data(arm_size=int(0.15*points_per_spiral_arm), seed=seed+2)

    data_generator = torch.Generator().manual_seed(seed)
    if batch_size in ['all', 'full_dataset']:
        batch_size = len(train_data)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, generator=data_generator)
    else:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=data_generator)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=len(val_data), shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, data_generator


def generate_inhomogeneous_poisson_spikes(rate, refractory_period=3):
    """
    Generate spike times from an inhomogeneous Poisson process using the thinning method.

    Example usage:
    ```python
    rate = 300 * np.ones(1000)  # Example rate in Hz
    spike_times = generate_inhomogeneous_poisson_spikes(rate, refractory_period=2)
    ```
    
    Parameters:
    -----------
    rate : numpy.ndarray
        Time series of firing rates in Hz, sampled at 1ms interval. Each element represents the instantaneous
        firing rate at that time point.
    refractory_period : float, optional
        Minimum interval between spikes in milliseconds
    
    Returns:
    --------
    list
        List of spike times in seconds
    """
    if len(rate) == 0:
        return []
    if np.any(rate < 0):
        raise ValueError("Firing rates must be non-negative")

    refractory_period_s = refractory_period / 1000.0  # Convert to seconds
    # rate = 1 / (1 / (rate+1e-10) - refractory_period_s)
    rate = rate / (1 - rate * refractory_period_s + 1e-5)    
    rate = np.clip(rate, 0, 50_000) # Avoid excessively high rates

    
    max_rate = np.max(rate) # Maximum firing rate for the homogeneous Poisson process 
    if max_rate == 0:
        return []
        
    # Generate homogeneous Poisson process at maximum rate
    total_time_s = len(rate) / 1000
    expected_events = max_rate * total_time_s

    n_events = int(np.ceil(expected_events * 2)) # Add some buffer to ensure we have enough events before thinning
    inter_spike_intervals = np.random.exponential(1.0 / max_rate, n_events) # Generate all inter-spike-intervals (drawn from exponential distribution)
    spike_times = np.cumsum(inter_spike_intervals)
    spike_times = spike_times[spike_times < total_time_s] # Reject events beyond the max simulation time
    spike_times = spike_times*1000  # Convert to milliseconds
    spike_times = list(spike_times)  # Convert to list for easier manipulation

    # Thinning: accept/reject based on instantaneous rate
    spike_times_filtered = []
    for t in spike_times:
        t_index = round(t)  # Convert from seconds to milliseconds
        if t_index == len(rate):
            continue # avoid index out of bounds

        instantaneous_rate = rate[t_index]
        accept = np.random.random() < (instantaneous_rate / max_rate)
        if len(spike_times_filtered) == 0:
            current_isi = np.inf
        else:
            current_isi = (t - spike_times_filtered[-1])

        if accept and current_isi >= refractory_period:
                spike_times_filtered.append(t)

    spike_times_filtered = np.array(spike_times_filtered).round().astype(int) # Round to nearest ms
    return spike_times_filtered


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

