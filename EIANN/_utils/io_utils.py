# *******************************************************************
# Functions to import and export data
# *******************************************************************

import torch
from torch.utils.data import DataLoader
import h5py
import os
import yaml
import torchvision




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
        file_path = root_dir+'/data/.plot_data.h5'

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
        file_path = root_dir + '/data/.plot_data.h5'

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
    
    return os.path.join(current_path, 'EIANN')


def get_MNIST_dataloaders(sub_dataloader_size=1000, classes=None, batch_size=1):
    """
    Load MNIST dataset into custom dataloaders with sample index
    """

    # Load dataset
    root_dir = get_project_root()
    tensor_flatten = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Lambda(torch.flatten)])
    MNIST_train_dataset = torchvision.datasets.MNIST(root=root_dir+'/data/datasets/', train=True, download=True,
                                                     transform=tensor_flatten)
    MNIST_test_dataset = torchvision.datasets.MNIST(root=root_dir+'/data/datasets/', train=False, download=True,
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

