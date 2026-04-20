import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import os, sys, math
from copy import deepcopy
import numpy as np
import click
import glob
import h5py
import gc

from EIANN import Network
from EIANN.utils import (read_from_yaml, write_to_yaml, analyze_simple_EIANN_epoch_loss_and_accuracy, \
    sort_by_val_history, recompute_validation_loss_and_accuracy, check_equilibration_dynamics, compute_test_loss_and_accuracy, \
    recompute_train_loss_and_accuracy, compute_test_loss_and_accuracy_history, sort_by_class_averaged_val_output,
                         get_binned_mean_population_attribute_history_dict)
from EIANN.plot import (plot_batch_accuracy, plot_train_loss_history, plot_validate_loss_history, plot_receptive_fields,
                        plot_representation_metrics)
from nested.utils import Context, str_to_bool
from nested.optimize_utils import update_source_contexts
from EIANN.optimize.network_config_updates import *
import EIANN.utils as utils

def get_split_mnist_dataset(data_dir, num_splits):
    tensor_flatten = T.Compose([T.ToTensor(), T.Lambda(torch.flatten)])
    MNIST_test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False,
                                                    download=False, transform=tensor_flatten)

    # split data
    num_classes = len(MNIST_test_dataset.classes)
    classes_per_task = num_classes // num_splits

    labels_in_tasks = [list(range(t, min(num_classes, t+classes_per_task))) for t in range(0, num_classes, classes_per_task)]

    test_datasets = [[] for _ in range(len(labels_in_tasks))]
    full_test_dataset = []

    for idx, (data, label) in enumerate(MNIST_test_dataset):
        target = torch.eye(num_classes)[label]
        task_membership = [i for i, labels in enumerate(labels_in_tasks) if label in labels][0]

        full_test_dataset.append((idx, data, target))
        test_datasets[task_membership].append((idx, data, target))

    test_dataloaders = []

    # put data into dataloaders
    for task_test in test_datasets:
        test_dataloaders.append(torch.utils.data.DataLoader(task_test, batch_size=len(task_test), shuffle=False))

    full_test_dataloader = torch.utils.data.DataLoader(full_test_dataset, batch_size=len(full_test_dataset), shuffle=False)

    return test_dataloaders, full_test_dataloader


@click.command()
@click.option("--model-folder-path", help="path to folder containing models trained after each phase")
@click.option("--config-file-path", help="path model config file")
@click.option("--data-dir", help="directory containing train/test data")
@click.option("--task", default='split_mnist', help="continual learning task")
@click.option("--num-splits", help="how many splits (or subtasks) the task was split into")
def main(model_folder_path, config_file_path, data_dir, task, num_splits):

    num_splits = int(num_splits)

    seeds = [seed for seed in os.listdir(model_folder_path) if os.path.isdir(os.path.join(model_folder_path, seed))]
    subtask_accuracy_per_seed = []
    overall_accuracy_per_seed = []

    task_test_loaders = full_test_loader = None

    if task == 'split_mnist':
        task_test_loaders, full_test_loader =  get_split_mnist_dataset(data_dir, num_splits)

    
    for seed in seeds:
        print(f'Computing Metrics for Seed {seed}')
        phase_weight_paths = sorted(glob.glob(os.path.join(model_folder_path, seed, '*.pkl')))
        seed_accuracies = []

        for weight_path in phase_weight_paths:
            
            network = utils.load_network(weight_path)
            phase_accuracies = []
            for i, test_loader in enumerate(task_test_loaders):
                test_loss, test_accuracy = compute_test_loss_and_accuracy(network, test_loader)
                phase_accuracies.append(float(test_accuracy))

            seed_accuracies.append(phase_accuracies)

        full_test_loss, full_test_accuracy = compute_test_loss_and_accuracy(network, full_test_loader)
        
        subtask_accuracy_per_seed.append(seed_accuracies)
        overall_accuracy_per_seed.append(float(full_test_accuracy))

    
    subtask_accuracy_per_seed = np.array(subtask_accuracy_per_seed)
    overall_accuracy_per_seed = np.array(overall_accuracy_per_seed)

    on_task_accuracies = np.zeros((len(seeds),))
    accuracies = np.zeros((len(seeds),))
    backward_transfers = np.zeros((len(seeds),))
    forward_transfers = np.zeros((len(seeds),))

    # get CL metrics as outlined here: https://arxiv.org/abs/1810.13166
    for i in range(num_splits):
        for j in range(num_splits):
            if i == j:
                on_task_accuracies += subtask_accuracy_per_seed[:, i, j]
            if i >= j:
                accuracies += subtask_accuracy_per_seed[:, i, j]
            if i > j:
                backward_transfers += (subtask_accuracy_per_seed[:, i, j] - subtask_accuracy_per_seed[:, j, j])
            if i < j:
                forward_transfers += subtask_accuracy_per_seed[:, i, j]
    
    on_task_accuracies /= (num_splits)
    accuracies /= (num_splits * (num_splits + 1) / 2)
    backward_transfers /= (num_splits * (num_splits - 1) / 2)
    forward_transfers /= (num_splits * (num_splits - 1) / 2)

    print(on_task_accuracies)
    print(accuracies)
    print(backward_transfers)
    print(forward_transfers)



if __name__ == '__main__': 
    main()