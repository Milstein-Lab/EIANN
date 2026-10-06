"""
Post-hoc evaluation of continual learning (class-incremental split MNIST) models.

For each seed folder in --model-folder-path, loads the network exported after each training phase and saves:
  - <output_dir>/<model_key>_accuracy.csv: long format, one row per (seed, phase, test_set), with columns
    model, seed, phase, test_set, accuracy. test_set is 'task<j>' (test set of task j), 'seen' (mean accuracy over
    tasks 0..phase) or 'all' (full 10-class test set). CSVs of different models can be concatenated and plotted together.
  - <output_dir>/<model_key>_activity.h5: class-averaged test activity of every population after each phase, nested as
    <seed>/phase<i>/average_pop_activity_dict/<pop> (num_classes, num_units) and <seed>/phase<i>/unit_labels_dict/<pop>.

Example (from the EIANN/ package directory):
python scripts/get_cl_metrics.py --model-folder-path=data/<network_name> --data-dir=data/datasets/MNIST_data/ \
    --num-splits=5 --label=<label>
"""
import torch
import torchvision
import torchvision.transforms as T
import os, re
import numpy as np
import pandas as pd
import click
import glob
import gc

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

    return test_dataloaders, full_test_dataloader, labels_in_tasks


def compute_task_incremental_test_loss_and_accuracy(network, test_dataloader, labels_in_tasks):
    idx, test_data, test_target = next(iter(test_dataloader))
    output = network.forward(test_data.to(network.device), no_grad=True)
    return utils.compute_task_incremental_loss_and_accuracy(output, test_target.to(network.device), labels_in_tasks,
                                                            network.criterion)


def get_phase_file_paths(seed_dir, label=None):
    """
    Find the network pickles exported after each phase, named <network_name>_phase<i>_<seed>_<data_seed>[_<label>].pkl.
    :param seed_dir: str
    :param label: str, optional; if None, all pickles in the folder must share the same label (or have none)
    :return: tuple of (list of str, sorted by phase; str '<seed>_<data_seed>')
    """
    pattern = re.compile(r'_phase(\d+)_(\d+)_(\d+)(?:_(.+))?\.pkl$')
    phase_files = {}
    found_labels = set()
    seed_keys = set()
    for file_path in glob.glob(os.path.join(seed_dir, '*.pkl')):
        match = pattern.search(os.path.basename(file_path))
        if match is None:
            continue
        phase, seed, data_seed, file_label = match.groups()
        found_labels.add(file_label)
        if label is not None and file_label != label:
            continue
        phase_files[int(phase)] = file_path
        seed_keys.add(f'{seed}_{data_seed}')

    if label is None and len(found_labels) > 1:
        raise ValueError(f'get_cl_metrics: {seed_dir} contains pickles with multiple labels {sorted(found_labels, key=str)}; '
                         f'specify one with --label')
    if len(seed_keys) > 1:
        raise ValueError(f'get_cl_metrics: {seed_dir} contains pickles from multiple seeds {sorted(seed_keys)}')
    if not phase_files:
        return [], None
    phases = sorted(phase_files)
    if phases != list(range(len(phases))):
        raise ValueError(f'get_cl_metrics: missing phases in {seed_dir}; found {phases}')
    return [phase_files[phase] for phase in phases], seed_keys.pop()


@click.command()
@click.option("--model-folder-path", required=True, help="path to folder with one sub-folder per seed of per-phase pickles")
@click.option("--data-dir", required=True, help="directory containing MNIST data")
@click.option("--num-splits", required=True, type=int, help="how many splits (or subtasks) the task was split into")
@click.option("--task-incremental", is_flag=True, default=False,
              help="evaluate each sample only within its own task's output units")
@click.option("--label", default=None, help="only use pickles whose names end in _<label>.pkl")
@click.option("--model-key", default=None, help="name for output files and the model column; "
                                                "default: <model folder name>[_<label>]")
@click.option("--output-dir", default='data/cl_metrics', help="directory to save the accuracy csv and activity h5 file")
def main(model_folder_path, data_dir, num_splits, task_incremental, label, model_key, output_dir):

    if model_key is None:
        model_key = os.path.basename(os.path.normpath(model_folder_path))
        if label is not None:
            model_key += f'_{label}'

    task_test_loaders, full_test_loader, labels_in_tasks = get_split_mnist_dataset(data_dir, num_splits)

    seed_dirs = sorted(seed for seed in os.listdir(model_folder_path)
                       if os.path.isdir(os.path.join(model_folder_path, seed)))
    accuracy_rows = []
    activity_dict = {}

    for seed_dir in seed_dirs:
        phase_file_paths, seed = get_phase_file_paths(os.path.join(model_folder_path, seed_dir), label)
        if not phase_file_paths:
            print(f'No phase pickles found in {seed_dir}; skipping')
            continue
        if len(phase_file_paths) != num_splits:
            raise ValueError(f'get_cl_metrics: expected {num_splits} phases in {seed_dir}, found {len(phase_file_paths)}')
        print(f'Computing metrics for seed {seed}')
        activity_dict[seed] = {}

        for phase, file_path in enumerate(phase_file_paths):
            network = utils.load_network(file_path, disp=False)

            task_accuracies = []
            for task_test_loader in task_test_loaders:
                if task_incremental:
                    _, test_accuracy = \
                        compute_task_incremental_test_loss_and_accuracy(network, task_test_loader, labels_in_tasks)
                else:
                    _, test_accuracy = utils.compute_test_loss_and_accuracy(network, task_test_loader)
                task_accuracies.append(float(test_accuracy))

            if task_incremental:
                _, full_test_accuracy = \
                    compute_task_incremental_test_loss_and_accuracy(network, full_test_loader, labels_in_tasks)
            else:
                _, full_test_accuracy = utils.compute_test_loss_and_accuracy(network, full_test_loader)

            test_set_accuracies = {f'task{task}': accuracy for task, accuracy in enumerate(task_accuracies)}
            test_set_accuracies['seen'] = float(np.mean(task_accuracies[:phase+1]))
            test_set_accuracies['all'] = float(full_test_accuracy)
            for test_set, accuracy in test_set_accuracies.items():
                accuracy_rows.append({'model': model_key, 'seed': seed, 'phase': phase, 'test_set': test_set,
                                      'accuracy': accuracy})

            average_pop_activity_dict, _, unit_labels_dict = \
                utils.compute_test_activity(network, full_test_loader, class_average=True, sort=False)
            activity_dict[seed][f'phase{phase}'] = {
                'average_pop_activity_dict': {pop: activity.cpu().numpy()
                                              for pop, activity in average_pop_activity_dict.items()},
                'unit_labels_dict': {pop: unit_labels.cpu().numpy() for pop, unit_labels in unit_labels_dict.items()}}

            del network
            gc.collect()

    if not accuracy_rows:
        raise ValueError(f'get_cl_metrics: no phase pickles found in {model_folder_path}')

    os.makedirs(output_dir, exist_ok=True)
    accuracy_df = pd.DataFrame(accuracy_rows)
    accuracy_file_path = os.path.join(output_dir, f'{model_key}_accuracy.csv')
    accuracy_df.to_csv(accuracy_file_path, index=False)
    activity_file_path = os.path.join(output_dir, f'{model_key}_activity.h5')
    utils.dict_to_hdf5(activity_dict, activity_file_path)
    print(f'Saved accuracies to {accuracy_file_path}')
    print(f'Saved class-averaged activity to {activity_file_path}')

    task_df = accuracy_df[accuracy_df.test_set.str.startswith('task')]
    accuracy_matrix = task_df.pivot_table(index='phase', columns='test_set', values='accuracy', aggfunc='mean')
    print(f'\nMean test accuracy across {accuracy_df.seed.nunique()} seed(s) (rows: phase, columns: test set):')
    print(accuracy_matrix.round(2).to_string())
    summary = accuracy_df[accuracy_df.test_set.isin(['seen', 'all'])].groupby(['test_set', 'phase']).accuracy.agg(
        ['mean', 'std']).unstack('test_set').swaplevel(axis=1).sort_index(axis=1)
    print('\nAccuracy on seen tasks and on the full test set:')
    print(summary.round(2).to_string())


if __name__ == '__main__':
    main()
