import torch
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import os
import pathlib
import h5py
import click
import gc
import codecs
import re
from sklearn.metrics.pairwise import cosine_similarity

import EIANN.utils as ut
import EIANN.plot as pt
import EIANN.network as nt



########################################################################################################
# Plot data generation (hdf5 and csv files of computed model analyses)
########################################################################################################

def load_model_dict(csv_file_path=None):
    # Load model specs from csv file
    root_dir = ut.get_project_root()
    if csv_file_path is None:
        csv_file_path = root_dir + "/EIANN/data/figure_model_specs.csv" 
    df = pd.read_csv(csv_file_path, index_col=0)
    df = df.map(lambda x: codecs.decode(x, 'unicode_escape') if isinstance(x, str) else x) # convert special characters like \n
    model_dict_all = df.transpose().to_dict()
    seeds = ["66049_257","66050_258", "66051_259", "66052_260", "66053_261"]
    for model_key in model_dict_all:
        model_dict_all[model_key]["seeds"] = seeds
    return model_dict_all


def generate_data_hdf5(config_path, saved_network_path, hdf5_path, recompute=None, variables_to_save='all'):
    '''
    Loads a network and saves plot-ready processed data into an hdf5 file.

    :param config_path_prefix: Path to config file directory
    :param saved_network_path_prefix: Path to directory containing pickled network
    :param model_dict: Dictionary containing model information for a single model
    :param hdf5_path: Path to hdf5 file to save data to
    '''

    # Build network
    network_name = os.path.basename(config_path).split('.')[0]
    pickle_filename = os.path.basename(saved_network_path).split('.')[0]
    network_seed, data_seed = pickle_filename.split('_')[-2:]
    assert network_seed.isdigit() and data_seed.isdigit(), f"network_seed and data_seed must be numbers, but got {network_seed} and {data_seed}"
    seed = f"{network_seed}_{data_seed}"
    network_config = ut.read_from_yaml(config_path)
    pop_names = set([key for subdict in network_config['layer_config'].values() for key in subdict.keys()])

    # Define which variables to compute
    if variables_to_save == 'all':
        variables_to_save = ['weights', 'accuracy', 'val_accuracy_history', 'test_accuracy_history', "test_accuracy_history_extended",
                            'average_pop_activity_dict', 'activity_dynamics', 'metrics_dict', 'robustness_to_pruning',
                            'val_loss_history', 'val_history_train_steps', 'test_loss_history',
                            'angle_vs_bp', 'feedback_weight_angle_history', 'sparsity_history', 'selectivity_history']

        if any('Dend' in pop_name for pop_name in pop_names):
            variables_to_save.extend("dendritic_state")
        if 'mnist' in config_path:
            variables_to_save.extend(['noise_sensitivity', 'final_receptive_fields'])
        elif 'spiral' in config_path:
            variables_to_save.extend(['spiral_decision_data_dict'])

    if "dendritic_state" in variables_to_save and not any('Dend' in pop_name for pop_name in pop_names):
        variables_to_save.remove("dendritic_state")

    if recompute==True:
        recompute = variables_to_save
    elif recompute==False:
        recompute = None

    # Open hdf5 and check if the relevant data already exists       
    if os.path.exists(hdf5_path): # If the file exists, check if the network data already exists or needs to be recomputed
        with h5py.File(hdf5_path, 'r') as file:
            if network_name in file.keys():
                if seed in file[network_name].keys():
                    if recompute == 'all':
                        print(f"Overwriting {network_name} {seed} in {hdf5_path}")
                    elif set(variables_to_save).issubset(file[network_name][seed].keys()) and recompute is None: # Don't recompute if all variables already exist
                        # print(f"Data for {network_name} {seed} exists in {hdf5_path}")
                        return
                    else:
                        print(f"Recomputing plot data for {network_name} {seed} in {hdf5_path}")
                        existing_vars = set(file[network_name][seed].keys())
                        if recompute is None:
                            recompute = []
                        variables_to_save = [var for var in variables_to_save if (var not in existing_vars) or (var in recompute)]

    print("-----------------------------------------------------------------------------")
    print(f"Variables to save: {variables_to_save}")
    print(f"Network: {network_name} {seed}")

    # Load the saved network pickle
    if not all('extended' in var for var in variables_to_save):  # don't load the regular network if we are only saving the extended training data
        network = ut.load_network(saved_network_path)
        network.seed = seed
        network.name = network_name
        if not hasattr(network, 'input_pop'):
            input_layer = list(network)[0]
            network.input_pop = next(iter(input_layer))

    # Load dataset
    if "mnist" in config_path and "fmnist" not in config_path:
        all_dataloaders = ut.get_MNIST_dataloaders(batch_size='full_dataset')
    elif "fmnist" in config_path:
        all_dataloaders = ut.get_FashionMNIST_dataloaders(batch_size='full_dataset')
    elif "spiral" in config_path:
        all_dataloaders = ut.get_spiral_dataloaders(batch_size='full_dataset')
    elif "cifar10" in config_path:
        all_dataloaders = ut.get_cifar10_dataloaders(batch_size='full_dataset')
    train_dataloader, val_dataloader, test_dataloader, data_generator = all_dataloaders

    ##################################################################
    ## Generate plot data

    if 'weights' in variables_to_save:
        weights_dict = {'initial_weights': {}, 'final_weights': {}}
        for proj in ['H1E_InputE', 'H2E_H1E']:
            proj_key = f"module_dict.{proj}.weight"
            weights_dict['initial_weights'][proj] = network.prev_param_history[0][proj_key]
            weights_dict['final_weights'][proj] = network.param_history[-1][proj_key]
        ut.save_plot_data(network.name, network.seed, data_key='weights', data=weights_dict, file_path=hdf5_path, overwrite=True)

    if 'average_pop_activity_dict' in variables_to_save:
        average_pop_activity_dict, pattern_labels, unit_labels_dict = ut.compute_test_activity(network, test_dataloader, class_average=True, sort=False)
        average_pop_activity_dict = {k: v.numpy() for k, v in average_pop_activity_dict.items()}
        ut.save_plot_data(network.name, network.seed, data_key='average_pop_activity_dict', data=average_pop_activity_dict, file_path=hdf5_path, overwrite=True)
        ut.save_plot_data(network.name, network.seed, data_key='pattern_labels', data=pattern_labels, file_path=hdf5_path, overwrite=True)
        ut.save_plot_data(network.name, network.seed, data_key='unit_labels_dict', data=unit_labels_dict, file_path=hdf5_path, overwrite=True)

    if set(['accuracy','test_loss_history', 'test_accuracy_history', 'val_loss_history', 'val_accuracy_history', 'val_history_train_steps']).intersection(variables_to_save):
        ut.save_plot_data(network.name, network.seed, data_key='val_loss_history',          data=network.val_loss_history,          file_path=hdf5_path, overwrite=True)
        ut.save_plot_data(network.name, network.seed, data_key='val_accuracy_history',      data=network.val_accuracy_history,      file_path=hdf5_path, overwrite=True)
        ut.save_plot_data(network.name, network.seed, data_key='val_history_train_steps',   data=network.val_history_train_steps,   file_path=hdf5_path, overwrite=True)
        
        ut.save_plot_data(network.name, network.seed, data_key='test_loss_history',         data=network.test_loss_history,         file_path=hdf5_path, overwrite=True)
        ut.save_plot_data(network.name, network.seed, data_key='test_accuracy_history',     data=network.test_accuracy_history,     file_path=hdf5_path, overwrite=True)

        pop_activity_dict, pattern_labels, unit_labels_dict = ut.compute_test_activity(network, test_dataloader, class_average=False, sort=True)
        ut.save_plot_data(network.name, network.seed, data_key='sorted_activity_dict', data=pop_activity_dict, file_path=hdf5_path, overwrite=True)
        ut.save_plot_data(network.name, network.seed, data_key='sorted_pattern_labels', data=pattern_labels, file_path=hdf5_path, overwrite=True)
        ut.save_plot_data(network.name, network.seed, data_key='sorted_unit_labels_dict', data=unit_labels_dict, file_path=hdf5_path, overwrite=True)

        output = pop_activity_dict[network.output_pop.fullname]
        percent_correct = ut.compute_test_accuracy_from_data(output, pattern_labels)
        ut.save_plot_data(network.name, network.seed, data_key='accuracy', data=percent_correct, file_path=hdf5_path, overwrite=True)
        
    if 'noise_sensitivity' in variables_to_save:
        noise_stds = np.arange(0, 1.1, 0.1)
        accuracy_list = ut.compute_noise_sensitivity(network, noise_stds=noise_stds)
        ut.save_plot_data(network.name, network.seed, data_key='noise_sensitivity', data=(noise_stds, accuracy_list), file_path=hdf5_path, overwrite=True)

    if 'robustness_to_pruning' in variables_to_save:
        fraction_to_prune, accuracy_list = ut.compute_robustness_to_pruning(network, test_dataloader, projections='all')
        ut.save_plot_data(network.name, network.seed, data_key='robustness_to_pruning', data=(fraction_to_prune, accuracy_list), file_path=hdf5_path, overwrite=True)

    if 'final_receptive_fields' in variables_to_save:
        rf_populations = [population for population in network.populations.values() if population.name == "E" and population.fullname != "InputE"]
        
        initial_state_dict = network.prev_param_history[0]
        network.load_state_dict(initial_state_dict)
        receptive_fields_dict = {}
        for population in rf_populations:
            receptive_fields_dict[population.fullname] = ut.compute_maxact_receptive_fields(population)
        ut.save_plot_data(network.name, network.seed, data_key='initial_receptive_fields', data=receptive_fields_dict, file_path=hdf5_path, overwrite=True)

        final_state_dict = network.param_history[-1]
        network.load_state_dict(final_state_dict)
        receptive_fields_dict = {}
        for population in rf_populations:
            receptive_fields_dict[population.fullname] = ut.compute_maxact_receptive_fields(population)
        ut.save_plot_data(network.name, network.seed, data_key='final_receptive_fields', data=receptive_fields_dict, file_path=hdf5_path, overwrite=True)

    # Sparsity, selectivity, and structure metrics
    if "metrics_dict" in variables_to_save:
        metrics_dict = {}
        initial_receptive_fields_dict = ut.hdf5_to_dict(file_path=hdf5_path, variable_name=f'{network_name}/{network_seed}_{data_seed}/initial_receptive_fields')
        final_receptive_fields_dict = ut.hdf5_to_dict(file_path=hdf5_path, variable_name=f'{network_name}/{network_seed}_{data_seed}/final_receptive_fields')
        for population in network.populations.values():
            if population.name == "E" and population.fullname != "InputE":
                if initial_receptive_fields_dict is not None:
                    initial_receptive_fields = torch.tensor(initial_receptive_fields_dict[population.fullname])
                if final_receptive_fields_dict is not None:
                    final_receptive_fields = torch.tensor(final_receptive_fields_dict[population.fullname])
            else:
                initial_receptive_fields = None
                final_receptive_fields = None
            metrics_dict[population.fullname] = ut.compute_representation_metrics(population, test_dataloader, final_receptive_fields, initial_receptive_fields)
        ut.save_plot_data(network.name, network.seed, data_key='metrics_dict', data=metrics_dict, file_path=hdf5_path, overwrite=True)

    # Angle vs Backprop
    if set(['angle_vs_bp','angle_vs_bp_stochastic']).intersection(variables_to_save):
        stored_history_step_size = torch.diff(network.param_history_steps)[-1]
        if "mnist" in config_path and "fmnist" not in config_path:
            comparison_config_path = os.path.join(os.path.dirname(config_path), "20231129_EIANN_2_hidden_mnist_bpDale_relu_SGD_config_G_complete_optimized.yaml")
        elif "fmnist" in config_path:
            comparison_config_path = os.path.join(os.path.dirname(config_path), "20250606_EIANN_2_hidden_fmnist_bpDale_relu_SGD_config_G_zero_bias_complete_optimized.yaml")
        elif "spiral" in config_path:
            comparison_config_path = os.path.join(os.path.dirname(config_path), "20250108_EIANN_2_hidden_spiral_bpDale_fixed_SomaI_learned_bias_config_complete_optimized.yaml")
        comparison_network = ut.build_EIANN_from_config(comparison_config_path, network_seed=network_seed)
        if not ut.network_architectures_match(network, comparison_network):
            print("WARNING: Network architectures do not match. Ignoring comparison network and computing angle vs the same network with Backprop conversion.")
            print(f"Network 1: {config_path}")
            print(f"Network 2: {comparison_config_path}")
            comparison_network = None

        # Compare overall angle after many train steps (stored_history_step_size)
        bpClone_network = ut.compute_alternate_dParam_history(train_dataloader, network, comparison_network, batch_size=stored_history_step_size)
        angles_stepaveraged = ut.compute_dW_angles_vs_BP(bpClone_network.predicted_dParam_history, bpClone_network.actual_dParam_history_stepaveraged)
        
        # Compare angles for one train step (batch_size=1)
        bpClone_network = ut.compute_alternate_dParam_history(train_dataloader, network, comparison_network, batch_size=1)
        angles_stochastic = ut.compute_dW_angles_vs_BP(bpClone_network.predicted_dParam_history, bpClone_network.actual_dParam_history)

        angles = {'stepaveraged': angles_stepaveraged, 'stochastic': angles_stochastic}
        ut.save_plot_data(network.name, network.seed, data_key='angle_vs_bp', data=angles, file_path=hdf5_path, overwrite=True)

    # Forward vs Backward weight angle (weight symmetry)
    if 'feedback_weight_angle_history' in variables_to_save:
        FF_FB_angles = ut.compute_feedback_weight_angle_history(network)
        ut.save_plot_data(network.name, network.seed, data_key='feedback_weight_angle_history', data=FF_FB_angles, file_path=hdf5_path, overwrite=True)

    # Dendritic state (local loss)
    if 'dendritic_state' in variables_to_save:
        steps, binned_mean_forward_dendritic_state = ut.get_binned_mean_population_attribute_history_dict(network, attr_name="forward_dendritic_state", bin_size=100, abs=True)
        dendritic_state = {'steps': steps, 'forward_dendritic_state': binned_mean_forward_dendritic_state}
        if binned_mean_forward_dendritic_state is not None:
            ut.save_plot_data(network.name, network.seed, data_key='dendritic_state', data=dendritic_state, file_path=hdf5_path, overwrite=True)

    # Sparsity and selectivity
    if 'sparsity_history' in variables_to_save or 'selectivity_history' in variables_to_save:
        sparsity_history_dict, selectivity_history_dict = ut.compute_sparsity_selectivity_history(network, test_dataloader)
        ut.save_plot_data(network.name, network.seed, data_key='sparsity_history', data=sparsity_history_dict, file_path=hdf5_path, overwrite=True)
        ut.save_plot_data(network.name, network.seed, data_key='selectivity_history', data=selectivity_history_dict, file_path=hdf5_path, overwrite=True)

    # Spiral decision boundary plots
    if 'spiral_decision_data_dict' in variables_to_save:
        spiral_decision_data_dict = ut.compute_spiral_decisions_data(network, test_dataloader)
        ut.save_plot_data(network.name, network.seed, data_key='spiral_decision_data_dict', data=spiral_decision_data_dict, file_path=hdf5_path, overwrite=True)

    # Network forward dynamics
    if 'activity_dynamics' in variables_to_save:
        pop_dynamics_dict = ut.compute_test_activity_dynamics(network, test_dataloader)
        ut.save_plot_data(network.name, network.seed, data_key='activity_dynamics', data=pop_dynamics_dict, file_path=hdf5_path, overwrite=True)

    # Load extended network pickle if needed
    if any('extended' in var for var in variables_to_save):
        if "cifar10" in config_path:
            saved_network_path = saved_network_path.replace('.pkl', '_10_epochs.pkl')
        else:
            saved_network_path = saved_network_path.replace('.pkl', '_extended.pkl')
        network = ut.load_network(saved_network_path)
        ut.save_plot_data(network_name, seed, data_key='val_history_train_steps_extended', data=network.val_history_train_steps, file_path=hdf5_path, overwrite=True)
        ut.save_plot_data(network_name, seed, data_key='test_accuracy_history_extended', data=network.test_accuracy_history, file_path=hdf5_path, overwrite=True)


def generate_hdf5_all_seeds(model_list, model_dict_all, dataset='MNIST', config_path_prefix=None, saved_network_path_prefix=None, hdf5_path_prefix=None, recompute=None, variables_to_save='all'):
    for model_key in model_list:
        model_dict = model_dict_all[model_key]
        network_name = model_dict['config'].split('.')[0]

        root_dir = ut.get_project_root()
        if config_path_prefix is None:
            config_path_prefix = root_dir + f"/EIANN/network_config/{dataset.lower()}/"
        config_path = config_path_prefix + model_dict['config']

        if hdf5_path_prefix is None:
            hdf5_path_prefix = root_dir + "/EIANN/data/model_hdf5_plot_data/"
        hdf5_path = hdf5_path_prefix + f"plot_data_{network_name}.h5"

        if saved_network_path_prefix is None:
            # Set path to Box data directory (default path based on OS)
            if os.name == "posix": # macOS or Linux
                username = os.environ.get("USER")
                saved_network_path_prefix = f"/Users/{username}/Library/CloudStorage/Box-Box/Milstein-Shared/EIANN exported data/2024 Manuscript V2/{dataset}/"
            elif os.name == "nt": # Windows
                username = os.environ.get("USERNAME")
                saved_network_path_prefix = f"C:/Users/{username}/Box/Milstein-Shared/EIANN exported data/2024 Manuscript V2/{dataset}/"

        if not os.path.exists(hdf5_path):
            # If the hdf5 is not available in local data directory, check in Box drive
            print("Local hdf5 not found, loading data from Box drive")
            box_hdf5_dir = pathlib.Path(saved_network_path_prefix).parents[1] / "2024 Figure data HDF5 files"
            box_hdf5_path = box_hdf5_dir / f"plot_data_{network_name}.h5"
            if box_hdf5_path.exists():
                print(f"Loading hdf5 from Box drive: {box_hdf5_path}")
                hdf5_path = str(box_hdf5_path)
            else:
                print("hdf5 not found in Box drive")

        for seed in model_dict['seeds']:
            saved_network_path = saved_network_path_prefix + network_name + f"_{seed}.pkl"
            generate_data_hdf5(config_path, saved_network_path, hdf5_path, recompute, variables_to_save)
            gc.collect()


########################################################################################################
# Multi-seed plotting functions
########################################################################################################

def plot_accuracy_all_seeds(data_dict, model_dict, ax, legend=True, extended=False):
    """
    Plot test accuracy for all seeds with shaded error bars
    """
    if extended:
        accuracy_all_seeds = [data_dict[seed]['test_accuracy_history_extended'] for seed in data_dict]
        val_steps = data_dict[next(iter(data_dict))]['val_history_train_steps_extended'][:]
    else:
        accuracy_all_seeds = [data_dict[seed]['test_accuracy_history'] for seed in data_dict]
        val_steps = data_dict[next(iter(data_dict))]['val_history_train_steps'][:]

    avg_accuracy = np.mean(accuracy_all_seeds, axis=0)
    error = np.std(accuracy_all_seeds, axis=0)
    ax.plot(val_steps, avg_accuracy, label=model_dict["label"], color=model_dict["color"])
    ax.fill_between(val_steps, avg_accuracy-error, avg_accuracy+error, alpha=0.3, color=model_dict["color"], linewidth=0)
    ax.set_ylim([0,100])
    ax.set_xlabel('Training step')
    ax.set_ylabel('Test accuracy (%)', labelpad=-1)
    if legend:
        legend = ax.legend(ncol=1, bbox_to_anchor=(0.2, 0.6), loc='upper left', fontsize=6)
        for line in legend.get_lines():
            line.set_linewidth(1.5)


def plot_error_all_seeds(data_dict, model_dict, ax, scale='log'):
    accuracy_all_seeds = [data_dict[seed]['test_accuracy_history'] for seed in data_dict]
    error_rate_all_seeds = [(100 - np.array(acc)) for acc in accuracy_all_seeds]
    avg_error_rate = np.mean(error_rate_all_seeds, axis=0)
    error = np.std(error_rate_all_seeds, axis=0)
    val_steps = data_dict[next(iter(data_dict))]['val_history_train_steps'][:]
    ax.plot(val_steps, avg_error_rate, label=model_dict["label"], color=model_dict["color"])
    ax.fill_between(val_steps, avg_error_rate-error, avg_error_rate+error, alpha=0.2, color=model_dict["color"], linewidth=0)
    ax.set_xlabel('Training step')
    ax.set_ylabel('Error Rate (%)', labelpad=0)
    if scale == 'log':
        ax.set_yscale('log')
        ax.set_ylim(0, 100)
        ax.set_yticks([10, 100], labels=['10%', '100%'])


def plot_metric_all_seeds(data_dict, model_dict, populations_to_plot, ax, metric_name, plot_type='cdf', side='both', plot_input=True):
    """
    Generalized function to plot a metric (sparsity, selectivity, or structure) across multiple random seeds.

    Parameters:
        data_dict (dict): Dictionary containing data for different seeds.
        model_dict (dict): Dictionary containing model metadata (e.g., name, color).
        populations_to_plot (list): List of population names to extract the metric from.
        ax (matplotlib.axes.Axes): Matplotlib axis to plot on.
        metric_name (str): Name of the metric to plot ('sparsity', 'selectivity', or 'structure').
        plot_type (str): Type of plot ('cdf' or 'bar').
    """
    metric_all_seeds = []
    metric_InputE = []
    for seed in data_dict:
        metric_one_seed = []
        for population in populations_to_plot:
            metric_one_seed.extend(data_dict[seed][f"metrics_dict"][population][metric_name])
        metric_all_seeds.append(metric_one_seed)
        metric_InputE.extend(data_dict[seed][f"metrics_dict"]["InputE"][metric_name])

    if sum(len(sublist) for sublist in metric_all_seeds) == 0:
        return

    if plot_type == 'cdf':
        pt.plot_cumulative_distribution(metric_all_seeds, ax=ax, label=model_dict["label"], color=model_dict["color"])
        ax.set_ylabel('Fraction of units' if metric_name in ['selectivity', 'structure'] else 'Fraction of patterns')
        ax.set_xlabel(metric_name.capitalize()) 
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_yticks([0, 1])

    elif plot_type == 'bar':
        avg_metric_per_seed = [np.mean(x) for x in metric_all_seeds]
        avg_metric = np.mean(avg_metric_per_seed)
        error = np.std(avg_metric_per_seed)

        if len(ax.patches) == 0 and plot_input:
            avg_input = np.mean(metric_InputE)
            error_input = np.std(metric_InputE)
            bar = ax.bar(-0.5, avg_input, color='gray', width=0.6, alpha=0.4)
            bar[0].set_label('Input')
            ax.errorbar(-0.5, avg_input, yerr=error_input, fmt='none', ecolor='k', capsize=0, linewidth=0.5)

        x = len(ax.patches)
        bar = ax.bar(x, avg_metric, color=model_dict["color"], width=0.6, alpha=0.4)
        bar[0].set_label(model_dict["label"])
        ax.errorbar(x, avg_metric, yerr=error, fmt='none', ecolor='k', capsize=0, linewidth=0.5)
        ax.set_ylabel(metric_name.capitalize())
        ax.set_ylim([0, 1])
        xticks = [-0.5] + [1 + i for i in range(x)]
        ax.set_xticks(xticks)
        xtick_labels = [patch.get_label() for patch in ax.patches]
        ax.set_xticklabels(xtick_labels, rotation=45, ha='right')

    elif plot_type == 'violin':
        # Pool all data into one list
        pooled_data = []
        seed_means = []
        for metric_one_seed in metric_all_seeds:
            seed_means.append(np.mean(metric_one_seed))
            pooled_data.extend(metric_one_seed)

        # Get existing labels (excluding default numerical labels) to set the x-axis positions
        labels = [t.get_text() for t in ax.get_xticklabels()]
        labels = [label for label in labels if not label.replace('.', '').isdigit()]  # Remove numerical labels

        if len(labels) == 0 and plot_input: # Plot InputE if it's the first violin plot
            parts = ax.violinplot(metric_InputE, positions=[0], showmeans=False, showmedians=False, showextrema=False, widths=0.7, side=side)
            parts['bodies'][0].set_alpha(0.65)
            parts['bodies'][0].set_facecolor('lightgray')
            mean_value = np.mean(metric_InputE)
            ax.scatter(0, mean_value, color='tomato', marker='o', s=5, zorder=5, edgecolors='w', linewidth=0.3)
            labels = labels + ['Input']

        if model_dict["label"] not in labels:
            # Update x-axis labels
            new_label = True
            labels = labels + [model_dict["label"]]
            ax.set_xticks(range(len(labels)))  # Set ticks explicitly
            ax.set_xticklabels(labels, rotation=45, ha='right', rotation_mode='anchor', va='center')
            ax.set_ylabel(metric_name.capitalize())
            ax.set_ylim([-0.03, 1.03])
            ax.set_yticks([0,0.5,1])
            ax.set_yticklabels([str(int(tick)) if tick in [0,1] else '' for tick in ax.get_yticks()])
        else: 
            new_label = False

        x = len(labels) - 1
        x_offset = 0
        if new_label == False:
            ax.vlines(x, -0.03, 1.03, color='w', linestyle='-', linewidth=0.8)
        if side == 'low':
            x_offset = -0.12
        elif side=='high':
            x_offset = 0.12

        # Create the violin plot
        parts = ax.violinplot(pooled_data, positions=[x], showmeans=False, showmedians=False, showextrema=False, widths=0.7, side=side)
        parts['bodies'][0].set_alpha(0.65)
        parts['bodies'][0].set_facecolor(model_dict["color"])

        # Scatter on a point with the mean and error bar
        mean_value = np.mean(seed_means)
        error = np.std(seed_means)
        ax.scatter(x+x_offset, mean_value, color='tomato', marker='o', s=5, zorder=5, edgecolors='w', linewidth=0.3)
        # ax.errorbar(x, mean_value, yerr=error, color='k', fmt='none', capsize=0, capthick=0.5, zorder=5)


def plot_dendritic_state_all_seeds(data_dict, model_dict, ax, scale='log'):
    if 'dendritic_state' not in data_dict[next(iter(data_dict.keys()))]:
        print(f"No dendritic state found for {model_dict['display_name']}")
        return
    dendstate_all_seeds = []
    for seed in model_dict['seeds']:
        dendstate_one_seed = data_dict[seed]['dendritic_state']['forward_dendritic_state']['all'][:]
        dendstate_all_seeds.append(dendstate_one_seed)
    avg_dendstate = np.mean(dendstate_all_seeds, axis=0)
    error = np.std(dendstate_all_seeds, axis=0)
    binned_mean_forward_dendritic_state_steps = data_dict[seed]['dendritic_state']['steps'][:]
    ax.plot(binned_mean_forward_dendritic_state_steps, avg_dendstate, label=model_dict["label"], color=model_dict["color"])
    ax.fill_between(binned_mean_forward_dendritic_state_steps, avg_dendstate-error, avg_dendstate+error, alpha=0.5, color=model_dict["color"], linewidth=0)
    ax.set_xlabel('Training step')
    ax.set_ylabel('Dendritic state')
    ax.set_ylim(bottom=-0.005, top=0.3)
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    # ax.set_yscale(scale)
    # ax.hlines(0, *ax.get_xlim(), color='black', linestyle='--', linewidth=1)
    

def plot_angle_vs_bp_all_seeds(data_dict, model_dict, ax, stochastic=True, error='std'):
    angle_all_seeds = []
    for seed in model_dict['seeds']:
        if stochastic:
            angle = data_dict[seed]['angle_vs_bp']['stochastic']['all_params'][:]
            # if np.isnan(angle).any(): # check if there are any NaNs in the array
            #     print(f"Warning: NaN values found in angle array for seed {seed}")
            angle = np.where(np.isnan(angle), 0, angle) # replace NaNs with 0
            bin_size = 3
            n = len(angle) // bin_size
            angle_trimmed = angle[:n * bin_size]
            angle = angle_trimmed.reshape(-1, bin_size).mean(axis=1) # Bin and average timepoints
            angle_all_seeds.append(angle)
        else:
            angle_all_seeds.append(data_dict[seed]['angle_vs_bp']['all_params'])
    avg_angle = np.nanmean(angle_all_seeds, axis=0)

    if error == 'std':
        error = np.nanstd(angle_all_seeds, axis=0)
    elif error == 'sem':
        error = np.nanstd(angle_all_seeds, axis=0) / np.sqrt(len(model_dict['seeds']))
    train_steps = data_dict[seed]['val_history_train_steps'][:]

    if stochastic:
        train_steps = train_steps[::bin_size]
    else:
        train_steps = train_steps[1:]

    ax.plot(train_steps, avg_angle, label=model_dict["label"], color=model_dict["color"])
    ax.fill_between(train_steps, avg_angle-error, avg_angle+error, alpha=0.5, color=model_dict["color"], linewidth=0)
    ax.grid(True, axis='y', color='gray', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Training step')
    ax.set_ylabel('Alignment angle\n(ΔW $\\measuredangle$ vs backprop)')
    ax.set_ylim([-5,max(100, np.nanmax(avg_angle+error))])
    ax.set_xlim([-train_steps[-1]/20, train_steps[-1]+1])
    ax.set_yticks(np.arange(0, 101, 30))
    ax.set_yticklabels([f'{y:.0f}°' for y in ax.get_yticks()])


def plot_angle_FB_all_seeds(data_dict, model_dict, ax, error='std'):
    # Plot angles between forward weights W vs backward weights B
    fb_angles_all_seeds = []
    for seed in model_dict['seeds']:
        angle = data_dict[seed]['feedback_weight_angle_history']['all_params'][:]
        fb_angles_all_seeds.append(angle)
    if len(fb_angles_all_seeds) == 0:
        print(f"No feedback weight angles found for {model_dict['label']}")
        return
    avg_angle = np.nanmean(fb_angles_all_seeds, axis=0)
    if error == 'std':
        error = np.nanstd(fb_angles_all_seeds, axis=0)
    elif error == 'sem':
        error = np.nanstd(fb_angles_all_seeds, axis=0) / np.sqrt(len(model_dict['seeds']))
    train_steps = data_dict[seed]['val_history_train_steps'][:]
    ax.grid(True, axis='y', color='gray', linewidth=0.5, alpha=0.3)
    if np.isnan(avg_angle).any():
        print(f"Warning: NaN values found in avg W vs B angle.")
    else:
        ax.plot(train_steps, avg_angle, color=model_dict['color'], label=model_dict['label'])
        ax.fill_between(train_steps, avg_angle-error, avg_angle+error, alpha=0.5, color=model_dict['color'], linewidth=0)
    ax.set_xlabel('Training step')
    ax.set_ylabel('Alignment angle \n(W $\\measuredangle$ B)')
    ax.set_xlabel('Training step')
    ax.set_ylim([-5,max(100, np.nanmax(avg_angle+error))])
    ax.set_xlim([-train_steps[-1]/20, train_steps[-1]+1])
    ax.set_yticks(np.arange(0, 101, 30))
    ax.set_yticklabels([f'{y:.0f}°' for y in ax.get_yticks()])


def plot_dimensionality_all_seeds(data_dict, model_dict, ax):
    # Plot dimensionality across E layers of the neural network
    dimensionality_all_seeds = []
    for seed in model_dict['seeds']:
        dimensionality_dict = data_dict[seed]['neural_dimensionality']
        # dimensionality_dict = data_dict[seed]['unit_RSM_dimensionality']
        dim_values = [val[()] for key, val in dimensionality_dict.items() if 'E' in key][::-1]  
        dimensionality_all_seeds.append(dim_values)

    labels = [name for name in dimensionality_dict if 'E' in name][::-1]
    avg_dim = np.mean(dimensionality_all_seeds, axis=0)
    error = np.std(dimensionality_all_seeds, axis=0) / np.sqrt(len(model_dict['seeds']))

    ax.plot(avg_dim, linestyle='-', color=model_dict['color'], label=model_dict['label'], linewidth=1)
    ax.fill_between(range(len(avg_dim)), avg_dim-error, avg_dim+error, alpha=0.3, color=model_dict['color'], linewidth=0)
    ax.scatter(range(len(avg_dim)), avg_dim, color=model_dict['color'], marker='o', s=3)
    ax.set_xticks(range(len(avg_dim)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Dimensionality \nof neural representation")

    # ax.grid(True, which='major', color='lightgray', linewidth=0.5)
    # ax.plot(avg_dim, range(len(avg_dim)), color=model_dict['color'], label=model_dict['label'], linewidth=1)
    # ax.fill_betweenx(range(len(avg_dim)), avg_dim-error, avg_dim+error, alpha=0.5, color=model_dict['color'], linewidth=0)
    # ax.set_yticks(range(len(avg_dim)))
    # ax.set_yticklabels(labels)
    # ax.set_xticks(np.arange(0, 40, 10))
    # ax.set_xlim(left=-3)
    # ax.set_xlabel("Dimensionality \nof neural representation")
    # ax.xaxis.set_ticks_position('top')
    # ax.xaxis.set_label_position('top')
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.tick_params(axis='both', length=0)


def plot_confusion_all_seeds(data_dict, model_dict, ax, population):
    between_class_similarity = {label: [] for label in range(10)}
    between_class_similarity_all = []
    for seed in model_dict['seeds']:
        # Calculate the receptive field similarity for each unit (the histogram will pool data across all model seeds)
        unit_labels_dict = data_dict[seed]['unit_labels_dict']
        unit_labels = unit_labels_dict[population][:]
        idx = np.argsort(unit_labels)
        unit_labels = unit_labels[idx]

        average_pop_activity = np.array(data_dict[seed]['average_pop_activity_dict'][population][:]).T
        sorted_pop_activity = average_pop_activity[idx]

        # Calculate within-class and between-class receptive field similarity (accumulate across all seeds)
        for label in range(10):
            class_idx = np.where(unit_labels == label)[0]
            max_activity_outside_class = np.max(sorted_pop_activity[class_idx][:, np.arange(10)!=label], axis=1)
            mean_activity_outside_class = np.mean(sorted_pop_activity[class_idx][:, np.arange(10)!=label], axis=1)
            confusion_ratio = max_activity_outside_class / (mean_activity_outside_class + 1e-10)
            between_class_similarity_all.extend(confusion_ratio)
            between_class_similarity[label].extend(confusion_ratio)

    for label in range(10):
        mean_val = np.mean(between_class_similarity[label])
        std_val = np.std(between_class_similarity[label])
        ax.bar(label, mean_val, width=0.8, label='Between-class' if label==0 else None, color=model_dict["color"], alpha=0.3)
        ax.errorbar(label, mean_val, yerr=std_val, fmt='none', ecolor=model_dict["color"], capsize=0, linewidth=0.5)

    ax.set_ylabel('Confusion ratio (non-\npreferred class selectivity)')
    ax.set_xticks(range(10))
    ax.set_xticklabels(range(10))
    ax.set_ylim(0, 8)
    ax.set_xlabel('Labels', labelpad=0)


########################################################################################################
# Multi-panel figure generation
########################################################################################################


def generate_model_summary_table(model_dict_all, model_list, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/saved_network_pickles/mnist/", save=None, recompute=None):
    mm = 1/25.4 #convert mm to inches
    num_rows = len(model_list)
    fig_height = num_rows*6.5*mm + 10*mm
    fig, ax = plt.subplots(figsize=(180*mm, fig_height))
    # fig, ax = plt.subplots(figsize=(5.5, 9))
    ax.axis('off')

    all_models = list(dict.fromkeys(model_list))
    generate_hdf5_all_seeds(all_models, model_dict_all, config_path_prefix, saved_network_path_prefix, recompute=recompute)

    columns = {'display_name': 0.17, 'Architecture': 0.12, 
               'Hidden Layers': 0.12, 'Algorithm': 0.12, 
               'W Learning Rule': 0.17, 'B Learning Rule': 0.17, 'Bias': 0.08}
    table_vals = []

    for i,model_key in enumerate(all_models):
        model_dict = model_dict_all[model_key]
        network_name = model_dict['config'].split('.')[0]
        hdf5_path = f"data/model_hdf5_plot_data/plot_data_{network_name}.h5"
        network_table_vals = [model_dict[col] for col in columns.keys() if col in model_dict]
        with h5py.File(hdf5_path, 'r') as f:
            # print(f"Generating table for {network_name}")
            data_dict = f[network_name]

            # Get the accuracy for each seed
            accuracy_all_seeds = []
            for seed in model_dict['seeds']:
                accuracy_all_seeds.append(data_dict[seed]['test_accuracy_history'][-1])
            avg_accuracy = np.mean(accuracy_all_seeds)
            std_accuracy = np.std(accuracy_all_seeds)
            sem_accuracy = std_accuracy / np.sqrt(len(accuracy_all_seeds))

            accuracy_all_seeds_extended = []
            for seed in model_dict['seeds']:
                accuracy_all_seeds_extended.append(data_dict[seed]['test_accuracy_history_extended'][-1])
            avg_accuracy_extended = np.mean(accuracy_all_seeds_extended)
            std_accuracy_extended = np.std(accuracy_all_seeds_extended)
            sem_accuracy_extended = std_accuracy_extended / np.sqrt(len(accuracy_all_seeds_extended))

            if 'MNIST' in saved_network_path_prefix:
                new_column_labels = ['MNIST Accuracy \n(20k samples)', 
                                     'MNIST Accuracy \n(50k samples)']
                network_table_vals += [f"{avg_accuracy:.2f} \u00b1 {sem_accuracy:.2f}", 
                                       f"{avg_accuracy_extended:.2f} \u00b1 {sem_accuracy_extended:.2f}"]
            elif 'spiral' in saved_network_path_prefix:
                new_column_labels = ['Spiral Accuracy \n(1 epoch)', 
                                     'Spiral Accuracy \n(10 epochs)']
                network_table_vals += [f"{avg_accuracy:.2f} \u00b1 {sem_accuracy:.2f}", 
                                       f"{avg_accuracy_extended:.2f} \u00b1 {sem_accuracy_extended:.2f}"]
                
        table_vals.append(network_table_vals)

    column_labels = list(columns.keys()) + new_column_labels
    column_labels[0] = ""
    col_widths = list(columns.values()) + [0.14, 0.14]
    
    table = ax.table(cellText=table_vals, colLabels=column_labels, cellLoc="center", loc="center", colWidths=col_widths)
    table.auto_set_font_size(False)
    
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_height(cell.get_height() * 1.5)
        cell.set_text_props(fontname='Arial', fontsize=6)
        if key[0] == 0: # Header row
            cell.set_facecolor([0.9 for i in range(3)])
            cell.set_text_props(weight='bold')
            cell.set_height(cell.get_height() * 1.2)
        elif key[0] % 2 == 0: # Even rows
            cell.set_facecolor([0.96 for i in range(3)]) # make even rows light grey

        if key[1] == 0: # First column
            cell.set_text_props(horizontalalignment='left', weight='semibold')

    if save:
        fig.savefig(f"figures/{save}.png", dpi=300)
        fig.savefig(f"figures/{save}.svg", dpi=300)
        fig.savefig(f"figures/{save}.tiff", dpi=300)


def generate_hyperparams_table(csv_filename, save):
    # Load model specs from csv file
    df = pd.read_csv(csv_filename)
    # df = df.map(lambda x: codecs.decode(x, 'unicode_escape') if isinstance(x, str) else x) # convert special characters like \n

    num_columns = len(df.columns)
    num_rows = len(df)

    mm = 1/25.4 #convert mm to inches
    fig_width = num_columns*22*mm
    fig_height = num_rows*5.8*mm
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    if fig_width > 8.5:
        print(f"WARNING: Table too wide ({fig_width} inch) to fit on one page. Consider reducing the number of columns.")
    if fig_height > 11:
        print(f"WARNING: Table too tall ({fig_height} inch) to fit on one page. Consider reducing the number of rows.")
    # fig, ax = plt.subplots(figsize=(5.5, 9))
    ax.axis('off')

    def round_if_numeric(x, decimals):
        try:
            f = float(x)
            return f"{f:.{decimals}f}"
        except (ValueError, TypeError):
            return x
    df = df.map(round_if_numeric, decimals=4)

    col_widths = [1/(0.83*num_columns)] * num_columns
    col_widths[0] *= 1.2
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center", colWidths=col_widths)
    table.auto_set_font_size(False)    
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_height(cell.get_height() * 1.3)
        cell.set_text_props(fontname='Arial', fontsize=6)
        if key[0] == 0: # Header row
            cell.set_facecolor([0.9 for i in range(3)])
            cell.set_text_props(weight='bold')
            cell.set_height(cell.get_height() * 1.2)
        elif key[0] % 2 == 0: # Even rows
            cell.set_facecolor([0.96 for i in range(3)]) # make even rows light grey

        if key[1] == 0: # First column
            cell.set_text_props(horizontalalignment='left')
            # cell.set_width(cell.get_width() * 1.5)
            
    fig.savefig(f"figures/{save}.png", dpi=300)
    fig.savefig(f"figures/{save}.svg", dpi=300)
    fig.savefig(f"figures/{save}.tiff", dpi=300)


def compare_RSM_properties(model_dict_all, model_list_heatmaps, model_list_metrics, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/mnist/", save=None, recompute=None):
    fig = plt.figure(figsize=(5.5, 9))
    axes = gs.GridSpec(nrows=4, ncols=3, figure=fig,
                       left=0.1,right=0.95,
                       top=0.95, bottom = 0.4,
                       wspace=0.5, hspace=0.5,
                       width_ratios=[1, 1, 1])
    
    all_models = list(dict.fromkeys(model_list_heatmaps + model_list_metrics))
    generate_hdf5_all_seeds(all_models, model_dict_all, config_path_prefix, saved_network_path_prefix, recompute=recompute)

    for row, model_key in enumerate(all_models):
        model_dict = model_dict_all[model_key]
        network_name = model_dict['config'].split('.')[0]
        hdf5_path = f"data/model_hdf5_plot_data/plot_data_{network_name}.h5"
        with h5py.File(hdf5_path, 'r') as f:
            data_dict = f[network_name]
            print(f"Generating plots for {model_dict['label']}")

            # Plot RSM heatmaps
            if model_key in model_list_heatmaps:
                seed = model_dict['seeds'][0] # example seed to plot

                pop_activity_dict = data_dict[seed]['sorted_activity_dict']
                # pattern_labels = data_dict[seed]['sorted_pattern_labels']
                unit_labels_dict = data_dict[seed]['sorted_unit_labels_dict']

                for col, pop_name in enumerate(['H1E','H2E']):
                    pop_activity = pop_activity_dict[pop_name][:]
                    neuron_similarity_matrix = cosine_similarity(pop_activity.T)
                    ax = fig.add_subplot(axes[row, col])
                    im = ax.imshow(neuron_similarity_matrix)
                    cbar = fig.colorbar(im, ax=ax)

                    if col==0:
                        ax.set_title(model_dict["label"], rotation=90, x=-0.6, y=0.5, ha='left', va='center')

                    ax.set_xlabel('Units')
                    ax.set_ylabel('Units')
                    num_units = neuron_similarity_matrix.shape[0]
                    if pop_name == 'OutputE':
                        x_ticks = np.arange(0, num_units)
                        ax.set_xticks(x_ticks)
                        # ax.set_xticklabels(range(0, num_units, 10))
                        y_ticks = np.arange(0, num_units)
                        ax.set_yticks(y_ticks)
                        # ax.set_yticklabels(range(0, num_units, 10))

                    unit_labels = unit_labels_dict[pop_name][:]
                    nan_idx = np.isnan(unit_labels)
                    pop_is_sorted = np.all(unit_labels[~nan_idx][:-1] <= unit_labels[~nan_idx][1:])
                    if pop_is_sorted:
                        for i in range(10):
                            class_idx = np.where(unit_labels == i)[0]
                            cmap = matplotlib.colormaps['tab20']
                            if len(class_idx) > 0:
                                class_boundary_start = class_idx[0]
                                class_boundary_end = class_idx[-1]+1
                                ax.add_patch(matplotlib.patches.Rectangle((class_boundary_start-0.5, class_boundary_start-0.5), class_boundary_end-class_boundary_start, class_boundary_end-class_boundary_start, fill=False, edgecolor=cmap(i), linewidth=0.5, facecolor=cmap(i)))
                

    if save:
        fig.savefig(f"figures/{save}.png", dpi=300)
        fig.savefig(f"figures/{save}.svg", dpi=300)


def compare_structure(model_dict_all, model_list_heatmaps, model_list_metrics, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/mnist/", save=None, recompute=None):
    '''
    Figure 1: Van_BP vs bpDale(learnedI)
        -> bpDale is more structured/sparse (focus on H1E metrics)

    Compare vanilla Backprop to networks with 'cortical' architecures (i.e. with somatic feedback inhibition). 
    '''

    fig = plt.figure(figsize=(5.5, 9))
    axes = gs.GridSpec(nrows=2, ncols=4, figure=fig,               
                       left=0.049,right=0.9,
                       top=0.9, bottom = 0.6,
                       wspace=0.15, hspace=0.5)
    axes_metrics = gs.GridSpec(nrows=2, ncols=3, figure=fig,            
                        left=0.15,right=1,
                        top=0.9, bottom = 0.65,
                        wspace=0.15, hspace=0.5)    
    ax_structure   = fig.add_subplot(axes_metrics[0, 2])

    all_models = list(dict.fromkeys(model_list_heatmaps + model_list_metrics))
    generate_hdf5_all_seeds(all_models, model_dict_all, config_path_prefix, saved_network_path_prefix, recompute=recompute)

    col = 0
    for model_key in all_models:
        model_dict = model_dict_all[model_key]
        network_name = model_dict['config'].split('.')[0]
        hdf5_path = f"data/model_hdf5_plot_data/plot_data_{network_name}.h5"
        with h5py.File(hdf5_path, 'r') as f:
            data_dict = f[network_name]
            print(f"Generating plots for {model_dict['label']}")
            populations_to_plot = ['H2E']

            if model_key in model_list_heatmaps:
                seed = model_dict['seeds'][0] # example seed to plot
                # populations_to_plot = [population for population in data_dict[seed]['average_pop_activity_dict'] if 'E' in population and population!='InputE']
                
                for row,population in enumerate(populations_to_plot):
                    # Receptive field plots
                    receptive_fields = torch.tensor(np.array(data_dict[seed][f"maxact_receptive_fields_{population}"]))
                    num_units = 10
                    ax = fig.add_subplot(axes[row, col])
                    ax.axis('off')
                    if row==0:
                        ax.set_title(model_dict["label"])
                    pos = ax.get_position()
                    new_left = pos.x0 - 0.01  # Move left boundary to the left
                    new_bottom = pos.y0 # Move bottom boundary up
                    new_height = pos.height  # Decrease height
                    new_width = pos.width - 0.036  # Decrease width 
                    ax.set_position([new_left, new_bottom, new_width, new_height])
                    rf_axes = gs.GridSpecFromSubplotSpec(4, 3, subplot_spec=ax, wspace=0., hspace=0.1)
                    ax_list = [fig.add_subplot(rf_axes[3,1])]
                    for j in range(num_units-1):
                        ax = fig.add_subplot(rf_axes[j])
                        ax_list.append(ax)
                        # box = matplotlib.patches.Rectangle((-0.5,-0.5), 28, 28, linewidth=0.5, edgecolor='k', facecolor='none', zorder=10)
                        # ax.add_patch(box)
                    preferred_classes = torch.argmax(torch.tensor(np.array(data_dict[seed]['average_pop_activity_dict'][population])), dim=1)
                    im = pt.plot_receptive_fields(receptive_fields, sort=True, ax_list=ax_list, preferred_classes=preferred_classes)
                    fig_width, fig_height = fig.get_size_inches()
                    cax = fig.add_axes([ax_list[0].get_position().x0-0.32/fig_width, ax.get_position().y0-0.2/fig_height, 0.04, 0.03/fig_height])
                    fig.colorbar(im, cax=cax, orientation='horizontal')

                col += 1

            if model_key in model_list_metrics:
                plot_metric_all_seeds(data_dict, model_dict, populations_to_plot=populations_to_plot, ax=ax_structure, metric_name='structure')

    if save is not None:
        fig.savefig(f"figures/{save}.png", dpi=300)
        fig.savefig(f"figures/{save}.svg", dpi=300)


def generate_metrics_plot(model_dict_all, model_list, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/mnist/", save=None, recompute=None): 
    # fig = plt.figure(figsize=(5.5, 4))
    fig = plt.figure(figsize=(7, 4))

    axes = gs.GridSpec(nrows=4, ncols=4, figure=fig, bottom=0.1, top=0.9, left=0.1, right=0.8, hspace=0.5, wspace=0.5)
    ax_accuracy = fig.add_subplot(axes[0,0])
    ax_structure = fig.add_subplot(axes[0,1])
    ax_dendstate = fig.add_subplot(axes[0,2])
    ax_angleBP_stoch = fig.add_subplot(axes[1,2])
    ax_sparsity = fig.add_subplot(axes[1,0])
    ax_selectivity = fig.add_subplot(axes[1,1])
    ax_FB_angles = fig.add_subplot(axes[2,0])
    ax_angleBP = fig.add_subplot(axes[2,1])
    ax_sparsity_hist = fig.add_subplot(axes[3,0])
    ax_selectivity_hist = fig.add_subplot(axes[3,1])
    ax_error_hist = fig.add_subplot(axes[3,2])

    all_models = list(dict.fromkeys(model_list))
    for model_key in all_models:
        model_dict = model_dict_all[model_key]
        config_path = config_path_prefix + model_dict['config']
        network_name = model_dict['config'].split('.')[0]
        hdf5_path = f"data/model_hdf5_plot_data/plot_data_{network_name}.h5"
        for seed in model_dict['seeds']:
            saved_network_path = saved_network_path_prefix + network_name + f"_{seed}.pkl"
            generate_data_hdf5(config_path, saved_network_path, hdf5_path, recompute=recompute)
            gc.collect()

    for i,model_key in enumerate(all_models):
        model_dict = model_dict_all[model_key]
        network_name = model_dict['config'].split('.')[0]
        hdf5_path = f"data/model_hdf5_plot_data/plot_data_{network_name}.h5"

        with h5py.File(hdf5_path, 'r') as f:
            data_dict = f[network_name]

            plot_angle_FB_all_seeds(data_dict, model_dict, ax=ax_FB_angles)
            plot_angle_vs_bp_all_seeds(data_dict, model_dict, ax=ax_angleBP, stochastic=False)
            plot_angle_vs_bp_all_seeds(data_dict, model_dict, ax=ax_angleBP_stoch, stochastic=True)
            plot_accuracy_all_seeds(data_dict, model_dict, ax=ax_accuracy)
            plot_error_all_seeds(data_dict, model_dict, ax=ax_error_hist)
            plot_dendritic_state_all_seeds(data_dict, model_dict, ax=ax_dendstate)

            if 'H1E' in data_dict[seed]['sparsity_history'] and 'H2E' in data_dict[seed]['sparsity_history']:
                plot_metric_all_seeds(data_dict, model_dict, populations_to_plot=['H1E','H2E'], ax=ax_selectivity, metric_name='selectivity', plot_type='violin')
                plot_metric_all_seeds(data_dict, model_dict, populations_to_plot=['H1E','H2E'], ax=ax_sparsity, metric_name='sparsity', plot_type='violin')
                plot_metric_all_seeds(data_dict, model_dict, populations_to_plot=['H1E','H2E'], ax=ax_structure, metric_name='structure', plot_type='violin')

                # Sparsity history
                val_steps = data_dict[seed]['val_history_train_steps'][:]
                sparsity_history_all_seeds = []
                for seed in data_dict:
                    H1E_sparsity_history = data_dict[seed]['sparsity_history']['H1E'][:]
                    H2E_sparsity_history = data_dict[seed]['sparsity_history']['H2E'][:]
                    sparsity_history = np.mean(np.stack([H1E_sparsity_history, H2E_sparsity_history]), axis=0)
                    sparsity_history_all_seeds.append(sparsity_history)
                avg_sparsity = np.mean(sparsity_history_all_seeds, axis=0)
                std_sparsity = np.std(sparsity_history_all_seeds, axis=0)
                ax_sparsity_hist.plot(val_steps, avg_sparsity, label=f"{model_dict['label']}", color=model_dict["color"])
                ax_sparsity_hist.fill_between(val_steps, avg_sparsity-std_sparsity, avg_sparsity+std_sparsity, alpha=0.2, color=model_dict["color"], linewidth=0)
                ax_sparsity_hist.set_xlabel('Training step')
                ax_sparsity_hist.set_ylabel('Sparsity')
                ax_sparsity_hist.set_ylim([0,1])

                # Selectivity history
                selectivity_history_all_seeds = []
                for seed in data_dict:
                    H1E_selectivity_history = data_dict[seed]['selectivity_history']['H1E'][:]
                    H2E_selectivity_history = data_dict[seed]['selectivity_history']['H2E'][:]
                    selectivity_history = np.mean(np.stack([H1E_selectivity_history, H2E_selectivity_history]), axis=0)
                    selectivity_history_all_seeds.append(selectivity_history)
                avg_selectivity = np.mean(selectivity_history_all_seeds, axis=0)
                std_selectivity = np.std(selectivity_history_all_seeds, axis=0)
                ax_selectivity_hist.plot(val_steps, avg_selectivity, label=f"{model_dict['label']}", color=model_dict["color"])
                ax_selectivity_hist.fill_between(val_steps, avg_selectivity-std_selectivity, avg_selectivity+std_selectivity, alpha=0.2, color=model_dict["color"], linewidth=0)
                ax_selectivity_hist.set_xlabel('Training step')
                ax_selectivity_hist.set_ylabel('Selectivity')
                ax_selectivity_hist.set_ylim([0,1])

    if save:
        fig.savefig(f"figures/{save}.png", dpi=300)
        fig.savefig(f"figures/{save}.svg", dpi=300)


########################################################################################################
# Main script
########################################################################################################

@click.command()
@click.option('--figure', default=None, help='Figure to generate')
@click.option('--recompute', default=None, help='Recompute plot data for a particular parameter')

def main(figure, recompute):
    # Load model specs from csv file
    csv_file_path = "data/figure_model_specs.csv" 
    df = pd.read_csv(csv_file_path, index_col=0)
    df = df.map(lambda x: codecs.decode(x, 'unicode_escape') if isinstance(x, str) else x) # convert special characters like \n
    model_dict_all = df.transpose().to_dict()

    # Set path to Box data directory (default path based on OS)
    if os.name == "posix":
        username = os.environ.get("USER")
        saved_network_path_prefix = f"/Users/{username}/Library/CloudStorage/Box-Box/Milstein-Shared/EIANN exported data/2024 Manuscript V2/"
    elif os.name == "nt":
        username = os.environ.get("USERNAME")
        saved_network_path_prefix = f"C:/Users/{username}/Box/Milstein-Shared/EIANN exported data/2024 Manuscript V2/"
    
    seeds = ["66049_257","66050_258", "66051_259", "66052_260", "66053_261"]
    for model_key in model_dict_all:
        model_dict_all[model_key]["seeds"] = seeds


    #-------------- Supplementary Tables --------------

    if figure in ["all", "T7"]:
        saved_network_path_prefix += "FMNIST/"
        figure_name = "FigT7_fmnist_table"
        model_list = ["fmnist_DTP_TCWN_hebbdend", "fmnist_DTP_WT_hebbdend", "fmnist_BTSP_TCWN_hebbdend", "fmnist_BTSP_WT_nobias_hebbdend", 
                      "fmnist_vanBP_nobias", "fmnist_bpDale_nobias", "fmnist_0hidden_vanBP_nobias", "fmnist_fixed_vanBP_nobias"]
        generate_model_summary_table(model_dict_all, model_list, saved_network_path_prefix=saved_network_path_prefix+"extended/", config_path_prefix="network_config/fmnist/", save=figure_name, recompute=recompute)

    #-------------- Other Figures --------------

    # Representational similarity analysis
    if figure in ["all", "rsm"]:
        saved_network_path_prefix += "MNIST/"
        model_list_heatmaps = ["bpDale_learned", "bpDale_fixed", "HebbWN_topsup", "bpLike_WT_hebbdend"]
        model_list_metrics = model_list_heatmaps
        figure_name = "Suppl_similarity_analysis"
        compare_RSM_properties(model_dict_all, model_list_heatmaps, model_list_metrics, save=figure_name, saved_network_path_prefix=saved_network_path_prefix, recompute=recompute)

    if figure in ["all", "structure"]:
        saved_network_path_prefix += "MNIST/"
        figure_name = "structure"
        model_list_heatmaps = ["vanBP", "bpDale_fixed", "bpLike_WT_hebbdend"]
        model_list_metrics = model_list_heatmaps
        compare_structure(model_dict_all, model_list_heatmaps, model_list_metrics, save=figure_name, saved_network_path_prefix=saved_network_path_prefix, recompute=recompute)

    if figure in ["all", "metrics"]:
        saved_network_path_prefix += "MNIST/"
        # model_list = ["vanBP", "bpDale_learned", "bpLike_fixedDend", "bpLike_hebbdend", "bpLike_hebbTD", "bpLike_FA"]
        # model_list = ["BTSP_WT_hebbdend", "BTSP_hebbTD_hebbdend", "BTSP_fixedTD_hebbdend"]
        # model_list = ["bpLike_hebbTD_hebbdend_eq", "bpLike_WT_hebbdend_eq", "bpLike_hebbTD_hebbdend", "bpLike_WT_hebbdend"]
        figure_name = "metrics_all_models"
        generate_metrics_plot(model_dict_all, model_list, save=figure_name, saved_network_path_prefix=saved_network_path_prefix, recompute=recompute)