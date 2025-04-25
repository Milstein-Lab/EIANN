import torch
import numpy as np
import scipy

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import os
import pathlib
import h5py
import click
import gc
import copy
import string
from reportlab.pdfgen import canvas
from sklearn.metrics.pairwise import cosine_similarity

import EIANN.utils as ut
import EIANN.plot as pt
import EIANN._network as nt

plt.rcParams.update({'font.size': 6,
                    'axes.spines.right': False,
                    'axes.spines.top':   False,
                    'axes.linewidth':    0.5,
                    'axes.labelpad':     2.0, 
                    'xtick.major.size':  2,
                    'xtick.major.width': 0.5,
                    'ytick.major.size':  2,
                    'ytick.major.width': 0.5,
                    'xtick.major.pad':   2,
                    'ytick.major.pad':   2,
                    'legend.frameon':       False,
                    'legend.handletextpad': 0.5,
                    'legend.handlelength': 0.8,
                    # 'legend.handleheight': 10,
                    'legend.labelspacing': 0.2,
                    'legend.columnspacing': 1.2,
                    'lines.linewidth': 0.5,
                    'figure.figsize': [10.0, 3.0],
                    'font.sans-serif': 'Avenir',
                    'svg.fonttype': 'none',
                    'text.usetex': False})



########################################################################################################

def generate_data_hdf5(config_path, saved_network_path, hdf5_path, recompute=None):
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
    if 'extended' in saved_network_path:
        network_seed, data_seed = pickle_filename.split('_')[-3:-1]
    assert network_seed.isdigit() and data_seed.isdigit(), f"network_seed and data_seed must be numbers, but got {network_seed} and {data_seed}"
    network = ut.build_EIANN_from_config(config_path, network_seed=network_seed)    
    seed = f"{network_seed}_{data_seed}"

    # Define which variables to compute
    variables_to_save = ['percent_correct', 'average_pop_activity_dict', 'activity_dynamics',
                         'val_loss_history', 'val_accuracy_history', 'val_history_train_steps', 'test_loss_history', 'test_accuracy_history',
                         'angle_vs_bp', 'angle_vs_bp_stochastic', 'feedback_weight_angle_history', 'sparsity_history', 'selectivity_history']
    if "Dend" in "".join(network.populations.keys()):
        variables_to_save.extend(["binned_mean_forward_dendritic_state", "binned_mean_forward_dendritic_state_steps"])
    if 'extended' in saved_network_path:
        variables_to_save.append('test_accuracy_history_extended')
    if 'mnist' in config_path:
        variables_to_save.extend([f"metrics_dict_{population.fullname}" for population in network.populations.values()])
        # variables_to_save.extend([f"maxact_receptive_fields_{population.fullname}" for population in network.populations.values() if population.name=='E' and population.fullname!='InputE'])
    if 'spiral' in config_path:
        variables_to_save.extend(['spiral_decision_data_dict'])

    # Open hdf5 and check if the relevant network data already exists       
    variables_to_recompute = []  
    if os.path.exists(hdf5_path): # If the file exists, check if the network data already exists or needs to be recomputed
        with h5py.File(hdf5_path, 'r') as file:
            if network_name in file.keys():
                if seed in file[network_name].keys():
                    if recompute == 'all':
                        print(f"Overwriting {network_name} {seed} in {hdf5_path}")
                        variables_to_recompute = variables_to_save                        
                    elif set(variables_to_save).issubset(file[network_name][seed].keys()) and recompute is None:
                        # print(f"Data for {network_name} {seed} exists in {hdf5_path}")
                        return
                    else:
                        print(f"Recomputing {network_name} {seed} in {hdf5_path}")
                        variables_to_recompute = [var for var in variables_to_save if var not in file[network_name][seed].keys()]
                else:
                    print(f"Computing data for name:{network_name}, seed:{seed} in {hdf5_path}")
                    variables_to_recompute = variables_to_save     
            else:
                print(f"Computing data for {network_name} {seed} in {hdf5_path}")
                variables_to_recompute = variables_to_save     
    else:
        print(f"Creating new data file {hdf5_path}")
        variables_to_recompute = variables_to_save
    
    if recompute is not None and recompute not in variables_to_recompute:
        variables_to_recompute.append(recompute)

    print("-----------------------------------------------------------------------------")

    print(variables_to_recompute)
    if not variables_to_recompute:
        print(f"No variables to recompute for {network_name} {seed} in {hdf5_path}")
        return        

    # Load the saved network pickle
    network = ut.load_network(saved_network_path)
    if type(network) != nt.Network:
        print("WARNING: Network pickle is not a Network object!")
        network = ut.build_EIANN_from_config(config_path, network_seed=network_seed)    
        ut.load_network_dict(network, saved_network_path)
    network.seed = seed
    network.name = network_name
    if not hasattr(network, 'input_pop'):
        input_layer = list(network)[0]
        network.input_pop = next(iter(input_layer))

    # Load dataset
    if 'mnist' in config_path:
        all_dataloaders = ut.get_MNIST_dataloaders(sub_dataloader_size=1000)
        train_dataloader, train_sub_dataloader, val_dataloader, test_dataloader, data_generator = all_dataloaders
    elif 'spiral' in config_path:
        all_dataloaders = ut.get_spiral_dataloaders(batch_size='full_dataset')
        train_dataloader, val_dataloader, test_dataloader, data_generator = all_dataloaders

    ##################################################################
    ## Generate plot data

    # Class-averaged activity
    if 'average_pop_activity_dict' in variables_to_recompute:
        average_pop_activity_dict, pattern_labels, unit_labels_dict = ut.compute_test_activity(network, test_dataloader, class_average=True, sort=False)
        ut.save_plot_data(network.name, network.seed, data_key='average_pop_activity_dict', data=average_pop_activity_dict, file_path=hdf5_path, overwrite=True)

    if 'percent_correct' in variables_to_recompute:
        pop_activity_dict, pattern_labels, unit_labels_dict = ut.compute_test_activity(network, test_dataloader, class_average=False, sort=True)
        ut.save_plot_data(network.name, network.seed, data_key='sorted_activity_dict', data=pop_activity_dict, file_path=hdf5_path, overwrite=True)
        ut.save_plot_data(network.name, network.seed, data_key='sorted_pattern_labels', data=pattern_labels, file_path=hdf5_path, overwrite=True)
        ut.save_plot_data(network.name, network.seed, data_key='sorted_unit_labels_dict', data=unit_labels_dict, file_path=hdf5_path, overwrite=True)

        output = pop_activity_dict[network.output_pop.fullname]
        percent_correct = ut.compute_test_accuracy(output, pattern_labels)
        ut.save_plot_data(network.name, network.seed, data_key='percent_correct', data=percent_correct, file_path=hdf5_path, overwrite=True)
        
    # Receptive fields and metrics
    for population in network.populations.values():
        if f"metrics_dict_{population.fullname}" in variables_to_recompute:
            receptive_fields = None
            if population.name == "E" and population.fullname != "InputE":
                receptive_fields = ut.compute_maxact_receptive_fields(population, export=True, export_path=hdf5_path, overwrite=True)
            metrics_dict = ut.compute_representation_metrics(population, test_dataloader, receptive_fields, export=True, export_path=hdf5_path, overwrite=True)

    # Angle vs Backprop
    if set(['angle_vs_bp','angle_vs_bp_stochastic']).intersection(variables_to_recompute):
        stored_history_step_size = torch.diff(network.param_history_steps)[-1]
        if 'mnist' in config_path:
            comparison_config_path = os.path.join(os.path.dirname(config_path), "20231129_EIANN_2_hidden_mnist_bpDale_relu_SGD_config_G_complete_optimized.yaml")
        elif 'spiral' in config_path:
            comparison_config_path = os.path.join(os.path.dirname(config_path), "20250108_EIANN_2_hidden_spiral_bpDale_fixed_SomaI_learned_bias_config_complete_optimized.yaml")
        comparison_network = ut.build_EIANN_from_config(comparison_config_path, network_seed=network_seed)
        if not ut.network_architectures_match(network, comparison_network):
            print("WARNING: Network architectures do not match. Ignoring comparison network and computing angle vs the same network with Backprop conversion.")
            print(f"Network 1: {config_path}")
            print(f"Network 2: {comparison_config_path}")
            comparison_network = None

        # Compare overall angle after many train steps (stored_history_step_size)
        bpClone_network = ut.compute_alternate_dParam_history(train_dataloader, network, comparison_network, batch_size=stored_history_step_size)
        angles = ut.compute_dW_angles_vs_BP(bpClone_network.predicted_dParam_history, bpClone_network.actual_dParam_history_stepaveraged)
        ut.save_plot_data(network.name, network.seed, data_key='angle_vs_bp', data=angles, file_path=hdf5_path, overwrite=True)
        
        # Compare angles for one train step (batch_size=1)
        bpClone_network = ut.compute_alternate_dParam_history(train_dataloader, network, comparison_network, batch_size=1)
        angles = ut.compute_dW_angles_vs_BP(bpClone_network.predicted_dParam_history, bpClone_network.actual_dParam_history)
        ut.save_plot_data(network.name, network.seed, data_key='angle_vs_bp_stochastic', data=angles, file_path=hdf5_path, overwrite=True)

    # Forward vs Backward weight angle (weight symmetry)
    if 'feedback_weight_angle_history' in variables_to_recompute:
        FF_FB_angles = ut.compute_feedback_weight_angle_history(network)
        ut.save_plot_data(network.name, network.seed, data_key='feedback_weight_angle_history', data=FF_FB_angles, file_path=hdf5_path, overwrite=True)

    # Binned dendritic state (local loss)
    if 'binned_mean_forward_dendritic_state' in variables_to_recompute:
        steps, binned_mean_forward_dendritic_state = ut.get_binned_mean_population_attribute_history_dict(network, attr_name="forward_dendritic_state", bin_size=100, abs=True)
        if binned_mean_forward_dendritic_state is not None:
            ut.save_plot_data(network.name, network.seed, data_key='binned_mean_forward_dendritic_state', data=binned_mean_forward_dendritic_state, file_path=hdf5_path, overwrite=True)
            ut.save_plot_data(network.name, network.seed, data_key='binned_mean_forward_dendritic_state_steps', data=steps, file_path=hdf5_path, overwrite=True)

    # Sparsity and selectivity
    if 'sparsity_history' in variables_to_recompute or 'selectivity_history' in variables_to_recompute:
        sparsity_history_dict, selectivity_history_dict = ut.compute_sparsity_selectivity_history(network, test_dataloader)
        ut.save_plot_data(network.name, network.seed, data_key='sparsity_history', data=sparsity_history_dict, file_path=hdf5_path, overwrite=True)
        ut.save_plot_data(network.name, network.seed, data_key='selectivity_history', data=selectivity_history_dict, file_path=hdf5_path, overwrite=True)

    # Loss and accuracy
    if set(['val_loss_history', 'val_accuracy_history', 'val_history_train_steps']).intersection(variables_to_recompute):
        ut.save_plot_data(network.name, network.seed, data_key='val_loss_history',          data=network.val_loss_history,          file_path=hdf5_path, overwrite=True)
        ut.save_plot_data(network.name, network.seed, data_key='val_accuracy_history',      data=network.val_accuracy_history,      file_path=hdf5_path, overwrite=True)
        ut.save_plot_data(network.name, network.seed, data_key='val_history_train_steps',   data=network.val_history_train_steps,   file_path=hdf5_path, overwrite=True)
    
    if set(['test_loss_history', 'test_accuracy_history']).intersection(variables_to_recompute):
        ut.save_plot_data(network.name, network.seed, data_key='test_loss_history',         data=network.test_loss_history,         file_path=hdf5_path, overwrite=True)
        ut.save_plot_data(network.name, network.seed, data_key='test_accuracy_history',     data=network.test_accuracy_history,     file_path=hdf5_path, overwrite=True)

    if 'test_accuracy_history_extended' in variables_to_recompute:
        ut.save_plot_data(network.name, network.seed, data_key='test_accuracy_history_extended', data=network.test_accuracy_history, file_path=hdf5_path, overwrite=True)
        ut.save_plot_data(network.name, network.seed, data_key='val_history_train_steps_extended', data=network.val_history_train_steps, file_path=hdf5_path, overwrite=True)

    # Spiral decision boundary plots
    if 'spiral_decision_data_dict' in variables_to_recompute:
        spiral_decision_data_dict = ut.compute_spiral_decisions_data(network, test_dataloader)
        ut.save_plot_data(network.name, network.seed, data_key='spiral_decision_data_dict', data=spiral_decision_data_dict, file_path=hdf5_path, overwrite=True)

    # Network forward dynamics
    if 'activity_dynamics' in variables_to_recompute:
        pop_dynamics_dict = ut.compute_test_activity_dynamics(network, test_dataloader)
        ut.save_plot_data(network.name, network.seed, data_key='activity_dynamics', data=pop_dynamics_dict, file_path=hdf5_path, overwrite=True)


def generate_hdf5_all_seeds(model_list, model_dict_all, config_path_prefix, saved_network_path_prefix, recompute=None):
    for model_key in model_list:
        model_dict = model_dict_all[model_key]
        config_path = config_path_prefix + model_dict['config']
        network_name = model_dict['config'].split('.')[0]
        hdf5_path = f"data/plot_data_{network_name}.h5"

        if not os.path.exists(hdf5_path):
            # If the hdf5 is not available in local data directory, check in Box drive
            print("Local hdf5 not found, loading data from Box drive")
            num_parent_dirs = 2 if "extended" in saved_network_path_prefix else 1
            box_hdf5_dir = pathlib.Path(saved_network_path_prefix).parents[num_parent_dirs] / "2024 Figure data HDF5 files"
            box_hdf5_path = box_hdf5_dir / f"plot_data_{network_name}.h5"
            if box_hdf5_path.exists():
                print(f"Loading hdf5 from Box drive: {box_hdf5_path}")
                hdf5_path = str(box_hdf5_path)
            else:
                print("hdf5 not found in Box drive")

        for seed in model_dict['seeds']:
            extended_flag = '_extended' if "extended" in saved_network_path_prefix else ""
            saved_network_path = saved_network_path_prefix + network_name + f"_{seed}{extended_flag}.pkl"
            generate_data_hdf5(config_path, saved_network_path, hdf5_path, recompute)
            gc.collect()


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
    ax.set_ylabel('Test accuracy (%)', labelpad=-2)
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
    # elif scale == 'linear':
    #     ax.set_ylim(0, 10)
    #     ax.set_yticks([0, 5, 10], labels=['0%', '5%', '10%'])


def plot_metric_all_seeds(data_dict, model_dict, populations_to_plot, ax, metric_name, plot_type='cdf', side='both'):
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
    for seed in data_dict:
        metric_one_seed = []
        for population in populations_to_plot:
            metric_one_seed.extend(data_dict[seed][f"metrics_dict_{population}"][metric_name])
        metric_all_seeds.append(metric_one_seed)

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
        x = len(ax.patches)
        print(x)

        bar = ax.bar(x, avg_metric, yerr=error, color=model_dict["color"], width=0.6, ecolor='red', alpha=0.8)
        bar[0].set_label(model_dict["label"])
        ax.set_ylabel(metric_name.capitalize())
        ax.set_ylim([0, 1])
        ax.set_xticks(range(x + 1))
        xtick_labels = [patch.get_label() for patch in ax.patches]
        ax.set_xticklabels(xtick_labels, rotation=45, ha='right')

    elif plot_type == 'violin':
        # Pool all data into one list
        # pooled_data = [value for sublist in metric_all_seeds for value in sublist]
        pooled_data = []
        seed_means = []
        for metric_one_seed in metric_all_seeds:
            seed_means.append(np.mean(metric_one_seed))
            pooled_data.extend(metric_one_seed)

        # Get existing labels (excluding default numerical labels) to set the x-axis positions
        labels = [t.get_text() for t in ax.get_xticklabels()]
        labels = [label for label in labels if not label.replace('.', '').isdigit()]  # Remove numerical labels
        if model_dict["label"] not in labels:
            # Update x-axis labels
            new_label = True
            labels = labels + [model_dict["label"]]
            ax.set_xticks(range(len(labels)))  # Set ticks explicitly
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel(metric_name.capitalize())
            ax.set_ylim([-0.03, 1.03])
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
        parts = ax.violinplot(pooled_data, positions=[x], showmeans=False, showmedians=False, showextrema=False, widths=0.9, side=side)
        parts['bodies'][0].set_alpha(0.65)
        parts['bodies'][0].set_facecolor(model_dict["color"])

        # Scatter on a point with the mean and error bar
        mean_value = np.mean(seed_means)
        error = np.std(seed_means) #/ np.sqrt(len(seed_means))
        # ax.scatter(x, mean_value, color='red', marker='o', s=5, zorder=5)
        ax.scatter(x+x_offset, mean_value, color='tomato', marker='o', s=5, zorder=5, edgecolors='w', linewidth=0.3)
        # ax.errorbar(x, mean_value, yerr=error, color='red', fmt='none', capsize=2, capthick=1, zorder=5)

        
def plot_dendritic_state_all_seeds(data_dict, model_dict, ax, scale='log'):
    if 'binned_mean_forward_dendritic_state_steps' not in data_dict[next(iter(data_dict.keys()))]:
        return
    dendstate_all_seeds = []
    for seed in model_dict['seeds']:
        dendstate_one_seed = data_dict[seed]['binned_mean_forward_dendritic_state']['all'][:]
        dendstate_all_seeds.append(dendstate_one_seed)
    avg_dendstate = np.mean(dendstate_all_seeds, axis=0)
    error = np.std(dendstate_all_seeds, axis=0)
    binned_mean_forward_dendritic_state_steps = data_dict[seed]['binned_mean_forward_dendritic_state_steps'][:]
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
            angle = data_dict[seed]['angle_vs_bp_stochastic']['all_params'][:]
            if np.isnan(angle).any(): # check if there are any NaNs in the array
                print(f"Warning: NaN values found in angle array for seed {seed}")
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
    ax.set_ylabel('Alignment angle\n(Δw     vs backprop)')
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
    ax.set_ylabel('Alignment angle \n(W     vs B)')
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


def plot_dynamics_all_seeds(data_dict, model_dict, ax):
    dynamics_all_seeds = {}
    for seed in model_dict['seeds']:
        pop_dynamics_dict = data_dict[seed]['activity_dynamics']
        for pop_name, pop_dynamics in pop_dynamics_dict.items():
            if pop_name not in dynamics_all_seeds:
                dynamics_all_seeds[pop_name] = []
            avg_dynamics = np.mean(pop_dynamics, axis=(1,2))
            avg_dynamics = avg_dynamics/avg_dynamics[-1]
            dynamics_all_seeds[pop_name].append(avg_dynamics)
    dynamics_all_seeds = {pop_name:np.array(dynamics) for pop_name, dynamics in dynamics_all_seeds.items()}

    axes = gs.GridSpecFromSubplotSpec(1, 3, subplot_spec=ax, wspace=0.2, hspace=0.2)
    fig = axes.figure
    ax_E = fig.add_subplot(axes[0])
    ax_I = fig.add_subplot(axes[1])
    all_axes = [ax_E, ax_I]

    E_populations = {pop_name:pop_dynamics for pop_name,pop_dynamics in dynamics_all_seeds.items() if 'E' in pop_name and 'Input' not in pop_name}
    # cmap = plt.get_cmap('Reds')
    cmap = plt.get_cmap('autumn_r')
    for i, (pop_name, pop_dynamics) in enumerate(E_populations.items()):
        avg_dynamics = np.mean(pop_dynamics, axis=(0))
        error = np.std(pop_dynamics, axis=0)
        # color = cmap(0.2 + 0.7*i/len(E_populations))
        color = cmap(0.2 + i/3)
        ax_E.plot(np.arange(1,16), avg_dynamics, label=pop_name, color=color)
        ax_E.fill_between(np.arange(1,16), avg_dynamics-error, avg_dynamics+error, alpha=0.2, linewidth=0,  color=color)
    legend = ax_E.legend(ncol=3, loc='upper left', bbox_to_anchor=(-0., 1.3), frameon=False, handlelength=0.8, handletextpad=0.5, columnspacing=1)
    for line in legend.get_lines():
        line.set_linewidth(2)

    I_populations = {pop_name:pop_dynamics for pop_name,pop_dynamics in dynamics_all_seeds.items() if 'SomaI' in pop_name}
    # cmap = plt.get_cmap('Blues')
    cmap = plt.get_cmap('winter_r')
    for i, (pop_name, pop_dynamics) in enumerate(I_populations.items()):
        avg_dynamics = np.mean(pop_dynamics, axis=(0))
        error = np.std(pop_dynamics, axis=0)
        # color = cmap(0.2 + 0.7*i/len(I_populations))
        color = cmap(0.2 + i/3)
        ax_I.plot(np.arange(1,16), avg_dynamics, label=pop_name, color=color)
        ax_I.fill_between(np.arange(1,16), avg_dynamics-error, avg_dynamics+error, alpha=0.2, linewidth=0,  color=color)
    legend = ax_I.legend(ncol=3, loc='upper left', bbox_to_anchor=(-0.1, 1.3), frameon=False, handlelength=0.8, handletextpad=0.5, columnspacing=1)
    for line in legend.get_lines():
        line.set_linewidth(2)

    DendI_populations = {pop_name:pop_dynamics for pop_name,pop_dynamics in dynamics_all_seeds.items() if 'DendI' in pop_name}
    if len(DendI_populations) > 0:
        ax_dendI = fig.add_subplot(axes[2])
        all_axes.append(ax_dendI)

        # cmap = plt.get_cmap('Blues')
        cmap = plt.get_cmap('winter_r')
        for i, (pop_name, pop_dynamics) in enumerate(DendI_populations.items()):
            avg_dynamics = np.mean(pop_dynamics, axis=(0))
            error = np.std(pop_dynamics, axis=0)
            color = cmap(0.2 + i/3)
            ax_dendI.plot(np.arange(1,16), avg_dynamics, label=pop_name, color=color)
            ax_dendI.fill_between(np.arange(1,16), avg_dynamics-error, avg_dynamics+error, alpha=0.2, linewidth=0,  color=color)
        legend = ax_dendI.legend(ncol=3, loc='upper left', bbox_to_anchor=(-0., 1.3), frameon=False, handlelength=0.8, handletextpad=0.5, columnspacing=1)
        for line in legend.get_lines():
            line.set_linewidth(2)
        
    for ax in all_axes:
        ax.set_xticks(np.arange(0, 16, 5))
        ax.set_yticks(np.arange(0, 3.1, 1))
        ax.set_ylim([-0.1, 3])
        ax.set_xlim([1, 15])
        ax.set_xticks([0, 5, 10, 15])
        ax.set_xlabel('Forward timestep')
        # ax.set_ylabel('Activity (norm.)')
    ax_E.set_ylabel('Activity (norm.)')
    ax_E.set_title(model_dict['label'], rotation=90, x=-0.25, y=0.4, va='center')


########################################################################################################

def plot_dynamics_example(model_dict_all, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/mnist/", save=None, recompute=False):
    model_key = "bpLike_WT_hebbdend_eq"
    model_dict = model_dict_all[model_key]
    network_name = model_dict['config'].split('.')[0] + "_dynamics"
    hdf5_path = f"data/plot_data_{network_name}.h5"

    # Open hdf5 and check if the dynamics data already exists      
    if not os.path.exists(hdf5_path):
        recompute = True

    if recompute in ['all', 'dendritic_dynamics_dict']:
        print(f"Computing dynamics for {network_name}...")
        saved_network_path = saved_network_path_prefix + "20240516_EIANN_2_hidden_mnist_BP_like_config_2L_66049_257_complete_dynamics.pkl"        
        if not os.path.exists(saved_network_path):
            config_path = config_path_prefix + "20240516_EIANN_2_hidden_mnist_BP_like_config_2L_complete_optimized_dynamics.yaml"
            network = ut.build_EIANN_from_config(config_path, network_seed=66049)
            train_dataloader, train_sub_dataloader, val_dataloader, test_dataloader, data_generator = ut.get_MNIST_dataloaders(sub_dataloader_size=20_000)
            data_generator.manual_seed(257)
            network.train(train_sub_dataloader, 
                            epochs=1,
                            samples_per_epoch=20_000,
                            store_history=True, 
                            store_dynamics=True,
                            store_params=True,
                            store_params_interval=(0,-1,100),
                            status_bar=True)
            ut.save_network(network, saved_network_path)
        else:
            network = ut.load_network(saved_network_path)
        dendritic_dynamics_dict = ut.compute_dendritic_state_dynamics(network)
        ut.save_plot_data(network.name, 'retrained_with_dynamics', data_key='dendritic_dynamics_dict', data=dendritic_dynamics_dict, file_path=hdf5_path, overwrite=True)
        ut.save_plot_data(network.name, 'retrained_with_dynamics', data_key='param_history_steps', data=network.param_history_steps, file_path=hdf5_path, overwrite=True)

    print("Generating figure...")

    fig = plt.figure(figsize=(5.5, 9))
    gs_axes = gs.GridSpec(nrows=2, ncols=1, figure=fig,                        
                       left=0.68,right=0.97,
                       top=0.86, bottom = 0.65,
                       wspace=0.3, hspace=0.5)
    axes = [fig.add_subplot(gs_axes[i,0]) for i in range(2)]
    with h5py.File(hdf5_path, 'r') as f:
        data_dict = f[network_name]['retrained_with_dynamics']
        dendritic_dynamics_dict  = data_dict['dendritic_dynamics_dict']
        param_history_steps = data_dict['param_history_steps'][:]

        pt.plot_network_dynamics_example(param_history_steps, dendritic_dynamics_dict, population="H2E", units=[0,7], t=5000, axes=axes)

    if save is not None:
        fig.savefig(f"figures/{save}.png", dpi=300)
        fig.savefig(f"figures/{save}.svg", dpi=300)


def plot_average_dynamics(model_dict_all, model_list, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/mnist/", save=None, recompute=False):
    fig = plt.figure(figsize=(5.5, 9))
    axes = gs.GridSpec(nrows=len(model_list), ncols=1, figure=fig,                       
                       left=0.1,right=0.95,
                       top=0.95, bottom=0.52,
                       wspace=1, hspace=0.8)
    
    all_models = model_list
    generate_hdf5_all_seeds(all_models, model_dict_all, config_path_prefix, saved_network_path_prefix, recompute=recompute)

    for i, model_key in enumerate(all_models):
        model_dict = model_dict_all[model_key]
        network_name = model_dict['config'].split('.')[0]
        hdf5_path = f"data/plot_data_{network_name}.h5"
        with h5py.File(hdf5_path, 'r') as f:
            data_dict = f[network_name]
            print(f"Generating plots for {model_dict['label']}")    
            plot_dynamics_all_seeds(data_dict, model_dict, ax=axes[i])

    if save is not None:
        fig.savefig(f"figures/{save}.png", dpi=300)
        fig.savefig(f"figures/{save}.svg", dpi=300)
            

def generate_fig2(model_dict_all, model_list_heatmaps, model_list_metrics, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/mnist/", save=None, recompute=None):
    fig = plt.figure(figsize=(5.5, 9))
    axes = gs.GridSpec(nrows=3, ncols=3, figure=fig,                    
                       left=0.049,right=0.95,
                       top=0.95, bottom=0.52,
                       wspace=0.2, hspace=0.4)
    metrics_axes = gs.GridSpec(nrows=3, ncols=3, figure=fig,                        
                       left=0.15,right=0.8,
                       top=0.95, bottom=0.6,
                       wspace=0.7, hspace=0.2,
                       width_ratios=[2.5,1,1])
    
    # fig = plt.figure(figsize=(5.5, 4))
    # axes = gs.GridSpec(nrows=3, ncols=3, figure=fig,                   
    #                    left=0.049,right=0.95,
    #                    top=0.95, bottom=0.08,
    #                    wspace=0.2, hspace=0.35)
    # metrics_axes = gs.GridSpec(nrows=3, ncols=4, figure=fig,                     
    #                    left=0.049,right=0.95,
    #                    top=0.95, bottom=0.25,
    #                    wspace=0.5, hspace=0.3,
    #                    width_ratios=[1.5, 1, 1, 1])

    ax_accuracy    = fig.add_subplot(metrics_axes[2, 0])  
    ax_selectivity = fig.add_subplot(metrics_axes[2, 1])
    ax_structure   = fig.add_subplot(metrics_axes[2, 2])

    all_models = list(dict.fromkeys(model_list_heatmaps + model_list_metrics)) # remove duplicates
    generate_hdf5_all_seeds(all_models, model_dict_all, config_path_prefix, saved_network_path_prefix, recompute=recompute)

    for i,model_key in enumerate(all_models):
        model_dict = model_dict_all[model_key]
        network_name = model_dict['config'].split('.')[0]
        hdf5_path = f"data/plot_data_{network_name}.h5"
        with h5py.File(hdf5_path, 'r') as f:
            data_dict = f[network_name]
            print(f"Generating plots for {model_dict['label']}")

            if model_key in model_list_heatmaps:
                seed = model_dict['seeds'][0] # example seed to plot
                population = 'H2E'

                # Activity plots: batch accuracy of each population to the test dataset
                ax = fig.add_subplot(axes[0, i])
                average_pop_activity_dict = data_dict[seed]['average_pop_activity_dict']
                num_units = average_pop_activity_dict[population].shape[1]

                pt.plot_batch_accuracy_from_data(average_pop_activity_dict, population=population, sort=True, ax=ax, cbar=True)
                ax.set_yticks([0,num_units-1])
                ax.set_yticklabels([1,num_units])
                ax.set_ylabel(f'{population} unit', labelpad=-8)
                ax.set_title(model_dict["name"])
                if i>0:
                    ax.set_ylabel('')
                    ax.set_yticklabels([])

                # Receptive field plots
                receptive_fields = torch.tensor(np.array(data_dict[seed][f"maxact_receptive_fields_{population}"]))
                num_units = 10
                temp_ax = fig.add_subplot(axes[1, i])
                pos = temp_ax.get_position()
                temp_ax.remove()
                rf_axes = fig.add_gridspec(5,5,left=pos.x0, right=pos.x1-0.04, bottom=pos.y0-0.02, top=pos.y1+0.01, wspace=0.1, hspace=0.1)
                ax_list = []
                for j in range(num_units):
                    _ax = fig.add_subplot(rf_axes[j])
                    ax_list.append(_ax)
                preferred_classes = np.argmax(average_pop_activity_dict[population], axis=0)
                im = pt.plot_receptive_fields(receptive_fields, sort=True, ax_list=ax_list, preferred_classes=preferred_classes)
                fig_width, fig_height = fig.get_size_inches()
                cax = fig.add_axes([ax_list[0].get_position().x0, _ax.get_position().y0-0.08/fig_height, 0.05, 0.03/fig_height])
                fig.colorbar(im, cax=cax, orientation='horizontal')

            if model_key in model_list_metrics:
                plot_accuracy_all_seeds(data_dict, model_dict, ax=ax_accuracy, legend=True)
                plot_metric_all_seeds(data_dict, model_dict, populations_to_plot=['H1E','H2E'], ax=ax_selectivity, metric_name='selectivity', plot_type='violin')
                plot_metric_all_seeds(data_dict, model_dict, populations_to_plot=['H1E','H2E'], ax=ax_structure, metric_name='structure', plot_type='violin')
                # plot_metric_all_seeds(data_dict, model_dict, populations_to_plot=['H1E'], ax=ax_selectivity, metric_name='selectivity', plot_type='violin', side='low')
                # plot_metric_all_seeds(data_dict, model_dict, populations_to_plot=['H1E'], ax=ax_structure, metric_name='structure', plot_type='violin', side='low')
                # plot_metric_all_seeds(data_dict, model_dict, populations_to_plot=['H2E'], ax=ax_selectivity, metric_name='selectivity', plot_type='violin', side='high')
                # plot_metric_all_seeds(data_dict, model_dict, populations_to_plot=['H2E'], ax=ax_structure, metric_name='structure', plot_type='violin', side='high')

    if save is not None:
        fig.savefig(f"figures/{save}.png", dpi=300)
        fig.savefig(f"figures/{save}.svg", dpi=300)


def fig4(model_dict_all, model_list_heatmaps, model_list_metrics, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/mnist/", save=None, recompute=None):
    fig = plt.figure(figsize=(5.5, 9))
    axes = gs.GridSpec(nrows=2, ncols=2, figure=fig,                    
                       left=0.04,right=0.6,
                       top=0.88, bottom=0.59,
                       wspace=0.1, hspace=0.3,
                       height_ratios=[1,1.2])
    metrics_axes = gs.GridSpec(nrows=2, ncols=2, figure=fig,                        
                       left=0.7,right=0.98,
                       top=0.99, bottom=0.72,
                       wspace=0.8, hspace=0.4)
    ax_accuracy    = fig.add_subplot(metrics_axes[0, :])  
    ax_selectivity = fig.add_subplot(metrics_axes[1, 0])
    ax_structure   = fig.add_subplot(metrics_axes[1, 1])

    all_models = list(dict.fromkeys(model_list_heatmaps + model_list_metrics)) # remove duplicates
    generate_hdf5_all_seeds(all_models, model_dict_all, config_path_prefix, saved_network_path_prefix, recompute=recompute)

    for i,model_key in enumerate(all_models):
        model_dict = model_dict_all[model_key]
        network_name = model_dict['config'].split('.')[0]
        hdf5_path = f"data/plot_data_{network_name}.h5"
        with h5py.File(hdf5_path, 'r') as f:
            data_dict = f[network_name]
            print(f"Generating plots for {model_dict['label']}")

            if model_key in model_list_heatmaps:
                seed = model_dict['seeds'][0] # example seed to plot
                population = 'H2E'

                # Activity plots: batch accuracy of each population to the test dataset
                ax = fig.add_subplot(axes[0, i])
                average_pop_activity_dict = data_dict[seed]['average_pop_activity_dict']
                num_units = average_pop_activity_dict[population].shape[1]

                pt.plot_batch_accuracy_from_data(average_pop_activity_dict, population=population, sort=True, ax=ax, cbar=True)
                ax.set_yticks([0,num_units-1])
                ax.set_yticklabels([1,num_units])
                ax.set_ylabel(f'{population} unit', labelpad=-8)
                ax.set_title(model_dict["name"])
                if i>0:
                    ax.set_ylabel('')
                    ax.set_yticklabels([])

                # Receptive field plots
                temp_ax = fig.add_subplot(axes[1, i])
                pos = temp_ax.get_position()
                temp_ax.remove()
                rf_axes = fig.add_gridspec(5,5,left=pos.x0, right=pos.x1-0.04, bottom=pos.y0, top=pos.y1, wspace=0.1, hspace=0.1)
                ax_list = []
                num_units = 10
                for j in range(num_units):
                    ax = fig.add_subplot(rf_axes[j])
                    ax_list.append(ax)
                preferred_classes = np.argmax(average_pop_activity_dict[population], axis=0)
                receptive_fields = torch.tensor(np.array(data_dict[seed][f"maxact_receptive_fields_{population}"]))
                im = pt.plot_receptive_fields(receptive_fields, sort=True, ax_list=ax_list, preferred_classes=preferred_classes)
                fig_width, fig_height = fig.get_size_inches()
                cax = fig.add_axes([ax_list[0].get_position().x0, ax.get_position().y0-0.08/fig_height, 0.05, 0.03/fig_height])
                fig.colorbar(im, cax=cax, orientation='horizontal')

            if model_key in model_list_metrics:
                plot_accuracy_all_seeds(data_dict, model_dict, ax=ax_accuracy, legend=True)
                plot_metric_all_seeds(data_dict, model_dict, populations_to_plot=['H1E','H2E'], ax=ax_selectivity, metric_name='selectivity', plot_type='violin')
                plot_metric_all_seeds(data_dict, model_dict, populations_to_plot=['H1E','H2E'], ax=ax_structure, metric_name='structure', plot_type='violin')
                # plot_metric_all_seeds(data_dict, model_dict, populations_to_plot=['H1E'], ax=ax_selectivity, metric_name='selectivity', plot_type='violin',side='low')
                # plot_metric_all_seeds(data_dict, model_dict, populations_to_plot=['H1E'], ax=ax_structure, metric_name='structure', plot_type='violin', side='low')
                # plot_metric_all_seeds(data_dict, model_dict, populations_to_plot=['H2E'], ax=ax_selectivity, metric_name='selectivity', plot_type='violin', side='high')
                # plot_metric_all_seeds(data_dict, model_dict, populations_to_plot=['H2E'], ax=ax_structure, metric_name='structure', plot_type='violin', side='high')

    if save is not None:
        fig.savefig(f"figures/{save}.png", dpi=300)
        fig.savefig(f"figures/{save}.svg", dpi=300)


def fig4_spirals(model_dict_all, model_list_spirals, model_list_metrics, spiral_type='scatter', config_path_prefix="network_config/spiral/", saved_network_path_prefix="data/spiral/", save=None, recompute=None):
    fig = plt.figure(figsize=(5.5, 9))
    axes = gs.GridSpec(nrows=1, ncols=4, figure=fig,                    
                       left=0.1,right=0.9,
                       top=0.9, bottom=0.8,
                       wspace=0.3, hspace=0.5,
                       width_ratios=[1,1,0.1,1.3])
    ax_accuracy =   fig.add_subplot(axes[0, 3])  

    all_models = list(dict.fromkeys(model_list_spirals + model_list_metrics))
    generate_hdf5_all_seeds(all_models, model_dict_all, config_path_prefix, saved_network_path_prefix, recompute=recompute)

    for model_idx, model_key in enumerate(all_models):
        model_dict = model_dict_all[model_key]
        network_name = model_dict['config'].split('.')[0]
        hdf5_path = f"data/plot_data_{network_name}.h5"
        with h5py.File(hdf5_path, 'r') as f:
            data_dict = f[network_name]
            print(f"Generating plots for {model_dict['label']}")
            seed = model_dict['seeds'][0] # example seed to plot

            if model_key in model_list_spirals:
                # Plot spirals
                ax = fig.add_subplot(axes[0, model_idx])
                decision_data = data_dict[seed]['spiral_decision_data_dict']
                pt.plot_spiral_decisions(decision_data, graph=spiral_type, ax=ax)
                ax.set_aspect('equal')
                ax.set_title(model_dict["label"], pad=4)
                ax.set_xlabel('x1')
                ax.set_ylabel('x2')

            # Plot metrics
            if model_key in model_list_metrics:                
                plot_accuracy_all_seeds(data_dict, model_dict, ax=ax_accuracy)

    if save:
        fig.savefig(f"figures/{save}.png", dpi=300)
        fig.savefig(f"figures/{save}.svg", dpi=300)


def compare_E_properties_full(model_dict_all, model_list_heatmaps, model_list_metrics, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/mnist/", save=None, recompute=None):
    '''
    Figure 1: Van_BP vs bpDale(learnedI)
        -> bpDale is more structured/sparse (focus on H1E metrics)

    Compare vanilla Backprop to networks with 'cortical' architecures (i.e. with somatic feedback inhibition). 
    '''

    fig = plt.figure(figsize=(5.5, 9))
    axes = gs.GridSpec(nrows=4, ncols=6, figure=fig,                        
                       left=0.049,right=1,
                       top=0.95, bottom = 0.5,
                       wspace=0.15, hspace=0.5)
    metrics_axes = gs.GridSpec(nrows=4, ncols=4, figure=fig,                 
                       left=0.049,right=0.95,
                       top=0.95, bottom = 0.48,
                       wspace=0.5, hspace=0.8)
    ax_accuracy    = fig.add_subplot(metrics_axes[3, 0])  
    ax_sparsity    = fig.add_subplot(metrics_axes[3, 1])
    ax_selectivity = fig.add_subplot(metrics_axes[3, 2])
    ax_structure   = fig.add_subplot(metrics_axes[3, 3])

    all_models = list(dict.fromkeys(model_list_heatmaps + model_list_metrics))
    generate_hdf5_all_seeds(all_models, model_dict_all, config_path_prefix, saved_network_path_prefix, recompute=recompute)

    col = 0
    for i,model_key in enumerate(all_models):
        model_dict = model_dict_all[model_key]
        network_name = model_dict['config'].split('.')[0]
        hdf5_path = f"data/plot_data_{network_name}.h5"
        with h5py.File(hdf5_path, 'r') as f:
            data_dict = f[network_name]
            print(f"Generating plots for {model_dict['label']}")

            if model_key in model_list_heatmaps:
                seed = model_dict['seeds'][0] # example seed to plot
                populations_to_plot = [population for population in data_dict[seed]['average_pop_activity_dict'] if 'E' in population and population!='InputE']
                for row,population in enumerate(populations_to_plot):
                    # Activity plots: batch accuracy of each population to the test dataset
                    ax = fig.add_subplot(axes[row, col*2])
                    average_pop_activity_dict = data_dict[seed]['average_pop_activity_dict']
                    pt.plot_batch_accuracy_from_data(average_pop_activity_dict, sort=True, population=population, ax=ax, cbar=False)
                    ax.set_yticks([0,average_pop_activity_dict[population].shape[0]-1])
                    ax.set_yticklabels([1,average_pop_activity_dict[population].shape[0]])
                    ax.set_ylabel(f'{population} unit', labelpad=-8)
                    if row==0:
                        ax.set_title(model_dict["label"])
                    if col>0:
                        ax.set_ylabel('')
                        ax.set_yticklabels([])

                    # Receptive field plots
                    receptive_fields = torch.tensor(np.array(data_dict[seed][f"maxact_receptive_fields_{population}"]))
                    num_units = 10
                    ax = fig.add_subplot(axes[row, col*2+1])
                    ax.axis('off')
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
                    cax = fig.add_axes([ax_list[0].get_position().x0-0.02/fig_width, ax.get_position().y0-0.25/fig_height, 0.04, 0.03/fig_height])
                    fig.colorbar(im, cax=cax, orientation='horizontal')
                col += 1

            if model_key in model_list_metrics:
                plot_accuracy_all_seeds(data_dict, model_dict, ax=ax_accuracy)
                plot_metric_all_seeds(data_dict, model_dict, populations_to_plot=['H1E','H2E'], ax=ax_selectivity, metric_name='selectivity', plot_type='violin')
                plot_metric_all_seeds(data_dict, model_dict, populations_to_plot=['H1E','H2E'], ax=ax_sparsity, metric_name='sparsity', plot_type='violin')
                plot_metric_all_seeds(data_dict, model_dict, populations_to_plot=['H1E','H2E'], ax=ax_structure, metric_name='structure', plot_type='violin')


    if save is not None:
        fig.savefig(f"figures/{save}.png", dpi=300)
        fig.savefig(f"figures/{save}.svg", dpi=300)


def compare_somaI_properties(model_dict_all, model_list_heatmaps, model_list_metrics, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/mnist/", save=None, recompute=None):
    fig = plt.figure(figsize=(5.5, 9))
    axes = gs.GridSpec(nrows=4, ncols=6, figure=fig,                       
                       left=0.049,right=0.98,
                       top=0.95, bottom = 0.5,
                       wspace=0.15, hspace=0.5)
    
    metrics_axes = gs.GridSpec(nrows=4, ncols=1, figure=fig,                     
                       left=0.6,right=0.78,
                       top=0.95, bottom = 0.5,
                       wspace=0., hspace=0.5)
    ax_accuracy    = fig.add_subplot(metrics_axes[0, 0])  
    ax_sparsity    = fig.add_subplot(metrics_axes[1, 0])
    ax_selectivity = fig.add_subplot(metrics_axes[2, 0])
    col = 0

    all_models = list(dict.fromkeys(model_list_heatmaps + model_list_metrics))
    generate_hdf5_all_seeds(all_models, model_dict_all, config_path_prefix, saved_network_path_prefix, recompute=recompute)

    for model_key in all_models:
        model_dict = model_dict_all[model_key]
        config_path = config_path_prefix + model_dict['config']
        pickle_basename = "_".join(model_dict['config'].split('_')[0:-2])
        network_name = model_dict['config'].split('.')[0]
        hdf5_path = f"data/plot_data_{network_name}.h5"

        with h5py.File(hdf5_path, 'r') as f:
            data_dict = f[network_name]
            
            print(f"Generating plots for {model_dict['label']}")
            seed = model_dict['seeds'][0] # example seed to plot
            populations_to_plot = [population for population in data_dict[seed]['average_pop_activity_dict'] if 'SomaI' in population]

            if model_key in model_list_heatmaps:
                for row,population in enumerate(populations_to_plot):
                    ## Activity plots: batch accuracy of each population to the test dataset
                    ax = fig.add_subplot(axes[row, col])
                    average_pop_activity_dict = data_dict[seed]['average_pop_activity_dict']
                    pt.plot_batch_accuracy_from_data(average_pop_activity_dict, sort=True, population=population, ax=ax, cbar=False)
                    ax.set_yticks([0,average_pop_activity_dict[population].shape[0]-1])
                    ax.set_yticklabels([1,average_pop_activity_dict[population].shape[0]])
                    ax.set_ylabel(f'{population} unit', labelpad=-8)
                    if row==0:
                        ax.set_title(model_dict["label"])
                    if col>0:
                        ax.set_ylabel('')
                        ax.set_yticklabels([])
                col += 1

            if model_key in model_list_metrics:
                plot_accuracy_all_seeds(data_dict, model_dict, ax=ax_accuracy)
                legend = ax_accuracy.legend(ncol=1, bbox_to_anchor=(1., 1.), loc='upper left', fontsize=6)
                for line in legend.get_lines():
                    line.set_linewidth(1.5)
                plot_metric_all_seeds(data_dict, model_dict, populations_to_plot=populations_to_plot, ax=ax_selectivity, metric_name='selectivity', plot_type='violin')
                plot_metric_all_seeds(data_dict, model_dict, populations_to_plot=populations_to_plot, ax=ax_sparsity, metric_name='sparsity', plot_type='violin')

    if save:
        fig.savefig(f"figures/{save}.png", dpi=300)
        fig.savefig(f"figures/{save}.svg", dpi=300)


def generate_fig3(model_dict_all, model_list_heatmaps, model_list_metrics, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/mnist/", save=None, recompute=None):
    fig = plt.figure(figsize=(5.5, 9))
    axes = gs.GridSpec(nrows=3, ncols=4, figure=fig,            
                       left=0.049,right=0.95,
                       top=0.95, bottom=0.5,
                       wspace=0.2, hspace=0.5,
                       width_ratios=[1, 1, 1, 0.1])
    
    metrics_axes = gs.GridSpec(nrows=3, ncols=4, figure=fig,
                       left=0.049,right=0.98,
                       top=0.95, bottom=0.55,
                       wspace=0.4, hspace=0.6,
                       width_ratios=[1, 1, 1, 0.4])

    ax_accuracy    = fig.add_subplot(metrics_axes[1, 0])
    ax_dendstate   = fig.add_subplot(metrics_axes[1, 1])
    ax_angle       = fig.add_subplot(metrics_axes[1, 2])
    ax_selectivity = fig.add_subplot(metrics_axes[1, 3])
    # ax_sparsity    = fig.add_subplot(metrics_axes[0, 3])

    all_models = list(dict.fromkeys(model_list_heatmaps + model_list_metrics))
    generate_hdf5_all_seeds(all_models, model_dict_all, config_path_prefix, saved_network_path_prefix, recompute=recompute)
    
    if "bpLike_WT_hebbdend" in model_dict_all: # Update model label for this figure
        model_dict_all["bpLike_WT_hebbdend"]["label"] = "Learned (Hebb)"
        
    for col, model_key in enumerate(all_models):
        model_dict = model_dict_all[model_key]
        network_name = model_dict['config'].split('.')[0]
        hdf5_path = f"data/plot_data_{network_name}.h5"

        with h5py.File(hdf5_path, 'r') as f:
            data_dict = f[network_name]
            print(f"Generating plots for {model_dict['label']}")
            populations_to_plot = ['H2DendI']

            # Plot heatmaps
            if model_key in model_list_heatmaps:
                for row,population in enumerate(populations_to_plot):
                    # Activity plots: batch accuracy of each population to the test dataset
                    ax = fig.add_subplot(axes[row, col])
                    seed = model_dict['seeds'][0] # example seed to plot
                    average_pop_activity_dict = data_dict[seed]['average_pop_activity_dict']
                    num_units = average_pop_activity_dict[population].shape[1]
                    pt.plot_batch_accuracy_from_data(average_pop_activity_dict, population=population, sort=True, ax=ax, cbar=True)

                    ax.set_yticks([0,num_units-1])
                    ax.set_yticklabels([1,num_units])
                    ax.set_ylabel(f'{population} unit', labelpad=-8)
                    if row==0:
                        ax.set_title(model_dict["name"], pad=3)
                    if col>0:
                        ax.set_ylabel('')
                        ax.set_yticklabels([])

            # Plot metrics
            if model_key in model_list_metrics:  
                plot_accuracy_all_seeds(data_dict, model_dict, ax=ax_accuracy)
                plot_dendritic_state_all_seeds(data_dict, model_dict, ax=ax_dendstate)
                plot_angle_vs_bp_all_seeds(data_dict, model_dict, ax=ax_angle)

                populations_to_plot = [population for population in data_dict[seed]['average_pop_activity_dict'] if 'DendI' in population]
                plot_metric_all_seeds(data_dict, model_dict, populations_to_plot=populations_to_plot, ax=ax_selectivity, metric_name='selectivity', plot_type='violin')
                # plot_metric_all_seeds(data_dict, model_dict, populations_to_plot=populations_to_plot, ax=ax_sparsity, metric_name='sparsity', plot_type='violin')

            legend = ax_accuracy.legend(ncol=1, bbox_to_anchor=(0.2, 0.6), loc='upper left', fontsize=6)
            for line in legend.get_lines():
                line.set_linewidth(1.5)

    if save:
        # fig.savefig(f"figures/{save}.png", dpi=300)
        fig.savefig(f"figures/{save}.svg", dpi=300)


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
        hdf5_path = f"data/plot_data_{network_name}.h5"
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
        hdf5_path = f"data/plot_data_{network_name}.h5"
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


def generate_fig5(model_dict_all, model_list, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/mnist/", save=None, recompute=None):
    fig = plt.figure(figsize=(5.5, 9))
    axes = gs.GridSpec(nrows=2, ncols=3, figure=fig,                       
                       left=0.1,right=0.9,
                       top=0.9, bottom = 0.6,
                       wspace=0.4, hspace=0.6,
                       height_ratios=[1, 0.7])
    ax_accuracy = fig.add_subplot(axes[1,0])
    ax_dendstate = fig.add_subplot(axes[1,1])
    ax_angle_vs_BP = fig.add_subplot(axes[1,2])

    axes = gs.GridSpecFromSubplotSpec(nrows=1, ncols=4, subplot_spec=axes[0,0:3], wspace=0.2)
    diagram_axes = [fig.add_subplot(axes[0,i]) for i in range(4)]
    for ax in diagram_axes: # decrease the height of the diagram axes
        ax.set_position([ax.get_position().x0, ax.get_position().y0, ax.get_position().width, ax.get_position().height*0.7])

    pt.plot_learning_rule_diagram(axes_list=diagram_axes)

    all_models = list(dict.fromkeys(model_list))
    generate_hdf5_all_seeds(all_models, model_dict_all, config_path_prefix, saved_network_path_prefix, recompute=recompute)

    model_dict_all["BTSP_WT_hebbdend"]["label"] = "BTSP"
    model_dict_all["bpLike_WT_hebbdend"]["label"] = "LDS"

    for model_key in all_models:
        model_dict = model_dict_all[model_key]
        config_path = config_path_prefix + model_dict['config']
        pickle_basename = "_".join(model_dict['config'].split('_')[0:-2])
        network_name = model_dict['config'].split('.')[0]
        hdf5_path = f"data/plot_data_{network_name}.h5"
        with h5py.File(hdf5_path, 'r') as f:
            data_dict = f[network_name]
            print(f"Generating plots for {model_dict['label']}")
            plot_accuracy_all_seeds(data_dict, model_dict, ax=ax_accuracy)
            plot_dendritic_state_all_seeds(data_dict, model_dict, ax=ax_dendstate)
            plot_angle_vs_bp_all_seeds(data_dict, model_dict, ax=ax_angle_vs_BP)

    legend = ax_accuracy.legend(ncol=4, bbox_to_anchor=(-0.1, 1.25), loc='upper left')
    for line in legend.get_lines():
        line.set_linewidth(1.5)

    if save is not None:
        fig.savefig(f"figures/{save}.png", dpi=300)
        fig.savefig(f"figures/{save}.svg", dpi=300)


def generate_fig6(model_dict_all, model_list1, model_list2, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/mnist/", save=None, recompute=None):
    fig = plt.figure(figsize=(5.5, 2.4))
    axes = gs.GridSpec(nrows=2, ncols=3, figure=fig,                    
                       left=0.26,right=0.95,
                       top=0.92, bottom = 0.15,
                       wspace=0.6, hspace=0.8)
    ax_accuracy1 = fig.add_subplot(axes[0,0])
    ax_angle_vs_BP1 = fig.add_subplot(axes[0,1])
    ax_FB_angle1 = fig.add_subplot(axes[0,2])
    ax_accuracy2 = fig.add_subplot(axes[1,0])
    ax_angle_vs_BP2 = fig.add_subplot(axes[1,1])
    ax_FB_angle2 = fig.add_subplot(axes[1,2])

    all_models = list(dict.fromkeys(model_list1 + model_list2))
    generate_hdf5_all_seeds(all_models, model_dict_all, config_path_prefix, saved_network_path_prefix, recompute=recompute)
    model_dict_all["bpLike_WT_hebbdend"]["label"] = "LDS"

    for i, model_key in enumerate(all_models):
        model_dict = model_dict_all[model_key]
        network_name = model_dict['config'].split('.')[0]
        hdf5_path = f"data/plot_data_{network_name}.h5"
        with h5py.File(hdf5_path, 'r') as f:
            data_dict = f[network_name]
            print(f"Generating plots for {model_dict['label']}")
            if model_key in model_list1:
                ax_accuracy = ax_accuracy1
                ax_angle_vs_BP = ax_angle_vs_BP1
                ax_FB_angle = ax_FB_angle1
            if model_key in model_list2:
                ax_accuracy = ax_accuracy2
                ax_angle_vs_BP = ax_angle_vs_BP2
                ax_FB_angle = ax_FB_angle2
            plot_accuracy_all_seeds(data_dict, model_dict, ax=ax_accuracy)
            plot_angle_vs_bp_all_seeds(data_dict, model_dict, ax=ax_angle_vs_BP, error='std')
            plot_angle_FB_all_seeds(data_dict, model_dict, ax=ax_FB_angle, error='std')

    legend = ax_accuracy1.legend(ncol=3, bbox_to_anchor=(-0.1, 1.3), loc='upper left')
    for line in legend.get_lines():
        line.set_linewidth(1.5)
    legend = ax_accuracy2.legend(ncol=3, bbox_to_anchor=(-0.1, 1.3), loc='upper left')
    for line in legend.get_lines():
        line.set_linewidth(1.5)

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
        hdf5_path = f"data/plot_data_{network_name}.h5"
        for seed in model_dict['seeds']:
            saved_network_path = saved_network_path_prefix + network_name + f"_{seed}.pkl"
            generate_data_hdf5(config_path, saved_network_path, hdf5_path, recompute=recompute)
            gc.collect()

    for i,model_key in enumerate(all_models):
        model_dict = model_dict_all[model_key]
        network_name = model_dict['config'].split('.')[0]
        hdf5_path = f"data/plot_data_{network_name}.h5"

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


def generate_summary_table(model_dict_all, model_list, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/mnist/", save=None, recompute=None):
    mm = 1/25.4 #convert mm to inches
    fig, ax = plt.subplots(figsize=(180*mm,170*mm))
    # fig, ax = plt.subplots(figsize=(5.5, 9))
    ax.axis('off')

    all_models = list(dict.fromkeys(model_list))
    generate_hdf5_all_seeds(all_models, model_dict_all, config_path_prefix, saved_network_path_prefix, recompute=recompute)

    networks = {}
    columns = []

    for i,model_key in enumerate(all_models):
        model_dict = model_dict_all[model_key]
        network_name = model_dict['config'].split('.')[0]
        hdf5_path = f"data/plot_data_{network_name}.h5"

        with h5py.File(hdf5_path, 'r') as f:
            print(f"Generating table for {network_name}")
            data_dict = f[network_name]

            if 'mnist' in saved_network_path_prefix:
                # Get the accuracy for each seed
                accuracy_all_seeds_20k = []
                for seed in model_dict['seeds']:
                    accuracy_all_seeds_20k.append(data_dict[seed]['test_accuracy_history'][-1])
                avg_accuracy_20k = np.mean(accuracy_all_seeds_20k)
                std_accuracy_20k = np.std(accuracy_all_seeds_20k)
                sem_accuracy_20k = std_accuracy_20k / np.sqrt(len(accuracy_all_seeds_20k))

                accuracy_all_seeds_50k = []
                for seed in model_dict['seeds']:
                    accuracy_all_seeds_50k.append(data_dict[seed]['test_accuracy_history_extended'][-1])
                avg_accuracy_50k = np.mean(accuracy_all_seeds_50k)
                std_accuracy_50k = np.std(accuracy_all_seeds_50k)
                sem_accuracy_50k = std_accuracy_50k / np.sqrt(len(accuracy_all_seeds_50k))

                networks[model_dict['label']] = {'MNIST Accuracy (20k samples)': f"{avg_accuracy_20k:.2f} \u00b1 {sem_accuracy_20k:.2f}",
                                                'MNIST Accuracy (50k samples)': f"{avg_accuracy_50k:.2f} \u00b1 {sem_accuracy_50k:.2f}"}

            elif 'spiral' in saved_network_path_prefix:
                accuracy_all_seeds_1_epoch = []
                for seed in model_dict['seeds']:
                    accuracy_all_seeds_1_epoch.append(data_dict[seed]['test_accuracy_history'][-1])
                avg_accuracy_1_epoch = np.mean(accuracy_all_seeds_1_epoch)
                # print(avg_accuracy_1_epoch)
                std_accuracy_1_epoch = np.std(accuracy_all_seeds_1_epoch)
                sem_accuracy_1_epoch = std_accuracy_1_epoch / np.sqrt(len(accuracy_all_seeds_1_epoch))

                accuracy_all_seeds_10_epochs = []
                for seed in model_dict['seeds']:
                    accuracy_all_seeds_10_epochs.append(data_dict[seed]['test_accuracy_history_extended'][-1])
                avg_accuracy_10_epochs = np.mean(accuracy_all_seeds_10_epochs)
                # print(avg_accuracy_10_epochs)
                std_accuracy_10_epochs = np.std(accuracy_all_seeds_10_epochs)
                sem_accuracy_10_epochs = std_accuracy_10_epochs / np.sqrt(len(accuracy_all_seeds_10_epochs))

                networks[model_dict['label']] = {'Spiral Accuracy (1 epoch)': f"{avg_accuracy_1_epoch:.2f} \u00b1 {sem_accuracy_1_epoch:.2f}",
                                                'Spiral Accuracy (10 epochs)': f"{avg_accuracy_10_epochs:.2f} \u00b1 {sem_accuracy_10_epochs:.2f}"}
                
        columns = list(model_dict.keys())
        columns.remove('config')
        columns.remove('color')
        columns.remove('seeds')

        for col in columns:
            networks[model_dict['label']].update({col: model_dict[col]})

    # Create a table from the networks dictionary
    table_vals = []
    column_labels = [header for header in list(networks.values())[0]]
    column_labels.insert(0, "")
    for network_name in networks:
        network_vals = [network_name]
        for col in networks[network_name]:
            network_vals.append(networks[network_name][col])
        table_vals.append(network_vals)

    table = ax.table(cellText=table_vals, colLabels=column_labels, cellLoc="center", loc="center")

    for key, cell in table.get_celld().items():
        cell.set_linewidth(0)
        if key[0] % 2 == 0:
            cell.set_facecolor([0.95 for i in range(3)])
        if key[0] == 0:
            cell.set_facecolor([0.9 for i in range(3)])
            cell.set_text_props(weight='bold')
        # cell.set_fontsize(20)
        # cell.set_height(cell.get_height() * 1.1)
        cell.set_text_props(fontname='Arial', fontsize=2)

    if save:
        fig.savefig(f"figures/{save}.png", dpi=300)
        fig.savefig(f"figures/{save}.svg", dpi=300)


def generate_spirals_figure(model_dict_all, model_list_heatmaps, model_list_metrics, spiral_type='scatter', config_path_prefix="network_config/spiral/", saved_network_path_prefix="data/spiral/", save=None, recompute=None):
    '''
    Plotting function for spiral dataset.
    Parameter ```spiral_type``` can be:
    - 'scatter': datapoints with incorrect preductions marked in red
    - 'decison': decision boundary map of the network's predictions
    '''
    cols = max(len(model_list_heatmaps), 3)

    fig = plt.figure(figsize=(cols * 3, cols * 2.5))
    axes = gs.GridSpec(nrows=3, ncols=cols, figure=fig,                    
                       left=0.03, right=0.97,
                       top=0.95, bottom=0.5,
                       wspace=0.5, hspace=0.7)

    spirals_row =   0
    heatmaps_row =  1 
    ax_accuracy =   fig.add_subplot(axes[2, 0])  
    ax_angle =      fig.add_subplot(axes[2, 1])
    ax_dendstate =  fig.add_subplot(axes[2, 2])

    all_models = list(dict.fromkeys(model_list_heatmaps + model_list_metrics))
    generate_hdf5_all_seeds(all_models, model_dict_all, config_path_prefix, saved_network_path_prefix, recompute=recompute)

    for model_idx, model_key in enumerate(all_models):
        model_dict = model_dict_all[model_key]
        network_name = model_dict['config'].split('.')[0]
        hdf5_path = f"data/plot_data_{network_name}.h5"
        with h5py.File(hdf5_path, 'r') as f:
            data_dict = f[network_name]
            print(f"Generating plots for {model_dict['label']}")
            seed = model_dict['seeds'][0] # example seed to plot

            # Plot heatmaps and spirals
            if model_key in model_list_heatmaps:
                ax = fig.add_subplot(axes[heatmaps_row, model_idx])
                if model_key != "vanBP_0_hidden_learned_bias_spiral":
                    population = 'H2E'
                    # populations_to_plot = [population for population in data_dict[seed]['average_pop_activity_dict']]
                    # Activity plots: batch accuracy of each population to the test dataset
                    average_pop_activity_dict = data_dict[seed]['average_pop_activity_dict']
                    pt.plot_batch_accuracy_from_data(average_pop_activity_dict, population=population, ax=ax, cbar=False)
                    ax.set_aspect('auto')
                    ax.set_yticks([0,average_pop_activity_dict[population].shape[0]-1])
                    ax.set_yticklabels([1,average_pop_activity_dict[population].shape[0]])
                    ax.set_ylabel(f'{population} unit', labelpad=-8)
                    ax.set_title(model_dict["label"], pad=3)

                # Plot spirals
                ax = fig.add_subplot(axes[spirals_row, model_idx])
                decision_data = data_dict[seed]['spiral_decision_data_dict']
                pt.plot_spiral_decisions(decision_data, graph=spiral_type, ax=ax)
                ax.set_aspect('equal')
                ax.set_title(model_dict["label"], pad=4)
                ax.set_xlabel('x1')
                ax.set_ylabel('x2')

            # Plot metrics
            if model_key in model_list_metrics:                
                plot_accuracy_all_seeds(data_dict, model_dict, ax=ax_accuracy)
                plot_dendritic_state_all_seeds(data_dict, model_dict, ax=ax_dendstate)
                plot_angle_vs_bp_all_seeds(data_dict, model_dict, ax=ax_angle)

    plt.tight_layout(rect=[0, 0.5, 1, 0.95])

    if save:
        fig.savefig(f"figures/{save}.png", dpi=300)
        fig.savefig(f"figures/{save}.svg", dpi=300)


from reportlab.pdfgen import canvas
import os
def images_to_pdf(image_paths, output_path, dpi=300):
    # Define US Letter page size in points
    letter_size = [8.5 * 72, 11 * 72]  # 612 x 792 points
    
    # Scale figure size based on dpi
    scale_factor = dpi / 300  # Reference is 300 dpi
    fig_size = [5.5 * 72 * scale_factor, 9 * 72 * scale_factor]
    
    fig_width = fig_size[0]
    fig_height = fig_size[1]
    
    margin_x = (letter_size[0] - fig_width) / 2
    margin_y = (letter_size[1] - fig_height) / 2
    
    # Create a canvas for the PDF
    c = canvas.Canvas(output_path, pagesize=letter_size)
    
    for img_path in image_paths:
        c.setPageSize(letter_size)
        
        # Draw the image on the page
        c.drawImage(img_path, margin_x, margin_y, width=fig_width, height=fig_height)
        
        # Add a caption with the image filename
        caption = os.path.basename(img_path)
        c.setFont("Helvetica", 12)
        caption_x = letter_size[0] * 0.35
        caption_y = letter_size[1] - margin_y * 0.7
        c.drawString(caption_x, caption_y, caption)
        
        c.showPage()  # Add a new page in the PDF for the next image
    
    c.save()




@click.command()
@click.option('--figure', default=None, help='Figure to generate')
@click.option('--recompute', default=None, help='Recompute plot data for a particular parameter')

def main(figure, recompute):
    model_dict_all = {
            ##########################
            # Backprop models
            ##########################
            "vanBP":       {"config": "20231129_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G_complete_optimized.yaml",
                            "name":   "Feedforward ANN",
                            "label":  "Backprop (ANN)",
                            "color":  "black",
                            "Architecture": "2-hidden",
                            "Algorithm":    "Backprop", # Error propagation scheme
                            "Learning rule":"Gradient descent"},

            "vanBP_0hidden": {"config": "20250103_EIANN_0_hidden_mnist_van_bp_relu_SGD_config_G_complete_optimized.yaml",
                              "label":  "Vanilla Backprop 0-hidden",
                              "color":  "black",
                              "Architecture": "",
                              "Algorithm": "",
                              "Learning Rule": ""},
            
            "vanBP_fixed_hidden": {"config":"20250108_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G_fixed_hidden_complete_optimized.yaml",
                                   "label": "Vanilla Backprop fixed hidden",
                                   "color": "black",
                                   "Architecture": "",
                                   "Algorithm": "",
                                   "Learning Rule": ""},

            "bpDale_learned":{"config": "20240419_EIANN_2_hidden_mnist_bpDale_relu_SGD_config_F_complete_optimized.yaml",
                              "name":   "bpDale (learned soma I)",
                              "label":  "Backprop (EIANN)",
                              "color":  "royalblue",
                              "Architecture": "",
                              "Algorithm": "",
                              "Learning Rule": ""},

            "bpDale_fixed":{"config": "20231129_EIANN_2_hidden_mnist_bpDale_relu_SGD_config_G_complete_optimized.yaml",
                            "name":   "bpDale (fixed soma I)",
                            "label":  "Backprop (EIANN)",
                            "color":  "cornflowerblue",
                            "Architecture": "",
                            "Algorithm": "",
                            "Learning Rule": ""},

            "bpDale_noI":  {"config": "20240919_EIANN_2_hidden_mnist_bpDale_noI_relu_SGD_config_G_complete_optimized.yaml",
                            "label": "Dale's Law\n(no soma I)",
                            "color": "royalblue",
                            "Architecture": "",
                            "Algorithm": "",
                            "Learning Rule": ""},

            ##########################
            # bpLike models
            ##########################
            "bpLike_WT_hebbdend":  {"config": "20241009_EIANN_2_hidden_mnist_BP_like_config_5J_complete_optimized.yaml",
                                    "name":   "bpLike_WT_hebbdend",
                                    "label":  "Dend. Target Prop.",
                                    "color":  "red",
                                    "Architecture": "",
                                    "Algorithm": "",
                                    "Learning Rule": ""},

            "bpLike_WT_hebbdend_eq":  {"config": "20240516_EIANN_2_hidden_mnist_BP_like_config_2L_complete_optimized.yaml",
                                    "color":  "red",
                                    "label":   "bpLike_WT_hebbdend_eq",
                                    "Architecture": "", 
                                    "Algorithm": "", 
                                    "Learning Rule": "",},

            # "bpLike_TC_hebbdend": {"config": "20241114_EIANN_2_hidden_mnist_BP_like_config_5J_learn_TD_HTC_2_complete_optimized.yaml",
            #                        "color": "green",
            #                        "label": "bpLike_TC_hebbdend"},  # TC applied to activity of wrong (bottom-up) unit instead of top-down unit

            "bpLike_WT_tempcont":  {"config": "20240508_EIANN_2_hidden_mnist_BP_like_config_1J_complete_optimized.yaml",
                                    "color": "red",
                                    "label": "BP-like (temp. cont.)",
                                    "Architecture": "",
                                    "Algorithm": "",
                                    "Learning Rule": ""},

            "bpLike_WT_localBP":   {"config": "20241113_EIANN_2_hidden_mnist_BP_like_config_5M_complete_optimized.yaml",
                                    "name":  "Backprop\n(Local loss func.)",
                                    "color": "black",
                                    "label": "Learned (Backprop)",
                                    "Architecture": "", 
                                    "Algorithm": "", 
                                    "Learning Rule": "",},

            "bpLike_WT_localBP_eq":{"config": "20240628_EIANN_2_hidden_mnist_BP_like_config_3M_complete_optimized.yaml",
                                    "color":  "black",
                                    "label":   "bpLike_WT_localBP_eq",
                                    "Architecture": "", 
                                    "Algorithm": "", 
                                    "Learning Rule": ""},

            "bpLike_WT_fixedDend": {"config": "20241113_EIANN_2_hidden_mnist_BP_like_config_5K_complete_optimized.yaml",
                                    "name": "bpLike_WT_fixedDend",
                                    "label":  "Fixed random",
                                    "color":  "gray",
                                    "Architecture": "", 
                                    "Algorithm": "", 
                                    "Learning Rule": ""},

            "bpLike_WT_fixedDend_eq":  {"config": "20240508_EIANN_2_hidden_mnist_BP_like_config_2K_complete_optimized.yaml",
                                        "color":  "gray",
                                        "label":   "bpLike_WT_fixedDend_eq",
                                        "Architecture": "", 
                                        "Algorithm": "", 
                                        "Learning Rule": ""},

            "bpLike_fixedTD_hebbdend": {"config": "20241114_EIANN_2_hidden_mnist_BP_like_config_5J_fixed_TD_complete_optimized.yaml",
                                        "name": "bpLike_fixedTD_hebbdend",
                                        "color": "gray",
                                        "label": "Fixed random B (LDS)",
                                        "Architecture": "", 
                                        "Algorithm": "", 
                                        "Learning Rule": ""},

            "bpLike_fixedTD_hebbdend_eq":  {"config": "20240830_EIANN_2_hidden_mnist_BP_like_config_2L_fixed_TD_complete_optimized.yaml",
                                            "color": "lightgray",
                                            "label": "bpLike_fixedTD_hebbdend_eq",
                                            "Architecture": "", 
                                            "Algorithm": "", 
                                            "Learning Rule": ""},

            "bpLike_hebbTD_hebbdend":{"config": "20241009_EIANN_2_hidden_mnist_BP_like_config_5J_learn_TD_HWN_1_complete_optimized.yaml",
                                    "color": "blue",
                                    "label": "bpLike_hebbTD_hebbdend",
                                    "Architecture": "", 
                                    "Algorithm": "", 
                                    "Learning Rule": "",},

            "bpLike_hebbTD_hebbdend_eq":{"config": "20240830_EIANN_2_hidden_mnist_BP_like_config_2L_learn_TD_HWN_3_complete_optimized.yaml",
                                        "color": "magenta",
                                        "label": "bpLike_hebbTD_hebbdend_eq",
                                        "Architecture": "", 
                                        "Algorithm": "", 
                                        "Learning Rule": "",},

            "bpLike_TCWN_hebbdend": {"config": "20241120_EIANN_2_hidden_mnist_BP_like_config_5J_learn_TD_HTCWN_2_complete_optimized.yaml",
                                     "name": "bpLike_TCWN_hebbdend",
                                     "label": "Learned B: TCH (LDS)",
                                     "color": "green",
                                     "Architecture": "", 
                                     "Algorithm": "", 
                                     "Learning Rule": "",}, # TC with weight norm

            ##########################
            # Biological models
            ##########################
            "HebbWN_topsup":       {"config": "20241105_EIANN_2_hidden_mnist_Top_Layer_Supervised_Hebb_WeightNorm_config_7_complete_optimized.yaml",
                                    "name":   "Top-supervised Hebb", #(bpLike in the top layer)
                                    "label":  "Hebb (EIANN)",
                                    "color":  "green",
                                    "Architecture": "", 
                                    "Algorithm": "", 
                                    "Learning Rule": ""},

            "Supervised_HebbWN_WT_hebbdend":{"config": "20240714_EIANN_2_hidden_mnist_Supervised_Hebb_WeightNorm_config_4_complete_optimized.yaml",
                                    "color": "olive",
                                    "label": "Supervised_HebbWN_WT_hebbdend",
                                    "Architecture": "", 
                                    "Algorithm": "", 
                                    "Learning Rule": ""}, 

            "SupHebbTempCont_WT_hebbdend": {"config": "20241125_EIANN_2_hidden_mnist_Hebb_Temp_Contrast_config_2_complete_optimized.yaml",
                                            "color": "mediumaquamarine",
                                            "label": "TCH",
                                            "Architecture": "", 
                                            "Algorithm": "", 
                                            "Learning Rule": ""}, # Like target propagation / temporal contrast on forward dW

            "Supervised_HebbWN_learned_somaI":{"config": "20240919_EIANN_2_hidden_mnist_Supervised_Hebb_WeightNorm_learn_somaI_config_4_complete_optimized.yaml",
                                                "color": "lime",
                                                "label": "Supervised HebbWN learned somaI",
                                                "Architecture": "", 
                                                "Algorithm": "", 
                                                "Learning Rule": ""},

            "Supervised_BCM_WT_hebbdend":  {"config": "20240723_EIANN_2_hidden_mnist_Supervised_BCM_config_4_complete_optimized.yaml",
                                            "color": "mediumpurple",
                                            "label": "BCM",
                                            "Architecture": "", 
                                            "Algorithm": "", 
                                            "Learning Rule": ""},

            "BTSP_WT_hebbdend":    {"config":"20241212_EIANN_2_hidden_mnist_BTSP_config_5L_complete_optimized.yaml",
                                    "name": "BTSP_WT_hebbdend",
                                    "label": "Symmetric B=W (BTSP)",
                                    "color": "orange",  
                                    "Architecture": "", 
                                    "Algorithm": "", 
                                    "Learning Rule": ""}, 

            # "BTSP_hebbTD_hebbdend": {"config": "20240905_EIANN_2_hidden_mnist_BTSP_config_3L_learn_TD_HWN_3_complete_optimized.yaml",
            #                         "color": "magenta",
            #                         "label": "BTSP_hebbTD_hebbdend"},

            "BTSP_fixedTD_hebbdend":{"config": "20241216_EIANN_2_hidden_mnist_BTSP_config_5L_fixed_TD_complete_optimized.yaml",
                                    "name": "BTSP_fixedTD_hebbdend",
                                    "label": "Fixed random B (BTSP)",
                                    "color": "gray",
                                    "Architecture": "", 
                                    "Algorithm": "", 
                                    "Learning Rule": ""},

            "BTSP_TCWN_hebbdend": {"config": "20241216_EIANN_2_hidden_mnist_BTSP_config_5L_learn_TD_HTCWN_3_complete_optimized.yaml",
                                    "name":  "BTSP_TCWN_hebbdend",
                                    "label": "Learned B: TCH (BTSP)",
                                    "color": "green",
                                    "Architecture": "", 
                                    "Algorithm": "", 
                                    "Learning Rule": ""}, # top-down learning with TempContrast+weight norm

            ##########################
            # Spirals dataset models
            ##########################
            "vanBP_0_hidden_learned_bias_spiral": {"config": "20250108_EIANN_0_hidden_spiral_van_bp_relu_learned_bias_config_complete_optimized.yaml",
                                            "color": "black",
                                            "label": "Vanilla Backprop 0-Hidden (Learned Bias)",
                                            "Architecture": "0-hidden", 
                                            "Algorithm": "Backprop", 
                                            "Learning Rule": "Gradient Descent",
                                            "Bias": "Learned"},

            "vanBP_2_hidden_learned_bias_spiral": {"config": "20250108_EIANN_2_hidden_spiral_van_bp_relu_learned_bias_config_complete_optimized.yaml",
                                                   "name": "Vanilla Backprop 2-Hidden (Learned Bias)",
                                                   "color": "black",
                                                   "label": "FF Backprop (learned bias)",
                                                   "Architecture": "2-hidden", 
                                                   "Algorithm": "Backprop", 
                                                   "Learning Rule": "Gradient Descent",
                                                   "Bias": "Learned"},

            "vanBP_2_hidden_zero_bias_spiral": {"config": "20250108_EIANN_2_hidden_spiral_van_bp_relu_zero_bias_config_complete_optimized.yaml",
                                        "name": "Vanilla Backprop 2-Hidden (Zero Bias)",
                                        "color": "gray",
                                        "label": "FF Backprop (no bias)",
                                        "Architecture": "2-hidden", 
                                        "Algorithm": "Backprop", 
                                        "Learning Rule": "Gradient Descent",
                                        "Bias": "Zero"},

            "bpDale_learned_bias_spiral": {"config": "20250108_EIANN_2_hidden_spiral_bpDale_fixed_SomaI_learned_bias_config_complete_optimized.yaml",
                                        "name": "Backprop + Dale's Law 2-Hidden (Learned Bias)",
                                        "color": "blue",
                                        "label": "Backprop, learned bias (EIANN)",
                                        "Architecture": "", 
                                        "Algorithm": "", 
                                        "Learning Rule": "",
                                        "Bias": "Learned"},

            "bpLike_DTC_learned_bias_spiral": {"config": "20250108_EIANN_2_hidden_spiral_BP_like_1_fixed_SomaI_learned_bias_config_complete_optimized.yaml",
                                        "color": "purple",
                                        "label": "Backprop Like (DTC) (Learned Bias)",
                                        "Architecture": "", 
                                        "Algorithm": "", 
                                        "Learning Rule": "",
                                        "Bias": "Learned"},

            "DTP_learned_bias_spiral": {"config": "20250108_EIANN_2_hidden_spiral_DTP_fixed_SomaI_learned_bias_config_complete_optimized.yaml",
                                        "color": "red",
                                        "label": "Dend. Target Prop., learned bias",
                                        "Architecture": "", 
                                        "Algorithm": "", 
                                        "Learning Rule": "",
                                        "Bias": "Learned"}, # optimized DendI fraction (~50%)

            "DTP_fixed_DendI_learned_bias_1_spiral": {"config": "20250217_EIANN_2_hidden_spiral_DTP_fixed_DendI_fixed_SomaI_learned_bias_1_config_complete_optimized.yaml",
                                        "color": "red",
                                        "label": "Dend Target Prop, fixed DendI/SomaI, learned bias (1)",
                                        "Architecture": "", 
                                        "Algorithm": "", 
                                        "Learning Rule": "",
                                        "Bias": "Learned"}, # optimized DendI fraction

            "DTP_fixed_DendI_learned_bias_2_spiral": {"config": "20250217_EIANN_2_hidden_spiral_DTP_fixed_DendI_fixed_SomaI_learned_bias_2_config_complete_optimized.yaml",
                                        "color": "red",
                                        "label": "Dend Target Prop, fixed DendI/SomaI, learned bias (2)", # fixed DendI fraction to 25%
                                        "Architecture": "", 
                                        "Algorithm": "", 
                                        "Learning Rule": "",
                                        "Bias": "Learned"}, 

            "DTP_fixed_DendI_learned_bias_3_spiral": {"config": "20250217_EIANN_2_hidden_spiral_DTP_fixed_DendI_fixed_SomaI_learned_bias_3_config_complete_optimized.yaml",
                                        "color": "red",
                                        "label": "Dend Target Prop, fixed DendI/SomaI, learned bias (3)", # fixed DendI fraction to 10%
                                        "Architecture": "", 
                                        "Algorithm": "", 
                                        "Learning Rule": "",
                                        "Bias": "Learned"}, 
        }


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


    #-------------- Main Figures --------------

    if figure == "all":
        # Generate HDF5 files for all models
        all_models = list(model_dict_all.keys())[9:]
        all_mnist_models = [model_key for model_key in all_models if "spiral" not in model_key]
        generate_hdf5_all_seeds(all_mnist_models, model_dict_all, config_path_prefix="network_config/mnist/", saved_network_path_prefix=saved_network_path_prefix+"MNIST/", recompute=recompute)
        all_spiral_models = [model_key for model_key in all_models if "spiral" in model_key]
        generate_hdf5_all_seeds(all_spiral_models, model_dict_all, config_path_prefix="network_config/spiral/", saved_network_path_prefix=saved_network_path_prefix+"spiral/", recompute=recompute)
        recompute = None

    # Diagrams + example dynamics
    if figure in ["all", "fig1"]:
        figure_name = "dynamics_example_plots"
        plot_dynamics_example(model_dict_all, save=figure_name, recompute=recompute)

    # Backprop models
    if figure in ["all", "fig2"]:
        saved_network_path_prefix += "MNIST/"
        model_list_heatmaps = ["vanBP", "bpDale_learned", "HebbWN_topsup"]
        model_list_metrics = model_list_heatmaps
        figure_name = "Fig2_vanBP_bpDale_hebb"
        generate_fig2(model_dict_all, model_list_heatmaps, model_list_metrics, save=figure_name, saved_network_path_prefix=saved_network_path_prefix, recompute=recompute)

    # Analyze DendI
    if figure in ["all", "fig3"]:
        saved_network_path_prefix += "MNIST/"
        model_list_heatmaps = ["bpLike_WT_fixedDend", "bpLike_WT_localBP", "bpLike_WT_hebbdend"]
        # model_list_metrics = model_list_heatmaps + ["bpDale_fixed"]
        model_list_metrics = model_list_heatmaps
        figure_name = "Fig3_dendI"
        generate_fig3(model_dict_all, model_list_heatmaps, model_list_metrics, save=figure_name, saved_network_path_prefix=saved_network_path_prefix, recompute=recompute)

    # Analyze E properties of main models
    if figure in ["all", "fig4"]:
        saved_network_path_prefix_m = saved_network_path_prefix + "MNIST/"
        model_list_heatmaps = ["bpDale_fixed", "bpLike_WT_hebbdend"]
        model_list_metrics = model_list_heatmaps
        figure_name = "Fig4_bpDale_bpLike"
        fig4(model_dict_all, model_list_heatmaps, model_list_metrics, save=figure_name, saved_network_path_prefix=saved_network_path_prefix_m, recompute=recompute)
        
        saved_network_path_prefix_s = saved_network_path_prefix + "spiral/"
        model_list_spirals = ["bpDale_learned_bias_spiral", "DTP_learned_bias_spiral"]        
        model_list_metrics = model_list_spirals + ["vanBP_2_hidden_zero_bias_spiral", "vanBP_2_hidden_learned_bias_spiral"]
        figure_name = "Fig4_Spirals"
        fig4_spirals(model_dict_all, model_list_spirals, model_list_metrics, spiral_type='decision', config_path_prefix='network_config/spiral/',
                                saved_network_path_prefix=saved_network_path_prefix_s, save=figure_name, recompute=recompute)

    # Biological learning rules (with WT/good gradients)
    if figure in ["all","fig5"]:
        saved_network_path_prefix += "MNIST/"
        model_list = ["bpLike_WT_hebbdend", "BTSP_WT_hebbdend", "Supervised_BCM_WT_hebbdend","SupHebbTempCont_WT_hebbdend"]
        figure_name = "Fig5_BTSP_BCM_HebbWN"
        generate_fig5(model_dict_all, model_list, save=figure_name, saved_network_path_prefix=saved_network_path_prefix, recompute=recompute)
        
    # Forward (W) vs backward (B) alignment angle
    if figure in ["all", "fig6"]:
        saved_network_path_prefix += "MNIST/"
        model_list1 = ["bpLike_WT_hebbdend", "bpLike_fixedTD_hebbdend", "bpLike_TCWN_hebbdend"]
        model_list2 = ["BTSP_WT_hebbdend", "BTSP_fixedTD_hebbdend", "BTSP_TCWN_hebbdend"]
        figure_name = "Fig6_WB_alignment_FA_bpLike_BTSP"
        generate_fig6(model_dict_all, model_list1, model_list2, save=figure_name, saved_network_path_prefix=saved_network_path_prefix, recompute=recompute)


    #-------------- Supplementary Figures --------------

    # S1: Population activity dynamics
    if figure in ['all', "dynamics"]:
        model_list = ["bpDale_learned", "bpDale_fixed", "HebbWN_topsup", "bpLike_WT_hebbdend"]
        # model_list = ["bpLike_WT_hebbdend"]
        figure_name = "Suppl_dynamics"
        plot_average_dynamics(model_dict_all, model_list, save=figure_name, saved_network_path_prefix=saved_network_path_prefix, recompute=recompute)
                  
    # Analyze somaI selectivity (S2: supplement to Fig.2)
    if figure in ["all", "S2"]:
        saved_network_path_prefix += "MNIST/"
        model_list_heatmaps = ["bpDale_learned", "bpDale_fixed", "HebbWN_topsup"]
        model_list_metrics = model_list_heatmaps
        figure_name = "FigS1_somaI"
        compare_somaI_properties(model_dict_all, model_list_heatmaps, model_list_metrics, save=figure_name, saved_network_path_prefix=saved_network_path_prefix, recompute=recompute)

    # S3: Representational similarity analysis
    if figure in ["all", "rsm"]:
        saved_network_path_prefix += "MNIST/"
        model_list_heatmaps = ["bpDale_learned", "bpDale_fixed", "HebbWN_topsup", "bpLike_WT_hebbdend"]
        model_list_metrics = model_list_heatmaps
        figure_name = "Suppl_similarity_analysis"
        compare_RSM_properties(model_dict_all, model_list_heatmaps, model_list_metrics, save=figure_name, saved_network_path_prefix=saved_network_path_prefix, recompute=recompute)

    # Supplementary Spirals Figure
    if figure in ["all", "spiral-suppl"]:
        saved_network_path_prefix += "spiral/"
        model_list_heatmaps = ["vanBP_0_hidden_learned_bias_spiral", "vanBP_2_hidden_learned_bias_spiral", 
                                "vanBP_2_hidden_zero_bias_spiral", "bpDale_learned_bias_spiral", 
                                "bpLike_DTC_learned_bias_spiral", "DTP_learned_bias_spiral", "DTP_fixed_DendI_learned_bias_1_spiral"]
        model_list_metrics = model_list_heatmaps
        figure_name = "Suppl1_Spirals"

        # Choose spiral_type='scatter' or 'decision'
        generate_spirals_figure(model_dict_all, model_list_heatmaps, model_list_metrics, spiral_type='decision', config_path_prefix='network_config/spiral/',
                                saved_network_path_prefix=saved_network_path_prefix, save=figure_name, recompute=recompute)

    if figure in ["all", "table"]:
        saved_network_path_prefix += "MNIST/"
        figure_name = "FigS3_mnist_table"
        model_list = ["vanBP", "bpDale_fixed", "bpDale_learned", "HebbWN_topsup", 
                      "bpLike_WT_fixedDend", "bpLike_WT_localBP", "bpLike_WT_hebbdend", 
                      "SupHebbTempCont_WT_hebbdend", "Supervised_BCM_WT_hebbdend", "BTSP_WT_hebbdend"]
        generate_summary_table(model_dict_all, model_list, saved_network_path_prefix=saved_network_path_prefix+"extended/", save=figure_name, recompute=recompute)
    
    if figure in ["all", "spiral-table"]:
        saved_network_path_prefix += "spiral/"
        figure_name = "spiral-table"
        model_list = ["vanBP_0_hidden_learned_bias_spiral", "vanBP_2_hidden_learned_bias_spiral", 
                    "vanBP_2_hidden_zero_bias_spiral", "bpDale_learned_bias_spiral", 
                    "bpLike_DTC_learned_bias_spiral", "DTP_learned_bias_spiral", "DTP_fixed_DendI_learned_bias_1_spiral"]
        generate_summary_table(model_dict_all, model_list, saved_network_path_prefix=saved_network_path_prefix+"extended/", config_path_prefix="network_config/spiral/", save=figure_name, recompute=recompute)

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


    # Combine figures into one PDF
    directory = "figures/"
    os.makedirs(directory, exist_ok=True)
    image_paths = [os.path.join(directory, figure) for figure in os.listdir(directory) if figure.endswith('.png') and figure.startswith('Fig')]
    image_paths.sort()
    images_to_pdf(image_paths=image_paths, output_path= directory+"all_figures.pdf")


if __name__=="__main__":
    main()
