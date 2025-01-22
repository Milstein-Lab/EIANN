import torch
import numpy as np
import scipy

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import os
import h5py
import click
import gc
import copy
from reportlab.pdfgen import canvas

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
                    'font.sans-serif': 'Helvetica',
                    'svg.fonttype': 'none',
                    'text.usetex': False})

'''
Figure 1: Diagram - How Dendritic Target Propagation (DendTP) works
    -> DendTP is a biologically plausible way to pass gradients to hidden layers
    -> Compute local loss with top-down nudges and dendI subtraction
    -> Diagram of nudges (+a+D) over time 

Figure 2: Dale's Law architecture is able to learn good representations (even with random somaI). Is self-organization enough? (No)
    -> vanBP vs bpDale(onlyE) vs bpDale(fixedI) vs Hebb(top-sup)
    -> (bpDale is more structured/sparse [H1E/H2E metrics])

Supplementary 1: learning somaI is not necessary
    -> bpDale(learned) vs bpDale(fixed) [somaI representations + accuracy/selectivity/sparsity]

How to compute accurate gradients with bio-plausible rule?
Figure 3: Hebbian learning rule enables dendritic cancellation of forward activity
    -> DendI representation: Random vs LocalBP vs HebbWN
    -> Plot Dend State over time
    -> Plot angle vs BP

Supplementary 2: E cell representations for dendritic models

Figure 4: Given good bio-gradients, what do different bio-motivated learning rules give?
    -> BTSP vs BCM vs HebbWN
    -> Representations/RFs for HiddenE + metrics plots

Supplement: Dend state + soma/dendI representations + angle vs BP for bio learning rule (+ RFs?)

Figure 5: Hebbian learning rule enables W/B alignment
    -> FA vs BTSP vs bpLike
    -> Plot W/B angle over time
    -> Plot angle vs BP + accuracy
    -> (Diagram + equations)

Supplementary: Alternate/Different approaches to W/B alignment
    -> If only consider postsynaptic D (not forward activity): perfect alignment (cf Burstprop)
    -> If only consider forward activity: alignment only if cov(a)=I (eg. Gaussian noise or one-hot)
    -> 'Coincidental' alignment: some alignment remains if cov(a)!=I, because of similarities in learned structure between PCA and BP

Supplementary: DendTP performs well across different tasks (cf Burstprop Fig6)
    -> Spirals requires biases (random fixed)
    -> Plot train accuracy in Spirals
    -> Plot test accuracy in Spirals
    

    
--------------------------------------------------------------------------------------
    
Figure 2: somaI representations in bpDale(learnedI) vs top-sup HebbWN(learnedI) vs unsup-HebbWN(learnedI)
    (top: OutputE + H1E, activity and receptive fields and metrics)
    (bottom: OutputSomaI + H1SomaI, activity and metrics)
    -> bpDale SomaI is unselective/unstructured, biological learning rule is not (focus on SomaI metrics)
    -> sup HebbWN performance is mediocre (representational collapse), we need a biological way to pass gradients to hidden layers

(switched to fixed somaI)
-> bpDale still performs with fixedI, HebbWN does not, but BCM does (focus on H1E metrics)
(Supp: bpDale(fixedI) vs top-sup HebbWN(fixedI) vs top-sup BCM(fixedI) vs unsup BCM(fixedI))

Passing gradients with apical dendrites:
Figure 3: dendritic_gating(bpLike) vs dendritic_gating w/ HebbWN dendritic learning
        -> Can compute local loss with top-down nudges and dendI subtraction
        -> dendritic subtraction is effective even with HebbWN

Figure 4: BTSP(weight transpose + HebbWN dendI) vs sup-HebbWN
        -> Local loss allows learning with a biological rule
        -> BTSP does better than Hebbian (Hebb too simple)
'''

########################################################################################################

def generate_data_hdf5(config_path, saved_network_path, data_file_path, overwrite=False, recompute=None):
    '''
    Loads a network and saves plot-ready processed data into an hdf5 file.

    :param config_path_prefix: Path to config file directory
    :param saved_network_path_prefix: Path to directory containing pickled network
    :param model_dict: Dictionary containing model information for a single model
    :param data_file_path: Path to hdf5 file to save data to
    '''

    # Build network
    network_name = os.path.basename(config_path).split('.')[0]
    pickle_filename = os.path.basename(saved_network_path)
    network_seed, data_seed = pickle_filename.split('_')[-3:-1]
                                                                                 
    network = ut.build_EIANN_from_config(config_path, network_seed=network_seed)    
    seed = f"{network_seed}_{data_seed}"

    variables_to_save = ['percent_correct', 'average_pop_activity_dict', 
                         'val_loss_history', 'val_accuracy_history', 'val_history_train_steps',
                         'test_loss_history', 'test_accuracy_history', 'angle_vs_bp', 'angle_vs_bp_stochastic',
                         'feedback_weight_angle_history', 'sparsity_history', 'selectivity_history']
    variables_to_save.extend([f"metrics_dict_{population.fullname}" for population in network.populations.values()])
    variables_to_save.extend([f"maxact_receptive_fields_{population.fullname}" for population in network.populations.values() if population.name=='E' and population.fullname!='InputE'])
    if "Dend" in "".join(network.populations.keys()):
        variables_to_save.extend(["binned_mean_forward_dendritic_state", "binned_mean_forward_dendritic_state_steps"])

    # Open hdf5 and check if the relevant network data already exists       
    variables_to_recompute = []  
    if os.path.exists(data_file_path): # If the file exists, check if the network data already exists or needs to be recomputed
        with h5py.File(data_file_path, 'r') as file:
            if network_name in file.keys():
                if seed in file[network_name].keys():
                    if overwrite:
                        print(f"Overwriting {network_name} {seed} in {data_file_path}")
                        variables_to_recompute = variables_to_save                        
                    elif set(variables_to_save).issubset(file[network_name][seed].keys()) and recompute is None:
                        return
                    else:
                        print(f"Recomputing {network_name} {seed} in {data_file_path}")
                        variables_to_recompute = [var for var in variables_to_save if var not in file[network_name][seed].keys()]
                else:
                    print(f"Computing data for {network_name} {seed} in {data_file_path}")
                    variables_to_recompute = variables_to_save     
            else:
                print(f"Computing data for {network_name} {seed} in {data_file_path}")
                variables_to_recompute = variables_to_save     
    else:
        print(f"Creating new data file {data_file_path}")
        variables_to_recompute = variables_to_save
    
    if recompute is not None and recompute not in variables_to_recompute:
        variables_to_recompute.append(recompute)

    print("-----------------------------------------------------------------------------")

    print(variables_to_recompute)
    if len(variables_to_recompute) > 0:
        overwrite = True
    else:
        print(f"No variables to recompute for {network_name} {seed} in {data_file_path}")
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
    all_dataloaders = ut.get_MNIST_dataloaders(sub_dataloader_size=1000)
    train_dataloader, train_sub_dataloader, val_dataloader, test_dataloader, data_generator = all_dataloaders

    ##################################################################
    ## Generate plot data
    # 1. Class-averaged activity
    if 'percent_correct' in variables_to_recompute:
        percent_correct, average_pop_activity_dict = ut.compute_test_activity(network, test_dataloader, sort=False, export=True, export_path=data_file_path, overwrite=overwrite)

    # 2. Receptive fields and metrics
    for population in network.populations.values():
        if f"metrics_dict_{population.fullname}" in variables_to_recompute:
            receptive_fields = None
            if population.name == "E" and population.fullname != "InputE":
                receptive_fields = ut.compute_maxact_receptive_fields(population, export=True, export_path=data_file_path, overwrite=overwrite)
            metrics_dict = ut.compute_representation_metrics(population, test_dataloader, receptive_fields, export=True, export_path=data_file_path, overwrite=overwrite)


    # Angle vs backprop
    if 'angle_vs_bp_stochastic' in variables_to_recompute:
        stored_history_step_size = torch.diff(network.param_history_steps)[-1]
        if 'H1SomaI' in network.populations:
            config_path2 = os.path.join(os.path.dirname(config_path), "20231129_EIANN_2_hidden_mnist_bpDale_relu_SGD_config_G_complete_optimized.yaml")
            network2 = ut.build_EIANN_from_config(config_path2, network_seed=network_seed)
            bpClone_network = ut.compute_alternate_dParam_history(train_dataloader, network, network2, batch_size=1, constrain_params=False)
        else:
            bpClone_network = ut.compute_alternate_dParam_history(train_dataloader, network, batch_size=1, constrain_params=False)
        angles = ut.compute_dW_angles_vs_BP(bpClone_network.predicted_dParam_history, bpClone_network.actual_dParam_history)
        ut.save_plot_data(network.name, network.seed, data_key='angle_vs_bp_stochastic', data=angles, file_path=data_file_path, overwrite=overwrite)

    if 'angle_vs_bp' in variables_to_recompute:
        stored_history_step_size = torch.diff(network.param_history_steps)[-1]
        if 'H1SomaI' in network.populations:
            config_path2 = os.path.join(os.path.dirname(config_path), "20231129_EIANN_2_hidden_mnist_bpDale_relu_SGD_config_G_complete_optimized.yaml")
            network2 = ut.build_EIANN_from_config(config_path2, network_seed=network_seed)
            bpClone_network = ut.compute_alternate_dParam_history(train_dataloader, network, network2, batch_size=stored_history_step_size, constrain_params=False)
        else:
            bpClone_network = ut.compute_alternate_dParam_history(train_dataloader, network, batch_size=stored_history_step_size, constrain_params=False)
        angles = ut.compute_dW_angles_vs_BP(bpClone_network.predicted_dParam_history, bpClone_network.actual_dParam_history_stepaveraged)
        ut.save_plot_data(network.name, network.seed, data_key='angle_vs_bp', data=angles, file_path=data_file_path, overwrite=overwrite)

    # Forward vs Backward weight angle (weight symmetry)
    if 'feedback_weight_angle_history' in variables_to_recompute:
        FF_FB_angles = ut.compute_feedback_weight_angle_history(network)
        ut.save_plot_data(network.name, network.seed, data_key='feedback_weight_angle_history', data=FF_FB_angles, file_path=data_file_path, overwrite=overwrite)

    # Binned dendritic state (local loss)
    if 'binned_mean_forward_dendritic_state' in variables_to_recompute:
        steps, binned_mean_forward_dendritic_state = ut.get_binned_mean_population_attribute_history_dict(network, attr_name="forward_dendritic_state", bin_size=100, abs=True)
        if binned_mean_forward_dendritic_state is not None:
            ut.save_plot_data(network.name, network.seed, data_key='binned_mean_forward_dendritic_state', data=binned_mean_forward_dendritic_state, file_path=data_file_path, overwrite=overwrite)
            ut.save_plot_data(network.name, network.seed, data_key='binned_mean_forward_dendritic_state_steps', data=steps, file_path=data_file_path, overwrite=overwrite)

    # Sparsity and selectivity
    if 'sparsity_history' in variables_to_recompute or 'selectivity_history' in variables_to_recompute:
        sparsity_history_dict, selectivity_history_dict = ut.compute_sparsity_selectivity_history(network, test_dataloader)
        ut.save_plot_data(network.name, network.seed, data_key='sparsity_history', data=sparsity_history_dict, file_path=data_file_path, overwrite=overwrite)
        ut.save_plot_data(network.name, network.seed, data_key='selectivity_history', data=selectivity_history_dict, file_path=data_file_path, overwrite=overwrite)

    # Loss and accuracy
    if any([var in variables_to_recompute for var in ['val_loss_history', 'val_accuracy_history', 'val_history_train_steps']]):
        ut.save_plot_data(network.name, network.seed, data_key='val_loss_history',          data=network.val_loss_history,          file_path=data_file_path, overwrite=overwrite)
        ut.save_plot_data(network.name, network.seed, data_key='val_accuracy_history',      data=network.val_accuracy_history,      file_path=data_file_path, overwrite=overwrite)
        ut.save_plot_data(network.name, network.seed, data_key='val_history_train_steps',   data=network.val_history_train_steps,   file_path=data_file_path, overwrite=overwrite)
    
    if any([var in variables_to_recompute for var in ['test_loss_history', 'test_accuracy_history']]):
        ut.save_plot_data(network.name, network.seed, data_key='test_loss_history',         data=network.test_loss_history,         file_path=data_file_path, overwrite=overwrite)
        ut.save_plot_data(network.name, network.seed, data_key='test_accuracy_history',     data=network.test_accuracy_history,     file_path=data_file_path, overwrite=overwrite)


def generate_data_all_seeds(all_models, model_dict_all, config_path_prefix, saved_network_path_prefix, overwrite=False, recompute=None):
    for model_key in all_models:
        model_dict = model_dict_all[model_key]
        config_path = config_path_prefix + model_dict['config']
        pickle_basename = "_".join(model_dict['config'].split('_')[0:-2])
        network_name = model_dict['config'].split('.')[0]
        data_file_path = f"data/plot_data_{network_name}.h5"
        for seed in model_dict['seeds']:
            saved_network_path = saved_network_path_prefix + pickle_basename + f"_{seed}_complete.pkl"
            generate_data_hdf5(config_path, saved_network_path, data_file_path, overwrite, recompute)
            gc.collect()

########################################################################################################

def plot_accuracy_all_seeds(data_dict, model_dict, ax):
    """
    Plot test accuracy for all seeds with shaded error bars
    """
    accuracy_all_seeds = [data_dict[seed]['test_accuracy_history'] for seed in data_dict]
    avg_accuracy = np.mean(accuracy_all_seeds, axis=0)
    error = np.std(accuracy_all_seeds, axis=0)
    val_steps = data_dict[next(iter(data_dict))]['val_history_train_steps'][:]
    ax.plot(val_steps, avg_accuracy, label=model_dict["name"], color=model_dict["color"])
    ax.fill_between(val_steps, avg_accuracy-error, avg_accuracy+error, alpha=0.2, color=model_dict["color"], linewidth=0)
    ax.set_ylim([0,100])
    ax.set_xlabel('Training step')
    ax.set_ylabel('Test accuracy (%)', labelpad=-2)
    legend = ax.legend(ncol=3, bbox_to_anchor=(-0.3, 1.4), loc='upper left', fontsize=6)
    for line in legend.get_lines():
        line.set_linewidth(1.5)


def plot_error_all_seeds(data_dict, model_dict, ax):
    accuracy_all_seeds = [data_dict[seed]['test_accuracy_history'] for seed in data_dict]
    error_rate_all_seeds = [(100 - np.array(acc)) for acc in accuracy_all_seeds]
    avg_error_rate = np.mean(error_rate_all_seeds, axis=0)
    error = np.std(error_rate_all_seeds, axis=0)
    val_steps = data_dict[next(iter(data_dict))]['val_history_train_steps'][:]
    ax.plot(val_steps, avg_error_rate, label=model_dict["name"], color=model_dict["color"])
    ax.fill_between(val_steps, avg_error_rate-error, avg_error_rate+error, alpha=0.2, color=model_dict["color"], linewidth=0)
    ax.set_xlabel('Training step')
    ax.set_ylabel('Error Rate (%)', labelpad=0)
    ax.set_yscale('log')
    ax.set_ylim(5, 100)
    ax.set_yticks([10, 100], labels=['10%', '100%'])


def plot_sparsity_all_seeds(data_dict, model_dict, populations_to_plot, ax, plot_type='cdf'):
    sparsity_all_seeds = []
    for seed in data_dict:
        sparsity_one_seed = []
        for population in populations_to_plot:
            sparsity_one_seed.extend(data_dict[seed][f"metrics_dict_{population}"]['sparsity'])
        sparsity_all_seeds.append(sparsity_one_seed)

    if plot_type == 'cdf':
        pt.plot_cumulative_distribution(sparsity_all_seeds, ax=ax, label=model_dict["name"], color=model_dict["color"])
        ax.set_ylabel('Fraction of patterns')
        ax.set_xlabel('Sparsity') # \n(1 - fraction of units active)')
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_yticks([0,1])
    elif plot_type == 'bar':
        avg_sparsity_per_seed = [np.mean(x) for x in sparsity_all_seeds]
        avg_sparsity = np.mean(avg_sparsity_per_seed)
        error = np.std(avg_sparsity_per_seed)
        x = len(ax.patches)
        bar = ax.bar(x, avg_sparsity, yerr=error, color=model_dict["color"], width=0.4, ecolor='red')
        bar[0].set_label(model_dict["name"])
        ax.set_ylabel('Sparsity')
        ax.set_ylim([0,1])
        ax.set_xticks(range(x+1))
        xtick_labels = [patch.get_label() for patch in ax.patches]
        ax.set_xticklabels(xtick_labels, rotation=45, ha='right')


def plot_selectivity_all_seeds(data_dict, model_dict, populations_to_plot, ax, plot_type='cdf'):
    selectivity_all_seeds = []
    for seed in data_dict:
        selectivity_one_seed = []
        for population in populations_to_plot:
            selectivity_one_seed.extend(data_dict[seed][f"metrics_dict_{population}"]['selectivity'])
        selectivity_all_seeds.append(selectivity_one_seed)

    if plot_type == 'cdf':
        pt.plot_cumulative_distribution(selectivity_all_seeds, ax=ax, label=model_dict["name"], color=model_dict["color"])
        ax.set_ylabel('Fraction of units')
        ax.set_xlabel('Selectivity') # \n(1 - fraction of active patterns)')
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_yticks([0,1])

    elif plot_type == 'bar':
        avg_selectivity_per_seed = [np.mean(x) for x in selectivity_all_seeds]
        avg_selectivity = np.mean(avg_selectivity_per_seed)
        error = np.std(avg_selectivity_per_seed)
        x = len(ax.patches)
        bar = ax.bar(x, avg_selectivity, yerr=error, color=model_dict["color"], width=0.4, ecolor='red')
        bar[0].set_label(model_dict["name"])
        ax.set_ylabel('Selectivity')
        ax.set_ylim([0,1])
        ax.set_xticks(range(x+1))
        xtick_labels = [patch.get_label() for patch in ax.patches]
        ax.set_xticklabels(xtick_labels, rotation=45, ha='right')


def plot_structure_all_seeds(data_dict, model_dict, ax, plot_type='cdf'):
    structure_all_seeds = []
    for seed in data_dict:
        structure_one_seed = []
        for population in ['H1E', 'H2E']:
            structure_one_seed.extend(data_dict[seed][f"metrics_dict_{population}"]['structure'])
        structure_all_seeds.append(structure_one_seed)

    if plot_type == 'cdf':
        pt.plot_cumulative_distribution(structure_all_seeds, ax=ax, label=model_dict["name"], color=model_dict["color"])
        ax.set_ylabel('Fraction of units')
        ax.set_xlabel("Structure (Moran's I)") # \n(Moran's I spatial autocorrelation)")
        ax.set_xlim([0,1])
    elif plot_type == 'bar':
        avg_structure_per_seed = [np.mean(x) for x in structure_all_seeds]
        avg_structure = np.mean(avg_structure_per_seed)
        error = np.std(avg_structure_per_seed)
        x = len(ax.patches)
        bar = ax.bar(x, avg_structure, yerr=error, color=model_dict["color"], width=0.4, ecolor='red')
        bar[0].set_label(model_dict["name"])
        ax.set_ylabel('Structure (Moran\'s I)')
        ax.set_ylim([0,1])
        ax.set_xticks(range(x+1))
        xtick_labels = [patch.get_label() for patch in ax.patches]
        ax.set_xticklabels(xtick_labels, rotation=45, ha='right')


def plot_dendritic_state_all_seeds(data_dict, model_dict, ax):
    if 'binned_mean_forward_dendritic_state_steps' not in data_dict[next(iter(data_dict.keys()))]:
        return
    dendstate_all_seeds = []
    for seed in model_dict['seeds']:
        dendstate_one_seed = data_dict[seed]['binned_mean_forward_dendritic_state']['all'][:]
        dendstate_all_seeds.append(dendstate_one_seed)
    avg_dendstate = np.mean(dendstate_all_seeds, axis=0)
    error = np.std(dendstate_all_seeds, axis=0)
    binned_mean_forward_dendritic_state_steps = data_dict[seed]['binned_mean_forward_dendritic_state_steps'][:]
    ax.plot(binned_mean_forward_dendritic_state_steps, avg_dendstate, label=model_dict["name"], color=model_dict["color"])
    ax.fill_between(binned_mean_forward_dendritic_state_steps, avg_dendstate-error, avg_dendstate+error, alpha=0.5, color=model_dict["color"], linewidth=0)
    ax.set_xlabel('Training step')
    ax.set_ylabel('Dendritic state')
    ax.set_yscale('log')
    # ax.set_ylim(bottom=-0., top=0.3)


def plot_angle_vs_bp_all_seeds(data_dict, model_dict, ax, stochastic=True):
    angle_all_seeds = []
    for seed in model_dict['seeds']:
        if stochastic:
            angle = data_dict[seed]['angle_vs_bp_stochastic']['all_params'][:]
            angle = scipy.ndimage.uniform_filter1d(angle, size=3) # Smooth with 3-point boxcar average
            angle_all_seeds.append(angle)
        else:
            angle_all_seeds.append(data_dict[seed]['angle_vs_bp']['all_params'])
    avg_angle = np.nanmean(angle_all_seeds, axis=0)
    error = np.nanstd(angle_all_seeds, axis=0)
    train_steps = data_dict[seed]['val_history_train_steps'][:]
    if not stochastic:
        train_steps = train_steps[1:]
    ax.plot(train_steps, avg_angle, label=model_dict["name"], color=model_dict["color"])
    ax.fill_between(train_steps, avg_angle-error, avg_angle+error, alpha=0.5, color=model_dict["color"], linewidth=0)
    ax.hlines(30, -1000, 20000, color='gray', linewidth=0.5, alpha=0.1)
    ax.hlines(60, -1000, 20000, color='gray', linewidth=0.5, alpha=0.1)
    ax.hlines(90, -1000, 20000, color='gray', linewidth=0.5, alpha=0.1)
    ax.set_xlabel('Training step')
    ax.set_ylabel('Angle vs BP')
    ax.set_ylim([0,100])
    ax.set_xlim([-1000,20000])
    ax.set_yticks(np.arange(0, 101, 30))


def plot_angle_FB_all_seeds(data_dict, model_dict, ax):
    # Plot angles between forward weights W vs backward weights B
    fb_angles_all_seeds = []
    for seed in model_dict['seeds']:
        for projection in data_dict[seed]["feedback_weight_angle_history"]:
            fb_angles_all_seeds.append(data_dict[seed]["feedback_weight_angle_history"][projection][:])

    if len(fb_angles_all_seeds) == 0:
        print(f"No feedback weight angles found for {model_dict['name']}")
        return
    avg_angles = np.mean(fb_angles_all_seeds, axis=0)
    std_angles = np.std(fb_angles_all_seeds, axis=0)
    train_steps = data_dict[seed]['val_history_train_steps'][:]
    ax.hlines(30, -1000, 20000, color='gray', linewidth=0.5, alpha=0.1)
    ax.hlines(60, -1000, 20000, color='gray', linewidth=0.5, alpha=0.1)
    ax.hlines(90, -1000, 20000, color='gray', linewidth=0.5, alpha=0.1)
    if np.isnan(avg_angles).any():
        print(f"Warning: NaN values found in avg W vs B angle.")
    else:
        ax.plot(train_steps, avg_angles, color=model_dict['color'], label=model_dict['name'])
        ax.fill_between(train_steps, avg_angles-std_angles, avg_angles+std_angles, alpha=0.5, color=model_dict['color'], linewidth=0)
    ax.set_xlabel('Training step')
    ax.set_ylabel('Angle \n(F vs B weights)')
    # ax.set_ylim(bottom=-2, top=90)
    # ax.set_yticks(np.arange(0, 91, 30))
    # ax.set_xlim([-1000,20000])
    ax.set_xlabel('Training step')
    ax.set_ylim([0,100])
    ax.set_xlim([-1000,20000])
    ax.set_yticks(np.arange(0, 101, 30))


########################################################################################################

def plot_dynamics_example(model_dict_all, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/mnist/", save=None, overwrite=False):
    model_key = "bpLike_WT_hebbdend_eq"
    model_dict = model_dict_all[model_key]
    network_name = model_dict['config'].split('.')[0] + "_dynamics"
    data_file_path = f"data/plot_data_{network_name}.h5"

    # data_file_path = "notebooks/saved_networks/test.h5"

    # Open hdf5 and check if the dynamics data already exists      
    recompute = False
    if not os.path.exists(data_file_path) or overwrite:
        recompute = True

    if recompute:
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
        ut.save_plot_data(network.name, 'retrained_with_dynamics', data_key='dendritic_dynamics_dict', data=dendritic_dynamics_dict, file_path=data_file_path, overwrite=overwrite)
        ut.save_plot_data(network.name, 'retrained_with_dynamics', data_key='param_history_steps', data=network.param_history_steps, file_path=data_file_path, overwrite=overwrite)

    print("Generating figure...")

    fig = plt.figure(figsize=(5.5, 9))
    gs_axes = gs.GridSpec(nrows=2, ncols=1,                        
                       left=0.68,right=0.97,
                       top=0.86, bottom = 0.65,
                       wspace=0.3, hspace=0.5)
    axes = [fig.add_subplot(gs_axes[i,0]) for i in range(2)]
    with h5py.File(data_file_path, 'r') as f:
        data_dict = f[network_name]['retrained_with_dynamics']
        dendritic_dynamics_dict  = data_dict['dendritic_dynamics_dict']
        param_history_steps = data_dict['param_history_steps'][:]

        pt.plot_network_dynamics_example(param_history_steps, dendritic_dynamics_dict, population="H2E", units=[0,7], t=5000, axes=axes)

    if save is not None:
        fig.savefig(f"figures/{save}.png", dpi=300)
        fig.savefig(f"figures/{save}.svg", dpi=300)


def compare_E_properties(model_dict_all, model_list_heatmaps, model_list_metrics, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/mnist/", save=None, overwrite=False):
    '''
    Figure 1: Van_BP vs bpDale(learnedI)
        -> bpDale is more structured/sparse (focus on H1E metrics)

    Compare vanilla Backprop to networks with 'cortical' architecures (i.e. with somatic feedback inhibition). 
    '''

    fig = plt.figure(figsize=(5.5, 9))
    axes = gs.GridSpec(nrows=4, ncols=6,                        
                       left=0.049,right=1,
                       top=0.95, bottom = 0.5,
                       wspace=0.15, hspace=0.5)
    metrics_axes = gs.GridSpec(nrows=4, ncols=4,                        
                       left=0.049,right=0.95,
                       top=0.95, bottom = 0.48,
                       wspace=0.5, hspace=0.8)
    ax_accuracy    = fig.add_subplot(metrics_axes[3, 0])  
    ax_sparsity    = fig.add_subplot(metrics_axes[3, 1])
    ax_selectivity = fig.add_subplot(metrics_axes[3, 2])
    ax_structure   = fig.add_subplot(metrics_axes[3, 3])

    all_models = list(dict.fromkeys(model_list_heatmaps + model_list_metrics))
    generate_data_all_seeds(all_models, model_dict_all, config_path_prefix, saved_network_path_prefix, overwrite=overwrite)

    col = 0
    for model_key in all_models:
        model_dict = model_dict_all[model_key]
        network_name = model_dict['config'].split('.')[0]
        data_file_path = f"data/plot_data_{network_name}.h5"
        with h5py.File(data_file_path, 'r') as f:
            data_dict = f[network_name]
            print(f"Generating plots for {model_dict['name']}")

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
                        ax.set_title(model_dict["name"])
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
                plot_sparsity_all_seeds(data_dict, model_dict, populations_to_plot=['H1E','H2E'], ax=ax_sparsity)
                plot_selectivity_all_seeds(data_dict, model_dict, populations_to_plot=['H1E','H2E'], ax=ax_selectivity)
                plot_structure_all_seeds(data_dict, model_dict, ax=ax_structure)

    if save is not None:
        fig.savefig(f"figures/{save}.png", dpi=300)
        fig.savefig(f"figures/{save}.svg", dpi=300)


def compare_somaI_properties(model_dict_all, model_list_heatmaps, model_list_metrics, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/mnist/", save=None, overwrite=False):
    fig = plt.figure(figsize=(5.5, 9))
    axes = gs.GridSpec(nrows=4, ncols=6,                        
                       left=0.049,right=0.98,
                       top=0.95, bottom = 0.5,
                       wspace=0.15, hspace=0.5)
    
    metrics_axes = gs.GridSpec(nrows=4, ncols=1,                        
                       left=0.6,right=0.78,
                       top=0.95, bottom = 0.5,
                       wspace=0., hspace=0.5)
    ax_accuracy    = fig.add_subplot(metrics_axes[0, 0])  
    ax_sparsity    = fig.add_subplot(metrics_axes[1, 0])
    ax_selectivity = fig.add_subplot(metrics_axes[2, 0])
    col = 0

    all_models = list(dict.fromkeys(model_list_heatmaps + model_list_metrics))
    generate_data_all_seeds(all_models, model_dict_all, config_path_prefix, saved_network_path_prefix, overwrite=overwrite)

    for model_key in all_models:
        model_dict = model_dict_all[model_key]
        config_path = config_path_prefix + model_dict['config']
        pickle_basename = "_".join(model_dict['config'].split('_')[0:-2])
        network_name = model_dict['config'].split('.')[0]
        data_file_path = f"data/plot_data_{network_name}.h5"

        with h5py.File(data_file_path, 'r') as f:
            data_dict = f[network_name]
            
            print(f"Generating plots for {model_dict['name']}")
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
                        ax.set_title(model_dict["name"])
                    if col>0:
                        ax.set_ylabel('')
                        ax.set_yticklabels([])
                col += 1

            if model_key in model_list_metrics:
                plot_accuracy_all_seeds(data_dict, model_dict, ax=ax_accuracy)
                legend = ax_accuracy.legend(ncol=1, bbox_to_anchor=(1., 1.), loc='upper left', fontsize=6)
                for line in legend.get_lines():
                    line.set_linewidth(1.5)
                plot_sparsity_all_seeds(data_dict, model_dict, populations_to_plot=populations_to_plot, ax=ax_sparsity)
                plot_selectivity_all_seeds(data_dict, model_dict, populations_to_plot=populations_to_plot, ax=ax_selectivity)

    if save:
        fig.savefig(f"figures/{save}.png", dpi=300)
        fig.savefig(f"figures/{save}.svg", dpi=300)


def compare_dendI_properties(model_dict_all, model_list_heatmaps, model_list_metrics, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/mnist/", save=None, overwrite=False):
    fig = plt.figure(figsize=(5.5, 9))
    axes = gs.GridSpec(nrows=4, ncols=6,                        
                       left=0.049,right=0.98,
                       top=0.95, bottom = 0.5,
                       wspace=0.15, hspace=0.5)
    
    metrics_axes = gs.GridSpec(nrows=4, ncols=3,
                       left=0.049,right=0.8,
                       top=0.95, bottom = 0.5,
                       wspace=0.3, hspace=0.6)
    ax_sparsity    = fig.add_subplot(metrics_axes[0, 0])
    ax_selectivity = fig.add_subplot(metrics_axes[0, 1])
    ax_accuracy    = fig.add_subplot(metrics_axes[0, 2])
    ax_dendstate   = fig.add_subplot(metrics_axes[1, 2])
    ax_angle       = fig.add_subplot(metrics_axes[2, 2])
    col = 0

    all_models = list(dict.fromkeys(model_list_heatmaps + model_list_metrics))
    generate_data_all_seeds(all_models, model_dict_all, config_path_prefix, saved_network_path_prefix, overwrite=overwrite)

    for model_key in all_models:
        model_dict = model_dict_all[model_key]
        network_name = model_dict['config'].split('.')[0]
        data_file_path = f"data/plot_data_{network_name}.h5"
        with h5py.File(data_file_path, 'r') as f:
            data_dict = f[network_name]
            print(f"Generating plots for {model_dict['name']}")
            seed = model_dict['seeds'][0] # example seed to plot
            populations_to_plot = [population for population in data_dict[seed]['average_pop_activity_dict'] if 'DendI' in population]

            # Plot heatmaps
            if model_key in model_list_heatmaps:
                for row,population in enumerate(populations_to_plot):
                    ## Activity plots: batch accuracy of each population to the test dataset
                    ax = fig.add_subplot(axes[row+1, col])
                    average_pop_activity_dict = data_dict[seed]['average_pop_activity_dict']
                    pt.plot_batch_accuracy_from_data(average_pop_activity_dict, sort=True, population=population, ax=ax, cbar=False)
                    ax.set_yticks([0,average_pop_activity_dict[population].shape[0]-1])
                    ax.set_yticklabels([1,average_pop_activity_dict[population].shape[0]])
                    ax.set_ylabel(f'{population} unit', labelpad=-8)
                    if row==0:
                        ax.set_title(model_dict["name"], pad=3)
                    if col>0:
                        ax.set_ylabel('')
                        ax.set_yticklabels([])
                col += 1

            # Plot metrics
            if model_key in model_list_metrics:                
                plot_accuracy_all_seeds(data_dict, model_dict, ax=ax_accuracy)
                # plot_sparsity_all_seeds(data_dict, model_dict, populations_to_plot=populations_to_plot, ax=ax_sparsity)
                # plot_selectivity_all_seeds(data_dict, model_dict, populations_to_plot=populations_to_plot, ax=ax_selectivity)
                plot_dendritic_state_all_seeds(data_dict, model_dict, ax=ax_dendstate)
                plot_angle_vs_bp_all_seeds(data_dict, model_dict, ax=ax_angle)
            legend = ax_accuracy.legend(ncol=3, bbox_to_anchor=(-2., 1.3), loc='upper left', fontsize=6)
            for line in legend.get_lines():
                line.set_linewidth(1.5)

    if save:
        fig.savefig(f"figures/{save}.png", dpi=300)
        fig.savefig(f"figures/{save}.svg", dpi=300)


def compare_angle_metrics(model_dict_all, model_list1, model_list2, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/mnist/", save=None, overwrite=False):
    fig = plt.figure(figsize=(5.5, 9))
    axes = gs.GridSpec(nrows=3, ncols=3,                        
                       left=0.1,right=0.9,
                       top=0.9, bottom = 0.6,
                       wspace=0.4, hspace=0.5)
    ax_accuracy1 = fig.add_subplot(axes[0,0])
    ax_angle_vs_BP1 = fig.add_subplot(axes[1,0])
    ax_FB_angle1 = fig.add_subplot(axes[2,0])
    ax_accuracy2 = fig.add_subplot(axes[0,1])
    ax_angle_vs_BP2 = fig.add_subplot(axes[1,1])
    ax_FB_angle2 = fig.add_subplot(axes[2,1])

    all_models = list(dict.fromkeys(model_list1 + model_list2))
    generate_data_all_seeds(all_models, model_dict_all, config_path_prefix, saved_network_path_prefix, overwrite=overwrite)

    for i, model_key in enumerate(all_models):
        model_dict = model_dict_all[model_key]
        network_name = model_dict['config'].split('.')[0]
        data_file_path = f"data/plot_data_{network_name}.h5"
        with h5py.File(data_file_path, 'r') as f:
            data_dict = f[network_name]
            print(f"Generating plots for {model_dict['name']}")
            if model_key in model_list1:
                ax_accuracy = ax_accuracy1
                ax_angle_vs_BP = ax_angle_vs_BP1
                ax_FB_angle = ax_FB_angle1
            if model_key in model_list2:
                ax_accuracy = ax_accuracy2
                ax_angle_vs_BP = ax_angle_vs_BP2
                ax_FB_angle = ax_FB_angle2
            plot_accuracy_all_seeds(data_dict, model_dict, ax=ax_accuracy)
            plot_angle_vs_bp_all_seeds(data_dict, model_dict, ax=ax_angle_vs_BP)
            plot_angle_FB_all_seeds(data_dict, model_dict, ax=ax_FB_angle)
    legend = ax_accuracy1.legend(ncol=1, bbox_to_anchor=(-0.1, 1.6), loc='upper left')
    for line in legend.get_lines():
        line.set_linewidth(1.5)
    legend = ax_accuracy2.legend(ncol=1, bbox_to_anchor=(-0.1, 1.6), loc='upper left')
    for line in legend.get_lines():
        line.set_linewidth(1.5)

    if save is not None:
        fig.savefig(f"figures/{save}.png", dpi=300)
        fig.savefig(f"figures/{save}.svg", dpi=300)


def compare_metrics_simple(model_dict_all, model_list, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/mnist/", save=None, overwrite=False):
    fig = plt.figure(figsize=(5.5, 3))
    axes = gs.GridSpec(nrows=2, ncols=3,                        
                       left=0.1,right=0.9,
                       top=0.9, bottom = 0.1,
                       wspace=0.4, hspace=0.5)
    ax_accuracy = fig.add_subplot(axes[1,0])
    ax_dendstate = fig.add_subplot(axes[1,1])
    ax_angle_vs_BP = fig.add_subplot(axes[1,2])

    all_models = list(dict.fromkeys(model_list))
    generate_data_all_seeds(all_models, model_dict_all, config_path_prefix, saved_network_path_prefix, overwrite=overwrite)

    for model_key in all_models:
        model_dict = model_dict_all[model_key]
        config_path = config_path_prefix + model_dict['config']
        pickle_basename = "_".join(model_dict['config'].split('_')[0:-2])
        network_name = model_dict['config'].split('.')[0]
        data_file_path = f"data/plot_data_{network_name}.h5"
        with h5py.File(data_file_path, 'r') as f:
            data_dict = f[network_name]
            print(f"Generating plots for {model_dict['name']}")
            plot_accuracy_all_seeds(data_dict, model_dict, ax=ax_accuracy)
            plot_dendritic_state_all_seeds(data_dict, model_dict, ax=ax_dendstate)
            plot_angle_vs_bp_all_seeds(data_dict, model_dict, ax=ax_angle_vs_BP)

    legend = ax_accuracy.legend(ncol=4, bbox_to_anchor=(-0.1, 1.4), loc='upper left')
    for line in legend.get_lines():
        line.set_linewidth(1.5)

    if save is not None:
        fig.savefig(f"figures/{save}.png", dpi=300)
        fig.savefig(f"figures/{save}.svg", dpi=300)


def generate_metrics_plot(model_dict_all, model_list, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/mnist/", save=None, overwrite=False): 
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
        pickle_basename = "_".join(model_dict['config'].split('_')[0:-2])
        network_name = model_dict['config'].split('.')[0]
        data_file_path = f"data/plot_data_{network_name}.h5"
        for seed in model_dict['seeds']:
            saved_network_path = saved_network_path_prefix + pickle_basename + f"_{seed}_complete.pkl"
            generate_data_hdf5(config_path, saved_network_path, data_file_path, overwrite)
            gc.collect()

    for i,model_key in enumerate(all_models):
        model_dict = model_dict_all[model_key]
        network_name = model_dict['config'].split('.')[0]
        data_file_path = f"data/plot_data_{network_name}.h5"

        with h5py.File(data_file_path, 'r') as f:
            data_dict = f[network_name]

            plot_angle_FB_all_seeds(data_dict, model_dict, ax=ax_FB_angles)
            plot_angle_vs_bp_all_seeds(data_dict, model_dict, ax=ax_angleBP, stochastic=False)
            plot_angle_vs_bp_all_seeds(data_dict, model_dict, ax=ax_angleBP_stoch, stochastic=True)
            plot_accuracy_all_seeds(data_dict, model_dict, ax=ax_accuracy)
            plot_error_all_seeds(data_dict, model_dict, ax=ax_error_hist)
            plot_dendritic_state_all_seeds(data_dict, model_dict, ax=ax_dendstate)

            if 'H1E' in data_dict[seed]['sparsity_history'] and 'H2E' in data_dict[seed]['sparsity_history']:
                plot_sparsity_all_seeds(data_dict, model_dict, populations_to_plot=['H1E','H2E'], ax=ax_sparsity)
                plot_selectivity_all_seeds(data_dict, model_dict, populations_to_plot=['H1E','H2E'], ax=ax_selectivity)
                plot_structure_all_seeds(data_dict, model_dict, ax=ax_structure)

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
                ax_sparsity_hist.plot(val_steps, avg_sparsity, label=f"{model_dict['name']}", color=model_dict["color"])
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
                ax_selectivity_hist.plot(val_steps, avg_selectivity, label=f"{model_dict['name']}", color=model_dict["color"])
                ax_selectivity_hist.fill_between(val_steps, avg_selectivity-std_selectivity, avg_selectivity+std_selectivity, alpha=0.2, color=model_dict["color"], linewidth=0)
                ax_selectivity_hist.set_xlabel('Training step')
                ax_selectivity_hist.set_ylabel('Selectivity')
                ax_selectivity_hist.set_ylim([0,1])

    if save:
        fig.savefig(f"figures/{save}.png", dpi=300)
        fig.savefig(f"figures/{save}.svg", dpi=300)


def generate_spirals_figure(model_dict_all, model_list, config_path_prefix="network_config/spiral/", saved_network_path_prefix="data/spiral/", save=None, overwrite=False):
    fig = plt.figure(figsize=(5.5, 9))
    axes = gs.GridSpec(nrows=2, ncols=4,                        
                       left=0.049,right=1,
                       top=0.95, bottom = 0.5,
                       wspace=0.15, hspace=0.5)
    ax_accuracy    = fig.add_subplot(axes[1, 0])  
    ax_dendstate    = fig.add_subplot(axes[1, 1])
    ax_angleBP = fig.add_subplot(axes[1, 2])

    all_models = list(dict.fromkeys(model_list))
    generate_data_all_seeds(all_models, model_dict_all, config_path_prefix, saved_network_path_prefix, overwrite=overwrite)


def images_to_pdf(image_paths, output_path):
    # Define US Letter page size in points
    letter_size = [8.5 * 72, 11 * 72]  # 612 x 792 points
    fig_size = [5.5 * 72, 9 * 72]  # 5.5 x 9 inches in points
    
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
        caption_x = letter_size[0]*0.35
        caption_y = letter_size[1] - margin_y*0.7
        c.drawString(caption_x, caption_y, caption)
        
        c.showPage()  # Add a new page in the PDF for the next image
    
    c.save()



@click.command()
@click.option('--figure', default=None, help='Figure to generate')
@click.option('--overwrite', is_flag=True, default=False, help='Overwrite existing network data in plot_data.hdf5 file')
@click.option('--generate-data', default=None, help='Generate HDF5 data files for plots')
@click.option('--recompute', default=None, help='Recompute plot data for a particular parameter')

def main(figure, overwrite, generate_data, recompute):
    model_dict_all = {
            ##########################
            # Backprop models
            ##########################
            "vanBP":       {"config": "20231129_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G_complete_optimized.yaml",
                            "color":  "black",
                            "name":   "Vanilla Backprop"},

            "vanBP_0hidden": {"config": "20250103_EIANN_0_hidden_mnist_van_bp_relu_SGD_config_G_complete_optimized.yaml",
                            "color": "black",
                            "name": "Vanilla Backprop 0-hidden"},

            
            "vanBP_fixed_hidden": {"config": "20250108_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G_fixed_hidden_complete_optimized.yaml",
                                   "color": "black",
                                   "name": "Vanilla Backprop fixed hidden"},

            "bpDale_learned":{"config": "20240419_EIANN_2_hidden_mnist_bpDale_relu_SGD_config_F_complete_optimized.yaml",
                            "color":  "blue",
                            "name":   "Backprop + Dale's Law (learned somaI)"},

            "bpDale_fixed":{"config": "20231129_EIANN_2_hidden_mnist_bpDale_relu_SGD_config_G_complete_optimized.yaml",
                            "color":  "cyan",
                            "name":   "Backprop + Dale's Law (fixed somaI)"},

            "bpDale_noI":  {"config": "20240919_EIANN_2_hidden_mnist_bpDale_noI_relu_SGD_config_G_complete_optimized.yaml",
                            "color": "blue",
                            "name": "Dale's Law (no somaI)"},

            ##########################
            # bpLike models
            ##########################
            "bpLike_WT_hebbdend":  {"config": "20241009_EIANN_2_hidden_mnist_BP_like_config_5J_complete_optimized.yaml",
                                    "color": "black",
                                    "name": "bpLike_WT_hebbdend"},

            "bpLike_WT_hebbdend_eq":  {"config": "20240516_EIANN_2_hidden_mnist_BP_like_config_2L_complete_optimized.yaml",
                                    "color":  "red",
                                    "name":   "bpLike_WT_hebbdend_eq"},

            "bpLike_hebbTD_hebbdend":{"config": "20241009_EIANN_2_hidden_mnist_BP_like_config_5J_learn_TD_HWN_1_complete_optimized.yaml",
                                    "color": "blue",
                                    "name": "bpLike_hebbTD_hebbdend"},

            "bpLike_hebbTD_hebbdend_eq":{"config": "20240830_EIANN_2_hidden_mnist_BP_like_config_2L_learn_TD_HWN_3_complete_optimized.yaml",
                                        "color": "magenta",
                                        "name": "bpLike_hebbTD_hebbdend_eq"},

            "bpLike_TCWN_hebbdend": {"config": "20241120_EIANN_2_hidden_mnist_BP_like_config_5J_learn_TD_HTCWN_2_complete_optimized.yaml",
                                   "color": "green",
                                   "name": "bpLike_TCWN_hebbdend"}, # TC with weight norm

            # "bpLike_TC_hebbdend": {"config": "20241114_EIANN_2_hidden_mnist_BP_like_config_5J_learn_TD_HTC_2_complete_optimized.yaml",
            #                        "color": "green",
            #                        "name": "bpLike_TC_hebbdend"},  # TC applied to activity of wrong (bottom-up) unit instead of top-down unit

            "bpLike_WT_localBP":   {"config": "20241113_EIANN_2_hidden_mnist_BP_like_config_5M_complete_optimized.yaml",
                                    "color": "orange",
                                    "name": "bpLike_WT_localBP"},

            "bpLike_WT_localBP_eq":{"config": "20240628_EIANN_2_hidden_mnist_BP_like_config_3M_complete_optimized.yaml",
                                    "color":  "black",
                                    "name":   "bpLike_WT_localBP_eq"},

            "bpLike_WT_fixedDend": {"config": "20241113_EIANN_2_hidden_mnist_BP_like_config_5K_complete_optimized.yaml",
                                    "color":  "gray",
                                    "name":   "bpLike_WT_fixedDend"},

            "bpLike_WT_fixedDend_eq":  {"config": "20240508_EIANN_2_hidden_mnist_BP_like_config_2K_complete_optimized.yaml",
                                        "color":  "gray",
                                        "name":   "bpLike_WT_fixedDend_eq"},

            "bpLike_fixedTD_hebbdend": {"config": "20241114_EIANN_2_hidden_mnist_BP_like_config_5J_fixed_TD_complete_optimized.yaml",
                                        "color": "lightblue",
                                        "name": "bpLike_fixedTD_hebbdend"},

            "bpLike_fixedTD_hebbdend_eq":  {"config": "20240830_EIANN_2_hidden_mnist_BP_like_config_2L_fixed_TD_complete_optimized.yaml",
                                            "color": "lightgray",
                                            "name": "bpLike_fixedTD_hebbdend_eq"},

            ##########################
            # Biological models
            ##########################
            "HebbWN_topsup":       {"config": "20241105_EIANN_2_hidden_mnist_Top_Layer_Supervised_Hebb_WeightNorm_config_7_complete_optimized.yaml",
                                    "color":  "green",
                                    "name":   "Top-supervised HebbWN"}, # bpLike in the top layer

            "Supervised_HebbWN_WT_hebbdend":{"config": "20240714_EIANN_2_hidden_mnist_Supervised_Hebb_WeightNorm_config_4_complete_optimized.yaml",
                                    "color": "olive",
                                    "name": "Supervised_HebbWN_WT_hebbdend"}, 

            "SupHebbTempCont_WT_hebbdend": {"config": "20241125_EIANN_2_hidden_mnist_Hebb_Temp_Contrast_config_2_complete_optimized.yaml",
                                            "color": "purple",
                                            "name": "Supervised Hebb Temp Contrast WT"}, # Like target propagation / temporal contrast on forward dW

            "Supervised_HebbWN_learned_somaI":{"config": "20240919_EIANN_2_hidden_mnist_Supervised_Hebb_WeightNorm_learn_somaI_config_4_complete_optimized.yaml",
                                                "color": "lime",
                                                "name": "Supervised HebbWN learned somaI"},

            "Supervised_BCM_WT_hebbdend":  {"config": "20240723_EIANN_2_hidden_mnist_Supervised_BCM_config_4_complete_optimized.yaml",
                                            "color": "deeppink",
                                            "name": "Supervised BCM"},

            "BTSP_WT_hebbdend":    {"config":"20240604_EIANN_2_hidden_mnist_BTSP_config_3L_complete_optimized.yaml",
                                    "color": "cyan",
                                    "name": "BTSP_WT_hebbdend"}, 

            "BTSP_5L":    {"config":"20241212_EIANN_2_hidden_mnist_BTSP_config_5L_complete_optimized.yaml",
                                    "color": "cyan",
                                    "name": "BTSP_5L"}, 

            "BTSP_5L_TD":    {"config":"20241216_EIANN_2_hidden_mnist_BTSP_config_5L_learn_TD_HTCWN_3_complete_optimized.yaml",
                                    "color": "cyan",
                                    "name": "BTSP_5L_TD"},

            "BTSP_5L_fixedTD":    {"config":"20241216_EIANN_2_hidden_mnist_BTSP_config_5L_fixed_TD_complete_optimized.yaml",
                                    "color": "cyan",
                                    "name": "BTSP_5L_fixedTD"},

            "BTSP_hebbTD_hebbdend": {"config": "20240905_EIANN_2_hidden_mnist_BTSP_config_3L_learn_TD_HWN_3_complete_optimized.yaml",
                                    "color": "magenta",
                                    "name": "BTSP_hebbTD_hebbdend"},

            "BTSP_fixedTD_hebbdend":{"config": "20240923_EIANN_2_hidden_mnist_BTSP_config_3L_fixed_TD_complete_optimized.yaml",
                                    "color": "black",
                                    "name": "BTSP_fixedTD_hebbdend"},

            "BTSP_TCWN_hebbdend": {"config": "20241126_EIANN_2_hidden_mnist_BTSP_config_3L_learn_TD_HTCWN_3_complete_optimized.yaml",
                                    "color": "green",
                                    "name": "BTSP_TCWN_hebbdend"}, # top-down learning with TempContrast+weight norm

            ##########################
            # Spirals dataset models
            ##########################

            "spirals_bpDale_nobias": {"config": "test_spiral_bpDale_learned_bias_complete_optimized.yaml",
                                "color": "black",
                                "name": "Backprop Dale, no bias"},
            
        }

    seeds = ["66049_257","66050_258", "66051_259", "66052_260", "66053_261"]
    for model_key in model_dict_all:
        model_dict_all[model_key]["seeds"] = seeds

    if recompute is not None and generate_data is None:
        generate_data = 'all'
        print(f"Recomputing data for {generate_data}")

    if generate_data is not None:
        config_path_prefix="network_config/mnist/"
        saved_network_path_prefix="data/mnist/"
        if generate_data == 'all':
            model_list = model_dict_all.keys()
        elif isinstance(generate_data, str):
            model_list = [generate_data]
        else:
            model_list = generate_data
        generate_data_all_seeds(model_list, model_dict_all, config_path_prefix, saved_network_path_prefix, overwrite, recompute)


    # Diagrams + example dynamics
    if figure == "fig1":
        figure_name = "dynamics_example_plots"
        plot_dynamics_example(model_dict_all, save=figure_name, overwrite=overwrite)

    # Backprop models
    if figure in ["all", "fig2"]:
        model_list_heatmaps = ["vanBP", "bpDale_learned", "HebbWN_topsup"]
        # model_list_heatmaps = ["bpLike_fixedTD_hebbdend", "bpLike_WT_hebbdend", "bpLike_TCWN_hebbdend"]
        # model_list_heatmaps = ["bpDale_fixed", "bpDale_learned", "bpLike_WT_hebbdend"]
        model_list_metrics = model_list_heatmaps
        figure_name = "Fig2_vanBP_bpDale_hebb"
        compare_E_properties(model_dict_all, model_list_heatmaps, model_list_metrics, save=figure_name, overwrite=overwrite)

    # Analyze somaI selectivity (supplement to Fig.2)
    elif figure in ["all", "S1"]:
        model_list_heatmaps = ["bpDale_learned", "bpDale_fixed", "HebbWN_topsup"]
        model_list_metrics = model_list_heatmaps
        figure_name = "FigS1_somaI"
        compare_somaI_properties(model_dict_all, model_list_heatmaps, model_list_metrics, save=figure_name, overwrite=overwrite)

    # Analyze DendI
    elif figure in ["all", "fig3"]:
        model_list_heatmaps = ["bpLike_WT_fixedDend", "bpLike_WT_localBP", "bpLike_WT_hebbdend"]
        model_list_metrics = ["bpDale_fixed", "bpDale_learned", "bpLike_WT_hebbdend"]
        # model_list_metrics = model_list_heatmaps #+ ["bpDale_fixed"]
        figure_name = "Fig3_dendI"
        compare_dendI_properties(model_dict_all, model_list_heatmaps, model_list_metrics, save=figure_name, overwrite=overwrite)

    # S2 (Supplement to Fig.3)
    elif figure in ["all", "S2"]:
        model_list_heatmaps = ["bpLike_WT_fixedDend", "bpLike_WT_localBP", "bpLike_WT_hebbdend"]
        model_list_metrics = model_list_heatmaps
        figure_name = "FigS2_Ecells_bpLike"
        compare_E_properties(model_dict_all, model_list_heatmaps, model_list_metrics, save=figure_name, overwrite=overwrite)

    # Biological learning rules (with WT/good gradients)
    elif figure in ["all","fig4"]:
        model_list = ["bpLike_WT_hebbdend", "SupHebbTempCont_WT_hebbdend", "Supervised_BCM_WT_hebbdend", "BTSP_WT_hebbdend"]
        figure_name = "Fig4_BTSP_BCM_HebbWN"
        compare_metrics_simple(model_dict_all, model_list, save=figure_name, overwrite=overwrite)

        # add diagrams (BCM, temp cont, btsp)

    # Supplement to Fig.4: "Supervised_HebbWN_WT_hebbdend" -> Performs badly because HWN is potentiation-only
        
        
    # Representations in bio-learning rules (Supplement to Fig.4)
    elif figure in ["all", "S3"]:
        pass

    # Forward (W) vs backward (B) alignment angle
    elif figure in ["all", "fig5"]:
        model_list1 = ["bpLike_WT_hebbdend", "bpLike_fixedTD_hebbdend", "bpLike_hebbTD_hebbdend", "bpLike_TCWN_hebbdend"]
        # model_list2 = ["bpLike_WT_hebbdend_eq", "bpLike_fixedTD_hebbdend_eq", "bpLike_hebbTD_hebbdend_eq"]
        model_list2 = ["BTSP_WT_hebbdend", "BTSP_hebbTD_hebbdend", "BTSP_fixedTD_hebbdend", "BTSP_TCWN_hebbdend"]
        figure_name = "Fig5_WB_alignment_FA_bpLike_BTSP"
        compare_angle_metrics(model_dict_all, model_list1, model_list2, save=figure_name, overwrite=overwrite)
        # add dendstate

    elif figure in ["all", "metrics"]:
        # model_list = ["vanBP", "bpDale_learned", "bpLike_fixedDend", "bpLike_hebbdend", "bpLike_hebbTD", "bpLike_FA"]
        # model_list = ["BTSP_WT_hebbdend", "BTSP_hebbTD_hebbdend", "BTSP_fixedTD_hebbdend"]
        # model_list = ["bpDale_learned", "bpLike_TCWN_hebbdend"]

        # model_list = ["bpLike_hebbTD_hebbdend_eq", "bpLike_WT_hebbdend_eq", "bpLike_hebbTD_hebbdend", "bpLike_WT_hebbdend"]
        figure_name = "metrics_all_models"
        generate_metrics_plot(model_dict_all, model_list, save=figure_name, overwrite=overwrite)


    # Combine figures into one PDF
    directory = "figures/"
    image_paths = [os.path.join(directory, figure) for figure in os.listdir(directory) if figure.endswith('.png') and figure.startswith('Fig')]
    image_paths.sort()
    images_to_pdf(image_paths=image_paths, output_path= directory+"all_figures.pdf")


if __name__=="__main__":
    main()
