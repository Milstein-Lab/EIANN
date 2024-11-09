import torch
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import os
import h5py
import click
import gc
import copy
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os

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

Supplement: Dend state + soma/dendI representations + angle vs BP for bio learning rule

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
        angles = ut.compute_dW_angles_vs_BP(bpClone_network.predicted_dParam_history, bpClone_network.actual_dParam_history, plot=True)
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
        steps, binned_attr_history_dict = ut.get_binned_mean_population_attribute_history_dict(network, attr_name="forward_dendritic_state", bin_size=100, abs=True)
        if binned_attr_history_dict is not None:
            ut.save_plot_data(network.name, network.seed, data_key='binned_mean_forward_dendritic_state', data=binned_attr_history_dict, file_path=data_file_path, overwrite=overwrite)
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


def generate_single_model_figure(model_dict, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/mnist/", save=True, overwrite=False):

    # Load data
    config_path = config_path_prefix + model_dict['config']
    pickle_basename = "_".join(model_dict['config'].split('_')[0:-2])
    network_name = model_dict['config'].split('.')[0]
    data_file_path = f"data/plot_data_{network_name}.h5"

    for seed in model_dict['seeds']:
        saved_network_path = saved_network_path_prefix + pickle_basename + f"_{seed}_complete.pkl"
        generate_data_hdf5(config_path, saved_network_path, data_file_path, overwrite)
    
    seed = model_dict['seeds'][0]

    # Plot figure
    fig = plt.figure(figsize=(5.5, 9))
    fig.suptitle(network_name, fontsize=12, y=1)

    axes = gs.GridSpec(nrows=6, ncols=3,
                        left=0.07,right=0.9,
                        top=0.96, bottom = 0.07,
                        wspace=0.3, hspace=0.8)

    axes_metrics = gs.GridSpec(nrows=6, ncols=3,
                    left=0.06,right=0.9,
                    top=0.96, bottom = 0.04,
                    wspace=0.3, hspace=0.6)
    ax_loss = fig.add_subplot(axes_metrics[0, 2])
    ax_accuracy = fig.add_subplot(axes_metrics[1, 2])
    ax_sparsity = fig.add_subplot(axes_metrics[2, 2])
    ax_selectivity = fig.add_subplot(axes_metrics[3, 2])
    ax_structure = fig.add_subplot(axes_metrics[4, 2])
    ax_discriminability = fig.add_subplot(axes_metrics[5, 2])

    with h5py.File(data_file_path, 'r') as file:
        data_dict = file[network_name]
    
        val_steps = data_dict[seed]['val_history_train_steps'][:]
        test_loss = data_dict[seed]['test_loss_history'][:]
        ax_loss.plot(val_steps, test_loss)
        ax_loss.set_xlabel('Training step')
        ax_loss.set_ylabel('Loss')

        test_accuracy = data_dict[seed]['test_accuracy_history'][:]
        ax_accuracy.plot(val_steps, test_accuracy)
        ax_accuracy.set_xlabel('Training step')
        ax_accuracy.set_ylabel('Accuracy')

        populations_to_plot = [population for population in data_dict[seed]['average_pop_activity_dict'] if population!='InputE']

        for i,population in enumerate(populations_to_plot):
            print(f"Plotting {population}")

            # Column 1: Average population activity (batch accuracy to the test dataset)
            ax = fig.add_subplot(axes[i, 0])
            pt.plot_batch_accuracy_from_data(data_dict[seed]['average_pop_activity_dict'], population=population, ax=ax)
            if i == 0:
                ax.set_title('Class-averaged unit activities')

            # Column 2: Example output receptive fields
            receptive_fields = torch.tensor(data_dict[seed][f"maxact_receptive_fields_{population}"])
            ax = fig.add_subplot(axes[i, 1])
            ax.axis('off')
            pos = ax.get_position()

            if receptive_fields.shape[0] > 20:
                num_units = 20
                new_left = pos.x0 - 0.02  # Move left boundary to the left
                new_bottom = pos.y0
                new_height = pos.height
                ax.set_position([new_left, new_bottom, pos.width, new_height])
                rf_axes = gs.GridSpecFromSubplotSpec(4, 5, subplot_spec=ax, wspace=0.1, hspace=0.1)
                ax_list = []
                for j in range(num_units):
                    ax = fig.add_subplot(rf_axes[j])
                    ax_list.append(ax)
                    box = matplotlib.patches.Rectangle((-0.5,-0.5), 28, 28, linewidth=0.5, edgecolor='k', facecolor='none', zorder=10)
                    ax.add_patch(box)
                # preferred_classes = torch.argmax(torch.tensor(data_dict[seed]['average_pop_activity_dict'][population]), dim=1)
                im = pt.plot_receptive_fields(receptive_fields, sort=True, ax_list=ax_list, preferred_classes=None)
            else:
                num_units = 10
                new_left = pos.x0 - 0.02  # Move left boundary to the left
                new_bottom = pos.y0 + 0.04 # Move bottom boundary up
                new_height = pos.height - 0.06  # Decrease height
                ax.set_position([new_left, new_bottom, pos.width, new_height])
                rf_axes = gs.GridSpecFromSubplotSpec(2, 5, subplot_spec=ax, wspace=0.1, hspace=0.1)
                ax_list = []
                for j in range(num_units):
                    ax = fig.add_subplot(rf_axes[j])
                    ax_list.append(ax)
                    box = matplotlib.patches.Rectangle((-0.5,-0.5), 28, 28, linewidth=0.5, edgecolor='k', facecolor='none', zorder=10)
                    ax.add_patch(box)
                im = pt.plot_receptive_fields(receptive_fields, sort=False, ax_list=ax_list)
            if i == 0:
                ax.set_title('Receptive fields')
                fig_width, fig_height = fig.get_size_inches()
                cax = fig.add_axes([ax_list[0].get_position().x0, ax.get_position().y0-0.1/fig_height,
                                    0.1, 0.05/fig_height])
                fig.colorbar(im, cax=cax, orientation='horizontal')

            # Column 3: Learning curves / metrics
            sparsity_all_seeds = [data_dict[seed][f"metrics_dict_{population}"]['sparsity'] for seed in data_dict]
            pt.plot_cumulative_distribution(sparsity_all_seeds, ax=ax_sparsity, label=population)
            ax_sparsity.set_ylabel('Fraction of patterns')
            ax_sparsity.set_xlabel('Sparsity') # \n(1 - fraction of units active)')
            ax_sparsity.legend()

            selectivity_all_seeds = [data_dict[seed][f"metrics_dict_{population}"]['selectivity'] for seed in data_dict]
            pt.plot_cumulative_distribution(selectivity_all_seeds, ax=ax_selectivity, label=population)
            ax_selectivity.set_ylabel('Fraction of units')
            ax_selectivity.set_xlabel('Selectivity') # \n(1 - fraction of active patterns)')

            if receptive_fields is not None:
                structure_all_seeds = [data_dict[seed][f"metrics_dict_{population}"]['structure'] for seed in data_dict]
                pt.plot_cumulative_distribution(structure_all_seeds, ax=ax_structure, label=population)
                ax_structure.set_ylabel('Fraction of units')
                ax_structure.set_xlabel("Structure") # \n(Moran's I spatial autocorrelation)")
            else:
                ax_structure.axis('off')

            discriminability_all_seeds = [data_dict[seed][f"metrics_dict_{population}"]['discriminability'] for seed in data_dict]
            pt.plot_cumulative_distribution(discriminability_all_seeds, ax=ax_discriminability, label=population)
            ax_discriminability.set_ylabel('Fraction of pattern pairs')
            ax_discriminability.set_xlabel('Discriminability') # \n(1 - cosine similarity)')

    print(f"Saving figure png/svg files")
    if save:
        fig.savefig(f"figures/{network_name}.png", dpi=300)
        fig.savefig(f"figures/{network_name}.svg", dpi=300)


########################################################################################################


def compare_E_properties(model_dict_all, model_list_heatmaps, model_list_metrics, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/mnist/", save=None, overwrite=False):
    '''
    Figure 1: Van_BP vs bpDale(learnedI)
        -> bpDale is more structured/sparse (focus on H1E metrics)

    Compare vanilla Backprop to networks with 'cortical' architecures (i.e. with somatic feedback inhibition). 
    '''

    fig = plt.figure(figsize=(5.5, 9))
    axes = gs.GridSpec(nrows=4, ncols=6,                        
                       left=0.049,right=1,
                       top=0.75, bottom = 0.3,
                       wspace=0.15, hspace=0.5)
    metrics_axes = gs.GridSpec(nrows=4, ncols=4,                        
                       left=0.049,right=1,
                       top=0.75, bottom = 0.28,
                       wspace=0.5, hspace=0.8)
    ax_accuracy    = fig.add_subplot(metrics_axes[3, 0])  
    ax_structure   = fig.add_subplot(metrics_axes[3, 1])
    ax_sparsity    = fig.add_subplot(metrics_axes[3, 2])
    ax_selectivity = fig.add_subplot(metrics_axes[3, 3])

    # fig = plt.figure(figsize=(8, 9))
    # axes = gs.GridSpec(nrows=4, ncols=6,                        
    #                    left=0.049,right=0.7,
    #                    top=0.75, bottom = 0.3,
    #                    wspace=0.15, hspace=0.5)
    # metrics_axes = gs.GridSpec(nrows=4, ncols=2,                        
    #                    left=0.7,right=1,
    #                    top=0.75, bottom = 0.3,
    #                    wspace=0.5, hspace=0.8)    
    # ax_accuracy    = fig.add_subplot(metrics_axes[1, 0])  
    # ax_structure   = fig.add_subplot(metrics_axes[1, 1])
    # ax_sparsity    = fig.add_subplot(metrics_axes[2, 0])
    # ax_selectivity = fig.add_subplot(metrics_axes[2, 1])    

    all_models = list(dict.fromkeys(model_list_heatmaps + model_list_metrics))
    
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

    col = 0
    for i, model_key in enumerate(all_models):
        model_dict = model_dict_all[model_key]
        config_path = config_path_prefix + model_dict['config']
        pickle_basename = "_".join(model_dict['config'].split('_')[0:-2])
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
                # Learning curves / metrics
                accuracy_all_seeds = [data_dict[seed]['test_accuracy_history'] for seed in data_dict]
                avg_accuracy = np.mean(accuracy_all_seeds, axis=0)
                error = np.std(accuracy_all_seeds, axis=0)
                val_steps = data_dict[seed]['val_history_train_steps'][:]
                ax_accuracy.plot(val_steps, avg_accuracy, label=model_dict["name"], color=model_dict["color"])
                ax_accuracy.fill_between(val_steps, avg_accuracy-error, avg_accuracy+error, alpha=0.2, color=model_dict["color"], linewidth=0)
                ax_accuracy.set_xlabel('Training step')
                ax_accuracy.set_ylabel('Test accuracy (%)', labelpad=-2)
                ax_accuracy.set_ylim([0,100])
                ax_accuracy.legend(handlelength=1, ncol=3, bbox_to_anchor=(-0.3, 1.5), loc='upper left', fontsize=6)
                legend = ax_accuracy.legend(ncol=3, bbox_to_anchor=(-0.3, 1.5), loc='upper left', fontsize=6)
                for line in legend.get_lines():
                    line.set_linewidth(1.5)

                sparsity_all_seeds = []
                for seed in data_dict:
                    sparsity_one_seed = []
                    for population in ['H1E', 'H2E']:
                        sparsity_one_seed.extend(data_dict[seed][f"metrics_dict_{population}"]['sparsity'])
                    sparsity_all_seeds.append(sparsity_one_seed)
                avg_sparsity_per_seed = [np.mean(x) for x in sparsity_all_seeds]
                avg_sparsity = np.mean(avg_sparsity_per_seed)
                error = np.std(avg_sparsity_per_seed)
                ax_sparsity.bar(i, avg_sparsity, yerr=error, color=model_dict["color"], width=0.4, label=model_dict["name"])
                ax_sparsity.set_ylabel('Sparsity')
                ax_sparsity.set_ylim([0,1])
                ax_sparsity.set_xticks([0,1,2])

                pt.plot_cumulative_distribution(sparsity_all_seeds, ax=ax_sparsity, label=model_dict["name"], color=model_dict["color"])
                ax_sparsity.set_ylabel('Fraction of patterns')
                ax_sparsity.set_xlabel('Sparsity') # \n(1 - fraction of units active)')
                ax_sparsity.set_xlim([0,1])

                selectivity_all_seeds = []
                for seed in data_dict:
                    selectivity_one_seed = []
                    for population in ['H1E', 'H2E']:
                        selectivity_one_seed.extend(data_dict[seed][f"metrics_dict_{population}"]['selectivity'])
                    selectivity_all_seeds.append(selectivity_one_seed)
                avg_selectivity_per_seed = [np.mean(x) for x in selectivity_all_seeds]
                avg_selectivity = np.mean(avg_selectivity_per_seed)
                error = np.std(avg_selectivity_per_seed)
                ax_selectivity.bar(i, avg_selectivity, yerr=error, color=model_dict["color"], width=0.4, label=model_dict["name"])
                ax_selectivity.set_ylabel('Selectivity')
                ax_selectivity.set_ylim([0,1])
                ax_selectivity.set_xticks([0,1,2])

                pt.plot_cumulative_distribution(selectivity_all_seeds, ax=ax_selectivity, label=model_dict["name"], color=model_dict["color"])
                ax_selectivity.set_ylabel('Fraction of units')
                ax_selectivity.set_xlabel('Selectivity') # \n(1 - fraction of active patterns)')
                ax_selectivity.set_xlim([0,1])

                receptive_fields = data_dict[seed][f"maxact_receptive_fields_{population}"]
                if receptive_fields is not None:
                    structure_all_seeds = []
                    for seed in data_dict:
                        structure_one_seed = []
                        for population in ['H1E', 'H2E']:
                            structure_one_seed.extend(data_dict[seed][f"metrics_dict_{population}"]['structure'])
                        structure_all_seeds.append(structure_one_seed)
                    # avg_structure_per_seed = [np.mean(x) for x in structure_all_seeds]
                    # avg_structure = np.mean(avg_structure_per_seed)
                    # error = np.std(avg_structure_per_seed)
                    # ax_structure.bar(i, avg_structure, yerr=error, color=model_dict["color"], width=0.4, label=model_dict["name"])
                    # ax_structure.set_ylabel('Structure')
                    # ax_structure.set_ylim([0,1])
                    # ax_structure.set_xticks([0,1,2])

                    pt.plot_cumulative_distribution(structure_all_seeds, ax=ax_structure, label=model_dict["name"], color=model_dict["color"])
                    ax_structure.set_ylabel('Fraction of units')
                    ax_structure.set_xlabel("Structure (Moran's I)") # \n(Moran's I spatial autocorrelation)")
                    ax_structure.set_xlim([0,1])
                else:
                    ax_structure.axis('off')

            # ax_sparsity.set_xticklabels([model_key for model_key in model_list_metrics], fontsize=6, rotation=45, ha='right', rotation_mode='anchor')
            # ax_selectivity.set_xticklabels([model_key for model_key in model_list_metrics], fontsize=6, rotation=45, ha='right', rotation_mode='anchor')
            # ax_structure.set_xticklabels([model_key for model_key in model_list_metrics], fontsize=6, rotation=45, ha='right', rotation_mode='anchor')


    if save is not None:
        fig.savefig(f"figures/{save}.png", dpi=300)
        fig.savefig(f"figures/{save}.svg", dpi=300)


def compare_somaI_properties(model_dict_all, model_list_heatmaps, model_list_metrics, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/mnist/", save=None, overwrite=False):

    fig = plt.figure(figsize=(5.5, 9))
    axes = gs.GridSpec(nrows=4, ncols=6,                        
                       left=0.049,right=1,
                       top=0.7, bottom = 0.25,
                       wspace=0.15, hspace=0.5)
    
    metrics_axes = gs.GridSpec(nrows=4, ncols=1,                        
                       left=0.6,right=0.78,
                       top=0.7, bottom = 0.25,
                       wspace=0., hspace=0.5)
    ax_accuracy    = fig.add_subplot(metrics_axes[0, 0])  
    ax_sparsity    = fig.add_subplot(metrics_axes[1, 0])
    ax_selectivity = fig.add_subplot(metrics_axes[2, 0])
    col = 0

    all_models = list(dict.fromkeys(model_list_heatmaps + model_list_metrics))

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
                ## Learning curves / metrics
                accuracy_all_seeds = [data_dict[seed]['val_accuracy_history'] for seed in data_dict]
                avg_accuracy = np.mean(accuracy_all_seeds, axis=0)
                error = np.std(accuracy_all_seeds, axis=0)
                val_steps = data_dict[seed]['val_history_train_steps'][:]
                ax_accuracy.plot(val_steps, avg_accuracy, label=model_dict["name"], color=model_dict["color"])
                ax_accuracy.fill_between(val_steps, avg_accuracy-error, avg_accuracy+error, alpha=0.2, color=model_dict["color"], linewidth=0)
                ax_accuracy.set_xlabel('Training step')
                ax_accuracy.set_ylabel('Accuracy', labelpad=-2)
                ax_accuracy.set_ylim([0,100])
                legend = ax_accuracy.legend(ncol=1, bbox_to_anchor=(1., 1.), loc='upper left', fontsize=6)
                for line in legend.get_lines():
                    line.set_linewidth(1.5)  # Adjust linewidth in the legend


                sparsity_all_seeds = []
                for seed in data_dict:
                    sparsity_one_seed = []
                    for population in populations_to_plot:
                        sparsity_one_seed.extend(data_dict[seed][f"metrics_dict_{population}"]['sparsity'])
                    sparsity_all_seeds.append(sparsity_one_seed)
                pt.plot_cumulative_distribution(sparsity_all_seeds, ax=ax_sparsity, label=model_dict["name"], color=model_dict["color"])
                ax_sparsity.set_ylabel('Fraction of patterns')
                ax_sparsity.set_xlabel('Sparsity') # \n(1 - fraction of units active)')

                selectivity_all_seeds = []
                for seed in data_dict:
                    selectivity_one_seed = []
                    for population in populations_to_plot:
                        selectivity_one_seed.extend(data_dict[seed][f"metrics_dict_{population}"]['selectivity'])
                    selectivity_all_seeds.append(selectivity_one_seed)
                pt.plot_cumulative_distribution(selectivity_all_seeds, ax=ax_selectivity, label=model_dict["name"], color=model_dict["color"])
                ax_selectivity.set_ylabel('Fraction of units')
                ax_selectivity.set_xlabel('Selectivity') # \n(1 - fraction of active patterns)')


    if save:
        fig.savefig(f"figures/{save}.png", dpi=300)
        fig.savefig(f"figures/{save}.svg", dpi=300)


def compare_dendI_properties(model_dict_all, model_list_heatmaps, model_list_metrics, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/mnist/", save=None, overwrite=False):

    fig = plt.figure(figsize=(5.5, 9))
    axes = gs.GridSpec(nrows=4, ncols=6,                        
                       left=0.049,right=1,
                       top=0.7, bottom = 0.25,
                       wspace=0.15, hspace=0.5)
    
    metrics_axes = gs.GridSpec(nrows=4, ncols=3,
                       left=0.049,right=0.8,
                       top=0.7, bottom = 0.25,
                       wspace=0.3, hspace=0.6)
    ax_sparsity    = fig.add_subplot(metrics_axes[1, 0])
    ax_selectivity = fig.add_subplot(metrics_axes[1, 1])

    ax_accuracy    = fig.add_subplot(metrics_axes[1, 2])
    ax_dendstate   = fig.add_subplot(metrics_axes[2, 2])
    ax_angle       = fig.add_subplot(metrics_axes[3, 2])
    col = 0

    all_models = list(dict.fromkeys(model_list_heatmaps + model_list_metrics))

    # for model_key in all_models:
    #     model_dict = model_dict_all[model_key]
    #     config_path = config_path_prefix + model_dict['config']
    #     pickle_basename = "_".join(model_dict['config'].split('_')[0:-2])
    #     network_name = model_dict['config'].split('.')[0]
    #     data_file_path = f"data/plot_data_{network_name}.h5"
    #     for seed in model_dict['seeds']:
    #         saved_network_path = saved_network_path_prefix + pickle_basename + f"_{seed}_complete.pkl"
    #         generate_data_hdf5(config_path, saved_network_path, data_file_path, overwrite)
    #         gc.collect()
        
    for model_key in all_models:
        model_dict = model_dict_all[model_key]
        network_name = model_dict['config'].split('.')[0]
        data_file_path = f"data/plot_data_{network_name}.h5"

        with h5py.File(data_file_path, 'r') as f:
            data_dict = f[network_name]

            print(f"Generating plots for {model_dict['name']}")
            seed = model_dict['seeds'][0] # example seed to plot
            populations_to_plot = [population for population in data_dict[seed]['average_pop_activity_dict'] if 'DendI' in population]

            if model_key in model_list_heatmaps:
                for row,population in enumerate(populations_to_plot):
                    ## Activity plots: batch accuracy of each population to the test dataset
                    ax = fig.add_subplot(axes[row+2, col])
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

            if model_key in model_list_metrics:
                # Learning curves / metrics]
                accuracy_all_seeds = [data_dict[seed]['test_accuracy_history'] for seed in model_dict['seeds']]
                avg_accuracy = np.mean(accuracy_all_seeds, axis=0)
                error = np.std(accuracy_all_seeds, axis=0)
                train_steps = data_dict[seed]['val_history_train_steps'][:]
                ax_accuracy.plot(train_steps, avg_accuracy, label=model_dict["name"], color=model_dict["color"])
                ax_accuracy.fill_between(train_steps, avg_accuracy-error, avg_accuracy+error, alpha=0.2, color=model_dict["color"], linewidth=0)
                ax_accuracy.set_xlabel('Training step')
                ax_accuracy.set_ylabel('Accuracy', labelpad=-2)
                ax_accuracy.set_ylim([0,100])
                legend = ax_accuracy.legend(ncol=3, bbox_to_anchor=(-2., 1.3), loc='upper left', fontsize=6)
                for line in legend.get_lines():
                    line.set_linewidth(1.5)  # Adjust linewidth in the legend
                
                sparsity_all_seeds = []
                for seed in model_dict['seeds']:
                    sparsity_one_seed = []
                    for population in populations_to_plot:
                        sparsity_one_seed.extend(data_dict[seed][f"metrics_dict_{population}"]['sparsity'])
                    sparsity_all_seeds.append(sparsity_one_seed)
                pt.plot_cumulative_distribution(sparsity_all_seeds, ax=ax_sparsity, label=model_dict["name"], color=model_dict["color"])
                ax_sparsity.set_ylabel('Fraction of patterns')
                ax_sparsity.set_xlabel('Sparsity') # \n(1 - fraction of units active)')
                ax_sparsity.set_xlim([0,1])

                selectivity_all_seeds = []
                for seed in model_dict['seeds']:
                    selectivity_one_seed = []
                    for population in populations_to_plot:
                        selectivity_one_seed.extend(data_dict[seed][f"metrics_dict_{population}"]['selectivity'])
                    selectivity_all_seeds.append(selectivity_one_seed)
                pt.plot_cumulative_distribution(selectivity_all_seeds, ax=ax_selectivity, label=model_dict["name"], color=model_dict["color"])
                ax_selectivity.set_ylabel('Fraction of units')
                ax_selectivity.set_xlabel('Selectivity') # \n(1 - fraction of active patterns)')
                ax_selectivity.set_xlim([0,1])
                
                dendstate_all_seeds = []
                for seed in model_dict['seeds']:
                    dendstate_one_seed = data_dict[seed]['binned_mean_forward_dendritic_state']['all'][:]
                    dendstate_all_seeds.append(dendstate_one_seed)
                avg_dendstate = np.mean(dendstate_all_seeds, axis=0)
                error = np.std(dendstate_all_seeds, axis=0)
                binned_mean_forward_dendritic_state_steps = data_dict[seed]['binned_mean_forward_dendritic_state_steps'][:]
                ax_dendstate.plot(binned_mean_forward_dendritic_state_steps, avg_dendstate, label=model_dict["name"], color=model_dict["color"])
                ax_dendstate.fill_between(binned_mean_forward_dendritic_state_steps, avg_dendstate-error, avg_dendstate+error, alpha=0.5, color=model_dict["color"], linewidth=0)
                ax_dendstate.set_xlabel('Training step')
                ax_dendstate.set_ylabel('Dendritic state')
                ax_dendstate.set_ylim(bottom=-0.0, top=0.06)

                angle_all_seeds = []
                for seed in model_dict['seeds']:
                    angle_all_seeds.append(data_dict[seed]['angle_vs_bp']['all_params'])
                avg_angle = np.mean(angle_all_seeds, axis=0)
                error = np.std(angle_all_seeds, axis=0)
                ax_angle.plot(avg_angle, label=model_dict["name"], color=model_dict["color"])

                ax_angle.plot(train_steps[1:], avg_angle, label=model_dict["name"], color=model_dict["color"])
                ax_angle.fill_between(train_steps[1:], avg_angle-error, avg_angle+error, alpha=0.5, color=model_dict["color"], linewidth=0)
                ax_angle.set_xlabel('Training step')
                ax_angle.set_ylabel('Angle vs BP')
                ax_angle.set_ylim([40,100])


    if save:
        fig.savefig(f"figures/{save}.png", dpi=300)
        fig.savefig(f"figures/{save}.svg", dpi=300)


def compare_angle_metrics(model_dict_all, model_list1, model_list2, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/mnist/", save=None, overwrite=False):
    fig = plt.figure(figsize=(5.5, 9))
    axes = gs.GridSpec(nrows=3, ncols=3,                        
                       left=0.1,right=0.9,
                       top=0.75, bottom = 0.3,
                       wspace=0.15, hspace=0.5)
    ax_accuracy1 = fig.add_subplot(axes[0,0])
    ax_angle_vs_BP1 = fig.add_subplot(axes[1,0])
    ax_FB_angle1 = fig.add_subplot(axes[2,0])
    ax_accuracy2 = fig.add_subplot(axes[0,1])
    ax_angle_vs_BP2 = fig.add_subplot(axes[1,1])
    ax_FB_angle2 = fig.add_subplot(axes[2,1])

    all_models = list(dict.fromkeys(model_list1 + model_list2))
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

    for i, model_key in enumerate(all_models):
        model_dict = model_dict_all[model_key]
        config_path = config_path_prefix + model_dict['config']
        pickle_basename = "_".join(model_dict['config'].split('_')[0:-2])
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
                
            # Plot accuracy
            accuracy_all_seeds = [data_dict[seed]['test_accuracy_history'] for seed in data_dict]
            avg_accuracy = np.mean(accuracy_all_seeds, axis=0)
            error = np.std(accuracy_all_seeds, axis=0)
            train_steps = data_dict[seed]['val_history_train_steps'][:]
            ax_accuracy.plot(train_steps, avg_accuracy, label=model_dict["name"], color=model_dict["color"])
            ax_accuracy.fill_between(train_steps, avg_accuracy-error, avg_accuracy+error, alpha=0.2, color=model_dict["color"], linewidth=0)
            ax_accuracy.set_xlabel('Training step')
            ax_accuracy.set_ylabel('Test accuracy (%)', labelpad=-2)
            ax_accuracy.set_ylim([0,100])
            ax_accuracy.legend(handlelength=1, ncol=3, bbox_to_anchor=(-0.3, 1.5), loc='upper left', fontsize=6)
            legend = ax_accuracy.legend(ncol=3, bbox_to_anchor=(-0.3, 1.5), loc='upper left', fontsize=6)
            for line in legend.get_lines():
                line.set_linewidth(1.5)

            # Plot angle vs BP
            angle_all_seeds = []
            from scipy.ndimage import gaussian_filter1d
            for seed in model_dict['seeds']:
                angle = data_dict[seed]['angle_vs_bp_stochastic']['all_params']
                sigma = 1
                smoothed_angle = gaussian_filter1d(angle, sigma)
                angle_all_seeds.append(smoothed_angle)
            avg_angle = np.nanmean(angle_all_seeds, axis=0)
            error = np.nanstd(angle_all_seeds, axis=0)
            ax_angle_vs_BP.plot(train_steps, avg_angle, label=model_dict["name"], color=model_dict["color"])
            ax_angle_vs_BP.fill_between(train_steps, avg_angle-error, avg_angle+error, alpha=0.5, color=model_dict["color"], linewidth=0)
            ax_angle_vs_BP.set_xlabel('Training step')
            ax_angle_vs_BP.set_ylabel('Angle vs BP (stoch.)')
            ax_angle_vs_BP.set_ylim([0,90])
            ax_angle_vs_BP.set_yticks(np.arange(0, 101, 30))
            # ax_angle_vs_BP.set_xlim([0,20000])

            # Plot angles: forward weights W vs backward weights B
            fb_angles_all_seeds = []
            for seed in model_dict['seeds']:
                for projection in data_dict[seed]["feedback_weight_angle_history"]:
                    fb_angles_all_seeds.append(data_dict[seed]["feedback_weight_angle_history"][projection][:])
            avg_angles = np.mean(fb_angles_all_seeds, axis=0)
            std_angles = np.std(fb_angles_all_seeds, axis=0)
            if np.isnan(avg_angles).any():
                print(f"Warning: NaN values found in avg_angles for {network_name}.")
            else:
                ax_FB_angle.plot(train_steps, avg_angles, color=model_dict['color'], label=model_dict['name'])
                ax_FB_angle.fill_between(train_steps, avg_angles-std_angles, avg_angles+std_angles, alpha=0.5, color=model_dict['color'], linewidth=0)
            ax_FB_angle.set_xlabel('Training step')
            ax_FB_angle.set_ylim(bottom=-2, top=90)
            ax_FB_angle.set_yticks(np.arange(0, 91, 30))
            ax_FB_angle.set_ylabel('Angle \n(F vs B weights)')


    if save is not None:
        fig.savefig(f"figures/{save}.png", dpi=300)
        fig.savefig(f"figures/{save}.svg", dpi=300)




def generate_metrics_plot(model_dict_all, model_list, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/mnist/", save=None, overwrite=False): 
    fig = plt.figure(figsize=(5.5, 4))
    # fig = plt.figure(figsize=(11, 8))
    # fig = plt.figure(figsize=(23.1, 16.8))
    fig = plt.figure(figsize=(7, 4))

    # axes = gs.GridSpec(nrows=4, ncols=4, figure=fig, bottom=0.1, top=0.9, left=0.1, right=0.99, hspace=0.5, wspace=0.5)
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

            # Forward vs backward angles
            fb_angles_all_seeds = []
            for seed in model_dict['seeds']:
                train_steps = data_dict[seed]['val_history_train_steps'][:]
                for projection in data_dict[seed]["feedback_weight_angle_history"]:
                    fb_angles_all_seeds.append(data_dict[seed]["feedback_weight_angle_history"][projection][:])
            ax = ax_FB_angles
            avg_angles = np.mean(fb_angles_all_seeds, axis=0)
            std_angles = np.std(fb_angles_all_seeds, axis=0)
            if np.isnan(avg_angles).any():
                print(f"Warning: NaN values found in avg_angles for {network_name}.")
            else:
                ax.plot(train_steps, avg_angles, color=model_dict['color'], label=model_dict['name'])
                ax.fill_between(train_steps, avg_angles-std_angles, avg_angles+std_angles, alpha=0.5, color=model_dict['color'], linewidth=0)
            ax.set_xlabel('Training step')
            ax.set_ylim(bottom=-2, top=90)
            ax.set_yticks(np.arange(0, 91, 30))
            ax.set_ylabel('Angle \n(F vs B weights)')

            # Angle vs BP
            angle_all_seeds = []
            for seed in model_dict['seeds']:
                angle_all_seeds.append(data_dict[seed]['angle_vs_bp']['all_params'])
            avg_angle = np.mean(angle_all_seeds, axis=0)
            error = np.std(angle_all_seeds, axis=0)
            ax_angleBP.plot(train_steps[1:], avg_angle, label=model_dict["name"], color=model_dict["color"])
            ax_angleBP.fill_between(train_steps[1:], avg_angle-error, avg_angle+error, alpha=0.5, color=model_dict["color"], linewidth=0)
            ax_angleBP.set_xlabel('Training step')
            ax_angleBP.set_ylabel('Angle vs BP')
            ax_angleBP.set_ylim([30,100])
            ax_angleBP.set_yticks(np.arange(30, 101, 30))

            angle_all_seeds = []
            from scipy.ndimage import gaussian_filter1d
            for seed in model_dict['seeds']:
                angle = data_dict[seed]['angle_vs_bp_stochastic']['all_params']
                # box_width = 3
                # smoothed_angle = np.convolve(angle, np.ones(box_width)/box_width, mode='same')
                sigma = 1
                smoothed_angle = gaussian_filter1d(angle, sigma)
                angle_all_seeds.append(smoothed_angle)
            avg_angle = np.nanmean(angle_all_seeds, axis=0)
            error = np.nanstd(angle_all_seeds, axis=0)
            ax_angleBP_stoch.plot(train_steps, avg_angle, label=model_dict["name"], color=model_dict["color"])
            ax_angleBP_stoch.fill_between(train_steps, avg_angle-error, avg_angle+error, alpha=0.5, color=model_dict["color"], linewidth=0)
            ax_angleBP_stoch.set_xlabel('Training step')
            ax_angleBP_stoch.set_ylabel('Angle vs BP (stoch.)')
            ax_angleBP_stoch.set_ylim([0,90])
            ax_angleBP_stoch.set_yticks(np.arange(0, 101, 30))
            # ax_angleBP_stoch.set_xlim([0,20000])

            # Learning curves / metrics
            accuracy_all_seeds = [data_dict[seed]['test_accuracy_history'] for seed in data_dict]
            avg_accuracy = np.mean(accuracy_all_seeds, axis=0)
            error = np.std(accuracy_all_seeds, axis=0)
            val_steps = data_dict[seed]['val_history_train_steps'][:]
            ax_accuracy.plot(val_steps, avg_accuracy, label=model_dict["name"], color=model_dict["color"])
            ax_accuracy.fill_between(val_steps, avg_accuracy-error, avg_accuracy+error, alpha=0.2, color=model_dict["color"], linewidth=0)
            ax_accuracy.set_xlabel('Training step')
            ax_accuracy.set_ylabel('Test accuracy (%)', labelpad=0)
            ax_accuracy.set_ylim([0,100])
            legend = ax_accuracy.legend(handlelength=1, handletextpad=0.5, ncol=3, bbox_to_anchor=(0., 1.5), loc='upper left', fontsize=6)
            for line in legend.get_lines():
                line.set_linewidth(1.5)  # Adjust linewidth in the legend

            sparsity_all_seeds = []
            for seed in data_dict:
                sparsity_one_seed = []
                for population in ['H1E', 'H2E']:
                    sparsity_one_seed.extend(data_dict[seed][f"metrics_dict_{population}"]['sparsity'])
                sparsity_all_seeds.append(sparsity_one_seed)
            pt.plot_cumulative_distribution(sparsity_all_seeds, ax=ax_sparsity, label=model_dict["name"], color=model_dict["color"])
            ax_sparsity.set_ylabel('Fraction of patterns')
            ax_sparsity.set_xlabel('Sparsity') # \n(1 - fraction of units active)')
            ax_sparsity.set_xlim([0,1])

            selectivity_all_seeds = []
            for seed in data_dict:
                selectivity_one_seed = []
                for population in ['H1E', 'H2E']:
                    selectivity_one_seed.extend(data_dict[seed][f"metrics_dict_{population}"]['selectivity'])
                selectivity_all_seeds.append(selectivity_one_seed)
            pt.plot_cumulative_distribution(selectivity_all_seeds, ax=ax_selectivity, label=model_dict["name"], color=model_dict["color"])
            ax_selectivity.set_ylabel('Fraction of units')
            ax_selectivity.set_xlabel('Selectivity') # \n(1 - fraction of active patterns)')
            ax_selectivity.set_xlim([0,1])

            receptive_fields = data_dict[seed][f"maxact_receptive_fields_{population}"]
            if receptive_fields is not None:
                structure_all_seeds = []
                for seed in data_dict:
                    structure_one_seed = []
                    for population in ['H1E', 'H2E']:
                        structure_one_seed.extend(data_dict[seed][f"metrics_dict_{population}"]['structure'])
                    structure_all_seeds.append(structure_one_seed)
                pt.plot_cumulative_distribution(structure_all_seeds, ax=ax_structure, label=model_dict["name"], color=model_dict["color"])
                ax_structure.set_ylabel('Fraction of units')
                ax_structure.set_xlabel("Structure (Moran's I)") # \n(Moran's I spatial autocorrelation)")
                ax_structure.set_xlim([0,1])
            else:
                ax_structure.axis('off')

            # Sparsity history
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

            # Dendritic state
            if 'binned_mean_forward_dendritic_state' in data_dict[seed]:
                dendstate_all_seeds = []
                for seed in model_dict['seeds']:
                        binned_mean_forward_dendritic_state_steps = data_dict[seed]['binned_mean_forward_dendritic_state_steps'][:]
                        dendstate_one_seed = data_dict[seed]['binned_mean_forward_dendritic_state']['all'][:]
                        dendstate_all_seeds.append(dendstate_one_seed)
                avg_dendstate = np.mean(dendstate_all_seeds, axis=0)
                error = np.std(dendstate_all_seeds, axis=0)            
                ax_dendstate.plot(binned_mean_forward_dendritic_state_steps, avg_dendstate, label=model_dict["name"], color=model_dict["color"])
                ax_dendstate.fill_between(binned_mean_forward_dendritic_state_steps, avg_dendstate-error, avg_dendstate+error, alpha=0.5, color=model_dict["color"], linewidth=0)
                ax_dendstate.set_xlabel('Training step')
                ax_dendstate.set_ylabel('Dendritic state')
                ax_dendstate.set_ylim(bottom=-0.0, top=0.06)
            

    if save:
        fig.savefig(f"figures/{save}.png", dpi=300)
        fig.savefig(f"figures/{save}.svg", dpi=300)





# def images_to_pdf(image_paths, output_path):
#     # Create a canvas for the PDF
#     c = canvas.Canvas(output_path, pagesize=letter)
#     width, height = letter  # US Letter size in points (612 x 792)

#     for img_path in image_paths:
#         # Open each image using PIL
#         img = Image.open(img_path)
#         # img_width, img_height = img.size
#         img_width, img_height = (5.5, 9)
        
#         # Set the page size to letter dimensions
#         c.setPageSize((width, height))
        
#         # Draw the image on the page (cropped if it's too large)
#         c.drawImage(img_path, 0, height - img_height, width=img_width, height=img_height)
        
#         # Add a caption with the image filename
#         caption = os.path.basename(img_path)  # Extract filename from the path
#         c.setFont("Helvetica", 10)  # Set font for the caption
#         c.drawString(10, 20, caption)  # Position the caption at the bottom-left
        
#         # Create a new page for the next image
#         c.showPage()
    
#     # Save the PDF
#     c.save()


def images_to_pdf(image_paths, output_path):
    # Create a canvas for the PDF
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    for img_path in image_paths:
        # Open each image using PIL
        img = Image.open(img_path)
        img_width, img_height = img.size
        c.setPageSize((width, height))
        
        # Draw the image on the canvas
        c.drawImage(img_path, 0, 0, width=width, height=height)
        
        # Add a caption with the image filename
        caption = os.path.basename(img_path)  # Extract filename from the path
        c.setFont("Helvetica", 10)  # Set font for the caption
        c.drawString(10, 20, caption)  # Position the caption at the bottom-left
        
        c.showPage()  # Add a new page in the PDF for the next image
    c.save()  # Save the PDF file





@click.command()
@click.option('--figure', default=None, help='Figure to generate')
@click.option('--overwrite', is_flag=True, default=False, help='Overwrite existing network data in plot_data.hdf5 file')
@click.option('--single-model', default=None, help='Model key for loading network data')
@click.option('--generate-data', default=None, help='Generate HDF5 data files for plots')
@click.option('--recompute', default=None, help='Recompute plot data for a particular parameter')

def main(figure, overwrite, single_model, generate_data, recompute):
    # pt.update_plot_defaults()
    seeds = ["66049_257","66050_258", "66051_259", "66052_260", "66053_261"]

    model_dict_all =    {# Networks with weight transpose on top-down weights
                        "vanBP":           {"config": "20231129_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G_complete_optimized.yaml",
                                            "color":  "black",
                                            "name":   "Vanilla Backprop"},

                        "bpDale_learned":  {"config": "20240419_EIANN_2_hidden_mnist_bpDale_relu_SGD_config_F_complete_optimized.yaml",
                                            "color":  "blue",
                                            "name":   "Backprop + Dale's Law (learned somaI)"},

                        "bpDale_fixed":    {"config": "20231129_EIANN_2_hidden_mnist_bpDale_relu_SGD_config_G_complete_optimized.yaml",
                                            "color":  "cyan",
                                            "name":   "Backprop + Dale's Law (fixed somaI)"},

                        "HebbWN_topsup":     {"config": "20240714_EIANN_2_hidden_mnist_Top_Layer_Supervised_Hebb_WeightNorm_config_4_complete_optimized.yaml",
                                            "color":  "green",
                                            "name":   "Top-supervised HebbWN"},

                        "BTSP":            {"config":"20240604_EIANN_2_hidden_mnist_BTSP_config_3L_complete_optimized.yaml",
                                            "color": "purple",
                                            "name": "BTSP"}, 

                        "HebbWN": {"config": "20240714_EIANN_2_hidden_mnist_Supervised_Hebb_WeightNorm_config_4_complete_optimized.yaml",
                                            "color": "cyan",
                                            "name": "HWN (B=W^T)"},

                        "bpDale_noI":     {"config": "20240919_EIANN_2_hidden_mnist_bpDale_noI_relu_SGD_config_G_complete_optimized.yaml",
                                           "color": "blue",
                                           "name": "Dale's Law (no somaI)"},


                        "bpLike_hebbdend": {"config": "20240516_EIANN_2_hidden_mnist_BP_like_config_2L_complete_optimized.yaml",
                                            "color":  "red",
                                            "name":   "Dendritic gating (Weight Symmetry)"}, # BP-local weight update rule with dendritic target propagation

                        "bpLike_localBP":  {"config": "20240628_EIANN_2_hidden_mnist_BP_like_config_3M_complete_optimized.yaml",
                                            "color":  "black",
                                            "name":   "Backprop (local loss func.))"},  # BP-local weight update rule with dendritic target propagation

                        "bpLike_fixedDend":{"config": "20240508_EIANN_2_hidden_mnist_BP_like_config_2K_complete_optimized.yaml",
                                            "color":  "gray",
                                            "name":   "Random"},     # BP-local weight update rule with dendritic target propagation


                        ## Networks with separate top-down weights (i.e. not weight transpose)

                        # 1. Consulting nudged activity for updating forward weights + HWN rules
                        "bpLike_FA":       {"config": "20240830_EIANN_2_hidden_mnist_BP_like_config_2L_fixed_TD_complete_optimized.yaml",
                                            "color": "lightgray",
                                            "name": "Random (Feedback Alignment)"},

                        "bpLike_learnedTD":{"config": "20240830_EIANN_2_hidden_mnist_BP_like_config_2L_learn_TD_HWN_3_complete_optimized.yaml",
                                            "color": "orange",
                                            "name": "Learned top-town (Hebb)"},

                        "BCM":             {"config": "20240723_EIANN_2_hidden_mnist_Supervised_BCM_config_4_complete_optimized.yaml",
                                            "color": "yellow",
                                            "name": "Supervised BCM"},
                                            
                        "BTSP_learnedTD": {"config": "20240905_EIANN_2_hidden_mnist_BTSP_config_3L_learn_TD_HWN_3_complete_optimized.yaml",
                                            "color": "magenta",
                                            "name": "BTSP learned Top-Down"},

                        "BTSP_FA":        {"config": "20240923_EIANN_2_hidden_mnist_BTSP_config_3L_fixed_TD_complete_optimized.yaml",
                                            "color": "orange",
                                            "name": "BTSP fixed TD(FA)"},

                        # 2. Consulting forward (un-nudged) activity for updating forward weights + HWN rules
                        "bpLike_learnedTD_nonudge":{"config": "20241009_EIANN_2_hidden_mnist_BP_like_config_5J_learn_TD_HWN_1_complete_optimized.yaml",
                                                    "color": "salmon",
                                                    "name": "Learned top-town (Hebb) no-nudge"},
                        
                        "HebbWN_learned_somaI": {"config": "20240919_EIANN_2_hidden_mnist_Supervised_Hebb_WeightNorm_learn_somaI_config_4_complete_optimized.yaml",
                                                 "color": "cyan",
                                                 "name": "Supervised HebbWN learned somaI"},

        }

    for model_key in model_dict_all:
        model_dict_all[model_key]["seeds"] = seeds

    if recompute is not None and generate_data is None:
        generate_data = 'all'
        # generate_data = ['bpLike_FA', 'bpLike_learnedTD', 'BTSP_learnedTD', 'BCM']
        print(f"Recomputing data for {generate_data}")

    if generate_data is not None:
        config_path_prefix="network_config/mnist/"
        saved_network_path_prefix="data/mnist/"
        if generate_data=='all':
            model_list = model_dict_all.keys()
        elif isinstance(generate_data, str):
            model_list = [generate_data]
        else:
            model_list = generate_data
        
        for model_key in model_list:
            model_dict = model_dict_all[model_key]
            config_path = config_path_prefix + model_dict['config']
            pickle_basename = "_".join(model_dict['config'].split('_')[0:-2])
            network_name = model_dict['config'].split('.')[0]
            data_file_path = f"data/plot_data_{network_name}.h5"
            for seed in model_dict['seeds']:
                saved_network_path = saved_network_path_prefix + pickle_basename + f"_{seed}_complete.pkl"
                generate_data_hdf5(config_path, saved_network_path, data_file_path, overwrite, recompute)
                gc.collect()

    if single_model is not None:
        if single_model=='all':
            for model_key in model_dict_all:
                generate_single_model_figure(model_dict=model_dict_all[model_key], save=True, overwrite=overwrite)
        else:
            generate_single_model_figure(model_dict=model_dict_all[single_model], save=True, overwrite=overwrite)

    if figure == "fig1":
        print("Diagram figure not implemented")

    if figure in ["all", "fig2"]:
        model_list_heatmaps = ["vanBP", "bpDale_fixed", "hebb_topsup"]
        model_list_metrics = model_list_heatmaps
        figure_name = "Fig2_vanBP_bpDale_hebb"
        compare_E_properties(model_dict_all, model_list_heatmaps, model_list_metrics, save=figure_name, overwrite=overwrite)

    elif figure in ["all", "S1"]:
        model_list_heatmaps = ["bpDale_learned", "bpDale_fixed"]
        model_list_metrics = ["bpDale_learned", "bpDale_fixed"]
        figure_name = "Fig2_supplement_somaI"
        compare_somaI_properties(model_dict_all, model_list_heatmaps, model_list_metrics, save=figure_name, overwrite=overwrite)

    elif figure in ["all", "fig3"]:
        model_list_heatmaps = ["bpLike_fixedDend", "bpLike_localBP", "bpLike_hebbdend"]
        model_list_metrics = model_list_heatmaps
        figure_name = "Fig3_dendI"
        compare_dendI_properties(model_dict_all, model_list_heatmaps, model_list_metrics, save=figure_name, overwrite=overwrite)

    elif figure in ["all", "S2"]:
        model_list_heatmaps = ["bpLike_fixedDend", "bpLike_localBP", "bpLike_hebbdend"]
        model_list_metrics = model_list_heatmaps
        figure_name = "FigS2_Ecells_bpLike"
        compare_E_properties(model_dict_all, model_list_heatmaps, model_list_metrics, save=figure_name, overwrite=overwrite)

    elif figure in ["all","fig4"]:
        #BTSP vs BCM vs HebbWN
        model_list_heatmaps = ["BTSP", "BCM", "HebbWN_topsup"]
        model_list_metrics = model_list_heatmaps
        figure_name = "Fig4_BTSP_BCM_HebbWN"
        compare_E_properties(model_dict_all, model_list_heatmaps, model_list_metrics, save=figure_name, overwrite=overwrite)

    # elif figure in ["all", "S3"]:
    #     pass

    # Figure 5: Hebbian learning rule enables W/B alignment
    #           -> Plot angles over time + accuracy
    #           -> (Diagram + equations)
    elif figure in ["all", "fig5"]:
        model_list1 = ["bpLike_hebbdend", "bpLike_FA", "bpLike_learnedTD"]
        model_list2 = ["BTSP", "BTSP_FA", "BTSP_learnedTD"]
        figure_name = "Fig5_WB_alignment_FA_bpLike_BTSP"
        compare_angle_metrics(model_dict_all, model_list1, model_list2, save=figure_name, overwrite=overwrite)
    

    elif figure in ["all", "metrics"]:
        model_list = ["vanBP", "bpDale_learned", "bpLike_fixedDend", "bpLike_hebbdend", "bpLike_learnedTD", "bpLike_FA"]
        figure_name = "metrics_all_models"
        generate_metrics_plot(model_dict_all, model_list, save=figure_name, overwrite=overwrite)

    # # Combine figures into one PDF
    # directory = "figures/"
    # image_paths = [os.path.join(directory, figure) for figure in os.listdir(directory) if figure.endswith('.png') and figure.startswith('Fig')]
    # images_to_pdf(image_paths=image_paths, output_path= "figures/all_figures.pdf")


if __name__=="__main__":
    main()
