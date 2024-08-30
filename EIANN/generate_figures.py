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
                    'legend.handletextpad': 0.1,
                    'lines.linewidth': 0.5,
                    'figure.figsize': [10.0, 3.0],
                    'font.sans-serif': 'Helvetica',
                    'svg.fonttype': 'none',
                    'text.usetex': False})

'''


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

Figure 5: BTSP(learned top-down W + HebbWN dendI)
'''



def generate_data_hdf5(config_path, saved_network_path, data_file_path, overwrite=False):
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
                         'test_loss_history', 'test_accuracy_history', 'angle_vs_bp']
    variables_to_save.extend([f"metrics_dict_{population.fullname}" for population in network.populations.values()])
    variables_to_save.extend([f"maxact_receptive_fields_{population.fullname}" for population in network.populations.values() if population.name=='E' and population.fullname!='InputE'])
    if "Dend" in "".join(network.populations.keys()):
        variables_to_save.extend(["binned_mean_forward_dendritic_state", "binned_mean_forward_dendritic_state_steps"])

    # Open hdf5 and check if the relevant network data already exists       
    variables_to_recompute = []  
    if os.path.exists(data_file_path):
        with h5py.File(data_file_path, 'r') as file:
            if network_name in file.keys():
                if seed in file[network_name].keys():
                    if overwrite:
                        print(f"Overwriting {network_name} {seed} in {data_file_path}")
                        variables_to_recompute = variables_to_save
                    elif set(variables_to_save).issubset(file[network_name][seed].keys()):
                        return
                    else:
                        variables_to_recompute = [var for var in variables_to_save if var not in file[network_name][seed].keys()]

    print("-----------------------------------------------------------------------------")

    print(variables_to_recompute)

    # Load the saved network pickle
    network = ut.load_network(saved_network_path)
    if type(network) != nt.Network:
        print("WARNING: Network pickle is not a Network object!")
        network = ut.build_EIANN_from_config(config_path, network_seed=network_seed)    
        ut.load_network_dict(network, saved_network_path)
    network.seed = seed
    network.name = network_name
    
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
        receptive_fields = None
        if population.name == "E" and population.fullname != "InputE":
            receptive_fields = ut.compute_maxact_receptive_fields(population, export=True, export_path=data_file_path, overwrite=overwrite)
        metrics_dict = ut.compute_representation_metrics(population, test_dataloader, receptive_fields, export=True, export_path=data_file_path, overwrite=overwrite)

    # Angle vs backprop
    if 'angle_vs_bp' in variables_to_recompute:
        stored_history_step_size = torch.diff(network.param_history_steps)[-1]
        if 'H1SomaI' in network.populations:
            config_path2 = os.path.join(os.path.dirname(config_path), "20231129_EIANN_2_hidden_mnist_bpDale_relu_SGD_config_G_complete_optimized.yaml")
            network2 = ut.build_EIANN_from_config(config_path2, network_seed=network_seed)
            bpClone_network = ut.compute_alternate_dParam_history(train_dataloader, network, network2, batch_size=stored_history_step_size, constrain_params=False)
        else:
            bpClone_network = ut.compute_alternate_dParam_history(train_dataloader, network, batch_size=stored_history_step_size, constrain_params=False)
        angles = ut.compute_dW_angles(bpClone_network.predicted_dParam_history, bpClone_network.actual_dParam_history_stepaveraged)
        ut.save_plot_data(network.name, network.seed, data_key='angle_vs_bp', data=angles, file_path=data_file_path, overwrite=overwrite)

    # 3. Binned dendritic state (local loss)
    if 'binned_mean_forward_dendritic_state' in variables_to_recompute:
        steps, binned_attr_history_dict = ut.get_binned_mean_population_attribute_history_dict(network, attr_name="forward_dendritic_state", bin_size=100, abs=True)
        if binned_attr_history_dict is not None:
            ut.save_plot_data(network.name, network.seed, data_key='binned_mean_forward_dendritic_state', data=binned_attr_history_dict, file_path=data_file_path, overwrite=overwrite)
            ut.save_plot_data(network.name, network.seed, data_key='binned_mean_forward_dendritic_state_steps', data=steps, file_path=data_file_path, overwrite=overwrite)

    # 3. Loss and accuracy
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


"""Figure 1: Van_BP vs bpDale(learnedI)
    -> bpDale is more structured/sparse (focus on H1E metrics)"""
def generate_Fig1(model_dict_all, model_list_heatmaps, model_list_metrics, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/mnist/", save=True, overwrite=False):
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
                
            ## Metrics plots
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
                    receptive_fields = torch.tensor(data_dict[seed][f"maxact_receptive_fields_{population}"])
                    ax = fig.add_subplot(axes[row, col*2+1])
                    ax.axis('off')
                    pos = ax.get_position()

                    if receptive_fields.shape[0] > 20:
                        num_units = 20
                        new_left = pos.x0 - 0.01  # Move left boundary to the left
                        new_bottom = pos.y0 + 0.005
                        new_height = pos.height - 0.005
                        ax.set_position([new_left, new_bottom, pos.width, new_height])
                        rf_axes = gs.GridSpecFromSubplotSpec(4, 5, subplot_spec=ax, wspace=0.1, hspace=0.1)
                        ax_list = []
                        for j in range(num_units):
                            ax = fig.add_subplot(rf_axes[j])
                            ax_list.append(ax)
                            # box = matplotlib.patches.Rectangle((-0.5,-0.5), 28, 28, linewidth=0.5, edgecolor='k', facecolor='none', zorder=10)
                            # ax.add_patch(box)
                        preferred_classes = torch.argmax(torch.tensor(data_dict[seed]['average_pop_activity_dict'][population]), dim=1)
                        im = pt.plot_receptive_fields(receptive_fields, sort=True, ax_list=ax_list, preferred_classes=preferred_classes)
                    else:
                        num_units = 10
                        new_left = pos.x0 - 0.01  # Move left boundary to the left
                        new_bottom = pos.y0 + 0.028 # Move bottom boundary up
                        new_height = pos.height - 0.045  # Decrease height
                        ax.set_position([new_left, new_bottom, pos.width, new_height])
                        rf_axes = gs.GridSpecFromSubplotSpec(2, 5, subplot_spec=ax, wspace=0.1, hspace=0.1)
                        ax_list = []
                        for j in range(num_units):
                            ax = fig.add_subplot(rf_axes[j])
                            ax_list.append(ax)
                            # box = matplotlib.patches.Rectangle((-0.5,-0.5), 28, 28, linewidth=0.5, edgecolor='k', facecolor='none', zorder=10)
                            # ax.add_patch(box)
                        preferred_classes = torch.argmax(torch.tensor(data_dict[seed]['average_pop_activity_dict'][population]), dim=1)
                        im = pt.plot_receptive_fields(receptive_fields, sort=False, ax_list=ax_list, preferred_classes=preferred_classes)
                    fig_width, fig_height = fig.get_size_inches()
                    cax = fig.add_axes([ax_list[0].get_position().x0+0.09, ax.get_position().y0-0.06/fig_height, 0.05, 0.03/fig_height])
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
                ax_accuracy.legend(handlelength=1, handletextpad=0.5, ncol=3, bbox_to_anchor=(-0.3, 1.5), loc='upper left', fontsize=6)

                sparsity_all_seeds = []
                for seed in data_dict:
                    sparsity_one_seed = []
                    for population in ['H1E', 'H2E']:
                        sparsity_one_seed.extend(data_dict[seed][f"metrics_dict_{population}"]['sparsity'])
                    sparsity_all_seeds.append(sparsity_one_seed)
                pt.plot_cumulative_distribution(sparsity_all_seeds, ax=ax_sparsity, label=model_dict["name"], color=model_dict["color"])
                ax_sparsity.set_ylabel('Fraction of patterns')
                ax_sparsity.set_xlabel('Sparsity') # \n(1 - fraction of units active)')

                selectivity_all_seeds = []
                for seed in data_dict:
                    selectivity_one_seed = []
                    for population in ['H1E', 'H2E']:
                        selectivity_one_seed.extend(data_dict[seed][f"metrics_dict_{population}"]['selectivity'])
                    selectivity_all_seeds.append(selectivity_one_seed)
                pt.plot_cumulative_distribution(selectivity_all_seeds, ax=ax_selectivity, label=model_dict["name"], color=model_dict["color"])
                ax_selectivity.set_ylabel('Fraction of units')
                ax_selectivity.set_xlabel('Selectivity') # \n(1 - fraction of active patterns)')

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
                    ax_structure.set_xlabel("Structure") # \n(Moran's I spatial autocorrelation)")
                else:
                    ax_structure.axis('off')


    if save:
        fig.savefig("figures/Fig1_vanBP_bpDale_hebb.png", dpi=600)
        fig.savefig("figures/Fig1_vanBP_bpDale_hebb.svg", dpi=600)



"""
Figure 2: bpDale(learnedI) vs top-sup HebbWN(learnedI) vs unsup-HebbWN(learnedI)
    (top: OutputE + H1E, activity and receptive fields and metrics)
    (bottom: OutputSomaI + H1SomaI, activity and metrics)
    -> bpDale SomaI is unselective/unstructured, biological learning rule is not (focus on SomaI metrics)
    -> sup HebbWN performance is mediocre (representational collapse), we need a biological way to pass gradients to hidden layers
"""
def generate_Fig2(model_dict_all, model_list_heatmaps, model_list_metrics, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/mnist/", save=True, overwrite=False):

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
                ax_accuracy.legend(handlelength=1, handletextpad=0.5, ncol=1, bbox_to_anchor=(1., 1.), loc='upper left', fontsize=6)

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
        fig.savefig("figures/Fig2_somaI.png", dpi=600)
        fig.savefig("figures/Fig2_somaI.svg", dpi=600)


def generate_Fig3(model_dict_all, model_list_heatmaps, model_list_metrics, config_path_prefix="network_config/mnist/", saved_network_path_prefix="data/mnist/", save=True, overwrite=False):

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
                    if row>0:
                        ax.set_xlabel('')
                        ax.set_xticklabels([])
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
                ax_accuracy.legend(handlelength=1, handletextpad=0.5, ncol=3, bbox_to_anchor=(-1., 1.3), loc='upper left', fontsize=6)
                sparsity_all_seeds = []
                for seed in model_dict['seeds']:
                    sparsity_one_seed = []
                    for population in populations_to_plot:
                        sparsity_one_seed.extend(data_dict[seed][f"metrics_dict_{population}"]['sparsity'])
                    sparsity_all_seeds.append(sparsity_one_seed)
                pt.plot_cumulative_distribution(sparsity_all_seeds, ax=ax_sparsity, label=model_dict["name"], color=model_dict["color"])
                ax_sparsity.set_ylabel('Fraction of patterns')
                ax_sparsity.set_xlabel('Sparsity') # \n(1 - fraction of units active)')

                selectivity_all_seeds = []
                for seed in model_dict['seeds']:
                    selectivity_one_seed = []
                    for population in populations_to_plot:
                        selectivity_one_seed.extend(data_dict[seed][f"metrics_dict_{population}"]['selectivity'])
                    selectivity_all_seeds.append(selectivity_one_seed)
                pt.plot_cumulative_distribution(selectivity_all_seeds, ax=ax_selectivity, label=model_dict["name"], color=model_dict["color"])
                ax_selectivity.set_ylabel('Fraction of units')
                ax_selectivity.set_xlabel('Selectivity') # \n(1 - fraction of active patterns)')
                
                dendstate_all_seeds = []
                for seed in model_dict['seeds']:
                    dendstate_one_seed = data_dict[seed]['binned_mean_forward_dendritic_state']['all']
                    dendstate_all_seeds.append(dendstate_one_seed)
                avg_dendstate = np.mean(dendstate_all_seeds, axis=0)
                error = np.std(dendstate_all_seeds, axis=0)
                binned_mean_forward_dendritic_state_steps = data_dict[seed]['binned_mean_forward_dendritic_state_steps'][:]
                ax_dendstate.plot(binned_mean_forward_dendritic_state_steps, avg_dendstate, label=model_dict["name"], color=model_dict["color"])
                ax_dendstate.fill_between(binned_mean_forward_dendritic_state_steps, avg_dendstate-error, avg_dendstate+error, alpha=0.5, color=model_dict["color"], linewidth=0)
                ax_dendstate.set_xlabel('Training step')
                ax_dendstate.set_ylabel('Dendritic state')
                ax_dendstate.set_ylim([-0.01,0.4])

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
        fig.savefig("figures/Fig3_DendI.png", dpi=600)
        fig.savefig("figures/Fig3_DendI.svg", dpi=600)



@click.command()
@click.option('--figure', default=None, help='Figure to generate')
@click.option('--overwrite', is_flag=True, default=False, help='Overwrite existing network data in plot_data.hdf5 file')
@click.option('--save',      is_flag=True, default=True, help='Save plots')
@click.option('--single-model', default=None, help='Model key for loading network data')

def main(figure, overwrite, save, single_model):
    # pt.update_plot_defaults()
    seeds = ["66049_257","66050_258", "66051_259", "66052_260", "66053_261"]

    model_dict =    {"vanBP":           {"config": "20231129_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G_complete_optimized.yaml",
                                        "color":  "black",
                                        "name":   "Vanilla Backprop"},

                    "bpDale_learned":   {"config": "20240419_EIANN_2_hidden_mnist_bpDale_relu_SGD_config_F_complete_optimized.yaml",
                                        "color":  "darkgray",
                                        "name":   "Backprop with \nDale's Law (learned I)"},

                    "bpDale_fixed":     {"config": "20231129_EIANN_2_hidden_mnist_bpDale_relu_SGD_config_G_complete_optimized.yaml",
                                        "color":  "lightgray",
                                        "name":   "Backprop with Dale's Law\n(fixed I)"},

                    "hebb":            {"config": "20240714_EIANN_2_hidden_mnist_Top_Layer_Supervised_Hebb_WeightNorm_config_4_complete_optimized.yaml",
                                        "color":  "cyan",
                                        "name":   "Supervised Hebb \n(w/ weight norm.)"},

                    "bpLike_hebb":     {"config": "20240516_EIANN_2_hidden_mnist_BP_like_config_2L_complete_optimized.yaml",
                                        "color":  "darkblue",
                                        "name":   "Hebb+WeightNorm"}, # BP-local weight update rule with dendritic target propagation

                    "bpLike_localBP":  {"config": "20240628_EIANN_2_hidden_mnist_BP_like_config_3M_complete_optimized.yaml",
                                        "color":  "black",
                                        "name":   "Local LossFunc"}, # BP-local weight update rule with dendritic target propagation

                    "bpLike_fixedDend": {"config": "20240508_EIANN_2_hidden_mnist_BP_like_config_2K_complete_optimized.yaml",
                                        "color":  "gray",
                                        "name":   "Fixed DendI"}, # BP-local weight update rule with dendritic target propagation

                    "BTSP":            {"config":"20240604_EIANN_2_hidden_mnist_BTSP_config_3L_complete_optimized.yaml",
                                        "color": "red",
                                        "name": "BTSP"},
                 }
    for model_key in model_dict:
        model_dict[model_key]["seeds"] = seeds

    if single_model is not None:
        if single_model=='all':
            for model_key in model_dict:
                generate_single_model_figure(model_dict=model_dict[model_key], save=save, overwrite=overwrite)
        else:
            generate_single_model_figure(model_dict=model_dict[single_model], save=save, overwrite=overwrite)

    if figure in ["all", "fig1"]:
        model_list_heatmaps = ["vanBP", "bpDale_learned", "hebb"]
        model_list_metrics = ["vanBP", "bpDale_learned", "hebb"]
        generate_Fig1(model_dict, model_list_heatmaps, model_list_metrics, save=save, overwrite=overwrite)

    elif figure in ["all", "fig2"]:
        model_list_heatmaps = ["bpDale_learned", "bpDale_fixed", "hebb"]
        model_list_metrics = ["bpDale_learned", "bpDale_fixed", "hebb"]
        generate_Fig2(model_dict, model_list_heatmaps, model_list_metrics, save=save, overwrite=overwrite)

    elif figure in ["all", "fig3"]:
        # model_list = ["bpLike_fixedDend", "bpLike_hebb", "bpLike_localBP"]
        model_list_heatmaps = ["BTSP", "bpLike_fixedDend", "bpLike_hebb"]
        model_list_metrics = ["BTSP", "bpLike_fixedDend", "bpLike_hebb"]
        generate_Fig3(model_dict, model_list_heatmaps, model_list_metrics, save=save, overwrite=overwrite)


if __name__=="__main__":
    main()
