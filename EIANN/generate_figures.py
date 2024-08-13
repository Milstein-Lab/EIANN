import torch
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import os
import h5py
import click

import EIANN.utils as ut
import EIANN.plot as pt

plt.rcParams.update({'font.size': 8,
                    'axes.spines.right': False,
                    'axes.spines.top':   False,
                    'axes.linewidth':    0.5,
                    'axes.labelpad':     2.0, 
                    'xtick.major.size':  2,
                    'xtick.major.width': 0.5,
                    'ytick.major.size':  2,
                    'ytick.major.width': 0.5,
                    'legend.frameon':       False,
                    'legend.handletextpad': 0.1,
                    'figure.figsize': [10.0, 3.0],
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



# def generate_data_hdf5(config_path, saved_network_path, data_file_path='data/plot_data.h5', overwrite=False):
def generate_data_hdf5(config_path_prefix, saved_network_path_prefix, model_dict, data_file_path='data/plot_data.h5', overwrite=False):
    '''
    Loads a network and saves plot-ready processed data into an hdf5 file.
    '''

    # Build network
    config_path = config_path_prefix + model_dict['config']
    saved_network_path = saved_network_path_prefix + model_dict['pickle']
    seed = model_dict['seed']

    network_name = os.path.basename(config_path).split('.')[0]
    network_seed = seed.split('_')[0]
    network = ut.build_EIANN_from_config(config_path, network_seed=network_seed)    
    network.seed = seed

    variables_to_save = ['percent_correct', 'average_pop_activity_dict', 'val_loss_history', 'val_accuracy_history', 'val_history_train_steps']
    variables_to_save.extend([f"metrics_dict_{population.fullname}" for population in network.populations.values()])
    variables_to_save.extend([f"maxact_receptive_fields_{population.fullname}" for population in network.populations.values()])
                             
    if os.path.exists(data_file_path):
        # Open hdf5 and check if the relevant network data already exists
        with h5py.File(data_file_path, 'r') as file:
            if network_name in file.keys():
                if seed in file[network_name].keys():
                    if overwrite:
                        print(f"Overwriting {network_name} {seed} in {data_file_path}")
                    elif set(variables_to_save).issubset(file[network_name][seed].keys()):
                        return

    try:
        network.load(saved_network_path)
    except:
        ut.rename_population(network, 'I', 'SomaI')
        network.load(saved_network_path)
    
    # Load dataset
    all_dataloaders = ut.get_MNIST_dataloaders(sub_dataloader_size=1000)
    train_dataloader, train_sub_dataloader, val_dataloader, test_dataloader, data_generator = all_dataloaders

    # Generate plot data

    # 1. Class-averaged activity
    percent_correct, average_pop_activity_dict = ut.compute_test_activity(network, test_dataloader, export=True, export_path=data_file_path, overwrite=overwrite)

    # 2. Receptive fields and metrics
    for population in network.populations.values():
        receptive_fields = ut.compute_maxact_receptive_fields(population, export=True, overwrite=overwrite)
        metrics_dict = ut.compute_representation_metrics(population, test_dataloader, receptive_fields, export=True, export_path=data_file_path, overwrite=overwrite)

    # 3. Loss and accuracy
    ut.save_plot_data(network.name, network.seed, data_key='val_loss_history', data=network.val_loss_history, file_path=data_file_path, overwrite=overwrite)
    ut.save_plot_data(network.name, network.seed, data_key='val_accuracy_history', data=network.val_accuracy_history, file_path=data_file_path, overwrite=overwrite)
    ut.save_plot_data(network.name, network.seed, data_key='val_history_train_steps', data=network.val_history_train_steps, file_path=data_file_path, overwrite=overwrite)


def generate_single_model_figure(config_path_prefix, saved_network_path_prefix, model_dict, save=True, overwrite=False):

    # Load data
    data_file_path = 'data/plot_data.h5'
    # ut.delete_plot_data('20231129_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G_optimized.yaml', 66049, data_file_path)
    # ut.delete_plot_data('20231129_EIANN_2_hidden_mnist_bpDale_relu_SGD_config_G_optimized.yaml', 66049, data_file_path)
    

    config_path = config_path_prefix + model_dict['config']
    saved_network_path = saved_network_path_prefix + model_dict['pickle']

    generate_data_hdf5(config_path_prefix, saved_network_path_prefix, model_dict, data_file_path, overwrite)
    network_name = os.path.splitext(os.path.basename(config_path))[0]
    seed = model_dict['seed']
    data_dict = ut.hdf5_to_dict(data_file_path)[network_name]
    num_populations = len(data_dict[seed]['average_pop_activity_dict'])

    # Plot figure
    mm = 1 / 25.4  # millimeters in inches
    fig = plt.figure(figsize=(180 * mm, 200 * mm))
    fig.suptitle(network_name, fontsize=12, y=1)

    axes = gs.GridSpec(nrows=6, ncols=3,
                        left=0.06,right=0.9,
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

    ax_loss.plot(data_dict[seed]['val_history_train_steps'], data_dict[seed]['val_loss_history'])
    ax_loss.set_xlabel('Training step')
    ax_loss.set_ylabel('Loss')

    ax_accuracy.plot(data_dict[seed]['val_history_train_steps'], data_dict[seed]['val_accuracy_history'])
    ax_accuracy.set_xlabel('Training step')
    ax_accuracy.set_ylabel('Accuracy')

    populations_to_plot = [population for population in data_dict[seed]['average_pop_activity_dict'] if population!='InputE']

    for i,population in enumerate(populations_to_plot):
        # Column 1: Average population activity (batch accuracy to the test dataset)
        ax = fig.add_subplot(axes[i, 0])
        pt.plot_batch_accuracy_from_data(data_dict[seed]['average_pop_activity_dict'], population=population, ax=ax)

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
        fig_width, fig_height = fig.get_size_inches()
        cax = fig.add_axes([ax_list[0].get_position().x0, ax.get_position().y0-0.1/fig_height, 
                            0.1, 0.05/fig_height])
        fig.colorbar(im, cax=cax, orientation='horizontal')

        # Column 3: Learning curves / metrics
        sparsity_all_seeds = [data_dict[seed][f"metrics_dict_{population}"]['sparsity'] for seed in data_dict]
        pt.plot_cumulative_distribution(sparsity_all_seeds, ax=ax_sparsity, label=population)
        ax_sparsity.set_ylabel('Fraction of patterns')
        ax_sparsity.set_xlabel('Sparsity \n(1 - fraction of units active)')
        ax_sparsity.legend()

        selectivity_all_seeds = [data_dict[seed][f"metrics_dict_{population}"]['selectivity'] for seed in data_dict]
        pt.plot_cumulative_distribution(selectivity_all_seeds, ax=ax_selectivity, label=population)
        ax_selectivity.set_ylabel('Fraction of units')
        ax_selectivity.set_xlabel('Selectivity \n(1 - fraction of active patterns)')
        ax_selectivity.legend()

        if receptive_fields is not None:
            structure_all_seeds = [data_dict[seed][f"metrics_dict_{population}"]['structure'] for seed in data_dict]
            pt.plot_cumulative_distribution(structure_all_seeds, ax=ax_structure, label=population)
            ax_structure.set_ylabel('Fraction of units')
            ax_structure.set_xlabel("Structure \n(Moran's I spatial autocorrelation)")
            ax_structure.legend()
        else:
            ax_structure.axis('off')

        discriminability_all_seeds = [data_dict[seed][f"metrics_dict_{population}"]['discriminability'] for seed in data_dict]
        pt.plot_cumulative_distribution(discriminability_all_seeds, ax=ax_discriminability, label=population)
        ax_discriminability.set_ylabel('Fraction of pattern pairs')
        ax_discriminability.set_xlabel('Discriminability \n(1 - cosine similarity)')
        ax_discriminability.legend()

    if save:
        fig.savefig(f"figures/{os.path.basename(saved_network_path)}.png", dpi=300)



"""Figure 1: Van_BP vs bpDale(learnedI)
    -> bpDale is more structured/sparse (focus on H1E metrics)"""
def generate_Fig1(model_dict, config_path_prefix, saved_network_path_prefix, save=True, overwrite=False):
    '''
    Figure 1: Van_BP vs bpDale(learnedI)
        -> bpDale is more structured/sparse (focus on H1E metrics)

    Compare vanilla Backprop to networks with 'cortical' architecures (i.e. with somatic feedback inhibition). 
    '''
    # ut.delete_plot_data('20231129_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G_optimized.yaml', 66049_257, file_path='data/plot_data.h5')
    # ut.delete_plot_data('20231129_EIANN_2_hidden_mnist_bpDale_relu_SGD_config_G_optimized.yaml', 66049, file_path='data/plot_data.h5')
    
    fig = plt.figure(figsize=(5.5, 9))
    axes = gs.GridSpec(nrows=6, ncols=4,
                    left=0.07,right=0.99,
                    top=0.98, bottom = 0.1,
                    wspace=0.4, hspace=0.6)
    
    ax_accuracy    = fig.add_subplot(axes[4, 0])  
    ax_structure   = fig.add_subplot(axes[4, 1])
    ax_sparsity    = fig.add_subplot(axes[5, 0])
    ax_selectivity = fig.add_subplot(axes[5, 1])

    # Load data
    data_file_path = 'data/plot_data.h5'

    for col, model_key in enumerate(model_dict):
        print(f"Generating plots for {model_key}")
        config_path = config_path_prefix + model_dict[model_key]['config']
        generate_data_hdf5(config_path_prefix, saved_network_path_prefix, model_dict[model_key], data_file_path, overwrite)

        network_name = os.path.splitext(os.path.basename(config_path))[0]
        seed = model_dict[model_key]['seed']
        data_dict = ut.hdf5_to_dict(data_file_path)[network_name]
        num_populations = len(data_dict[seed]['average_pop_activity_dict'])

        ax_accuracy.plot(data_dict[seed]['val_history_train_steps'], data_dict[seed]['val_accuracy_history'], label=model_dict[model_key]["name"])
        ax_accuracy.set_xlabel('Training step')
        ax_accuracy.set_ylabel('Accuracy')
        ax_accuracy.legend(handlelength=1, handletextpad=0.5, ncol=2, bbox_to_anchor=(-0.1, 1.3), loc='upper left')

        populations_to_plot = [population for population in data_dict[seed]['average_pop_activity_dict'] if population!='InputE' and 'E' in population]

        for row,population in enumerate(populations_to_plot):
            ## Activity plots: batch accuracy of each population to the test dataset
            ax = fig.add_subplot(axes[row*2, col+2])
            pt.plot_batch_accuracy_from_data(data_dict[seed]['average_pop_activity_dict'], population=population, ax=ax, cbar=False)
            if row==0:
                ax.set_title(model_dict[model_key]["name"], fontsize=10)


            ## Receptive field plots
            receptive_fields = torch.tensor(data_dict[seed][f"maxact_receptive_fields_{population}"])
            ax = fig.add_subplot(axes[1+row*2, col+2])
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
                    # box = matplotlib.patches.Rectangle((-0.5,-0.5), 28, 28, linewidth=0.5, edgecolor='k', facecolor='none', zorder=10)
                    # ax.add_patch(box)
                im = pt.plot_receptive_fields(receptive_fields, sort=False, ax_list=ax_list)
            fig_width, fig_height = fig.get_size_inches()
            cax = fig.add_axes([ax_list[0].get_position().x0, ax.get_position().y0-0.1/fig_height, 
                                0.1, 0.05/fig_height])
            fig.colorbar(im, cax=cax, orientation='horizontal')



        ## Learning curves / metrics
        sparsity_all_seeds = []
        for seed in data_dict:
            sparsity_one_seed = []
            for population in ['H1E', 'H2E']:
                sparsity_one_seed.extend(data_dict[seed][f"metrics_dict_{population}"]['sparsity'])
            sparsity_all_seeds.append(sparsity_one_seed)
        pt.plot_cumulative_distribution(sparsity_all_seeds, ax=ax_sparsity, label=model_dict[model_key]["name"])
        ax_sparsity.set_ylabel('Fraction of patterns')
        ax_sparsity.set_xlabel('Sparsity') # \n(1 - fraction of units active)')

        selectivity_all_seeds = []
        for seed in data_dict:
            selectivity_one_seed = []
            for population in ['H1E', 'H2E']:
                selectivity_one_seed.extend(data_dict[seed][f"metrics_dict_{population}"]['selectivity'])
            selectivity_all_seeds.append(selectivity_one_seed)
        pt.plot_cumulative_distribution(selectivity_all_seeds, ax=ax_selectivity, label=model_dict[model_key]["name"])
        ax_selectivity.set_ylabel('Fraction of units')
        ax_selectivity.set_xlabel('Selectivity') # \n(1 - fraction of active patterns)')

        if receptive_fields is not None:
            structure_all_seeds = []
            for seed in data_dict:
                structure_one_seed = []
                for population in ['H1E', 'H2E']:
                    structure_one_seed.extend(data_dict[seed][f"metrics_dict_{population}"]['structure'])
                structure_all_seeds.append(structure_one_seed)
            pt.plot_cumulative_distribution(structure_all_seeds, ax=ax_structure, label=model_dict[model_key]["name"])
            ax_structure.set_ylabel('Fraction of units')
            ax_structure.set_xlabel("Structure") # \n(Moran's I spatial autocorrelation)")
        else:
            ax_structure.axis('off')


    if save:
        fig.savefig("figures/Fig1_vanBP_bpDale.png", dpi=300)
        fig.savefig("figures/Fig1_vanBP_bpDale.svg", dpi=300)



"""
Figure 2: bpDale(learnedI) vs top-sup HebbWN(learnedI) vs unsup-HebbWN(learnedI)
    (top: Output+H1 E, activity and receptive fields and metrics)
    (bottom: Output+H1 SomaI, activity and metrics)
    -> bpDale SomaI is unselective/unstructured, biological learning rule is not (focus on SomaI metrics)
    -> sup HebbWN performance is mediocre (representational collapse), we need a biological way to pass gradients to hidden layers
"""
def generate_Fig2(model_dict, config_path_prefix, saved_network_path_prefix, save=True, overwrite=False):
    fig = plt.figure(figsize=(5.5, 9))
    axes = gs.GridSpec(nrows=6, ncols=4,
                    left=0.07,right=0.99,
                    top=0.98, bottom = 0.1,
                    wspace=0.4, hspace=0.6)
    
    ax_accuracy    = fig.add_subplot(axes[4, 0])  
    ax_structure   = fig.add_subplot(axes[4, 1])
    ax_sparsity    = fig.add_subplot(axes[5, 0])
    ax_selectivity = fig.add_subplot(axes[5, 1])

    # Load data
    data_file_path = 'data/plot_data.h5'





@click.command()
@click.option('--figure', default='all', help='Figure to generate')
@click.option('--overwrite', is_flag=True, default=False, help='Overwrite existing network data in plot_data.hdf5 file')
@click.option('--save',      is_flag=True, default=True, help='Save plots')
@click.option('--single-model', default=None, help='Model key for loading network data')

def main(figure, overwrite, save, single_model):
    # pt.update_plot_defaults()

    # all_dataloaders = ut.get_MNIST_dataloaders(sub_dataloader_size=1000)

    # generate_Fig1(all_dataloaders, save=True)


    model_dict = {"vanBP":      {"config": "20231129_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G_optimized.yaml", 
                                 "pickle": "20231129_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G_66049_257_complete.pkl",
                                 "seed": "66049_257",
                                 "color": "black",
                                 "name": "Vanilla BP"},
                  "bpDale":     {"config": "20231129_EIANN_2_hidden_mnist_bpDale_relu_SGD_config_G_optimized.yaml", 
                                 "pickle": "20231129_EIANN_2_hidden_mnist_bpDale_relu_SGD_config_G_66049_257_complete.pkl",
                                 "seed": "66049_257",
                                 "color": "green",
                                 "name": "Backprop with Dale's Law"},
                 }
    
    config_path_prefix = "network_config/mnist/"
    saved_network_path_prefix = "data/mnist/"

    if single_model is not None:
        if single_model=='all':
            for model_key in model_dict:
                generate_single_model_figure(config_path_prefix, saved_network_path_prefix, model_dict[model_key], save, overwrite)
        else:
            generate_single_model_figure(config_path_prefix, saved_network_path_prefix, model_dict[single_model], save, overwrite)

    if figure in ["all", "Fig1"]:
        model_subdict = {"vanBP": model_dict["vanBP"], 
                         "bpDale": model_dict["bpDale"]}
        generate_Fig1(model_subdict, config_path_prefix, saved_network_path_prefix, save, overwrite)



if __name__=="__main__":
    main()