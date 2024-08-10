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
Figure 1: Van_BP vs bpDale(learnedI)
        -> bpDale is more structured/sparse (focus on H1E metrics)

Figure 2: bpDale(learnedI) vs top-sup HebbWN(learnedI) vs unsup-HebbWN(learnedI)
        (top: Output+H1 E, activity and receptive fields and metrics)
        (bottom: Output+H1 SomaI, activity and metrics)
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

Figure 5: BTSP(learned top-down W + HebbWN dendI)
'''



def generate_data_hdf5(config_path, saved_network_path, data_file_path='data/plot_data.h5', overwrite=False):
    '''
    Loads a network and saves plot-ready processed data into an hdf5 file.
    '''

    # Build network
    network_name = os.path.basename(config_path).split('.')[0]
    network_seed = int(os.path.basename(saved_network_path).split('.')[0].split('_')[-2])
    data_seed = int(os.path.basename(saved_network_path).split('.')[0].split('_')[-1])
    seed = f"{network_seed}_{data_seed}"

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



def generate_single_model_figure(config_path, saved_network_path, save=True, overwrite=False):

    # Load data
    data_file_path = 'data/plot_data.h5'
    # ut.delete_plot_data('20231120_EIANN_1_hidden_mnist_van_bp_relu_SGD_config_G_optimized', 66049, data_file_path)

    generate_data_hdf5(config_path, saved_network_path, data_file_path, overwrite)
    network_name = os.path.splitext(os.path.basename(config_path))[0]
    network_seed = int(os.path.basename(saved_network_path).split('.')[0].split('_')[-2])
    data_seed = int(os.path.basename(saved_network_path).split('.')[0].split('_')[-1])
    seed = f"{network_seed}_{data_seed}"
    data_dict = ut.hdf5_to_dict(data_file_path)[network_name]
    num_populations = len(data_dict[seed]['average_pop_activity_dict'])

    # Plot figure
    mm = 1 / 25.4  # millimeters in inches
    fig = plt.figure(figsize=(180 * mm, 200 * mm))
    fig.suptitle(network_name, fontsize=12, y=1)

    axes = gs.GridSpec(nrows=5, ncols=3,
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

    for i,population in enumerate(data_dict[seed]['average_pop_activity_dict']):
        if population == 'InputE':
            continue

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

        # metrics_dict = data_dict[seed][f"representation_metrics_{population}"]

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



def generate_Fig1(all_dataloaders, show=True, save=False):
    '''
    Compare vanilla Backprop to networks with 'cortical' architecures (i.e. with somatic feedback inhibition). 
    All networks have 1 hidden layer with 500 E units and 50 somaI units.
    '''

    model_dict = {"vanBP":      {"config": "20231120_EIANN_1_hidden_mnist_van_bp_relu_SGD_config_G_optimized.yaml", 
                                 "pickle": "20231120_EIANN_1_hidden_mnist_van_bp_relu_SGD_config_G_66049_257.pkl",
                                 "color": "black",
                                 "name": "Vanilla BP"},
                  "bpDale":     {"config": "20231018_EIANN_1_hidden_mnist_bpDale_relu_SGD_config_G_optimized.yaml", 
                                 "pickle": "20231018_EIANN_1_hidden_mnist_bpDale_relu_SGD_config_G_66049_257.pkl",
                                 "color": "green",
                                 "name": "Backprop with Dale's Law"},
                 }

    mm = 1 / 25.4  # millimeters in inches
    fig = plt.figure(figsize=(180 * mm, 200 * mm))
    axes = gs.GridSpec(nrows=4, ncols=3,
                    left=0.06,right=0.9,
                    top=0.96, bottom = 0.04,
                    wspace=0.3, hspace=0.5)
    
    axes_metrics = gs.GridSpec(nrows=6, ncols=3,
                    left=0.06,right=0.9,
                    top=0.96, bottom = 0.04,
                    wspace=0.3, hspace=0.6)
    ax_loss = fig.add_subplot(axes_metrics[0, 2])
    ax_accuracy = fig.add_subplot(axes_metrics[1, 2])
    ax_H1E_sparsity = fig.add_subplot(axes_metrics[2, 2])
    ax_H1E_selectivity = fig.add_subplot(axes_metrics[3, 2])
    ax_H1E_structure = fig.add_subplot(axes_metrics[4, 2])
    ax_H1E_discriminability = fig.add_subplot(axes_metrics[5, 2])

    load = False
    overwrite = False

    train_dataloader, train_sub_dataloader, val_dataloader, test_dataloader, data_generator = all_dataloaders

    for i,model_name in enumerate(model_dict):
        # Build network model
        config_path_prefix = "network_config/mnist/"
        config_path = config_path_prefix + model_dict[model_name]["config"]
        network_seed = 66049
        network = ut.build_EIANN_from_config(config_path, network_seed=network_seed)
        
        if load:
            # Load network from pickle (and use to rerun analyses)
            saved_network_path_prefix = "data/mnist/"
            saved_network_path = saved_network_path_prefix + model_dict[model_name]["pickle"]
            try:
                network.load(saved_network_path)
            except:
                ut.rename_population(network, 'I', 'SomaI')
                network.load(saved_network_path)


        # Plot 1: Average population activity of OutputE (batch accuracy to the test dataset)
        ax = fig.add_subplot(axes[i*2, 0])
        pt.plot_batch_accuracy(network, test_dataloader, population=network.Output.E, ax=ax, export=True, overwrite=overwrite)
        ax.set_title(model_dict[model_name]["name"], fontsize=10)


        # Plot 2: Example output receptive fields
        population = network.Output.E
        num_units = 10
        ax = fig.add_subplot(axes[i*2, 1])
        ax.axis('off')
        if i == 0:
            ax.set_title("Receptive fields", fontsize=10)

        pos = ax.get_position()
        new_left = pos.x0 - 0.02  # Move left boundary to the left
        new_bottom = pos.y0 + 0.06 # Move bottom boundary up
        new_height = pos.height - 0.09  # Decrease height
        ax.set_position([new_left, new_bottom, pos.width, new_height])
        rf_axes = gs.GridSpecFromSubplotSpec(2, 5, subplot_spec=ax, wspace=0.1, hspace=0.1)

        ax_list = []
        for j in range(num_units):
            ax = fig.add_subplot(rf_axes[j])
            ax_list.append(ax)
            ax.axis('off')
        
        receptive_fields = ut.compute_maxact_receptive_fields(population, export=True, overwrite=overwrite)
        im = pt.plot_receptive_fields(receptive_fields, sort=False, ax_list=ax_list)
        fig_width, fig_height = fig.get_size_inches()
        # cax = fig.add_axes([0.03, ax.get_position().y0-0.1/fig_height, 0.1, 0.05/fig_height])

        cax = fig.add_axes([ax_list[0].get_position().x0, ax.get_position().y0-0.1/fig_height, 
                            0.1, 0.05/fig_height])
        fig.colorbar(im, cax=cax, orientation='horizontal')


        # Plot 3: Average population activity of H1E
        ax = fig.add_subplot(axes[1+i*2, 0])
        pt.plot_batch_accuracy(network, test_dataloader, population=network.H1.E, ax=ax, export=True, overwrite=overwrite)


        # Plot 4: Example hidden receptive fields
        population = network.H1.E
        num_units = 20
        ax = fig.add_subplot(axes[1+i*2, 1])
        ax.axis('off')

        pos = ax.get_position()
        # new_bottom = pos.y0 + 0.0  # Move bottom boundary up
        # new_width = pos.width + 0.04  # Increase width
        # new_height = pos.height - 0.01  # Decrease height
        new_left = pos.x0 - 0.02  # Move left boundary to the left
        new_bottom = pos.y0
        new_height = pos.height

        ax.set_position([new_left, new_bottom, pos.width, new_height])
        rf_axes = gs.GridSpecFromSubplotSpec(4, 5, subplot_spec=ax, wspace=0.1, hspace=0.1)

        ax_list = []
        for j in range(num_units):
            ax = fig.add_subplot(rf_axes[j])
            ax_list.append(ax)
            ax.axis('off')
        
        receptive_fields_H1E = ut.compute_maxact_receptive_fields(population, export=True, overwrite=overwrite)
        im = pt.plot_receptive_fields(receptive_fields_H1E, sort=True, ax_list=ax_list)
        fig_width, fig_height = fig.get_size_inches()
        cax = fig.add_axes([ax_list[0].get_position().x0, ax.get_position().y0-0.1/fig_height, 
                            0.1, 0.05/fig_height])
        fig.colorbar(im, cax=cax, orientation='horizontal')


        # Plot 4: Learning curves / metrics
        population = network.H1.E
        metrics_dict = ut.compute_representation_metrics(population, test_dataloader, receptive_fields_H1E, export=True, overwrite=overwrite)

        # loss_history = ut.load_plot_data(network.name, network.seed, data_key='loss_history')
        # ax_loss.plot(network.loss_history)

        ax_H1E_sparsity.hist(metrics_dict['sparsity'],50)
        ax_H1E_sparsity.set_ylabel('num patterns')
        ax_H1E_sparsity.set_xlabel('Sparsity \n(1 - fraction active units)')

        ax_H1E_selectivity.hist(metrics_dict['selectivity'],50)
        ax_H1E_selectivity.set_ylabel('num units')
        ax_H1E_selectivity.set_xlabel('Selectivity \n(1 - fraction active patterns)')

        ax_H1E_discriminability.hist(metrics_dict['discriminability'], 50)
        ax_H1E_discriminability.set_ylabel('pattern pairs')
        ax_H1E_discriminability.set_xlabel('Discriminability \n(1 - cosine similarity)')

        if receptive_fields is not None:
            ax_H1E_structure.hist(metrics_dict['structure'], 50)
            ax_H1E_structure.set_ylabel('num units')
            ax_H1E_structure.set_xlabel("Structure \n(Moran's I spatial autocorrelation)")
        else:
            ax_H1E_structure.axis('off')


        # ax = fig.add_subplot(axes[0, 2])
        # ax.plot(network.loss_history)
        # ax.set_ylabel('Train loss')
        # ax.set_xlabel('Training steps')

        print("-------------------------------------------")

    # if show:
    #     plt.show()

    if save:
        fig.savefig("figures/Fig1.png", dpi=300)
        fig.savefig("figures/Fig1.svg", dpi=300)


# make "overwrite" a click command
@click.command()
@click.option('--overwrite', is_flag=True, default=False, help='Overwrite existing network data in plot_data.hdf5 file')
@click.option('--show', is_flag=True, default=False, help='Show plots')
@click.option('--save', is_flag=True, default=True, help='Save plots')

def main(overwrite, show, save):
    # pt.update_plot_defaults()

    # all_dataloaders = ut.get_MNIST_dataloaders(sub_dataloader_size=1000)

    # generate_Fig1(all_dataloaders, save=True)


    model_dict = {"vanBP":      {"config": "20231120_EIANN_1_hidden_mnist_van_bp_relu_SGD_config_G_optimized.yaml", 
                                 "pickle": "20231120_EIANN_1_hidden_mnist_van_bp_relu_SGD_config_G_66049_257.pkl",
                                 "color": "black",
                                 "name": "Vanilla BP"},
                  "bpDale":     {"config": "20231018_EIANN_1_hidden_mnist_bpDale_relu_SGD_config_G_optimized.yaml", 
                                 "pickle": "20231018_EIANN_1_hidden_mnist_bpDale_relu_SGD_config_G_66049_257.pkl",
                                 "color": "green",
                                 "name": "Backprop with Dale's Law"},
                 }
    
    for model in model_dict:
        config_path = "network_config/mnist/" + model_dict[model]["config"]
        saved_network_path = "data/mnist/" + model_dict[model]["pickle"]
        generate_single_model_figure(config_path, saved_network_path, save, overwrite)


if __name__=="__main__":
    main()