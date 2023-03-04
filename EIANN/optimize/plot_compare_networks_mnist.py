import numpy as np
import h5py
import matplotlib.pyplot as plt
import EIANN.utils as utils
import EIANN.plot as pt

pt.update_plot_defaults()


def plot_metrics_comparison(model_list, data_dict, title_dict, legend_dict, model_names_dict):
    fig, ax = plt.subplots(1, 4, figsize=[12, 5])

    label_list = []
    for x, model_name in enumerate(model_list):
        model_name = model_names_dict[model_name]
        mean_sparsity = []
        mean_selectivity = []
        mean_discriminability = []
        mean_structure = []
        for seed in data_dict[model_name]:
            mean_sparsity.append(np.mean(data_dict[model_name][seed]['metrics']['sparsity']))
            mean_selectivity.append(np.mean(data_dict[model_name][seed]['metrics']['selectivity']))
            mean_discriminability.append(np.mean(data_dict[model_name][seed]['metrics']['discriminability']))
            mean_structure.append(np.mean(data_dict[model_name][seed]['metrics']['structure']))

        # Plot bar graph with error bars
        ax[0].bar(x, np.mean(mean_sparsity), yerr=np.std(mean_sparsity), width=0.5, color=legend_dict[model_name][1])
        ax[1].bar(x, np.mean(mean_selectivity), yerr=np.std(mean_selectivity), width=0.5, color=legend_dict[model_name][1])
        ax[2].bar(x, np.mean(mean_discriminability), yerr=np.std(mean_discriminability), width=0.5, color=legend_dict[model_name][1])
        ax[3].bar(x, np.mean(mean_structure), yerr=np.std(mean_structure), width=0.5, color=legend_dict[model_name][1])
        label_list.append(legend_dict[model_name][0])

    ax[0].set_title('Sparsity')
    ax[1].set_title('Selectivity')
    ax[2].set_title('Discriminability')
    ax[3].set_title('Structure')

    for i in range(4):
        ax[i].set_xticks(np.arange(len(data_dict)))
        ax[i].set_xticklabels(label_list, rotation=-45, ha="left", rotation_mode="anchor")

    fig.tight_layout()


def plot_accuracy_comparison(model_list, data_dict, title_dict, legend_dict, model_names_dict):
    fig, ax = plt.subplots(figsize=[6, 5])

    for model_name in model_list:
        model_name = model_names_dict[model_name]
        accuracy = []
        for seed in data_dict[model_name]:
            accuracy.append(data_dict[model_name][seed]['metrics']['test_accuracy'])

        # Plot accuracy line plot with shaded error
        x = data_dict[model_name][seed]['metrics']['test_loss_steps']
        mean_accuracy = np.mean(accuracy, axis=0)
        error = np.std(accuracy, axis=0)
        ax.plot(x, mean_accuracy, color=legend_dict[model_name][1])
        ax.fill_between(x, mean_accuracy-error, mean_accuracy+error,
                        color=legend_dict[model_name][1], alpha=0.2, label=legend_dict[model_name][0])

    ax.set_title('Accuracy')
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Accuracy (%)')
    plt.legend()


data_file_path = 'data/20230303_exported_output_EIANN_1_hidden_mnist.hdf5'
data_dict = utils.hdf5_to_dict(data_file_path)

model_list = ['BP', 'BP_Dale', 'Hebb', 'BTSP']

title_dict = {}
legend_dict = {}
model_names_dict = {}
for model_name in data_dict:
    if 'van_bp' in model_name:
        title_dict[model_name] = 'Backprop'
        legend_dict[model_name] = ('Backprop', 'k')
        model_names_dict['BP'] = model_name
    elif 'Dale' in model_name:
        title_dict[model_name] = 'Backprop (EI)'
        legend_dict[model_name] = ('Backprop (EI)', 'b')
        model_names_dict['BP_Dale'] = model_name
    if 'Hebb' in model_name:
        title_dict[model_name] = 'Hebb'
        legend_dict[model_name] = ('Hebb', 'r')
        model_names_dict['Hebb'] = model_name
    elif 'BTSP' in model_name:
        title_dict[model_name] = 'Top-Down Dendritic Gating'
        legend_dict[model_name] = ('Dendritic Gating', 'c')
        model_names_dict['BTSP'] = model_name

plot_metrics_comparison(model_list, data_dict, title_dict, legend_dict, model_names_dict)
plot_accuracy_comparison(model_list, data_dict, title_dict, legend_dict, model_names_dict)
plt.show()