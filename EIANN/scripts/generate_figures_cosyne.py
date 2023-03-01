
import numpy as np
import matplotlib.pyplot as plt
import click

import EIANN.utils as utils
import EIANN.plot as plot


def plot_metrics_comparison(models_list, legend_dict, path):

    metrics_dict = utils.import_metrics_data(path)

    fig, ax = plt.subplots(1, 4, figsize=[12, 5])

    label_list = []
    for x, model_name in enumerate(models_list):
        print(x, model_name)
        mean_sparsity = np.mean(metrics_dict[model_name]['sparsity'])
        ax[0].bar(x, mean_sparsity, width=0.5, color=legend_dict[model_name][1])

        mean_selectivity = np.mean(metrics_dict[model_name]['selectivity'])
        ax[1].bar(x, mean_selectivity, width=0.5, color=legend_dict[model_name][1])

        mean_discriminability = np.mean(metrics_dict[model_name]['discriminability'])
        ax[2].bar(x, mean_discriminability, width=0.5, color=legend_dict[model_name][1])

        mean_structure = np.mean(metrics_dict[model_name]['structure'])
        ax[3].bar(x, mean_structure, width=0.5, color=legend_dict[model_name][1])

        label_list.append(legend_dict[model_name][0])

    ax[0].set_title('Sparsity')
    ax[1].set_title('Selectivity')
    ax[2].set_title('Discriminability')
    ax[3].set_title('Structure')

    for i in range(4):
        ax[i].set_xticks(np.arange(len(metrics_dict)))
        ax[i].set_xticklabels(label_list, rotation=-45, ha="left", rotation_mode="anchor")

    fig.tight_layout()


@click.command()
@click.option("--show", is_flag=True)
@click.option("--path", default='../notebooks/saved_networks/model_metrics.hdf5')
def main(path, show=False):
    plot.update_plot_defaults()

    models_list = ['EIANN_1_hidden_mnist_backprop_relu_SGD_config.yaml',
                   '20230102_EIANN_1_hidden_mnist_bpDale_softplus_config.yaml',
                   '20230214_1_hidden_mnist_Supervised_Gjorgjieva_Hebb_C.yaml',
                   '20230220_1_hidden_mnist_BTSP_Clone_Dend_I_4.yaml']

    legend_dict = {'EIANN_1_hidden_mnist_backprop_relu_SGD_config.yaml':        ('Standard Backprop', 'k'),
                   '20230102_EIANN_1_hidden_mnist_bpDale_softplus_config.yaml': ("Backprop w/ Dale's Law", 'grey'),
                   '20230214_1_hidden_mnist_Supervised_Gjorgjieva_Hebb_C.yaml': ('Hebb w/ weight norm.', 'purple'),
                   '20230220_1_hidden_mnist_BTSP_Clone_Dend_I_4.yaml':          ('Dendritic Gating', 'r')}

    plot_metrics_comparison(models_list, legend_dict, path)

    if show:
        plt.show()



if __name__ == '__main__':
    main(standalone_mode=False)