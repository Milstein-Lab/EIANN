from plot_compare_networks_utils import *


model_list = ['van_BP', 'BP_1_inh', 'Hebb', 'BTSP_C']

title_dict = {'van_BP': 'Backprop',
              'BP_1_inh': 'Backprop (EI)',
              'Hebb': 'Hebb',
              'BTSP_C': 'Top-Down Dendritic Gating'}

data_file_path_dict = \
    {'BTSP_C': 'data/20221103_EIANN_1_hidden_exported_data.hdf5',
     'Hebb': 'data/20221012_EIANN_1_hidden_exported_data.hdf5',
     'BP_1_inh': 'data/20221103_EIANN_1_hidden_exported_data.hdf5',
     'van_BP': 'data/20221103_EIANN_1_hidden_exported_data.hdf5'}

legend_dict =  {'BP_1_inh': ('Backprop (EI)', 'b'),
                'van_BP': ('Backprop', 'k'),
                'Hebb': ('Hebb', 'r'),
                'BTSP_C': ('Dendritic Gating', 'c')}

example_index_dict = {'van_BP': 0, 'BP_1_inh': 0, 'Hebb': 0, 'BTSP_C': 0}

activity_dict, metrics_dict = unpack_data(model_list, data_file_path_dict)

plot_n_choose_k_task()

plot_activity(activity_dict, title_dict, example_index_dict, model_list[:1], label_pop=False)
plot_activity(activity_dict, title_dict, example_index_dict, model_list[1:])

plot_activation_funcs()

plot_metrics(metrics_dict, legend_dict, model_list[:1])
plot_metrics(metrics_dict, legend_dict, model_list[:2])
plot_metrics(metrics_dict, legend_dict, model_list[:3])
plot_metrics(metrics_dict, legend_dict, model_list)


sparsity_dict, discriminability_dict = analyze_hidden_representations(activity_dict)

plot_summary_comparison(sparsity_dict, discriminability_dict, legend_dict, model_list[:1])
plot_summary_comparison(sparsity_dict, discriminability_dict, legend_dict, model_list[:2])
plot_summary_comparison(sparsity_dict, discriminability_dict, legend_dict, model_list[:3])
plot_summary_comparison(sparsity_dict, discriminability_dict, legend_dict, model_list)

plt.show()