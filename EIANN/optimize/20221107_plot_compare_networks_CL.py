from plot_compare_networks_utils import *


model_list = ['van_BP', 'BP_1_inh', 'BTSP_C']

title_dict = {'van_BP': 'Backprop',
              'BP_1_inh': 'Backprop (EI)',
              'BTSP_C': 'Top-Down Dendritic Gating'
              }

data_file_path_dict = \
    {'BTSP_C': 'data/20221107_EIANN_1_hidden_CL_exported_data.hdf5',
     'BP_1_inh': 'data/20221107_EIANN_1_hidden_CL_exported_data.hdf5',
     'van_BP': 'data/20221107_EIANN_1_hidden_CL_exported_data.hdf5'
     }

legend_dict =  {'BP_1_inh': ('Backprop (EI)', 'b'),
                'van_BP': ('Backprop', 'k'),
                'BTSP_C': ('Dendritic Gating', 'c')}

example_index_dict = {'van_BP': 0, 'BP_1_inh': 0, 'BTSP_C': 1}

activity_dict, metrics_dict = unpack_data_CL(model_list, data_file_path_dict)

plot_activity_CL(activity_dict, title_dict, example_index_dict, model_list)

plot_metrics_CL(metrics_dict, legend_dict, [model_list[0]])
plot_metrics_CL(metrics_dict, legend_dict, model_list[:2])
plot_metrics_CL(metrics_dict, legend_dict, model_list)

"""

plot_metrics(metrics_dict, legend_dict, model_list[:3])



sparsity_dict, discriminability_dict = analyze_hidden_representations(activity_dict)

plot_summary_comparison(sparsity_dict, discriminability_dict, legend_dict, [model_list[0]])
plot_summary_comparison(sparsity_dict, discriminability_dict, legend_dict, model_list[:2])
plot_summary_comparison(sparsity_dict, discriminability_dict, legend_dict, model_list[:3])
plot_summary_comparison(sparsity_dict, discriminability_dict, legend_dict, model_list)
"""
plt.show()
