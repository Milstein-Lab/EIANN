from plot_compare_networks_utils import *
import EIANN.utils as ut

model_list = ['BP_1_inh', 'BTSP_C']

title_dict = {'BP_1_inh': 'Backprop',
              'BTSP_C': 'BTSP'}

data_file_path_dict = \
    {'BTSP_C': 'data/20221103_EIANN_1_hidden_exported_data.hdf5',
     'BP_1_inh': 'data/20221103_EIANN_1_hidden_exported_data.hdf5'}

legend_dict =  {'BP_1_inh': ('Backprop', 'k'),
                'BTSP_C': ('BTSP', 'r')}

example_index_dict = {'BP_1_inh': 0, 'BTSP_C': 0}

activity_dict, metrics_dict = unpack_data(model_list, data_file_path_dict)

model_list.append('ideal')
title_dict['ideal'] = 'ideal encoding'
example_index_dict['ideal'] = 0
activity_dict['ideal'] = {'Input': {'E': [torch.eye(21)]},
                          'H1': {'E': [ut.n_hot_patterns(2, 7).T]},
                          'Output': {'E': [torch.eye(21)]}}
metrics_dict['ideal'] = {'accuracy': np.zeros([5,300]),
                         'loss': np.zeros([5,300])}
legend_dict['ideal'] = ('ideal', 'c')

plot_activity(activity_dict, title_dict, example_index_dict, model_list)
model_list.pop(-1)
activity_dict.pop('ideal', None)

plot_metrics(metrics_dict, legend_dict, model_list)

sparsity_dict, discriminability_dict = analyze_hidden_representations(activity_dict)

plot_summary_comparison(sparsity_dict, discriminability_dict, legend_dict, model_list)

plt.show()