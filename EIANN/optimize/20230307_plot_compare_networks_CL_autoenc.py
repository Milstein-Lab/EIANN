from plot_compare_networks_utils import *


model_list = ['vanilla_backprop', 'bpDale_1_inh_softplus', 'Supervised_Gjorgjieva_Hebb_C', 'BTSP_D1']

title_dict = {'vanilla_backprop': 'Backprop',
              'bpDale_1_inh_softplus': 'Backprop (Dale)',
              'Supervised_Gjorgjieva_Hebb_1_inh_static_C': 'Supervised Hebb',
              'Supervised_Gjorgjieva_Hebb_C': 'Supervised Hebb',
              'BTSP_D1': 'Dendritic gating'}

default_data_file_path = 'data/autoenc/20230307_exported_output_EIANN_1_hidden_CL_autoenc.hdf5'
data_file_path_dict = {}
for model_label in title_dict:
    data_file_path_dict[model_label] = default_data_file_path

legend_dict =  {'vanilla_backprop': ('Backprop', 'k'),
                'bpDale_1_inh_softplus': ('Backprop (Dale)', 'r'),
                'Supervised_Gjorgjieva_Hebb_1_inh_static_C': ('Supervised Hebb', 'purple'),
                'Supervised_Gjorgjieva_Hebb_C': ('Supervised Hebb', 'purple'),
                'BTSP_D1': ('Dendritic Gating', 'c')}

example_index_dict = {'vanilla_backprop': 0,
                      'bpDale_1_inh_softplus': 0,
                      'Supervised_Gjorgjieva_Hebb_1_inh_static_C': 0,
                      'Supervised_Gjorgjieva_Hebb_C': 0,
                      'BTSP_D1': 2}

activity_dict, metrics_dict = unpack_data_CL(model_list, data_file_path_dict)

plot_activity_CL(activity_dict, title_dict, example_index_dict, model_list)

plot_metrics_CL(metrics_dict, legend_dict, model_list)

plt.show()
