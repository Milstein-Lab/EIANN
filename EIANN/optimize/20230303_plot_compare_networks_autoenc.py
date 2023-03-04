from plot_compare_networks_utils import *


model_list = ['vanilla_backprop', 'bpDale_1_inh_softplus', 'Gjorgjieva_Hebb_1_inh_static_C',
              'Supervised_Gjorgjieva_Hebb_1_inh_static_C', 'BTSP_D1']

title_dict = {'vanilla_backprop': 'Backprop',
              'bpDale_1_inh_softplus': 'Backprop (Dale)',
              'Gjorgjieva_Hebb_1_inh_static_C': 'Hebb',
              'Supervised_Gjorgjieva_Hebb_1_inh_static_C': 'Supervised Hebb',
              'BTSP_D1': 'Dendritic gating'}

data_file_path_dict = \
    {'vanilla_backprop': 'data/autoenc/20230303_exported_output_EIANN_1_hidden_autoenc.hdf5',
     'bpDale_1_inh_softplus': 'data/autoenc/20230303_exported_output_EIANN_1_hidden_autoenc.hdf5',
     'Gjorgjieva_Hebb_1_inh_static_C': 'data/autoenc/20230303_exported_output_EIANN_1_hidden_autoenc.hdf5',
     'Supervised_Gjorgjieva_Hebb_1_inh_static_C': 'data/autoenc/20230303_exported_output_EIANN_1_hidden_autoenc.hdf5',
     'BTSP_D1': 'data/autoenc/20230303_exported_output_EIANN_1_hidden_autoenc.hdf5'}

legend_dict =  {'vanilla_backprop': ('Backprop', 'k'),
                'bpDale_1_inh_softplus': ('Backprop (Dale)', 'r'),
                'Gjorgjieva_Hebb_1_inh_static_C': ('Hebb', 'brown'),
                'Supervised_Gjorgjieva_Hebb_1_inh_static_C': ('Supervised Hebb', 'purple'),
                'BTSP_D1': ('Dendritic gating', 'c')}

example_index_dict = {'vanilla_backprop': 0,
                      'bpDale_1_inh_softplus': 0,
                      'Gjorgjieva_Hebb_1_inh_static_C': 0,
                      'Supervised_Gjorgjieva_Hebb_1_inh_static_C': 0,
                      'BTSP_D1': 0}

activity_dict, metrics_dict = unpack_data(model_list, data_file_path_dict)

plot_n_choose_k_task()

plot_activity(activity_dict, title_dict, example_index_dict, model_list[:1], label_pop=False)
plot_activity(activity_dict, title_dict, example_index_dict, model_list[1:])

plot_activation_funcs()

plot_metrics(metrics_dict, legend_dict, model_list[:1])
plot_metrics(metrics_dict, legend_dict, model_list[:2])
plot_metrics(metrics_dict, legend_dict, model_list[:3])
plot_metrics(metrics_dict, legend_dict, model_list[:4])
plot_metrics(metrics_dict, legend_dict, model_list)


sparsity_dict, discriminability_dict = analyze_hidden_representations(activity_dict)

plot_summary_comparison(sparsity_dict, discriminability_dict, legend_dict, model_list[:1])
plot_summary_comparison(sparsity_dict, discriminability_dict, legend_dict, model_list[:2])
plot_summary_comparison(sparsity_dict, discriminability_dict, legend_dict, model_list[:3])
plot_summary_comparison(sparsity_dict, discriminability_dict, legend_dict, model_list[:4])
plot_summary_comparison(sparsity_dict, discriminability_dict, legend_dict, model_list)

plt.show()