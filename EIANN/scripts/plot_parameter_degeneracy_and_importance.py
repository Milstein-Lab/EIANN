from nested.optimize_utils import *

opt_file_path_dict = {'BP_like_5J': '../optimize/data/mnist/20241011_152454_nested_optimization_history_PopulationAnnealing_EIANN_2_hidden_mnist_BP_like_5J_298410265168692003819576414887770517830.hdf5'
                      }
                  
order_dict = {'BP_like_5J': 1000}
best_param_file_path = '../optimize/optimize_params/mnist/20240816_nested_optimize_2_hidden_mnist_params.yaml'
best_param_dict = read_from_yaml(best_param_file_path)
best_x_dict = {model_name: best_param_dict[model_name] for model_name in opt_file_path_dict}
labels = ['Best', 'M1', 'M2', 'M3', 'M4']
titles_dict = {'BP_like_5J': 'Dendritic Target Propagation\nSymmetric (B=W^T)'
               }

yticklabels_dict = {
    'H1_E_Input_E_init_weight_scale': 'W$_{\mathrm{init}}$ (H1)',
    'H_E_E_learning_rate': '$\eta$, W (H1/H2)',
    'H1_E_H1_SomaI_init_weight_scale': 'Y$_{\mathrm{init}}$ (SomaI, H1)',
    'H1_E_H1_DendI_init_weight_scale': 'Y$_{\mathrm{init}}$ (DendI, H1)',
    'H_E_DendI_learning_rate': '$\eta$, Y (DendI, H1/H2)',
    'H1_E_H2_E_weight_scale': 'B$_{\mathrm{scale}}$ (H1)',
    'H1_SomaI_Input_E_init_weight_scale': 'W$_{\mathrm{init}}$ (SomaI, H1)',
    'H1_SomaI_H1_E_init_weight_scale': 'Q$_{\mathrm{init}}$ (SomaI, H1)',
    'H1_SomaI_H1_SomaI_init_weight_scale': 'R$_{\mathrm{init}}$ (SomaI, H1)',
    'H1_DendI_H1_E_weight_scale': 'Q$_{\mathrm{sum}}$ (DendI, H1)',
    'H1_DendI_H1_DendI_weight_scale': 'R$_{\mathrm{sum}}$ (DendI, H1)',
    'H2_E_H1_E_init_weight_scale': 'W$_{\mathrm{init}}$ (H2)',
    'H2_E_H2_SomaI_init_weight_scale': 'Y$_{\mathrm{init}}$ (SomaI, H2)',
    'H2_E_H2_DendI_init_weight_scale': 'Y$_{\mathrm{init}}$ (DendI, H2)',
    'H2_E_Output_E_weight_scale': 'B$_{\mathrm{scale}}$ (H2)',
    'H2_SomaI_H1_E_init_weight_scale': 'W$_{\mathrm{init}}$ (SomaI, H2)',
    'H2_SomaI_H2_E_init_weight_scale': 'Q$_{\mathrm{init}}$ (SomaI, H2)',
    'H2_SomaI_H2_SomaI_init_weight_scale': 'R$_{\mathrm{init}}$ (SomaI, H2)',
    'H2_DendI_H2_E_weight_scale': 'Q$_{\mathrm{sum}}$ (DendI, H2)',
    'H2_DendI_H2_DendI_weight_scale': 'R$_{\mathrm{sum}}$ (DendI, H2)',
    'Output_E_H2_E_init_weight_scale': 'W$_{\mathrm{init}}$ (Output)',
    'Output_E_H2_E_learning_rate': '$\eta$, W (Output)',
    'Output_E_Output_I_init_weight_scale': 'Y$_{\mathrm{init}}$ (SomaI, Output)',
    'Output_I_H2_E_init_weight_scale': 'W$_{\mathrm{init}}$ (SomaI, Output)',
    'Output_I_Output_E_init_weight_scale': 'Q$_{\mathrm{init}}$ (SomaI, Output)',
    'Output_I_Output_I_init_weight_scale': 'R$_{\mathrm{init}}$ (SomaI, Output)',
    'DendI_E_learning_rate': '$\eta$, Q (DendI, H1/H2)',
    'DendI_DendI_learning_rate': '$\eta$, R (DendI, H1/H2)'
    }
    
requested_ids_dict = {'BP_like_5J': [27351, 28461, 22309, 28785]
                      }

for model_name in opt_file_path_dict:
    opt_results = OptimizationReport(file_path=opt_file_path_dict[model_name])
    best_x = param_dict_to_array(best_x_dict[model_name], opt_results.param_names)
    group = opt_results.get_marder_group(plot=True, order=order_dict[model_name])
    min_param_vals = opt_results.min_param_vals
    max_param_vals = opt_results.max_param_vals

    model_ids = [indiv.model_id for indiv in group]
    requested_ids = requested_ids_dict[model_name]
    selected_x = [best_x]
    for requested_id in requested_ids:
        index = model_ids.index(requested_id)
        selected_x.append(group[index].x)
    selected_x = np.array(selected_x)
    normalized_selected_x = [normalize_dynamic(selected_x[:, i], min_param_vals[i], max_param_vals[i]) for i in
                            range(len(min_param_vals))]
    normalized_selected_x = np.array(normalized_selected_x).T
    
    fig, axis = plt.subplots(figsize=(7., 6.))
    plot_param_indexes = []
    plot_param_names = []
    for i, param_name in enumerate(opt_results.param_names):
        if param_name in yticklabels_dict:
            plot_param_indexes.append(i)
            plot_param_names.append(param_name)
    plot_param_indexes = np.array(plot_param_indexes)
    y_vals = list(reversed(range(len(plot_param_names))))
    for label, this_x in zip(labels, normalized_selected_x):
        axis.scatter(this_x[plot_param_indexes], y_vals, label=label)
    axis.set_xlabel('Normalized parameter value')
    axis.set_yticks(y_vals)
    # y_ticklabels = opt_results.param_names
    y_ticklabels = [yticklabels_dict[param_name] for param_name in plot_param_names]
    axis.set_yticklabels(y_ticklabels)
    axis.set_xlim((0., 1.))
    axis.legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
    fig.suptitle(titles_dict[model_name])
    clean_axes(axis)
    # fig.subplots_adjust(left=0.4, top=0.95)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.show()
plt.show()