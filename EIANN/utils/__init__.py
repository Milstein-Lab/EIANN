"""
Utility functions for the EIANN project.
"""

from .activations import (
    set_activation,
    srelu, 
    linear, 
    get_scaled_rectified_sigmoid_orig, get_scaled_rectified_sigmoid, 
)

from .data_utils import (
    write_to_yaml,
    read_from_yaml,
    export_metrics_data,
    import_metrics_data,
    hdf5_to_dict,
    dict_to_hdf5,
    load_plot_data,
    save_plot_data,
    get_project_root,
    get_MNIST_dataloaders,
    n_choose_k,
    n_hot_patterns,
    get_diag_argmax_row_indexes,
    sort_by_val_history,
    sort_by_class_averaged_val_output,
    sort_unsupervised_by_test_batch_autoenc,
    sort_unsupervised_by_best_epoch,
    compute_test_loss_and_accuracy_single_batch,
    compute_test_loss_and_accuracy_history,
    recompute_validation_loss_and_accuracy,
    recompute_train_loss_and_accuracy,
    get_optimal_sorting,
    recompute_history,
    analyze_simple_EIANN_epoch_loss_and_accuracy,
    test_EIANN_autoenc_config,
    test_EIANN_CL_config,
)

from .weight_inits import scaled_kaiming_init, half_kaiming_init

from .network_utils import (
    build_EIANN_from_config,
    build_clone_network,
    change_learning_rule_to_backprop,
    rename_population,
    recursive_dict_rename,
    nested_convert_scalars,
    count_dict_elements,
)

from .representation_analysis import (
    compute_average_activity,
    compute_test_activity,
    compute_dParam_history,
    compute_sparsity_history,
    compute_selectivity_history,
    spatial_structure_similarity_fft,
    compute_rf_structure,
    compute_morans_I,
    compute_diag_fisher,
    compute_representation_metrics,
    compute_act_weighted_avg,
    compute_maxact_receptive_fields,
    compute_unit_receptive_field,
    compute_PSD,
    check_equilibration_dynamics,
)


__all__ = [
    "srelu", "linear", "get_scaled_rectified_sigmoid", "get_scaled_rectified_sigmoid_orig", "set_activation",
    "n_choose_k", "n_hot_patterns",
    "get_diag_argmax_row_indexes",
    "sort_by_val_history",
    "sort_by_class_averaged_val_output",
    "sort_unsupervised_by_test_batch_autoenc",
    "sort_unsupervised_by_best_epoch",
    "compute_test_loss_and_accuracy_single_batch",
    "compute_test_loss_and_accuracy_history",
    "recompute_validation_loss_and_accuracy",
    "recompute_train_loss_and_accuracy",
    "get_optimal_sorting",
    "recompute_history",
    "analyze_simple_EIANN_epoch_loss_and_accuracy",
    get_project_root,
    "get_MNIST_dataloaders",
    "scaled_kaiming_init", "half_kaiming_init",
    "write_to_yaml", "read_from_yaml",
    "export_metrics_data", "import_metrics_data",
    "hdf5_to_dict", "dict_to_hdf5",
    "load_plot_data", "save_plot_data",
    "build_EIANN_from_config", "build_clone_network",
    "change_learning_rule_to_backprop", "rename_population",
    "recursive_dict_rename", "count_dict_elements",
    "nested_convert_scalars",
    "compute_average_activity",
    "compute_test_activity",
    "compute_dParam_history",
    "compute_representation_metrics", "compute_sparsity_history", "compute_selectivity_history",
    "compute_rf_structure", "spatial_structure_similarity_fft", "compute_morans_I",
    "compute_act_weighted_avg", "compute_maxact_receptive_fields", "compute_unit_receptive_field",
    "compute_PSD",
    "check_equilibration_dynamics",
    "compute_diag_fisher",
    "test_EIANN_CL_config", "test_EIANN_autoenc_config",    
]
