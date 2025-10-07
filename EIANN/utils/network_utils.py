import EIANN.network as nt
import EIANN.utils as ut
import EIANN.external as external
import os
import pickle
import dill
import datetime
import torch
import matplotlib.pyplot as plt
import numpy as np


def build_EIANN_from_config(config_path, network_seed=42, config_format='normal'):
    '''
    Build an EIANN network from a config file
    '''        
    network_config = ut.read_from_yaml(config_path)
    layer_config = network_config['layer_config']
    projection_config = network_config['projection_config']
    if config_format == 'simplified':
        projection_config = convert_projection_config_dict(projection_config)
        layer_config = convert_layer_config_dict(layer_config)
    training_kwargs = network_config['training_kwargs']
    
    try:
        network = nt.Network(layer_config, projection_config, seed=network_seed, **training_kwargs)
    except:
        projection_config = convert_projection_config_dict(projection_config)
        layer_config = convert_layer_config_dict(layer_config)
        network = nt.Network(layer_config, projection_config, seed=network_seed, **training_kwargs)
    
    network.name = os.path.splitext(os.path.basename(config_path))[0]
    return network


def convert_config_dict(simple_format_dict):    
    layer_config = simple_format_dict['layer_config']
    simple_format_dict['layer_config'] = convert_layer_config_dict(layer_config)
    projection_config = simple_format_dict['projection_config']
    projection_config_keys = list(projection_config.keys())
    if ("." in str(projection_config_keys[0])): 
        simple_format_dict['projection_config'] = convert_projection_config_dict(projection_config)
    return simple_format_dict


def convert_projection_config_dict(simple_format_dict):
    """
    Convert a projection config with simplified format (formatted as "layer.population":{}) to the extended format with nested dicts (formatted as "layer": {"population": {}})
    """
    extended_format_dict = {}
    
    for layer_fullname, subdictionary in simple_format_dict.items():
        layer_name, population_name = layer_fullname.split('.')
        
        if layer_name not in extended_format_dict: # If the first part of the split key isn't in the extended format dictionary, add it
            extended_format_dict[layer_name] = {}

        if population_name not in extended_format_dict[layer_name]: # If the second part of the split key isn't in the sub-dictionary, add it
            extended_format_dict[layer_name][population_name] = {}
        
        # Iterate over the items in the sub-dictionary
        for pre_layer_fullname, subsubdictionary in subdictionary.items():
            pre_layer_name, pre_pop_name = pre_layer_fullname.split('.')
            
            if pre_layer_name not in extended_format_dict[layer_name][population_name]: # If the first part of the split key isn't already in the sub-sub-dictionary, add it
                extended_format_dict[layer_name][population_name][pre_layer_name] = {}
            
            # Add the second part of the split key to the sub-sub-dictionary, converting 'None' string values to Python None
            extended_format_dict[layer_name][population_name][pre_layer_name][pre_pop_name] = {}

            # Translate projection properties in the subsubdictionary
            for k, v in subsubdictionary.items():
                if k == 'type':
                    if v.lower() in ['e', 'exc', 'excitatory']:
                        extended_format_dict[layer_name][population_name][pre_layer_name][pre_pop_name]['weight_bounds'] = [0, None]
                    elif v.lower() in ['i', 'inh', 'inhibitory']:
                        extended_format_dict[layer_name][population_name][pre_layer_name][pre_pop_name]['weight_bounds'] = [None, 0]
                else:
                    extended_format_dict[layer_name][population_name][pre_layer_name][pre_pop_name][k] = None if v == 'None' else v
    
    return extended_format_dict


def convert_layer_config_dict(layer_config_dict):
    """
    Convert a layer config with simplified format to the extended format
    """
    for layer in layer_config_dict:
        for population in layer_config_dict[layer]:
            if 'bias' in layer_config_dict[layer][population]: # Allows for syntax like bias: 'uniform(0,1)'
                bias_distribution = layer_config_dict[layer][population]['bias']
                bias_init = bias_distribution.split('(')[0] + '_' 
                init_args = bias_distribution.split('(')[1].split(')')[0].split(',')
                init_args = [float(arg) for arg in init_args]
                
                del layer_config_dict[layer][population]['bias']
                layer_config_dict[layer][population]['include_bias'] = True
                layer_config_dict[layer][population]['bias_init'] = bias_init
                layer_config_dict[layer][population]['bias_init_args'] = init_args
    return layer_config_dict       


def save_network(network, path=None, dir='saved_networks', file_name_base=None, disp=True, overwrite=False):
    if path is None:
        if file_name_base is None:
            file_name_base = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(dir, f'{file_name_base}.pkl')
    
    dir = os.path.dirname(path)
    if dir != '':
        os.makedirs(dir, exist_ok=True)
    
    if os.path.exists(path):
        if overwrite is True:
            print(f"WARNING: File '{path}' already exists. Overwriting...")
        else:
            print(f"WARNING: File '{path}' already exists, new file not saved. Use overwrite=True to overwrite.")
            return

    with open(path, 'wb') as f:
        dill.dump(network, f)
    if disp:
        print(f"Saved network to '{path}'")


def load_network(path, disp=True):
    """
    Load a neural network from a file using dill serialization.

    Parameters
    ----------
    path : str
        Path to the file containing the serialized network.
    disp : bool, optional
        Whether to display loading status messages. Default is True.

    Returns
    -------
    Network
        The loaded neural network object with attribute histories re-registered.
    """
    if disp:
        print(f"Loading network from '{path}'")
    with open(path, 'rb') as f:
        network = dill.load(f)
    for layer in network:
        for population in layer:
            for attr_name in population.attribute_history_dict:
                population.register_attribute_history(attr_name)
            for projection in population:
                for attr_name in projection.attribute_history_dict:
                    projection.register_attribute_history(attr_name)
    if disp:
        print(f"Network successfully loaded from '{path}'")
    return network
    

def save_network_dict(network, path=None, dir='saved_networks', file_name_base=None, disp=True):
    """
    Save a neural network to a file as a dictionary using pickle serialization.

    Parameters
    ----------
    network : Network
        The neural network object to save.
    path : str, optional
        Full path to the output file. If None, path is constructed from dir and 
        file_name_base. Default is None.
    dir : str, optional
        Directory where the file will be saved. Only used if path is None. 
        Default is 'saved_networks'.
    file_name_base : str, optional
        Base name for the output file. If None, uses current timestamp. Only 
        used if path is None. Default is None.
    disp : bool, optional
        Whether to display saving status messages. Default is True.
    """
    if path is None:
        if file_name_base is None:
            file_name_base = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path = '%s/%s.pkl' % (dir, file_name_base)
        if not os.path.exists(dir):
            os.makedirs(dir)
            
    elif os.path.exists(path):
        print(f"WARNING: File '{path}' already exists. Overwriting...")

    network.params_to_save.extend(['param_history', 'param_history_steps', 'prev_param_history', 'sample_order',
                                'target_history', 'sorted_sample_indexes', 'loss_history', 'val_output_history',
                                'val_loss_history', 'val_history_train_steps', 'val_accuracy_history',
                                'val_target', 'attribute_history_dict', 'forward_dendritic_state'])
    
    data_dict = {'network': {param_name: value for param_name, value in network.__dict__.items()
                                if param_name in network.params_to_save},
                    'layers': {},
                    'populations': {},
                    'final_state_dict': network.state_dict()}

    for layer in network:
        layer_data = {param_name: value for param_name, value in layer.__dict__.items()
                        if param_name in network.params_to_save}
        data_dict['layers'][layer.name] = layer_data

        for population in layer:
            population_data = {param_name: value for param_name, value in population.__dict__.items()
                                if param_name in network.params_to_save}
            data_dict['populations'][population.fullname] = population_data

    with open(path, 'wb') as file:
        pickle.dump(data_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    if disp:
        print(f'Model saved to {path}')


def load_network_dict(network, path):
    """
    Load network parameters from a pickled dictionary file into an existing network.

    Parameters
    ----------
    network : :class:`EIANN.network.Network`
        The existing network object to load parameters into.
    path : str
        Path to the pickle file containing the network dictionary data.
    """
    print(f"Loading model data from '{path}'...")
    with open(path, 'rb') as file:
        data_dict = pickle.load(file)

    print('Loading parameters into the network...')
    network.__dict__.update(data_dict['network'])

    for layer in network:
        layer_data = data_dict['layers'][layer.name]
        layer.__dict__.update(layer_data)
        for population in layer:
            population_data = data_dict['populations'][population.fullname]
            population.__dict__.update(population_data)

    network.load_state_dict(data_dict['final_state_dict'])
    print(f"Model successfully loaded from '{path}'")


def build_clone_network(network, backprop=True):
    """
    Build a clone network from an existing network, with the option to change the learning rule to backprop

    Parameters
    ----------
    network : :class:`EIANN.network.Network`
        The source network to clone from.
    backprop : bool, optional
        Whether to change the learning rule to backpropagation and ensure 
        backward_steps >= 1. Default is True.

    Returns
    -------
    Network
        A new network object cloned from the source network with the same 
        configuration and seed.
    """
    layer_config = network.layer_config
    projection_config = network.projection_config
    training_kwargs = network.training_kwargs
    seed = network.seed
    if backprop:
        change_learning_rule_to_backprop(projection_config)
        if 'backward_steps' not in training_kwargs or training_kwargs['backward_steps'] < 1:
            training_kwargs['backward_steps'] = 3
    clone_network = nt.Network(layer_config, projection_config, seed=seed, **training_kwargs)
    return clone_network


def network_architectures_match(network, comparison_network):
    '''
    Compare architecture of 2 networks. In order to transfer params to the comparison_network, the comparison_network must have at least the same parameters as the original network.
    Any additional parameters in the original network will be ignored.

    E.g. if the original network has a DendI population, it will be ignored in the comparison, but if the comparison network has a neuron population not present in the original 
    network the networks don't match and the function will return False.
    '''
    for name, param in comparison_network.state_dict().items():
        if name not in network.state_dict():
            return False
        elif network.state_dict()[name].shape != param.shape:
            return False
    return True


def change_learning_rule_to_backprop(projection_config):
    '''
    Recursively update the learning rule to 'Backprop' for all projections that have a learning rule specified.
    '''
    for key, value in projection_config.items():
        if key == 'learning_rule_kwargs':
            projection_config['learning_rule_kwargs'] = {'learning_rate': projection_config['learning_rule_kwargs']['learning_rate']}
        if isinstance(value, dict):
            change_learning_rule_to_backprop(value)
        elif key == 'learning_rule':
            if value not in [None, 'None', 'Backprop']:
                projection_config[key] = 'Backprop'


def rename_population(network, old_name, new_name):
    recursive_dict_rename(network.__dict__, old_name, new_name)

    # Rename populations in module_dict
    for key in list(network.module_dict):
        post_pop, pre_pop = key.split('_')

        for layer_name in list(network.layers):
            if pre_pop.startswith(layer_name):
                pre_pop_name = pre_pop[len(layer_name):]
                if pre_pop_name == old_name:
                    pre_pop = layer_name+new_name
            if post_pop.startswith(layer_name):
                post_pop_name = post_pop[len(layer_name):]
                if post_pop_name == old_name:
                    post_pop = layer_name+new_name
        new_key = f'{post_pop}_{pre_pop}'
        if new_key != key:
            network.module_dict[new_key] = network.module_dict.pop(key)

    # Rename populations in parameter_dict
    for key in list(network.parameter_dict):
        pop_fullname, param_name = key.split('_')

        for layer_name in list(network.layers):
            if pop_fullname.startswith(layer_name):
                pop_name = pop_fullname[len(layer_name):]
                if pop_name == old_name:
                    pop_fullname = layer_name+new_name
        new_key = f'{pop_fullname}_{param_name}'
        if new_key != key:
            network.parameter_dict[new_key] = network.parameter_dict.pop(key)

    # Rename populations in layers and projections
    for layer in network:
        recursive_dict_rename(layer.__dict__, old_name, new_name)
        for population in layer:
            if population.name == old_name:
                population.name = new_name
                population.fullname = layer.name+new_name
            recursive_dict_rename(population.__dict__, old_name, new_name)

            for projection in population:
                projection.name = f'{projection.post.layer.name}{projection.post.name}_{projection.pre.layer.name}{projection.pre.name}'


def recursive_dict_rename(my_dict, old_name, new_name):
    for key in list(my_dict):
        if key == old_name:
            my_dict[new_name] = my_dict.pop(old_name)
        elif isinstance(my_dict[key], dict):
            recursive_dict_rename(my_dict[key], old_name, new_name)
    return 


def set_new_activation(network, activation, population='all', activation_kwargs=None):
    """
    Set a new activation function for a population or all populations in the network
    
    Parameters
    ----------
    network : :class:`EIANN.network.Network`
        The neural network object to set the activation function for.
    activation : str
        The name of the activation function to set.
    population : str or list or :class:`EIANN.network.Population`, optional
        The population or populations to set the activation function for. Default is 'all'.
    activation_kwargs : dict, optional
        The keyword arguments for the activation function. Default is None.
    """

    # Set callable activation function
    if isinstance(activation, str):
        activation_name = activation
        if hasattr(ut, activation):
            activation = getattr(ut, activation)
        elif hasattr(torch.nn.functional, activation):
            activation = getattr(torch.nn.functional, activation) 
        elif hasattr(external, activation):
            activation = getattr(external, activation)
    elif hasattr(activation, '__name__'):
        activation_name = activation.__name__
    else:
        activation_name = None

    if not callable(activation):
        raise RuntimeError \
            ('Population: callable for activation: %s must be imported' % activation)
    if activation_kwargs is None:
        activation_kwargs = {}

    if population in [None, 'all']:
        populations = network.populations.values()
    elif isinstance(population, str):
        populations = [network.populations[population]]
    elif isinstance(population, list):
        populations = population
    else:
        populations = [population]


    for population in populations:
        population.activation = lambda x: activation(x, **activation_kwargs)
        population.activation.name = activation_name
        population.activation.kwargs = activation_kwargs


def recompute_history(network, output_sorting):
    """
    Re-compute activity history, loss history, and weight+bias history
    with new sorting of the output units
    """
    output_pop = network.output_pop

    # Sort activity history
    if output_pop.activity_history.dim() > 2:
        output_pop.activity_history.data = output_pop.activity_history[:, :, output_sorting]
    else:
        output_pop.activity_history.data = output_pop.activity_history[:, output_sorting]

    for t in range(len(network.param_history)):
        # TODO: why is this starting with index == -1?
        # Recompute loss history
        if output_pop.activity_history.dim() > 2:
            output = output_pop.activity_history[t-1, -1, :]
        else:
            output = output_pop.activity_history[t - 1, :]
        target = network.target_history[t-1]
        network.loss_history[t-1] = network.criterion(output, target)

        # Sort weights going to and from the output population
        for proj in output_pop.incoming_projections.values():
            sorted_weights = network.param_history[t][f'module_dict.{proj.name}.weight'][output_sorting,:]
            network.param_history[t][f'module_dict.{proj.name}.weight'] = sorted_weights

        for proj in output_pop.outgoing_projections.values():
            sorted_weights = network.param_history[t][f'module_dict.{proj.name}.weight'][:,output_sorting]
            network.param_history[t][f'module_dict.{proj.name}.weight'] = sorted_weights

        # Sort output bias
        sorted_bias = network.param_history[t][f'parameter_dict.{output_pop.fullname}_bias'][output_sorting]
        network.param_history[t][f'parameter_dict.{output_pop.fullname}_bias'] = sorted_bias

    # Update network with re-sorted weights from final state
    network.load_state_dict(network.param_history[-1])


def get_binned_mean_population_attribute_history_dict(network, attr_name, bin_size=100, abs=False):
    all_pop_attr_history_list = []
    binned_attr_history_dict = {}
    num_patterns = network.output_pop.activity_history.shape[0]
    num_bins = num_patterns // bin_size
    excess = num_patterns % bin_size
    steps = torch.arange(bin_size, bin_size * (num_bins + 1), bin_size)
    
    for pop_name, pop in network.populations.items():
        attr_history = pop.get_attribute_history(attr_name)
        if attr_history is None:
            continue
        attr_history = attr_history.detach().clone()
        if excess > 0:
            attr_history = attr_history[:-excess]
        num_units = pop.size
        binned_attr_history = attr_history.reshape(num_bins, bin_size, num_units)
        if abs:
            binned_attr_history = torch.abs(binned_attr_history)
        binned_attr_history = torch.mean(binned_attr_history, dim=1)
        all_pop_attr_history_list.append(binned_attr_history)
        binned_attr_history_dict[pop_name] = torch.mean(binned_attr_history, dim=1)
    
    binned_attr_history_tensor = torch.concatenate(all_pop_attr_history_list, dim=1)
    binned_attr_history_dict['all'] = torch.mean(binned_attr_history_tensor, dim=1)
    return steps, binned_attr_history_dict


def get_optimal_sorting(network, test_dataloader, plot=False):
    """
    Find optimal output unit sorting by measuring test loss on re-sorted activity throughout training history.

    This function evaluates the network at every point in its training history, computes the optimal
    sorting of output units based on average class responses, and returns the sorting that yields
    the minimum loss.

    Parameters
    ----------
    network : object
        Neural network object containing output_pop, param_history, val_target, criterion, and 
        forward method.
    test_dataloader : torch.utils.data.DataLoader
        Test data loader that must contain a single large batch for evaluation.
    plot : bool, optional
        Whether to generate a plot showing optimal loss history with the minimum loss point 
        highlighted. Default is False.

    Returns
    -------
    min_loss_sorting : torch.Tensor
        Indices representing the optimal sorting of output units that minimizes test loss.
    """
    assert len(test_dataloader)==1, 'Dataloader must have a single large batch'
    output_pop = network.output_pop

    optimal_loss_history = []
    sorting_history = []
    history_len = output_pop.activity_history.shape[0]
    idx, test_data, test_target = next(iter(test_dataloader))

    from tqdm.autonotebook import tqdm
    for t in tqdm(range(history_len)):
        network.load_state_dict(network.param_history[t])
        output = network.forward(test_data, no_grad=True)  # row=patterns, col=units

        # Get average output for each label class
        num_units = network.val_target.shape[1]
        num_labels = num_units
        avg_output = torch.zeros(num_labels, num_units)
        targets = torch.argmax(network.val_target, dim=1)  # convert from 1-hot vector to int label
        for label in range(num_labels):
            label_idx = torch.where(targets == label)  # find all instances of given label
            avg_output[label, :] = torch.mean(output[label_idx], dim=0)

        # Find optimal output unit (column) sorting given average responses
        optimal_sorting = get_diag_argmax_row_indexes(avg_output.T)
        sorted_activity = avg_output[:, optimal_sorting]
        optimal_loss = network.criterion(sorted_activity, torch.eye(num_units))
        optimal_loss_history.append(optimal_loss)
        sorting_history.append(optimal_sorting)

        # Pick timepoint with lowest sorted loss
    optimal_loss_history = torch.stack(optimal_loss_history)
    min_loss_idx = torch.argmin(optimal_loss_history)
    min_loss_sorting = sorting_history[min_loss_idx]

    if plot:
        plt.scatter(min_loss_idx,torch.min(optimal_loss_history),color='red')
        plt.plot(optimal_loss_history)
        plt.title('optimal loss history (re-sorted for each point)')
        plt.show()

    return min_loss_sorting


def get_diag_argmax_row_indexes(data):
    """
    Sort the rows of a square matrix such that whenever row argmax and col argmax are equal, that value appears
    on the diagonal. Returns row indexes.

    :param data: 2d array; square matrix
    :return: array of int
    """
    data = np.array(data)
    if data.shape[0] != data.shape[1]:
        raise Exception('get_diag_argmax_row_indexes: data must be a square matrix')
    dim = data.shape[0]
    avail_row_indexes = list(range(dim))
    avail_col_indexes = list(range(dim))
    final_row_indexes = np.empty_like(avail_row_indexes)
    while(len(avail_col_indexes) > 0):
        row_selectivity = np.zeros_like(avail_row_indexes)
        row_max = np.max(data[avail_row_indexes, :][:, avail_col_indexes], axis=1)
        row_mean = np.mean(data[avail_row_indexes,:][:,avail_col_indexes], axis=1)
        nonzero_indexes = np.where(row_mean > 0)
        row_selectivity[nonzero_indexes] = row_max[nonzero_indexes] / row_mean[nonzero_indexes]

        row_index = avail_row_indexes[np.argsort(row_selectivity)[-1]]
        col_index = avail_col_indexes[np.argmax(data[row_index,avail_col_indexes])]
        final_row_indexes[col_index] = row_index
        avail_row_indexes.remove(row_index)
        avail_col_indexes.remove(col_index)
    return final_row_indexes


def sort_by_val_history(network, val_dataloader, plot=False):
    """
    Find the sorting giving the best argmax across the full validation history

    :param network:
    :param plot:
    :return: tuple (int, tensor): index of the point with lowest loss (index relative only to the validation points);
        optimal sorting indices for the point with lowest loss
    """
    output_pop = network.output_pop

    num_units = output_pop.size
    num_labels = num_units
    num_patterns = network.val_output_history.shape[1]
    
    sorting_history = []
    optimal_loss_history = []
    optimal_accuracy_history = []
    sorted_idx_history = []
    
    _, _, val_target = next(iter(val_dataloader))
    targets = torch.argmax(val_target, dim=1)  # convert from 1-hot vector to int label
    
    for output in network.val_output_history:
        # Get average output for each label class
        avg_output = torch.zeros(num_labels, num_units)
        for label in range(num_labels):
            label_idx = torch.where(targets == label)  # find all instances of given label
            avg_output[label, :] = torch.mean(output[label_idx], dim=0)

        # Find optimal output unit (column) sorting given average responses
        optimal_sorting = get_diag_argmax_row_indexes(avg_output.T)
        sorted_activity = output[:, optimal_sorting]
        optimal_loss = network.criterion(sorted_activity, val_target)
        optimal_loss_history.append(optimal_loss.item())
        optimal_accuracy = 100 * torch.sum(torch.argmax(sorted_activity, dim=1) == targets) / num_patterns
        optimal_accuracy_history.append(optimal_accuracy.item())
        sorting_history.append(optimal_sorting)

    # Pick timepoint with lowest sorted loss
    optimal_loss_history = torch.tensor(optimal_loss_history)
    min_loss_idx = torch.argmin(optimal_loss_history)
    min_loss_sorting = sorting_history[min_loss_idx]

    if plot:
        fig = plt.figure()
        plt.scatter(min_loss_idx, torch.min(optimal_loss_history), color='red')
        plt.plot(optimal_loss_history)
        plt.title('optimal loss history (re-sorted for each point)')
        fig.show()

    return min_loss_idx, min_loss_sorting


def sort_by_class_averaged_val_output(network, val_dataloader, index=-1):
    """
    Find the sorting for the class-averaged output at the given index of the validation history

    :param network:
    :param val_dataloader:
    :param index: int
    :return: sorted_output_idx: array of int
    """
    num_units = network.output_pop.size
    num_labels = num_units
    
    output = network.val_output_history[index]
    
    # Get average output for each label class
    avg_output = torch.zeros(num_labels, num_units)
    _, _, val_targets = next(iter(val_dataloader))
    targets = torch.argmax(val_targets, dim=1)  # convert from 1-hot vector to int label
    for label in range(num_labels):
        label_idx = torch.where(targets == label)  # find all instances of given label
        avg_output[label, :] = torch.mean(output[label_idx], dim=0)
    
    # Find optimal output unit (column) sorting given average responses
    sorted_output_idx = get_diag_argmax_row_indexes(avg_output.T)
    
    return sorted_output_idx


def sort_unsupervised_by_test_batch_autoenc(network, test_dataloader):
    """
    Run a test batch and return the output unit sorting that best matches the output activity to the test labels.
    :param network:
    :param test_dataloader:
    :return: tensor of int
    """
    assert len(test_dataloader) == 1, 'Dataloader must have a single large batch'

    network.test(test_dataloader)

    # Find optimal output unit (column) sorting
    optimal_sorting = get_diag_argmax_row_indexes(network.output_pop.activity.T)

    return optimal_sorting


def sort_unsupervised_by_best_epoch(network, target, plot=False):

    output_pop = network.output_pop

    dynamic_epoch_loss_history = []
    sorted_idx_history = []

    if output_pop.activity_history.dim() > 2:
        output_history = output_pop.activity_history[network.sorted_sample_indexes, -1, :]
    else:
        output_history = output_pop.activity_history[network.sorted_sample_indexes, :]
    start = 0
    while start < output_history.shape[0]:
        end = start + target.shape[0]
        epoch_output = output_history[start:end, :]
        sorted_idx = get_diag_argmax_row_indexes(epoch_output.T)
        loss = network.criterion(epoch_output[:, sorted_idx], target)
        dynamic_epoch_loss_history.append(loss)
        sorted_idx_history.append(sorted_idx)
        start += target.shape[0]

    best_index = np.where(dynamic_epoch_loss_history == np.min(dynamic_epoch_loss_history))[0][0]
    sorted_idx = sorted_idx_history[best_index]
    epoch_loss_history = []
    start = 0
    while start < output_pop.activity_history.shape[0]:
        end = start + target.shape[0]
        epoch_output = output_history[start:end, :]
        loss = network.criterion(epoch_output[:, sorted_idx], target)
        epoch_loss_history.append(loss)
        start += target.shape[0]

    if plot:
        fig = plt.figure()
        plt.plot(dynamic_epoch_loss_history, label='Dynamic')
        plt.plot(epoch_loss_history, label='Sorted by peak')
        plt.xlabel('Training epochs')
        plt.ylabel('MSE loss')
        plt.title('Epoch training loss')
        plt.legend(loc='best', frameon=False)
        fig.tight_layout()
        fig.show()

    return sorted_idx


def recompute_validation_loss_and_accuracy(network, val_dataloader, sorted_output_idx, store=False):
    """
    Recompute validation loss and accuracy using sorted output indices.

    This function applies the provided sorting to the validation output history and 
    recomputes loss and accuracy metrics for each batch in the history.

    Parameters
    ----------
    network : object
        Neural network object containing val_output_history and criterion attributes.
    val_dataloader : torch.utils.data.DataLoader
        Validation data loader containing target data.
    sorted_output_idx : array-like
        Indices for sorting output units to reorder the validation output history.
    store : bool, optional
        Whether to store the sorted results back to the network object. If True, updates
        network.val_output_history, network.val_loss_history, and network.val_accuracy_history.
        Default is False.

    Returns
    -------
    sorted_val_loss_history : torch.Tensor
        Loss values computed for each batch using sorted output.
    sorted_val_accuracy_history : torch.Tensor
        Accuracy percentages computed for each batch using sorted output.
    """

    # Sort output history
    val_output_history = network.val_output_history[:, :, sorted_output_idx]

    # Recompute loss
    sorted_val_loss_history = []
    sorted_val_accuracy_history = []
    num_patterns = val_output_history.shape[1]
    _, _, val_target = next(iter(val_dataloader))
    targets = torch.argmax(val_target, dim=1)
    for batch_output in val_output_history:
        loss = network.criterion(batch_output, val_target).item()
        accuracy = 100 * torch.sum(torch.argmax(batch_output, dim=1) == targets) / num_patterns

        sorted_val_loss_history.append(loss)
        sorted_val_accuracy_history.append(accuracy.item())

    sorted_val_loss_history = torch.tensor(sorted_val_loss_history)
    sorted_val_accuracy_history = torch.tensor(sorted_val_accuracy_history)

    if store:
        network.val_output_history = val_output_history
        network.val_loss_history = sorted_val_loss_history
        network.val_accuracy_history = sorted_val_accuracy_history

    return sorted_val_loss_history, sorted_val_accuracy_history


def recompute_train_loss_and_accuracy(network, sorted_output_idx=None, bin_size=100, plot=False, title=None):
    """
    Recompute training loss and accuracy with optional output sorting and binning.

    This function processes the training output history by optionally applying output unit 
    sorting and binning the data to compute average loss and accuracy over training steps.
    The binned results are stored in the network object.

    Parameters
    ----------
    network : object
        Neural network object containing Output.E.activity_history, target_history, and 
        criterion attributes.
    sorted_output_idx : array-like, optional
        Indices for sorting output units. If None, uses original ordering. Default is None.
    bin_size : int, optional
        Number of samples per bin for averaging loss and accuracy. Default is 100.
    plot : bool, optional
        Whether to generate plots showing training loss and accuracy over time. 
        Default is False.
    title : str, optional
        Additional title text to append to plot titles. Default is None.

    Returns
    -------
    binned_train_loss_steps : torch.Tensor
        Step indices corresponding to each bin.
    sorted_loss_history : torch.Tensor
        Binned loss values over training steps.
    sorted_accuracy_history : torch.Tensor
        Binned accuracy percentages over training steps.
    """

    # Sort output history
    if network.Output.E.activity_history.dim() > 2:
        output_history = network.Output.E.activity_history[:, -1, :]
    else:
        output_history = network.Output.E.activity_history
    if sorted_output_idx is not None:
        output_history = output_history[:, sorted_output_idx]
    target_history = network.target_history
    num_units = output_history.shape[1]
    num_patterns = output_history.shape[0]

    # Bin output history to compute average loss & accuracy over training
    num_bins = num_patterns // bin_size
    excess = num_patterns % bin_size
    if excess > 0:
        output_history = output_history[:-excess]
        target_history = target_history[:-excess]
    
    binned_output_history = output_history.reshape(num_bins, bin_size, num_units)
    binned_target_history = target_history.reshape(num_bins, bin_size, num_units)
    binned_train_loss_steps = torch.arange(bin_size, bin_size * (num_bins + 1), bin_size)

    # Recompute loss
    sorted_loss_history = []
    sorted_accuracy_history = []

    for (batch_output, batch_target) in zip(binned_output_history, binned_target_history):
        loss = network.criterion(batch_output, batch_target).item()
        predictions = torch.argmax(batch_output, dim=1)
        labels = torch.argmax(batch_target, dim=1)
        accuracy = 100 * torch.sum(predictions == labels) / bin_size

        sorted_loss_history.append(loss)
        sorted_accuracy_history.append(accuracy.item())

    sorted_loss_history = torch.tensor(sorted_loss_history)
    sorted_accuracy_history = torch.tensor(sorted_accuracy_history)

    if title is None:
        title_str = ''
    else:
        title_str = ': %s' % str(title)
    if plot:
        fig = plt.figure()
        plt.plot(binned_train_loss_steps, sorted_loss_history)

        plt.title('Train Loss%s' % title_str)
        plt.ylabel('Loss')
        plt.xlabel('Train steps')
        plt.ylim((0, plt.ylim()[1]))
        fig.show()

        fig = plt.figure()
        plt.plot(binned_train_loss_steps, sorted_accuracy_history)
        plt.title('Train accuracy%s' % title_str)
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Train steps')
        plt.ylim((0, max(100, plt.ylim()[1])))
        fig.show()
    
    network.binned_train_loss_steps = binned_train_loss_steps
    network.binned_sorted_train_loss_history = sorted_loss_history
    network.binned_sorted_train_accuracy_history = sorted_accuracy_history
    
    return binned_train_loss_steps, sorted_loss_history, sorted_accuracy_history

