import EIANN._network as nt
import EIANN.utils as ut
import os
import pickle
import dill
import datetime


def build_EIANN_from_config(config_path, network_seed=42, config_format='normal'):
    '''
    Build an EIANN network from a config file
    '''
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


def save_network(network, path=None, dir='saved_networks', file_name_base=None, disp=True):
    if path is None:
        if file_name_base is None:
            file_name_base = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path = '%s/%s.pkl' % (dir, file_name_base)
        if not os.path.exists(dir):
            os.makedirs(dir)
    elif os.path.exists(path):
        print(f"WARNING: File '{path}' already exists. Overwriting...")

    with open(path, 'wb') as f:
        dill.dump(network, f)
    if disp:
        print(f"Saved network to '{path}'")


def load_network(filepath):
    print(f"Loading network from '{filepath}'")
    with open(filepath, 'rb') as f:
        network = dill.load(f)
    for layer in network:
        for population in layer:
            for attr_name in population.attribute_history_dict:
                population.register_attribute_history(attr_name)
            for projection in population:
                for attr_name in projection.attribute_history_dict:
                    projection.register_attribute_history(attr_name)
    print(f"Network successfully loaded from '{filepath}'")
    return network
    

def save_network_dict(network, path=None, dir='saved_networks', file_name_base=None, disp=True):
    """

    :param path: str (path to file)
    :param dir: str (path to dir)
    :param file_name_base: str
    :param disp: str
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


def load_network_dict(network, filepath):
    print(f"Loading model data from '{filepath}'...")
    with open(filepath, 'rb') as file:
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
    print(f"Model successfully loaded from '{filepath}'")


def build_clone_network(network, backprop=True):
    '''
    Build a clone network from an existing network, with the option to change the learning rule to backprop
    '''
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


def count_dict_elements(dict1, leaf=0):
    nodes = dict1.keys()
    for node in nodes:
        subnode = dict1[node]
        if isinstance(subnode, dict):
            leaf = count_dict_elements(subnode, leaf)
        else:
            leaf += 1
    return leaf
