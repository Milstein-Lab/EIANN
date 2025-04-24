import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from skimage import metrics
import scipy.stats as stats
import copy
from tqdm.autonotebook import tqdm
from scipy import signal

from EIANN.utils import data_utils, network_utils
import EIANN.plot as pt

    
def compute_test_activity(network, test_dataloader, class_average:bool, sort:bool):
    """
    Compute activity for all populations in the network on the test dataloader
    """
    assert len(test_dataloader)==1, 'Dataloader must have a single large batch'
    idx, data, target = next(iter(test_dataloader))
    network.forward(data, no_grad=True)
    pattern_labels = torch.argmax(target, dim=1)
    sorted_pattern_labels, pattern_sort_idx = torch.sort(pattern_labels) 
    if class_average:
        num_labels = target.shape[1]
        sorted_pattern_labels = torch.arange(num_labels)

    # Compute sorted/averaged test activity for each population
    reversed_populations = list(reversed(network.populations.values())) # start with the output population
    pop_activity_dict = {}
    unit_labels_dict = {}    
    for i, population in enumerate(reversed_populations):
        pop_activity = population.activity

        if class_average: 
            # Average activity across the patterns (rows) for each class label
            num_units = pop_activity.shape[1]
            avg_pop_activity = torch.zeros(num_labels, num_units)
            for label in range(num_labels):
                label_idx = torch.where(pattern_labels == label)
                avg_pop_activity[label] = torch.mean(pop_activity[label_idx], dim=0)
            pop_activity = avg_pop_activity

        if sort:
            if not class_average:
                # Sort patterns (rows) of pop_activity by label
                pop_activity = pop_activity[pattern_sort_idx]

            # Sort neurons (columns) by argmax of activity
            if population is network.output_pop:
                unit_sort_idx = torch.arange(0, network.output_pop.size)
                unit_labels = pattern_labels[torch.argmax(pop_activity, dim=0)]
            else:
                silent_unit_idx = torch.where(torch.sum(pop_activity, dim=0) == 0)[0]
                active_unit_idx = torch.where(torch.sum(pop_activity, dim=0) > 0)[0]
                preferred_input_active = sorted_pattern_labels[torch.argmax(pop_activity[:,active_unit_idx], dim=0)]
                unit_labels, sort_idx = torch.sort(preferred_input_active)
                unit_sort_idx = torch.cat([active_unit_idx[sort_idx], silent_unit_idx])
                unit_labels = torch.cat([unit_labels, torch.zeros(len(silent_unit_idx))*torch.nan])
            pop_activity = pop_activity[:,unit_sort_idx]
        else:
            unit_labels = sorted_pattern_labels[torch.argmax(pop_activity, dim=0)]
            
        pop_activity_dict[population.fullname] = pop_activity
        unit_labels_dict[population.fullname] = unit_labels
    
    if sort:
        pattern_labels = sorted_pattern_labels

    return pop_activity_dict, pattern_labels, unit_labels_dict


def compute_average_activity(pop_activity_dict, pattern_labels):
    """
    Compute average activity for each population
    """
    # Compute average activity for each population
    avg_pop_activity_dict = {}
    for pop_name, pop_activity in pop_activity_dict.items():
        avg_pop_activity = []
        for class_label in torch.unique(pattern_labels):
            class_idx = torch.where(pattern_labels == class_label)
            avg_class_activity = torch.mean(pop_activity[class_idx], dim=0)
            avg_pop_activity.append(avg_class_activity)
        avg_pop_activity_dict[pop_name] = torch.stack(avg_pop_activity)

    return avg_pop_activity_dict


def compute_test_accuracy(output, labels):
    percent_correct = 100 * torch.sum(torch.argmax(output, dim=1) == labels) / len(labels)
    percent_correct = torch.round(percent_correct, decimals=2)       
    return percent_correct 


def compute_test_activity_dynamics(network, test_dataloader):
    assert len(test_dataloader)==1, 'Dataloader must have a single large batch'
    idx, data, target = next(iter(test_dataloader))
    network.forward(data, no_grad=True, store_dynamics=True)
    reversed_populations = list(reversed(network.populations.values())) # start with the output population
    pop_dynamics_dict = {population.fullname: torch.stack(population.forward_steps_activity) for population in reversed_populations}
    return pop_dynamics_dict


def compute_dParam_history(network):
    dParam_history = {name: [] for name in network.state_dict()}

    for i in range(len(network.param_history) - 1):
        state_dict1 = network.param_history[i]
        state_dict2 = network.param_history[i + 1]

        for param_name, param_val1, param_val2 in zip(state_dict1.keys(), state_dict1.values(), state_dict2.values()):
            d_param = param_val2 - param_val1
            dParam_history[param_name].append(d_param)

    for name, value in dParam_history.items():
        dParam_history[name] = torch.stack(value)

    return dParam_history


def compute_alternate_dParam_history(dataloader, network, network2=None, save_path=None, batch_size=None, constrain_params=None):
    """
    Iterate through the parameter history and compute both the actual dParam at each training step 
    and the alternate dParam predicted from either a backprop/gradient descent step or from an identical 
    network with a custom learning rule

    :param dataloader:
    :param network: network with param_history
    :param network2: (optional) network with a custom learning rule
    :param save_path: (optional) path to save network clone with new param_history
    :param batch_size: (optional) batch size to use for alternate dParam computation
    :param constrain_params: (optional) constrain weights and biases to valid range (e.g. to obey Dale's law)
    :return: network object with dParam_history, alternate_dParam_history
    """
    assert len(network.param_history)>0, 'Network must have param_history'
    assert len(dataloader)==1, 'Dataloader must have a single large batch'

    idx, data, target = next(iter(dataloader))

    if batch_size is None:
        print('Warning: batch_size not specified, default batch_size is full dataset')
        sample_data = data
        sample_target = target.squeeze()

    if network2 is None: # Turn on gradient tracking to compute backprop dW
        test_network = network_utils.build_clone_network(network, backprop=True)
    else:
        test_network = network_utils.build_clone_network(network2, backprop=False)
        if "Backprop" in str(test_network.backward_methods):
            assert test_network.backward_steps > 0, "Backprop network must have backward_steps>0!"
    test_network.batch_size = batch_size
    test_network.constrain_params = constrain_params

    # Align param_history and prev_param_history (exclude initial params)
    if len(network.prev_param_history)==0: # if interval step is 1
        print('WARNING: network.prev_param_history is empty, using network.param_history instead')
        prev_param_history = network.param_history[:-1] # exclude final params
        param_history = network.param_history[1:] # exclude initial params
        param_history_steps = network.param_history_steps[1:]
    else:
        prev_param_history = network.prev_param_history
        param_history = network.param_history
        param_history_steps = network.param_history_steps

    # Create a list of all the E-to-E projections to compare
    forward_E_params = []
    for proj_name, param in network.projections.items(): 
        pre_name, post_name = proj_name.split('_')
        if pre_name[-1]=='E' and post_name[-1]=='E' and param.weight.is_learned and not param.update_phase=='B':
            forward_E_params.append(proj_name)

    actual_dParam_history_dict = {name:[] for name,param in network.named_parameters() if name.split('.')[1] in forward_E_params}
    actual_dParam_history_all = []
    actual_dParam_history_dict_stepaveraged = {name:[] for name,param in network.named_parameters() if name.split('.')[1] in forward_E_params}
    actual_dParam_history_stepaveraged_all = []
    predicted_dParam_history_dict = {name:[] for name,param in network.named_parameters() if name.split('.')[1] in forward_E_params}
    predicted_dParam_history_all = []

    for t in tqdm(range(len(param_history))):  
        # Load params into network
        state_dict = prev_param_history[t]
        if network2 is not None: # Select only params that are in both networks
            state_dict = {name:param for name,param in state_dict.items() if name in test_network.state_dict()}
        test_network.load_state_dict(state_dict)
        test_network.prev_param_history.append(copy.deepcopy(state_dict))
 
        # Compute forward pass (using the same data sample order stored in the original network)
        if batch_size is not None:
            sample_idx = network.sample_order[param_history_steps[t]:param_history_steps[t]+batch_size]            
            sample_data = data[sample_idx]
            sample_target = target[sample_idx]
        output = test_network.forward(sample_data)
        loss = test_network.criterion(output, sample_target)   

        # Compute backward pass param update
        if network2 is None:
            # Regular backprop update
            test_network.zero_grad()
            loss.backward() 
            test_network.optimizer.step()    
            if test_network.constrain_params==True:
                test_network.constrain_weights_and_biases()
        else:
            # Backward update specified by learning rules in network2
            for backward in test_network.backward_methods:
                backward(test_network, output, sample_target)

            # Step weights and biases
            for layer in test_network:
                for population in layer:
                    if population.include_bias:
                        population.bias_learning_rule.step()
                    for projection in population:
                        projection.learning_rule.step()
            if test_network.constrain_params is None or test_network.constrain_params==True:
                test_network.constrain_weights_and_biases()

        new_state_dict = test_network.state_dict() 
        test_network.param_history.append(copy.deepcopy(new_state_dict))

        # Compute the predicted dParam (from the test network)
        dParam_vec = []
        for key in predicted_dParam_history_dict:
            dParam = (new_state_dict[key]-state_dict[key])
            dParam = dParam/(torch.norm(dParam)+1e-10)
            predicted_dParam_history_dict[key].append(dParam)
            dParam_vec.append(dParam.flatten())
        predicted_dParam_history_all.append(torch.cat(dParam_vec))
        
        # Compute the actual dParam of the first network
        next_state_dict = param_history[t]
        dParam_vec = []
        for key in actual_dParam_history_dict:
            dParam = (next_state_dict[key]-state_dict[key])
            dParam = dParam/(torch.norm(dParam)+1e-10)
            actual_dParam_history_dict[key].append(dParam)
            dParam_vec.append(dParam.flatten())
        actual_dParam_history_all.append(torch.cat(dParam_vec))

        # Compute the actual dParam of the first network (step-averaged), computed between consecutive saved checkpoints
        if t<len(param_history)-1:
            next_state_dict = prev_param_history[t+1]
            dParam_vec = []
            for key in actual_dParam_history_dict_stepaveraged:
                dParam = (next_state_dict[key]-state_dict[key])
                actual_dParam_history_dict_stepaveraged[key].append(dParam)
                dParam_vec.append(dParam.flatten())
            actual_dParam_history_stepaveraged_all.append(torch.cat(dParam_vec))

    predicted_dParam_history_dict['all_params'] = predicted_dParam_history_all
    actual_dParam_history_dict['all_params'] = actual_dParam_history_all
    actual_dParam_history_dict_stepaveraged['all_params'] = actual_dParam_history_stepaveraged_all

    test_network.predicted_dParam_history = predicted_dParam_history_dict
    test_network.actual_dParam_history = actual_dParam_history_dict
    test_network.actual_dParam_history_stepaveraged = actual_dParam_history_dict_stepaveraged

    if save_path is not None:
        network_utils.save_network(test_network, save_path)

    return test_network


def compute_vector_angle(vector1, vector2):
    '''
    Compute the angle between two vectors.
    '''
    vector1, vector2 = vector1.double(), vector2.double() # increase the precision to reduce floating point errors
    dot_product = torch.dot(vector1, vector2)
    norm_product = torch.norm(vector1) * torch.norm(vector2)
    cos_angle = dot_product / norm_product
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)  # clamp to avoid floating point rounding errors
    angle_rad = torch.acos(cos_angle)
    angle_deg = torch.rad2deg(angle_rad)
    return angle_deg.type(torch.float32)


def compute_dW_angles_vs_BP(predicted_dParam_history, actual_dParam_history, plot=False, binarize=False, only_updated_params=False):

    '''
    Compute the angle between the actual and predicted parameter updates (dW) for each training step.
    The angle is computed as the arccosine of the dot product between the two vectors, normalized by the product of their norms (resulting in a value between 0 and 180 degrees).

    :param test_network: network generated from compute_alternate_dParam_history (with actual_dParam_history and predicted_dParam_history)
    :param plot: bool, plot the angle for each parameter
    :return: dictionary of angles (in degrees)
    '''
    print('Computing angles between actual and predicted parameter updates...')
    
    if plot:
        n_params = len(actual_dParam_history)
        fig, axes = plt.subplots(n_params, 1, figsize=(8,n_params*2))
        ax_top = axes[0]
  
    angles = {}
    for i, param_name in enumerate(actual_dParam_history):
        angles[param_name] = []
        
        for t, (predicted_dParam, actual_dParam) in enumerate(zip(predicted_dParam_history[param_name], actual_dParam_history[param_name])):
            # Compute angle between parameter update (dW) vectors
            predicted_dParam = predicted_dParam.flatten()
            actual_dParam = actual_dParam.flatten()
            if binarize:
                predicted_dParam = torch.sign(predicted_dParam)
                actual_dParam = torch.sign(actual_dParam)
            if only_updated_params:
                updated_idx = torch.where(actual_dParam != 0)
                # if len(updated_idx[0]) != len(actual_dParam):
                #     print(f"Percentage updated = {len(updated_idx[0])/len(actual_dParam)*100:.2f}%")
                predicted_dParam = predicted_dParam[actual_dParam != 0]
                actual_dParam = actual_dParam[actual_dParam != 0]
                
            angle = compute_vector_angle(predicted_dParam, actual_dParam)
            angles[param_name].append(angle)
            if torch.isnan(angle):
                print(f'Warning: angle is NaN at step {t}, {param_name}, t={t}, Pred. norm.={torch.norm(predicted_dParam)}, Actual norm.={torch.norm(actual_dParam)}')
                # return predicted_dParam, actual_dParam

        if plot:
            ax = axes[n_params-(i+1)]
            ax.plot(angles[param_name])
            ax_top.plot(angles[param_name], color='gray', alpha=0.3)
            ax.plot([0, len(angles[param_name])], [90, 90], color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Training step')
            ax.set_ylabel('Angle between \nlearning rules (degrees)')

            max_angle = max(95, np.nanmax(angles[param_name]))
            ax.set_ylim(bottom=-5, top=max_angle)
            ax.set_yticks(np.arange(0, max_angle+1, 30))
            if i == n_params-1:
                ax.set_ylim(bottom=-5, top=120)
                ax.set_yticks(np.arange(0, 121, 30))

            avg_angle = np.nanmean(angles[param_name])
            ax.text(0.03, 0.12, f'Avg angle = {avg_angle:.2f} degrees', transform=ax.transAxes)
            if '.' in param_name:
                param_name = param_name.split('.')[1]
            ax.set_title(param_name)
            plt.tight_layout()
            fig.show()

    return angles


def compute_feedback_weight_angle_history(network, plot=False, ax=None):
    '''
    Compute the angle between the actual and predicted parameter updates (dW) for each training step.
    '''
    layers = list(network.layers)
    angles = {f"{layer}E_{next_layer}E": [] for layer, next_layer in zip(layers[1:-1], layers[2:])}
    angles["all_params"] = []
    
    for params in network.param_history:
        forward_weights_all = []
        backward_weights_all = []
        for i, layer in enumerate(layers[1:-1], start=1):
            next_layer = layers[i+1]
            forward_weights = params[f"module_dict.{next_layer}E_{layer}E.weight"].flatten()
            forward_weights_all.append(forward_weights/(torch.norm(forward_weights)+1e-10))
            backward_projection_name = f"module_dict.{layer}E_{next_layer}E.weight"
            if backward_projection_name in params:
                backward_weights = params[backward_projection_name].T.flatten()
                backward_weights_all.append(backward_weights/(torch.norm(backward_weights)+1e-10))
            else:
                return []
            angle = compute_vector_angle(forward_weights, backward_weights)
            angles[f"{layer}E_{next_layer}E"].append(angle.item())

        if len(forward_weights_all)>0 and len(backward_weights_all)>0:
            angle = compute_vector_angle(torch.cat(forward_weights_all), torch.cat(backward_weights_all))
            angles["all_params"].append(angle.item())
        else: 
            return []

    if plot:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5,3))
        else:
            fig = ax.get_figure()
        steps = network.param_history_steps
        for i,projection_pair in enumerate(angles):
            ax.plot(steps,angles[projection_pair], label=projection_pair)
        ax.legend()
        ax.set_xlabel('Training step')
        ax.set_ylabel('Angle \n(forward vs backward weights)')
        ax.set_ylim(bottom=-2, top=90)
        ax.set_yticks(np.arange(0, 91, 30))
    return angles


def compute_feedback_dW_angle_history(network, plot=False, ax=None):
    '''
    Compute the angle between the actual and predicted parameter updates (dW) for each training step.
    '''
    layers = list(network.layers)
    angles = {f"{layer}E_{next_layer}E": [] for layer, next_layer in zip(layers[1:-1], layers[2:])}
    angles["all_params"] = []
    
    for params, prev_params in zip(network.param_history, network.prev_param_history):
        forward_weights_all = []
        backward_weights_all = []
        for i, layer in enumerate(layers[1:-1], start=1):
            next_layer = layers[i+1]
            forward_dW = params[f"module_dict.{next_layer}E_{layer}E.weight"].flatten() - prev_params[f"module_dict.{next_layer}E_{layer}E.weight"].flatten()

            forward_weights_all.append(forward_dW/(torch.norm(forward_dW)+1e-10))
            backward_projection_name = f"module_dict.{layer}E_{next_layer}E.weight"
            if backward_projection_name in params:
                backward_dW = params[backward_projection_name].T.flatten() - prev_params[backward_projection_name].T.flatten()
                backward_weights_all.append(backward_dW/(torch.norm(backward_dW)+1e-10))
            else:
                return []
            angle = compute_vector_angle(forward_dW, backward_dW)
            angles[f"{layer}E_{next_layer}E"].append(angle.item())

        angle = compute_vector_angle(torch.cat(forward_weights_all), torch.cat(backward_weights_all))
        angles["all_params"].append(angle.item())

    if plot:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5,3))
        else:
            fig = ax.get_figure()
        steps = network.param_history_steps
        for i,projection_pair in enumerate(angles):
            ax.plot(steps,angles[projection_pair], label=projection_pair)
        ax.legend()
        ax.set_xlabel('Training step')
        ax.set_ylabel('Angle \n(forward vs backward weights)')
        ax.set_ylim(bottom=-2, top=90)
        ax.set_yticks(np.arange(0, 91, 30))
    return angles


def recompute_dParam_history_all(network):
    '''
    Recompute the 'all_params' key in dParam history (in case any changes are made to the dParam_history dicts)
    '''

    actual_dParam_history_all = []
    actual_dParam_history_stepaveraged_all = []
    predicted_dParam_history_all = []

    for t in range(len(network.param_history)):
        # Compute the predicted dParam (from the test network)
        dParam_vec = []
        for name,dParam_history in network.predicted_dParam_history.items():
            dParam_vec.append(dParam_history[t].flatten())
        predicted_dParam_history_all.append(torch.cat(dParam_vec))

        # Compute the actual dParam of the first network
        dParam_vec = []
        for name,dParam_history in network.actual_dParam_history.items():
            dParam_vec.append(dParam_history[t].flatten())
        actual_dParam_history_all.append(torch.cat(dParam_vec))

        # Compute the actual dParam of the first network (step-averaged), computed between consecutive saved checkpoints
        if t<len(network.param_history)-1:
            dParam_vec = []
            for name,dParam_history in network.actual_dParam_history_stepaveraged.items():
                dParam_vec.append(dParam_history[t].flatten())
            actual_dParam_history_stepaveraged_all.append(torch.cat(dParam_vec))

    network.actual_dParam_history['all_params'] = actual_dParam_history_all
    network.actual_dParam_history_stepaveraged['all_params'] = actual_dParam_history_stepaveraged_all
    network.predicted_dParam_history['all_params'] = predicted_dParam_history_all


def compute_sparsity_selectivity_history(network, test_dataloader):
    sparsity_history_dict = {population: [] for population in network.populations}
    selectivity_history_dict = {population: [] for population in network.populations}
    idx, data, target = next(iter(test_dataloader))
    for params in tqdm(network.param_history):
        network.load_state_dict(params)
        network.forward(data, no_grad=True)
        for pop_name in network.populations:
            if "Input" in pop_name:
                continue
            activity = network.populations[pop_name].activity
            avg_sparsity = torch.mean(compute_sparsity(activity))
            avg_selectivity = torch.mean(compute_selectivity(activity))
            sparsity_history_dict[pop_name].append(avg_sparsity.item())
            selectivity_history_dict[pop_name].append(avg_selectivity.item())
    return sparsity_history_dict, selectivity_history_dict


def compute_sparsity(population_activity):
    """
    Sparsity metric from (Vinje & Gallant 2000): https://www.science.org/doi/10.1126/science.287.5456.1273
    """
    num_units = population_activity.shape[1] #dims: 0=batch/samples, 1=units
    activity_fraction = (torch.sum(population_activity,dim=1) / num_units) ** 2 / (torch.sum((population_activity**2 / num_units),dim=1)+1e-10)
    sparsity = (1 - activity_fraction) / (1 - 1 / num_units)
    sparsity[torch.where(torch.sum(population_activity, dim=1) == 0.)] = 0. #Set sparsity to 0 if all units are inactive
    return sparsity


def compute_selectivity(population_activity):
    """
    Selectivity metric from (Vinje & Gallant 2000): https://www.science.org/doi/10.1126/science.287.5456.1273
    """
    num_patterns = population_activity.shape[0] #dims: 0=batch/samples, 1=units
    activity_fraction = (torch.sum(population_activity, dim=0) / num_patterns)**2 / torch.sum(population_activity**2 / num_patterns, dim=0)
    selectivity = (1 - activity_fraction) / (1 - 1 / num_patterns)
    selectivity[torch.where(torch.sum(population_activity, dim=0) == 0.)] = 0.
    return selectivity


def compute_discriminability(population_activity):
    silent_pattern_idx = np.where(torch.sum(population_activity, dim=1) == 0.)[0]
    similarity_matrix = cosine_similarity(population_activity)
    similarity_matrix[silent_pattern_idx,:] = 1
    similarity_matrix[:,silent_pattern_idx] = 1
    similarity_matrix_idx = np.tril_indices_from(similarity_matrix, -1) # select values below diagonal
    similarity = similarity_matrix[similarity_matrix_idx]
    discriminability = 1 - similarity
    return discriminability


def compute_representational_similarity_matrix(pop_activity_dict, population='all'):
    """
    Compute the representational similarity matrix between patterns and between units.
    """
    pattern_similarity_matrix_dict = {}
    neuron_similarity_matrix_dict = {}

    if population == 'all':
        pop_activity_dict = pop_activity_dict
    else:
        pop_activity_dict = {population: pop_activity_dict[population]}

    for pop_name, pop_activity in pop_activity_dict.items():
        pattern_similarity_matrix_dict[pop_name] = cosine_similarity(pop_activity)
        neuron_similarity_matrix_dict[pop_name] = cosine_similarity(pop_activity.T)

    return pattern_similarity_matrix_dict, neuron_similarity_matrix_dict


def compute_within_class_representational_similarity(network, test_dataloader, population='all'):
    """
    Compute cosine similarity between patterns and between units
    """
    percent_correct, pop_activity_dict, pattern_labels, unit_labels_dict = compute_test_activity(network, test_dataloader, class_average=False, sort=True)
    pattern_similarity_matrix_dict, neuron_similarity_matrix_dict = compute_representational_similarity_matrix(pop_activity_dict, population)

    within_class_pattern_similarity_dict = {}
    between_class_pattern_similarity_dict = {}
    within_class_unit_similarity_dict = {}
    between_class_unit_similarity_dict = {}
    num_classes = len(np.unique(pattern_labels))

    for pop_name in pattern_similarity_matrix_dict:
        within_class_pattern_similarity_dict[pop_name] = []
        between_class_pattern_similarity_dict[pop_name] = []
        within_class_unit_similarity_dict[pop_name] = []
        between_class_unit_similarity_dict[pop_name] = []
        
        for class_label in range(num_classes):
            within_similarity_matrix = pattern_similarity_matrix_dict[pop_name][pattern_labels == class_label, :][:, pattern_labels == class_label]
            lower_idx = np.tril_indices_from(within_similarity_matrix, -1)
            within_class_pattern_similarity_dict[pop_name].append(within_similarity_matrix[lower_idx])

            between_similarity_matrix = pattern_similarity_matrix_dict[pop_name][pattern_labels != class_label, :][:, pattern_labels == class_label]
            between_class_pattern_similarity_dict[pop_name].append(between_similarity_matrix.flatten())

            within_similarity_matrix = neuron_similarity_matrix_dict[pop_name][unit_labels_dict[pop_name] == class_label, :][:, unit_labels_dict[pop_name] == class_label]
            lower_idx = np.tril_indices_from(within_similarity_matrix, -1)
            within_class_unit_similarity_dict[pop_name].append(within_similarity_matrix[lower_idx])

            between_similarity_matrix = neuron_similarity_matrix_dict[pop_name][unit_labels_dict[pop_name] != class_label, :][:, unit_labels_dict[pop_name] == class_label]
            between_class_unit_similarity_dict[pop_name].append(between_similarity_matrix.flatten())

    return within_class_pattern_similarity_dict, between_class_pattern_similarity_dict, within_class_unit_similarity_dict, between_class_unit_similarity_dict


def compute_dimensionality_from_RSM(representational_similarity_matrix):
    """
    Estimate the effective intrinsic dimensionality of a neural population
    using the participation ratio (PR) of the eigenvalue spectrum of a 
    representational similarity matrix (RSM).

    Parameters
    ----------
    RSM : np.ndarray
        A square representational similarity matrix (RSM),
        where each entry RSM[i, j] indicates the similarity between the tuning
        vectors of neurons (or stimulus patterns) i and j. Typically computed using cosine 
        similarity or correlation between neural response vectors.

    Returns
    -------
    float
        The participation ratio, defined as:

            PR = (sum_i λ_i)^2 / sum_i (λ_i^2)

        where λ_i are the eigenvalues of the RSM. This quantity estimates the 
        effective number of orthogonal dimensions required to describe the 
        representational space encoded by the population.

        A value of:
        - ~1 indicates complete redundancy or representational collapse 
          (all neurons behave identically),
        - ~n_neurons indicates complete decorrelation (each neuron codes 
          independently).

    Notes
    -----
    This approach is based on methods for characterizing the 
    representational capacity or dimensionality of neural population codes.

    See:
    - Kriegeskorte, N., Mur, M., & Bandettini, P. A. (2008). Representational similarity analysis—connecting the branches of systems neuroscience. *Frontiers in Systems Neuroscience*, 2, 4. https://doi.org/10.3389/neuro.06.004.2008
    - Yamins, D. L., et al. (2014). Performance-optimized hierarchical models predict neural responses in higher visual cortex. *PNAS*, 111(23), 8619–8624. https://doi.org/10.1073/pnas.1403112111
    """
    representational_similarity_matrix = 0.5 * (representational_similarity_matrix + representational_similarity_matrix.T) # Ensure symmetry for numerical stability (to avoid floating point rounding errors)

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(representational_similarity_matrix)

    # Remove small negative eigenvalues due to numerical precision
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    # Compute participation ratio
    pr = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
    return pr


def compute_dimensionality_from_activity(pop_activity_dict):
    """
    Compute the intrinsic dimensionality (participation ratio) of neural representations across layers.

    For each population (e.g., layer) in the provided dictionary, the function performs PCA on the
    (pattern x neuron) activity matrix and calculates the participation ratio of the eigenvalue spectrum.

    This metric estimates the effective number of dimensions used to encode stimulus representations to assess representational capacity or collapse.

    The participation ratio is defined as:
        (sum_i λ_i)^2 / sum_i (λ_i^2)
    where λ_i are the eigenvalues of the covariance matrix of the activity.

    Parameters
    ----------
    pop_activity_dict : dict[str, np.ndarray]
        Dictionary where each key is a layer name and each value is a 2D numpy array of shape 
        (n_patterns, n_neurons), representing the activation of that population across inputs.

    Returns
    -------
    dim_dict : dict[str, float]
        Dictionary mapping each layer name to its estimated intrinsic dimensionality (participation ratio).

    Reference
    ---------
    - Gao, P. et al. (2017). A theory of multineuronal dimensionality, dynamics and measurement. bioRxiv.
      https://doi.org/10.1101/214262
    - Stringer, C. et al. (2019). High-dimensional geometry of population responses in visual cortex. Nature.
      https://doi.org/10.1038/s41586-019-1346-5
    """
    dim_dict = {}
    for population_name, X in pop_activity_dict.items():
        # Center the data across patterns (rows)
        X = np.array(X)
        X_centered = X - np.mean(X, axis=0, keepdims=True)
        
        # Compute covariance matrix
        cov = np.cov(X_centered, rowvar=False)  # shape: (n_neurons, n_neurons)

        # Compute eigenvalues
        eigvals = np.linalg.eigvalsh(cov)  # sorted ascending
        eigvals = np.clip(eigvals, 0, None) # clip to avoid tiny negatives due to numerical error

        # Compute participation ratio
        numerator = np.sum(eigvals) ** 2
        denominator = np.sum(eigvals ** 2)
        dim = numerator / denominator if denominator > 0 else 0.0
        dim_dict[population_name] = dim

    return dim_dict


def spatial_structure_similarity_fft(img1, img2):
    '''
    Compute the structural similarity of two images based on the correlation of their 2D spatial frequency distributions
    :param img1: 2D numpy array of pixels
    :param img2: 2D numpy array of pixels
    :return:
    '''
    # Compute the 2D spatial frequency distribution of each image
    freq1 = np.abs(np.fft.fftshift(np.fft.fft2(img1 - np.mean(img1))))
    freq2 = np.abs(np.fft.fftshift(np.fft.fft2(img2 - np.mean(img2))))

    # Compute the frequency correlation
    spatial_structure_similarity =  signal.correlate2d(freq1, freq2, mode='valid')[0][0]

    return spatial_structure_similarity


def compute_rf_structure(receptive_fields, dimensions=None, method='moran'):
    structure_ls = []
    for unit_rf in receptive_fields:
        similarity_to_noise = 0

        if dimensions is None:
            rf_width = rf_height = 28
        else:
            rf_width, rf_height = dimensions

        if torch.all(unit_rf == 0): # if receptive field is all zeros
            structure_ls.append(np.nan)

        else:
            if method == 'moran':
                    # Calculate Moran's I (spatial autocorrelation)
                    spatial_autocorrelation = np.abs(compute_morans_I(unit_rf.view(rf_width, rf_height).detach().numpy()))
                    structure_ls.append(spatial_autocorrelation)
            else:
                for i in range(3):  # structural similarity to noise (averaged across 3 random noise images)
                    # noise = np.random.uniform(min(unit_rf), max(unit_rf), (rf_width, rf_height))
                    noise = unit_rf.clone()[torch.randperm(rf_width * rf_height)].view(rf_width, rf_height).numpy() # generate "noise" by random permutation of original unit_rf
                    if method == 'fft':
                        reference_correlation = spatial_structure_similarity_fft(noise, noise)
                        similarity_to_noise += spatial_structure_similarity_fft(unit_rf.view(rf_width, rf_height).numpy(), noise) / reference_correlation
                    elif method == 'ssim':
                        similarity_to_noise += metrics.structural_similarity(unit_rf.view(rf_width, rf_height).numpy(), noise)
                structure_ls.append(1 - similarity_to_noise/3)
    
    return np.array(structure_ls)


def compute_morans_I(array, kernel_size=1):
    """
    Compute the Global Moran's I for a 2D numpy array using a specified kernel size.
    
    Parameters:
    array (numpy.ndarray): A 2D numpy array representing the spatial data.
    kernel_size (int): The size of the kernel to define the local neighborhood.
                       Default is 1, which corresponds to the 8 neighboring values.
    
    Returns:
    float: The Global Moran's I value.
    
    Moran's I formula:
    I = (N / W) * (sum_i sum_j w_ij * (x_i - x_bar) * (x_j - x_bar)) / (sum_i (x_i - x_bar)^2)
    where:
    - N is the total number of pixels,
    - W is the sum of all weights,
    - x_i and x_j are pixel values,
    - x_bar is the mean of all pixel values,
    - w_ij is the weight between pixel i and pixel j.
    """
    assert kernel_size > 0, 'Kernel size must be greater than 0'
    assert kernel_size % 1 == 0, 'Kernel size must be an integer'
    assert array.ndim == 2, 'Input array must be 2D'
    assert kernel_size < min(array.shape), 'Kernel size must be less than the smallest dimension of the input array'
    
    # Total number of pixels
    N = array.size
    
    # Mean of the input array (x̄)
    mean_val = np.mean(array)
    
    # Global sum of squared deviations from the mean (∑(xᵢ - x̄)²)
    global_deviation = np.sum((array - mean_val) ** 2)
    
    # Define the kernel for the weights matrix
    kernel_shape = (2 * kernel_size + 1, 2 * kernel_size + 1)
    weights_kernel = np.ones(kernel_shape)
    weights_kernel[kernel_size, kernel_size] = 0  # Set self-weight to 0 (w_ii = 0)
    
    # Compute the sum of all weights (W)
    W = np.sum(weights_kernel) * N
    
    # # Compute the local deviations from the mean
    deviations = array - mean_val
    
    # # Compute the weighted sum of cross-products using convolution
    weighted_cross_sum = signal.convolve2d(deviations, weights_kernel, mode='same', boundary='fill', fillvalue=0) * deviations

    # Sum all weighted cross-products
    total_sum = np.sum(weighted_cross_sum)
    
    # Compute the Global Moran's I using the provided equation
    morans_I = (N / W) * (total_sum / global_deviation)
    
    return morans_I


def compute_diag_fisher(network, train_dataloader_CL1_full):
    '''
    Compute the diagonal of the Fisher Information Matrix for the network, for implementation of the EWC continual learning algorithm.
    '''
    assert len(train_dataloader_CL1_full) == 1, "The dataloader should only have one batch"
    idx, data, target = next(iter(train_dataloader_CL1_full))

    params_with_grad = {name:param for name,param in network.named_parameters() if param.requires_grad}
    output = network(data)
    label = torch.argmax(target, dim=1)
    loss = torch.nn.NLLLoss()(output, label)
    # loss = network.criterion(output, target)
    loss.backward()
    
    diag_fisher = {name: param.grad.data**2 for name, param in params_with_grad.items()} # Diagonal of the Fisher Information Matrix
    return diag_fisher


def compute_representation_metrics(population, test_dataloader, receptive_fields=None, plot=False, export=False,
                                   export_path=None, overwrite=False, dimensions=None):
    """
    Compute representation metrics for a population of neurons
    :param population: Population object
    :param test_dataloader:
    :param receptive_fields: (optional) receptive fields for each neuron
    :param plot: bool
    :param export: bool
    :param export_path: str (path)
    :param overwrite: bool
    :param dimensions: tuple of int
    :return: dictionary of metrics
    """

    if export and overwrite is False:
        assert hasattr(population.network, 'name'), 'Network must have a name attribute to load/export data'
        metrics_dict = data_utils.load_plot_data(population.network.name, population.network.seed,
                                                data_key=f'metrics_dict_{population.fullname}', file_path=export_path)
        if metrics_dict is not None:
            return metrics_dict

    network = population.network
    idx, data, target = next(iter(test_dataloader))
    data.to(network.device)
    network.forward(data, no_grad=True)

    # total_act = torch.sum(population.activity, dim=0)
    # active_units_idx = torch.where(total_act > 1e-10)[0] # Only consider units that are active at least once
    selectivity = compute_selectivity(population.activity)
    sparsity = compute_sparsity(population.activity)
    discriminability = compute_discriminability(population.activity)

    # Compute structure
    if receptive_fields is not None:
        # receptive_fields = receptive_fields[active_units_idx]
        structure = compute_rf_structure(receptive_fields, dimensions=dimensions)
    else:
        structure = []

    metrics_dict = {'sparsity': sparsity, 
                    'selectivity': selectivity,
                    'discriminability': discriminability, 
                    'structure': structure}

    if plot:
        pt.plot_representation_metrics(metrics_dict)

    if export:
        assert hasattr(population.network, 'name'), 'Network must have a name attribute to load/export data'
        data_utils.save_plot_data(population.network.name, network.seed,
                                  data_key=f'metrics_dict_{population.fullname}', data=metrics_dict,
                                  file_path=export_path, overwrite=overwrite)

    return metrics_dict


def compute_act_weighted_avg(population, dataloader):
    """
    Compute activity-weighted average input for every unit in the population

    :param population:
    :param dataloader:
    :return:
    """

    idx, data, target = next(iter(dataloader))
    network = population.network

    network.forward(data, no_grad=True)  # compute unit activities in forward pass
    pop_activity = population.activity
    weighted_avg_input = (data.T @ pop_activity) / (pop_activity.sum(axis=0) + 0.0001) # + epsilon to avoid div-by-0 error
    weighted_avg_input = weighted_avg_input.T

    network.forward(weighted_avg_input, no_grad=True)  # compute unit activities in forward pass
    activity_preferred_inputs = population.activity.detach().clone()

    return weighted_avg_input, activity_preferred_inputs


def compute_maxact_receptive_fields(population, num_units=None, sigmoid=False, softplus=False, export=False,
                                    export_path=None, overwrite=False, test_dataloader=None):
    """
    Use the 'activation maximization' method to compute receptive fields for all units in the population

    :param population:
    :param num_units:
    :param sigmoid: if True, use sigmoid activation function for the input images;
                    if False, returns unfiltered receptive fields and activities from act_weighted_avg images
    :param softplus: if True, use softplus activation function for the input images;
    :param export: bool
    :param export_path: str (path)
    :param overwrite: bool
    :param test_dataloader:
    :return:
    """

    if export and overwrite is False:
        # Check if receptive fields and activity_preferred_inputs have already been computed and saved in the data hdf5 file
        assert hasattr(population.network, 'name'), 'Network must have a name attribute to load/export data'
        receptive_fields = data_utils.load_plot_data(population.network.name, population.network.seed,
                                                     data_key=f'maxact_receptive_fields_{population.fullname}',
                                                     file_path=export_path)
        if receptive_fields is not None:
            return torch.tensor(receptive_fields)

    # Otherwise, compute receptive fields
    num_random_initializations = 1000
    network = population.network

    # seed = network.seed
    # torch.manual_seed(seed)

    if network.backward_steps == 0:
        network.backward_steps = 3

    if num_units is None or num_units>population.size:
        num_units = population.size

    input_size = population.network.Input.E.size
    all_images = [torch.empty(num_units, input_size).uniform_(-0.01,0.01)]

    if test_dataloader is None:
        train_dataloader, train_sub_dataloader, val_dataloader, test_dataloader, data_generator = (
            data_utils.get_MNIST_dataloaders(sub_dataloader_size=20_000))
    idx, data, target = next(iter(test_dataloader))

    print("Optimizing receptive field images...")

    for i in tqdm(range(num_random_initializations)):   
        # input_images = torch.empty(num_units, input_size).uniform_(0,1)
        # input_images = torch.empty(num_units, input_size).normal_(mean=0,std=10)
        random_sample = data[np.random.choice(len(data))]
        input_images = random_sample.expand(num_units, -1)

        input_images.requires_grad = True
        loss_history = []

        if sigmoid:
            im = torch.sigmoid((input_images-0.5)*10)
            network.forward(im)  # compute unit activities in forward pass
        elif softplus:
            im = torch.nn.functional.softplus(input_images)
            network.forward(im)  # compute unit activities in forward pass
        else:
            network.forward(input_images)  # compute unit activities in forward pass
        pop_activity = population.activity[:,0:num_units]
        loss = torch.sum(-torch.diagonal(pop_activity))
        loss_history.append(loss.detach().numpy())
        loss.backward()
        all_images.append(-input_images.grad.detach().clone())

    receptive_fields = torch.mean(torch.stack(all_images), dim=0)
    if sigmoid:
        receptive_fields = torch.sigmoid((receptive_fields-0.5)*10)
    elif softplus:
        receptive_fields = torch.nn.functional.softplus(receptive_fields)

    if export:
        # Save receptive fields and activity_preferred_inputs to data hdf5 file
        assert hasattr(population.network, 'name'), 'Network must have a name attribute to load/export data'
        data_utils.save_plot_data(population.network.name, population.network.seed,
                                  data_key=f'maxact_receptive_fields_{population.fullname}', data=receptive_fields,
                                  file_path=export_path, overwrite=overwrite)
    
    return receptive_fields


def compute_unit_receptive_field(population, dataloader, unit):
    """
    Use the 'activation maximization' method to compute receptive fields for all units in the population

    :param population:
    :param dataloader:
    :param num_units:
    :return:
    """

    idx, data, target = next(iter(dataloader))
    learning_rate = 0.1
    num_steps = 10000
    network = population.network

    # turn on network gradients
    if network.forward(data[0]).requires_grad == False:
        network.backward_steps = 1
        for param in network.parameters():
            param.requires_grad = True

    weighted_avg_input = compute_act_weighted_avg(population, dataloader)

    input_image = weighted_avg_input[unit]
    input_image.requires_grad = True
    optimizer = torch.optim.SGD([input_image], lr=learning_rate)

    print("Optimizing receptive field images...")
    for step in tqdm(range(num_steps)):
        network.forward(input_image)  # compute unit activities in forward pass
        unit_activity = population.activity[unit]
        loss = -torch.log(unit_activity + 0.001)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return input_image.detach()


def compute_PSD(receptive_field, plot=False):
    '''
    Compute the power spectral density of a receptive field image
    Function based on https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/

    :param receptive_field: 2D numpy array of pixels
    :param plot: bool
    :return: frequencies, spectral_power, peak_spatial_frequency
    '''

    # Take Fourier transform of the receptive field
    fourier_image = np.fft.fftn(receptive_field)
    fourier_amplitudes = np.abs(fourier_image)**2

    # Get frequencies corresponding to signal PSD
    # (bin the results of the Fourier analysis by contstructing an array of wave vector norms)
    npix = receptive_field.shape[0] # this only works for a square image
    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knorm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knorm = knorm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    # Create the frequency power spectrum
    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knorm, fourier_amplitudes,
                                         statistic = "mean",
                                         bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    peak_spatial_frequency = np.argmax(Abins)
    spectral_power = Abins
    frequencies = kvals

    if plot:
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(receptive_field)
        ax[1].loglog(kvals, Abins)
        ax[1].set_xlabel("Spatial Frequency $k$ [pixels]")
        ax[1].set_ylabel("Power per Spatial Frequency $P(k)$")
        plt.show()

    return frequencies, spectral_power, peak_spatial_frequency


def check_equilibration_dynamics(network, dataloader, equilibration_activity_tolerance, store_num_steps=None,
                                 disp=False, plot=False):
    """

    :param network: :class:'Network'
    :param dataloader: :class:'torch.DataLoader'
    :param equilibration_activity_tolerance: float in [0, 1]
    :param store_num_steps: int
    :param disp: bool
    :param: plot: bool
    :return: float
    """
    idx, data, targets = next(iter(dataloader))
    network.forward(data, store_dynamics=True, no_grad=True, store_num_steps=store_num_steps)
    
    residuals = 0
    
    if plot:
        max_rows = 1
        for layer in network:
            max_rows = max(max_rows, len(layer.populations))
        cols = len(network.layers) - 1
        fig, axes = plt.subplots(max_rows, cols, figsize=(3.2 * cols, 3. * max_rows))
        if max_rows == 1:
            if cols == 1:
                axes = [[axes]]
            else:
                axes = [axes]
        elif cols == 1:
            axes = [[axis] for axis in axes]

    for i, layer in enumerate(network):
        if i > 0:
            col = i - 1
            for row, population in enumerate(layer):
                if len(population.forward_steps_activity) == 1:
                    return 0
                # for memory efficiency
                average_activity = torch.tensor([torch.mean(step) for step in population.forward_steps_activity])
                population.forward_steps_activity = []
                if plot:
                    this_axis = axes[row][col]
                    this_axis.plot(average_activity)
                    this_axis.set_xlabel('Equilibration time steps')
                    this_axis.set_ylabel('Average population activity')
                    this_axis.set_title('%s.%s' % (layer.name, population.name))
                    this_axis.set_ylim((0., this_axis.get_ylim()[1]))
                equil_mean = torch.mean(average_activity[-2:])
                if equil_mean > 0:
                    equil_delta = torch.abs(average_activity[-1] - average_activity[-2])
                    equil_error = equil_delta/equil_mean
                    if equil_error > equilibration_activity_tolerance:
                        if disp:
                            print('population: %s failed check_equilibration_dynamics: %.2f' %
                                  (population.fullname, equil_error))
                        residuals += equil_error
    if plot:
        fig.suptitle('Activity dynamics')
        fig.tight_layout()
        fig.show()
    return residuals


def compute_dendritic_state_dynamics(network):
    print('Computing dendritic state dynamics from param and activity history...')
    dendritic_dynamics_dict = {}

    for i, (param_step, state_dict) in tqdm(enumerate(zip(network.param_history_steps, network.prev_param_history))):
        network.load_state_dict(state_dict)

        # Compute dendritic state history dynamics
        for population in network.populations.values():
            if not hasattr(population,'dendritic_state'):
                continue
            elif not hasattr(population, 'forward_dendritic_state_history_dynamics'):
                # Initialize dendritic state history dynamics
                population.forward_dendritic_state_history_dynamics = torch.zeros_like(population.activity_history[network.param_history_steps])
                population.backward_dendritic_state_history_dynamics = torch.zeros_like(population.activity_history[network.param_history_steps])

            dendritic_dynamics_dict[population.fullname] = {"forward_dendritic_state_history_dynamics": population.forward_dendritic_state_history_dynamics,
                                                            "backward_dendritic_state_history_dynamics": population.backward_dendritic_state_history_dynamics,
                                                            "activity_history": population.activity_history,
                                                            "backward_activity_history": population.backward_activity_history}

            if population is network.output_pop:
                population.backward_dendritic_state_history_dynamics[i] = population.backward_dendritic_state_history[param_step] # Output dendritic state is fixed to output error, does not have dynamics
            else:
                for projection in population:
                    if projection.compartment in ['dend', 'dendrite']:
                        assert hasattr(projection.pre, 'backward_activity_history'), f"{projection.pre.fullname} does not have backward activity history"
                        assert projection.pre.backward_activity_history is not None, f"{projection.pre.fullname} does not have backward activity history"
                        if projection.direction in ['forward', 'F']:
                            population.forward_dendritic_state_history_dynamics[i] += projection(projection.pre.activity_history[param_step])
                            population.backward_dendritic_state_history_dynamics[i] += projection(projection.pre.backward_activity_history[param_step])
                        elif projection.direction in ['recurrent', 'R']:
                            assert hasattr(projection.pre, 'backward_activity_history'), f"{projection.pre.fullname} does not have backward activity history"
                            population.forward_dendritic_state_history_dynamics[i,1:] += projection(projection.pre.activity_history[param_step,:-1])
                            population.backward_dendritic_state_history_dynamics[i,0] += projection(projection.pre.activity_history[param_step,-1]) # Start of backward phase: recurrent connections refer to last activity of forward phase
                            population.backward_dendritic_state_history_dynamics[i,1:] += projection(projection.pre.backward_activity_history[param_step,:-1])

    for key, value in dendritic_dynamics_dict.items():
        dendritic_dynamics_dict[key] = {k: v.detach().clone().numpy() for k, v in value.items() if v is not None}
    return dendritic_dynamics_dict


def compute_spiral_decisions_data(network, test_dataloader):
    '''
    Get the correct and wrong indices to plot the spiral loss landscape (both ways)
    '''
    # Test batch inputs
    idx, data, target = next(iter(test_dataloader))

    # Predicted labels after training 
    test_outputs = network.forward(data).detach().cpu()
    _, predicted_labels = torch.max(test_outputs, 1)

    # Test labels
    target = torch.squeeze(target)
    _, test_labels = torch.max(target, 1)

    # Accuracy
    accuracy = (predicted_labels == test_labels).sum().item() / len(test_labels)

    # Data needed for graphing
    correct_indices = (predicted_labels == test_labels).nonzero().squeeze()
    wrong_indices = (predicted_labels != test_labels).nonzero().squeeze()

    # For decision map boundary plot
    meshgrid_size = 1000
    x_max = 2.0
    arms = 4
    eps=1e-3

    ii, jj = torch.meshgrid(torch.linspace(-x_max, x_max, meshgrid_size),
							torch.linspace(-x_max, x_max, meshgrid_size),
							indexing="ij")
    X_all = torch.cat([ii.unsqueeze(-1),
					   jj.unsqueeze(-1)],
					   dim=-1).view(-1, 2)
    y_pred = network.forward(X_all)

    decision_map = torch.argmax(y_pred, dim=1)
    # decision_value, decision_map = torch.max(y_pred, dim=1)
    y_prob = torch.nn.functional.softmax(y_pred, dim=1)
    decision_value, _ = torch.max(y_prob, dim=1)

    decision_map = decision_map.view(meshgrid_size, meshgrid_size).detach().cpu()
    decision_map = decision_map.T
    decision_value = decision_value.view(meshgrid_size, meshgrid_size).detach().cpu()
    decision_value = decision_value.T

    # Return data for plotting
    decision_data = {
        "inputs": data,
        "test_labels": test_labels,
        "accuracy": accuracy,
        "correct_indices": correct_indices,
        "wrong_indices": wrong_indices,
        "decision_map": decision_map,
        "decision_value": decision_value
    }

    return decision_data

