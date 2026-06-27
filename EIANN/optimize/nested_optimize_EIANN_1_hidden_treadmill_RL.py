import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os, sys, math
from copy import deepcopy
import numpy as np
import h5py
import gc

from EIANN import Q_Network
from EIANN.environments import Cue_Treadmill
from EIANN.utils import (read_from_yaml, write_to_yaml, analyze_simple_EIANN_epoch_loss_and_accuracy, \
    sort_by_val_history, recompute_validation_loss_and_accuracy, check_equilibration_dynamics, \
    recompute_train_loss_and_accuracy, compute_test_loss_and_accuracy_history, sort_by_class_averaged_val_output,
                         get_binned_mean_population_attribute_history_dict)
from EIANN.plot_rl import plot_validation_rewards, plot_final_q_vals, plot_actions_over_training, \
    plot_hidden_state_cross_correlation, plot_equilibration_dynamics, plot_treadmill_hidden_activity
from nested.utils import Context, str_to_bool
from nested.optimize_utils import update_source_contexts
from EIANN.optimize.network_rl_config_updates import *
import EIANN.utils as utils


context = Context()

# run 5 random seeds in parallel:
# mpirun -n 6 python -m mpi4py.futures -m nested.analyze --framework=mpi \
#   --config-file-path=optimize/config/mnist/nested_optimize_EIANN_1_hidden_mnist_BTSP_config_D1.yaml \
#   --param-file-path=optimize/config/mnist/20230301_nested_optimize_mnist_1_hidden_1_inh_params.yaml --model-key=BTSP_D1 --output-dir=optimize/data --label=btsp \
#   --export --store_history=True --retrain=False --full_analysis=True --status_bar=True

# mpirun -n 6 python -m mpi4py.futures -m nested.analyze --framework=mpi \
#   --config-file-path=optimize/config/mnist/nested_optimize_EIANN_1_hidden_mnist_bpDale_softplus_SGD_1_inh_config_A.yaml \
#   --param-file-path=optimize/config/mnist/20230301_nested_optimize_mnist_1_hidden_1_inh_params.yaml --model-key=bpDale_softplus_1_inh_A --output-dir=optimize/data --label=bpDale \
#   --export --export-file-path=multiseed_mnist_metrics.hdf5 --store_history=True --retrain=False --full_analysis=True --status_bar=True

# python -m nested.analyze --framework=serial \
# --config-file-path=optimize/optimize_config/treadmill_RL/20260428_nested_optimize_EIANN_2_hidden_treadmill_RL_van_bp_relu_SGD_config_G.yaml \
# --param-file-path=optimize/optimize_params/treadmill_RL/20260610_treadmill_RL_params.yaml --model-key=van_bp_relu_SGD \
# --output-dir=data --compute_receptive_fields=False --num_instances=1 --store_history=True \
# --retrain=True --status_bar=True --plot --disp --debug

# python -m nested.analyze --framework=serial \
# --config-file-path=optimize/optimize_config/treadmill_RL/20231129_nested_optimize_EIANN_2_hidden_treadmill_RL_bpDale_relu_SGD_config_G.yaml \
# --output-dir=data --compute_receptive_fields=False --num_instances=1 --store_history=True \
# --retrain=True --status_bar=True --plot --disp --debug



def config_controller():
    if 'debug' not in context():
        context.debug = False
    else:
        context.debug = str_to_bool(context.debug)


def config_worker():
    context.seed_start = int(context.seed_start)
    context.num_instances = int(context.num_instances)
    context.network_id = int(context.network_id)
    context.task_id = int(context.task_id)
    context.data_seed_start = int(context.data_seed_start)
    context.status_bar = str_to_bool(context.status_bar)
    if 'train_online' not in context():
        context.train_online = False
    else:
        context.train_online = str_to_bool(context.train_online)
    if 'debug' not in context():
        context.debug = False
    else:
        context.debug = str_to_bool(context.debug)
    if 'verbose' not in context():
        context.verbose = False
    else:
        context.verbose = str_to_bool(context.verbose)
    if 'interactive' not in context():
        context.interactive = False
    else:
        context.interactive = str_to_bool(context.interactive)
    if 'meta' not in context():
        context.meta = False
    else:
        context.meta = str_to_bool(context.meta)
    if 'eval_accuracy' not in context():
        context.eval_accuracy = 'final'
    else:
        context.eval_accuracy = str(context.eval_accuracy)
    if 'store_history' not in context():
        context.store_history = False
    else:
        context.store_history = str_to_bool(context.store_history)
    if 'store_dynamics' not in context():
        context.store_dynamics = False
    else:
        context.store_dynamics = str_to_bool(context.store_dynamics)
        
    if 'store_params' not in context():
        context.store_params = False
    else:
        context.store_params = str_to_bool(context.store_params)
    if context.debug:
        context.store_num_steps = None
    elif 'store_num_steps' not in context():
        if context.store_dynamics:
            context.store_num_steps = None
        else:
            context.store_num_steps = 2
    else:
        context.store_num_steps = int(context.store_num_steps)
    if 'full_analysis' not in context():
        context.full_analysis = False
    else:
        context.full_analysis = str_to_bool(context.full_analysis)
    if 'equilibration_activity_tolerance' not in context():
        context.equilibration_activity_tolerance = 0.2
    else:
        context.equilibration_activity_tolerance = float(context.equilibration_activity_tolerance)
    if 'compute_receptive_fields' not in context():
        context.compute_receptive_fields = False
    else:
        context.compute_receptive_fields = str_to_bool(context.compute_receptive_fields)
    if 'constrain_equilibration_dynamics' not in context():
        context.constrain_equilibration_dynamics = True
    else:
        context.constrain_equilibration_dynamics = str_to_bool(context.constrain_equilibration_dynamics)
    # Name exported files and plots by the optimize config basename. nested does not forward
    # config_file_path to worker contexts, so recover it from the command-line args when absent.
    if 'config_file_path' in context() and context.config_file_path is not None:
        config_file_path = context.config_file_path
    else:
        config_file_path = ''
        for i, arg in enumerate(sys.argv):
            if arg.startswith(('--config-file-path=', '--config_file_path=')):
                config_file_path = arg.split('=', 1)[1]
            elif arg in ('--config-file-path', '--config_file_path') and i + 1 < len(sys.argv):
                config_file_path = sys.argv[i + 1]
    context.run_name = config_file_path.split('/')[-1].split('.')[0]
    if 'export_network_config_file_path' not in context():
        if context.label is None:
            context.export_network_config_file_path = f"{context.output_dir}/{context.run_name}_optimized.yaml"
        else:
            context.export_network_config_file_path = f"{context.output_dir}/{context.run_name}_{context.label}_optimized.yaml"
    if 'retrain' not in context():
        context.retrain = True
    else:
        context.retrain = str_to_bool(context.retrain)
    if 'plot_initial' not in context():
        context.plot_initial = False
    else:
        context.plot_initial - str_to_bool(context.plot_initial)
    if 'save_plots' not in context():
        context.save_plots = False
    else:
        context.save_plots = str_to_bool(context.save_plots)
    if 'save_plots_dir' not in context():
        context.save_plots_dir = f"{context.output_dir}/rl_plots"
    if 'model_key' not in context():
        # nested consumes the recognized --model-key/-k option on the controller and does not forward it
        # to worker contexts, so recover it directly from the command-line args (default to empty).
        context.model_key = ''
        for i, arg in enumerate(sys.argv):
            if arg.startswith(('--model-key=', '--model_key=', '-k=')):
                context.model_key = arg.split('=', 1)[1]
            elif arg in ('--model-key', '--model_key', '-k') and i + 1 < len(sys.argv):
                context.model_key = sys.argv[i + 1]
    if 'include_dend_loss_objective' not in context():
        context.include_dend_loss_objective = False
    else:
        context.include_dend_loss_objective = str_to_bool(context.include_dend_loss_objective)
    if 'include_equilibration_dynamics_objective' not in context():
        context.include_equilibration_dynamics_objective = False
    else:
        context.include_equilibration_dynamics_objective = str_to_bool(context.include_equilibration_dynamics_objective)
    if 'include_behavior_loss_objective' not in context():
        context.include_behavior_loss_objective = False
    else:
        context.include_behavior_loss_objective = str_to_bool(context.include_behavior_loss_objective)
    if 'store_history_interval' not in context():
        context.store_history_interval = None
    
    context.train_episodes = int(context.train_episodes)
    
    history_interval = max(int(context.train_episodes / 10), 50)
    if 'store_params_interval' not in context():
        context.store_params_interval = (0, -1, history_interval)
    
    if context.full_analysis:
        context.val_interval = (0, -1, history_interval)
        context.store_params_interval = context.val_interval
        context.store_params = True
        context.store_num_steps = None
        if context.store_history_interval is not None:
            context.store_history_interval = context.val_interval
    
    if context.include_dend_loss_objective:
        if not context.store_history:
            context.store_history = True
            if context.store_history_interval is None:
                context.store_history_interval = context.val_interval
    
    if 'data_file_path' in context():
        context.base_data_file_path = context.data_file_path
    else:
        context.base_data_file_path = f"{context.output_dir}/{context.run_name}.pkl"
    
    network_config = read_from_yaml(context.network_config_file_path)
    context.layer_config = network_config['layer_config']
    context.projection_config = network_config['projection_config']
    context.training_kwargs = network_config['training_kwargs']
    
    # Set up treadmill environments
    context.treadmill_length = int(context.treadmill_length)
    context.treadmill_cue_position = int(context.treadmill_cue_position)
    context.treadmill_cue_length = int(context.treadmill_cue_length)
    context.treadmill_reward_length = int(context.treadmill_reward_length)
    context.treadmill_reward_positions = [int(x) for x in context.treadmill_reward_positions]
    context.treadmill_reward_values = [float(x) for x in context.treadmill_reward_values]

    context.environments = []
    for i, reward_position in enumerate(context.treadmill_reward_positions):
        context.environments.append(
            Cue_Treadmill(
                length = context.treadmill_length,
                cue_position = context.treadmill_cue_position, 
                cue_length = context.treadmill_cue_length,
                reward_position = reward_position,
                reward_length = context.treadmill_reward_length,
                reward_positions = context.treadmill_reward_positions,
                cue_number = i,
                total_cues = len(context.treadmill_reward_positions),
                reward_values = context.treadmill_reward_values
            )
        )

def get_mean_forward_dend_loss(network, num_steps, abs=True):
    """
    
    :param network:
    :param num_steps: int
    :param: abs: bool
    :return: tensor
    """
    attr_name = 'forward_dendritic_state'
    all_pop_attr_history_list = []
    
    for pop_name, pop in network.populations.items():
        attr_history = pop.get_attribute_history(attr_name)
        if attr_history is None:
            continue
        attr_history = attr_history.detach().clone()
        if abs:
            attr_history = torch.abs(attr_history)
        all_pop_attr_history_list.append(attr_history)
    
    all_pop_attr_history_tensor = torch.concatenate(all_pop_attr_history_list, dim=1)
    mean_attr_history = torch.mean(all_pop_attr_history_tensor, dim=1)
    
    return torch.mean(mean_attr_history[-num_steps:]).item()


def get_random_seeds():
    network_seeds = [int.from_bytes((context.network_id, context.task_id, instance_id), byteorder='big')
                     for instance_id in range(context.seed_start, context.seed_start + context.num_instances)]
    data_seeds = [int.from_bytes((context.network_id, instance_id), byteorder='big')
                     for instance_id in range(context.data_seed_start, context.data_seed_start + context.num_instances)]
    if context.debug:
        print('network_seeds:', network_seeds, 'data_seeds:', data_seeds)
        sys.stdout.flush()
    return [network_seeds, data_seeds]


def compute_features(x, seed, data_seed, model_id=None, export=False, plot=False):
    """

    :param x: array of float
    :param seed: int
    :param data_seed: int
    :param model_id: str
    :param export: bool
    :param plot: bool
    :return: dict
    """
    update_source_contexts(x, context)

    param_dict = param_array_to_dict(x, context.param_names)
        
    
    network = Q_Network(context.layer_config, context.projection_config, seed=seed, **context.training_kwargs)
    
    if export:
        config_dict = {'layer_config': context.layer_config,
                       'projection_config': context.projection_config,
                       'training_kwargs': context.training_kwargs}
        write_to_yaml(context.export_network_config_file_path, config_dict, convert_scalars=True)
        if context.disp:
            print('nested_optimize_EIANN_1_hidden_mnist: pid: %i exported network config to %s' %
                  (os.getpid(), context.export_network_config_file_path))
    
    if plot:
        try:
            network.Output.E.H1.E.initial_weight = network.Output.E.H1.E.weight.data.detach().clone()
            network.H1.E.Output.E.initial_weight = network.H1.E.Output.E.weight.data.detach().clone()
        except:
            pass
        if context.plot_initial:
            title = 'Initial (%i, %i)' % (seed, data_seed)
            # plot_batch_accuracy(network, test_dataloader, population='all', title=title) # IMPLEMENT MAYBE
    
    if not context.retrain:
        network = utils.load_network(context.data_file_path)
        if context.disp:
            print('nested_optimize_EIANN_1_hidden_treadmill_RL: pid: %i loaded network history from %s' %
                  (os.getpid(), context.data_file_path))
    else:
        if context.debug:
            import time
            current_time = time.time()
        if context.train_online:
            network.train_online(environments=context.environments, epsilon=param_dict['epsilon'], epsilon_decay=param_dict['epsilon_decay'],
                        gamma=param_dict['gamma'], episodes=context.train_episodes,
                        val_interval=context.val_interval,  # e.g. (-201, -1, 10),
                        store_history=context.store_history,
                        store_dynamics=context.store_dynamics, store_history_interval=context.store_history_interval,
                        store_params=context.store_params, store_params_interval=context.store_params_interval,
                        status_bar=context.status_bar, meta=context.meta)
        else:
            network.train(environments=context.environments, epsilon=param_dict['epsilon'], epsilon_decay=param_dict['epsilon_decay'],
                        gamma=param_dict['gamma'], episodes=context.train_episodes,
                        val_interval=context.val_interval,  # e.g. (-201, -1, 10),
                        store_history=context.store_history,
                        store_dynamics=context.store_dynamics, store_history_interval=context.store_history_interval,
                        store_params=context.store_params, store_params_interval=context.store_params_interval,
                        status_bar=context.status_bar, meta=context.meta)

    
    if plot:
        try:
            from EIANN.plot import plot_FB_weight_alignment
            plot_FB_weight_alignment(network.Output.E.H1.E, network.H1.E.Output.E)
        except:
            pass


    max_reward_idx = torch.argmax(network.val_reward_history)
    sorted_output_idx = None
    sorted_val_reward_history = network.val_reward_history

    
    # Select for stability by computing mean accuracy in a window after the best validation step
    val_stepsize = int(context.val_interval[2])
    num_val_steps_accuracy_window = int(context.num_training_steps_accuracy_window) // val_stepsize
    
    if context.eval_accuracy == 'final':
        final_reward = torch.mean(sorted_val_reward_history[-num_val_steps_accuracy_window:])
        
        results = {'reward': final_reward}

    elif context.eval_accuracy == 'best':
        if max_reward_idx + num_val_steps_accuracy_window > len(
                sorted_val_reward_history):  # if best loss too close to the end
            best_reward_window = torch.mean(sorted_val_reward_history[-num_val_steps_accuracy_window:])

        else:
            best_reward_window = \
                torch.mean(sorted_val_reward_history[max_reward_idx:max_reward_idx + num_val_steps_accuracy_window])
        
        results = {'reward': best_reward_window}
    else:
        raise Exception('nested_optimize_EIANN_1_hidden_mnist: eval_accuracy must be final or best, not %s' %
                        context.eval_accuracy)

    if np.isnan(results['reward']) or np.isinf(results['reward']):
        if context.debug and context.interactive:
            context.update(locals())
        return dict()
    
    if context.include_dend_loss_objective:
        if context.store_history_interval is None:
            dend_loss_window = int(context.num_training_steps_accuracy_window)
        else:
            dend_loss_window = num_val_steps_accuracy_window
        mean_forward_dend_loss = get_mean_forward_dend_loss(network, dend_loss_window)
        results['mean_forward_dend_loss'] = mean_forward_dend_loss

    # behavior proxy: we want the ratio of wrong location licks to be lower for the closer treadmill
    if context.include_behavior_loss_objective:
        if not hasattr(network, 'val_action_history') or len(network.val_action_history) == 0:
            print('plot_actions_over_training: network has no val_action_history; skipping')
        else:
            action_history = np.asarray(network.val_action_history)  # [n_val_steps, n_environments, length]
            t0_position = context.environments[0].reward_position
            t1_position = context.environments[1].reward_position

            action_ratio = action_history[:, 0, t1_position].sum() / (action_history[:, 1, t0_position].sum() +  + 1e-6)
            if t0_position > t1_position:
                action_ratio = 1 / action_ratio
        
            results['behavior_ratio'] = action_ratio
            
    
    if plot or context.save_plots:
        if context.model_key:
            title = str(context.model_key)
        else:
            title = 'Final (%i, %i)' % (seed, data_seed)
        plot_names = ['validation_rewards', 'final_q_vals', 'actions_over_training',
                      'cross_correlation_H1E', 'cross_correlation_H2E', 'hidden_activity']
        if context.save_plots:
            plot_prefix = f"{context.save_plots_dir}/{context.run_name}_{seed}_{data_seed}"
            if context.label is not None:
                plot_prefix += f"_{context.label}"
            save_paths = {name: f"{plot_prefix}_{name}.png" for name in plot_names}
        else:
            save_paths = {name: None for name in plot_names}
        plot_validation_rewards(network, title=title, save_path=save_paths['validation_rewards'])
        plot_final_q_vals(network, context.environments, title=title, save_path=save_paths['final_q_vals'], meta=context.meta)
        plot_actions_over_training(network, context.environments, title=title,
                                   save_path=save_paths['actions_over_training'])
        plot_hidden_state_cross_correlation(network, context.environments, 'H1E', title=f'{title} (H1E)',
                                            save_path=save_paths['cross_correlation_H1E'], meta=context.meta)
        plot_hidden_state_cross_correlation(network, context.environments, 'H2E', title=f'{title} (H2E)',
                                            save_path=save_paths['cross_correlation_H2E'], meta=context.meta)
        plot_treadmill_hidden_activity(network, context.environments, population_names=('H1E', 'H2E'),
                                       title=title, save_path=save_paths['hidden_activity'], meta=context.meta)
        if context.constrain_equilibration_dynamics or context.debug:
            # store_num_steps left as None to capture the full forward_steps settling trace
            plot_equilibration_dynamics(network, context.environments, title=title, meta=context.meta)
        
    if context.debug:
        activities = network.get_treadmill_hidden_activity(context.environments, population_name='H2E', meta=context.meta)

        for pos in range(len(activities[0])):
            print('Position {}'.format(pos))
            print(np.round(activities[0][pos], 2))
            print(np.round(activities[1][pos], 2))
    
    if export:
        base_data_file_path_prefix = context.base_data_file_path.split('.')[0]
        if context.label is None:
            this_data_file_path = f"{base_data_file_path_prefix}_{seed}_{data_seed}.pkl"
        else:
            this_data_file_path = f"{base_data_file_path_prefix}_{seed}_{data_seed}_{context.label}.pkl"
        
        utils.save_network(network, path=this_data_file_path, disp=False)
        if context.disp:
            print('nested_optimize_EIANN_1_hidden_mnist: pid: %i exported network history to %s' %
                  (os.getpid(), this_data_file_path))
    
    if not context.interactive:
        del network
        gc.collect()
    else:
        context.update(locals())

    results['reward'] = -1 * results['reward'] # for optimization

    return results


def filter_features(primitives, current_features, model_id=None, export=False, plot=False):

    features = {}
    for instance_features in primitives:
        for key, val in instance_features.items():
            if np.isnan(val) or np.isinf(val):
                return dict()
            if key not in features:
                features[key] = []
            features[key].append(val)
    for key, val in features.items():
        features[key] = np.mean(val)

    return features


def get_objectives(features, model_id=None, export=False, plot=False):
    objectives = {}
    for key, val in features.items():
        if 'accuracy' in key:
            objectives[key] = 100. - val
        elif key == 'mean_forward_dend_loss':
            objectives[key] = np.abs(val)
        else:
            objectives[key] = val
    return features, objectives
