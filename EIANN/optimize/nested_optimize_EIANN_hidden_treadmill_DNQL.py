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
    plot_hidden_state_cross_correlation, plot_equilibration_dynamics, plot_treadmill_hidden_activity, \
    plot_region_cross_correlation_over_training, plot_region_cross_correlation_by_population, \
    plot_hidden_state_cross_correlation_over_training, plot_loss, plot_qnext_treadmill_hidden_activity
from nested.utils import Context, str_to_bool
from nested.optimize_utils import update_source_contexts
from EIANN.optimize.network_dnql_config_updates import *
from EIANN.optimize.train.dnql import train, train_online
import EIANN.utils as utils


context = Context()

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
    if 'include_xcorr_loss_objective' not in context():
        context.include_xcorr_loss_objective = False
    else:
        context.include_xcorr_loss_objective = str_to_bool(context.include_xcorr_loss_objective)
    if 'xcorr_threshold' not in context():
        context.xcorr_threshold = 0.5
    else:
        context.xcorr_threshold = float(context.xcorr_threshold)
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
    
    # Dual-network Q learning: load a separate config for each network
    q_network_config = read_from_yaml(context.q_network_config_file_path)
    context.q_layer_config = q_network_config['layer_config']
    context.q_projection_config = q_network_config['projection_config']
    context.q_training_kwargs = q_network_config['training_kwargs']

    qnext_network_config = read_from_yaml(context.qnext_network_config_file_path)
    context.qnext_layer_config = qnext_network_config['layer_config']
    context.qnext_projection_config = qnext_network_config['projection_config']
    context.qnext_training_kwargs = qnext_network_config['training_kwargs']
    
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
        
    
    # Dual-network Q learning: `network` is the Q network (owns val_reward_history and
    # drives all downstream features/plots); qnext_network is the auxiliary next-Q predictor.
    # Distinct seeds so the two networks do not initialize identically.
    q_network = Q_Network(context.q_layer_config, context.q_projection_config, seed=seed,
                        **context.q_training_kwargs)
    qnext_network = Q_Network(context.qnext_layer_config, context.qnext_projection_config, seed=seed,
                              **context.qnext_training_kwargs)

    if export:
        q_config_dict = {'layer_config': context.q_layer_config,
                         'projection_config': context.q_projection_config,
                         'training_kwargs': context.q_training_kwargs}
        qnext_config_dict = {'layer_config': context.qnext_layer_config,
                             'projection_config': context.qnext_projection_config,
                             'training_kwargs': context.qnext_training_kwargs}
        qnext_export_path = context.export_network_config_file_path.replace('.yaml', '_qnext.yaml')
        write_to_yaml(context.export_network_config_file_path, q_config_dict, convert_scalars=True)
        write_to_yaml(qnext_export_path, qnext_config_dict, convert_scalars=True)
        if context.disp:
            print('nested_optimize_EIANN_1_hidden_treadmill_DNQL: pid: %i exported network configs to %s, %s' %
                  (os.getpid(), context.export_network_config_file_path, qnext_export_path))
    
    if plot:
        try:
            q_network.Output.E.H1.E.initial_weight = q_network.Output.E.H1.E.weight.data.detach().clone()
            q_network.H1.E.Output.E.initial_weight = q_network.H1.E.Output.E.weight.data.detach().clone()
        except:
            pass
        if context.plot_initial:
            title = 'Initial (%i, %i)' % (seed, data_seed)
            # plot_batch_accuracy(network, test_dataloader, population='all', title=title) # IMPLEMENT MAYBE
    
    if not context.retrain:
        # Load both networks. qnext was saved to the sibling '_qnext.pkl' path (see train.dnql).
        q_network = utils.load_network(context.data_file_path)
        qnext_network = utils.load_network(context.data_file_path.replace('.pkl', '_qnext.pkl'))
        if context.disp:
            print('nested_optimize_EIANN_hidden_treadmill_DNQL: pid: %i loaded network histories from %s (+ _qnext)' %
                  (os.getpid(), context.data_file_path))
    else:
        if context.debug:
            import time
            current_time = time.time()
        # Dual-network Q learning training. `train_dnql` is the training method you are
        # implementing on Q_Network: it drives both networks (Q + next-Q predictor),
        # applying each network's learning rules independently. train_online is passed
        # through so the method can branch on the online/offline schedule internally.

        if context.train_online:
            pass
        else:
            train(q_network=q_network, qnext_network=qnext_network,
                        environments=context.environments, epsilon=param_dict['epsilon'], epsilon_decay=param_dict['epsilon_decay'],
                        gamma=param_dict['gamma'], episodes=context.train_episodes,
                        val_interval=context.val_interval,  # e.g. (-201, -1, 10),
                        store_history=context.store_history,
                        store_dynamics=context.store_dynamics, store_history_interval=context.store_history_interval,
                        store_params=context.store_params, store_params_interval=context.store_params_interval,
                        status_bar=context.status_bar, save_to_file=None, meta=context.meta)

    
    if plot:
        try:
            from EIANN.plot import plot_FB_weight_alignment
            plot_FB_weight_alignment(q_network.Output.E.H1.E, q_network.H1.E.Output.E)
        except:
            pass


    max_reward_idx = torch.argmax(q_network.val_reward_history)
    sorted_output_idx = None
    sorted_val_reward_history = q_network.val_reward_history

    
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
        mean_forward_dend_loss = get_mean_forward_dend_loss(q_network, dend_loss_window)
        results['mean_forward_dend_loss'] = mean_forward_dend_loss

    # behavior proxy: we want the ratio of wrong location licks to be lower for the closer treadmill
    if context.include_behavior_loss_objective:
        if not hasattr(q_network, 'val_action_history') or len(q_network.val_action_history) == 0:
            print('network has no val_action_history; skipping')
        else:
            action_history = np.asarray(q_network.val_action_history)  # [n_val_steps, n_environments, length]
            t0_position = context.environments[0].reward_position
            t1_position = context.environments[1].reward_position

            action_ratio = action_history[:, 0, t1_position].sum() / (action_history[:, 1, t0_position].sum() +  + 1e-6)
            if t0_position > t1_position:
                action_ratio = 1 / action_ratio
        
            results['behavior_ratio'] = action_ratio

    if context.include_xcorr_loss_objective:
        if not hasattr(q_network, 'val_cross_correlation_history') or len(q_network.val_action_history) == 0:
            print('q network has no val_cross_correlation_history; skipping')

        if not hasattr(q_network, 'val_cross_correlation_matrix_history') or len(q_network.val_action_history) == 0:
            print('q network has no val_cross_correlation_matrix_history; skipping')
        
        population = 'H2E'
        pre_r1_corrs = np.array([x[population][0]['pre_r1'] for x in q_network.val_cross_correlation_history])
        pre_r2_corrs = np.array([x[population][0]['pre_r2'] for x in q_network.val_cross_correlation_history])


        pre_r1_decorr_indeces = np.where(pre_r1_corrs <= context.xcorr_threshold)[0]
        pre_r2_decorr_indeces = np.where(pre_r2_corrs <= context.xcorr_threshold)[0]

        if pre_r1_decorr_indeces.size > 0:
            pre_r1_decorr_time = pre_r1_decorr_indeces[0]
        else:
            pre_r1_decorr_time = len(pre_r1_corrs)

        if pre_r2_decorr_indeces.size > 0:
            pre_r2_decorr_time = pre_r2_decorr_indeces[0]
        else:
            pre_r2_decorr_time = len(pre_r2_corrs)

        time_to_decorr_ratio = pre_r2_decorr_time / (pre_r1_decorr_time + 1e-6) # we want r2_decorr_time to be < pre_r1_decorr_time

        final_avg_xcorr = np.nanmean(q_network.val_cross_correlation_matrix_history[-1][population])

        results['final_avg_xcorr'] = final_avg_xcorr
        results['time_to_decorr_ratio'] = time_to_decorr_ratio

    
    if plot or context.save_plots:
        if context.model_key:
            title = str(context.model_key)
        else:
            title = 'Final (%i, %i)' % (seed, data_seed)
        # distinguish which network each figure belongs to
        q_title = 'Q — %s' % title
        qnext_title = 'QNext — %s' % title
        plot_names = ['validation_rewards', 'final_q_vals', 'actions_over_training',
                      'cross_correlation_H1E', 'cross_correlation_H2E', 'hidden_activity',
                      'region_cross_correlation', 'region_cross_correlation_by_population',
                      'cross_correlation_over_training', 'q_loss', 'qnext_loss', 'qnext_hidden_activity']
        if context.save_plots:
            plot_prefix = f"{context.save_plots_dir}/{context.run_name}_{seed}_{data_seed}"
            if context.label is not None:
                plot_prefix += f"_{context.label}"
            save_paths = {name: f"{plot_prefix}_{name}.png" for name in plot_names}
        else:
            save_paths = {name: None for name in plot_names}
        plot_validation_rewards(q_network, title=q_title, save_path=save_paths['validation_rewards'])
        plot_final_q_vals(q_network, context.environments, title=q_title, save_path=save_paths['final_q_vals'], meta=context.meta)
        plot_actions_over_training(q_network, context.environments, title=q_title,
                                   save_path=save_paths['actions_over_training'])
        plot_treadmill_hidden_activity(q_network, context.environments, population_names=('H1E', 'H2E'),
                                       title=q_title, save_path=save_paths['hidden_activity'], meta=context.meta)
        plot_region_cross_correlation_over_training(q_network, populations=('H1E', 'H2E'), title=q_title,
                                                    save_path=save_paths['region_cross_correlation'])
        plot_region_cross_correlation_by_population(q_network, populations=('H1E', 'H2E'), title=q_title,
                                                    save_path=save_paths['region_cross_correlation_by_population'])
        plot_hidden_state_cross_correlation_over_training(q_network, context.environments, populations=('H1E', 'H2E'),
                                                          n_timepoints=4, title=q_title,
                                                          save_path=save_paths['cross_correlation_over_training'])
        # QNext network figures
        plot_loss(q_network, title=q_title, save_path=save_paths['q_loss'])
        plot_loss(qnext_network, title=qnext_title, save_path=save_paths['qnext_loss'])
        plot_qnext_treadmill_hidden_activity(q_network, qnext_network, context.environments,
                                             shared_population='H1E', population_names=('H1E',),
                                             title=qnext_title, save_path=save_paths['qnext_hidden_activity'],
                                             meta=context.meta)
        if context.constrain_equilibration_dynamics or context.debug:
            # store_num_steps left as None to capture the full forward_steps settling trace
            plot_equilibration_dynamics(q_network, context.environments, title=q_title, meta=context.meta)
        
    if context.debug:
        activities = q_network.get_treadmill_hidden_activity(context.environments, population_name='H2E', meta=context.meta)

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
        
        qnext_data_file_path = this_data_file_path.replace('.pkl', '_qnext.pkl')
        utils.save_network(q_network, path=this_data_file_path, disp=False)
        utils.save_network(qnext_network, path=qnext_data_file_path, disp=False)
        if context.disp:
            print('nested_optimize_EIANN_1_hidden_treadmill_DNQL: pid: %i exported network histories to %s, %s' %
                  (os.getpid(), this_data_file_path, qnext_data_file_path))
    
    if not context.interactive:
        del q_network
        del qnext_network
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
