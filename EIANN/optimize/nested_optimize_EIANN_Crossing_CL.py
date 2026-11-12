import os
import gc
import numpy as np

from EIANN import Network
from EIANN.utils import read_from_yaml, write_to_yaml
from EIANN.reinforcement_learning.environments import TaskSchedule, ScheduledEnv, make_crossing_tasks
from EIANN.reinforcement_learning.treadmill_metrics import block_rewards, seen_track_rewards
from EIANN.reinforcement_learning.crossing_metrics import block_outcome_rates
from nested.utils import Context, str_to_bool
from nested.optimize_utils import update_source_contexts
from EIANN.optimize.network_rl_config_updates import *
import EIANN.reinforcement_learning as reinforcement_learning
import EIANN.utils as utils


context = Context()

# Crossing_CL: continual-learning gridworld task (design: reinforcement_learning/20261104_crossing_CL_design.md).
# A 4x4 MiniGrid room split by a wall with one gap; three identical goal tiles on the far side, of which one pays.
# Reaching any tile ends the episode. The gap's row is the cue for which tile pays: tasks A, B, C differ in both route
# and rewarded tile (CrossingChoiceEnv, make_crossing_tasks). The protocol is the same as nested_optimize_EIANN_2ACDC_CL:
# blocks of episodes_per_task episodes on one task (A, B, C, A, ...), num_cycles times; each block is one algo.learn()
# call followed by algo.end_task(). Exploration across task switches is set by algorithm_kwargs['epsilon_schedule']
# ('fixed', 'decay' or 'reset'; see QLearning).
# Objectives, as in 2ACDC_CL: for every block b, phase{b}_reward is the greedy reward averaged over the tasks seen so far,
# over the last num_episodes_reward_window episodes of the block. Features only: phase{b}_reward_task{j} (the block x
# task reward matrix), reward (the mean of the phase rewards), and per block and task the success rate (rewarded tile)
# and choice accuracy (success / fraction of episodes that reached any tile): phase{b}_{success,choice}_task{j}, with
# phase{b}_success as the mean success over the tasks seen so far.

# train and export a single seed (must be run from the EIANN package directory):
# python -m nested.analyze --framework=serial \
#   --config-file-path=optimize/optimize_config/Crossing_CL/20261104_nested_optimize_EIANN_2_hidden_Crossing_CL_van_bp_relu_SGD_config.yaml \
#   --param-file-path=optimize/optimize_params/Crossing_CL/20261104_Crossing_CL_params.yaml --model-key=van_bp_relu_SGD \
#   --output-dir=data/Crossing_CL --label=van_bp_relu_SGD --num_instances=1 --export --disp

# added param: episodes_per_task (int) consecutive episodes on each task in a block
# added param: num_cycles (int) passes through all tasks; train_episodes = episodes_per_task * num_tasks * num_cycles
# added param: num_episodes_reward_window (int) last episodes of each block that phase rewards are averaged over
# added param: algorithm (str) name of an RLAlgorithm in EIANN.reinforcement_learning, e.g. QLearning or OnlineQLearning
# added param: algorithm_kwargs (dict) fixed arguments of the algorithm, e.g. epsilon_schedule; update_context functions
#   add the optimized ones (e.g. update_q_learning_kwargs)
# added param: crossing_kwargs (dict) passed to make_crossing_tasks (CrossingChoiceEnv arguments shared by all tasks)
# added param: retrain (bool) with False, loads the network exported by a previous --export run instead of training it
#   (default to true)
# added param: plot_dir (str) with --plot, saves the figures to this folder instead of showing them (default to None)
# added param: plot_format (str) file format of the saved figures (default to svg)
# post-hoc analysis and videos of learning: notebooks/Crossing_CL_analysis.ipynb


def config_worker():
    context.seed_start = int(context.seed_start)
    context.num_instances = int(context.num_instances)
    context.network_id = int(context.network_id)
    context.task_id = int(context.task_id)
    context.data_seed_start = int(context.data_seed_start)
    context.status_bar = str_to_bool(context.status_bar)
    context.episodes_per_task = int(context.episodes_per_task)
    context.num_cycles = int(context.num_cycles)
    context.num_episodes_reward_window = int(context.num_episodes_reward_window)
    context.algorithm_class = getattr(reinforcement_learning, context.algorithm)

    if 'interactive' not in context():
        context.interactive = False
    else:
        context.interactive = str_to_bool(context.interactive)
    if 'retrain' not in context():
        context.retrain = True
    else:
        context.retrain = str_to_bool(context.retrain)
    if 'crossing_kwargs' not in context() or context.crossing_kwargs is None:
        context.crossing_kwargs = {}
    if 'plot_dir' not in context():
        context.plot_dir = None
    if 'plot_format' not in context():
        context.plot_format = 'svg'

    context.network_name = context.network_config_file_path.split('/')[-1].split('.')[0]
    if 'export_network_config_file_path' not in context():
        if context.label is None:
            context.export_network_config_file_path = f"{context.output_dir}/{context.network_name}_optimized.yaml"
        else:
            context.export_network_config_file_path = \
                f"{context.output_dir}/{context.network_name}_{context.label}_optimized.yaml"

    network_config = read_from_yaml(context.network_config_file_path)
    context.layer_config = network_config['layer_config']
    context.projection_config = network_config['projection_config']
    context.training_kwargs = network_config['training_kwargs']

    context.tasks = make_crossing_tasks(**context.crossing_kwargs)
    context.num_tasks = len(context.tasks)
    context.num_blocks = context.num_tasks * context.num_cycles
    context.train_episodes = context.episodes_per_task * context.num_blocks
    context.max_reward = context.tasks[0].correct_reward


def get_random_seeds():
    network_seeds = [int.from_bytes((context.network_id, context.task_id, instance_id), byteorder='big')
                     for instance_id in range(context.seed_start, context.seed_start + context.num_instances)]
    data_seeds = [int.from_bytes((context.network_id, instance_id), byteorder='big')
                  for instance_id in range(context.data_seed_start, context.data_seed_start + context.num_instances)]
    return [network_seeds, data_seeds]


def get_data_file_path(seed, data_seed):
    """
    Path of the pickled network of one instance. Used both to export it and to load it back with retrain=False.

    :param seed: int
    :param data_seed: int
    :return: str
    """
    label = '' if context.label is None else f"_{context.label}"
    return f"{context.output_dir}/{context.network_name}/{context.network_name}_{seed}_{data_seed}{label}.pkl"


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
    data_file_path = get_data_file_path(seed, data_seed)

    if context.retrain:
        network = Network(context.layer_config, context.projection_config, seed=seed, **context.training_kwargs)
        algo = context.algorithm_class(network, persistent_state=network.persistent_state, seed=data_seed,
                                       **context.algorithm_kwargs)
        # one learn() per block, as MNIST CL trains one phase at a time; end_task() tells CL learning rules (SI, EWC)
        # that the task switched, and resets exploration with epsilon_schedule='reset'. Each block trains on one task
        # and validates on all
        for block in range(context.num_blocks):
            env = ScheduledEnv(TaskSchedule(context.tasks, order=[block % context.num_tasks]))
            algo.learn(env, context.episodes_per_task, status_bar=context.status_bar,
                       val_interval=context.val_interval)
            algo.end_task()
        # kept on the network so that it is exported with it
        network.rl_history = {'val_rewards': algo.val_rewards, 'val_episodes': algo.val_episodes,
                              'val_laps': algo.val_laps, 'episode_returns': algo.episode_returns,
                              'episode_lengths': algo.episode_lengths, 'loss_history': algo.loss_history,
                              'crossing_kwargs': context.crossing_kwargs,
                              'episodes_per_task': context.episodes_per_task, 'num_cycles': context.num_cycles,
                              'num_tasks': context.num_tasks}
    else:
        if not os.path.isfile(data_file_path):
            raise FileNotFoundError('nested_optimize_EIANN_Crossing_CL: retrain=False requires a network exported by '
                                    'a previous --export run; not found: %s' % data_file_path)
        network = utils.load_network(data_file_path, disp=context.disp)

    # rewards are sums of environment rewards, so only the loss shows that the weights diverged
    if not np.isfinite(network.rl_history['loss_history'][-1]):
        if context.interactive:
            context.update(locals())
        return dict()

    # entry [b, j]: greedy reward on task j at the end of block b; block b trains on task b % num_tasks
    rl_history = network.rl_history
    num_tasks = rl_history['num_tasks']
    num_blocks = num_tasks * rl_history['num_cycles']
    task_rewards = [[lap['episode_return'] for lap in laps] for laps in rl_history['val_laps']]
    block_args = (rl_history['val_episodes'], rl_history['episodes_per_task'], num_blocks,
                  context.num_episodes_reward_window)
    reward_matrix = block_rewards(task_rewards, *block_args)
    rates = block_outcome_rates(rl_history['val_laps'], *block_args)
    # like the cumulative val set of MNIST CL: the mean over the tasks seen so far (0..block)
    phase_rewards = seen_track_rewards(reward_matrix)
    phase_success = seen_track_rewards(rates['success'])
    results = {}
    for block in range(num_blocks):
        results[f'phase{block}_reward'] = phase_rewards[block]
        results[f'phase{block}_success'] = phase_success[block]
        for task in range(num_tasks):
            results[f'phase{block}_reward_task{task}'] = reward_matrix[block, task]
            results[f'phase{block}_success_task{task}'] = rates['success'][block, task]
            # NaN when no tile was reached; filter_features skips it instead of failing the instance
            results[f'phase{block}_choice_task{task}'] = rates['choice'][block, task]
    # summary only (not an objective)
    results['reward'] = np.mean([results[f'phase{block}_reward'] for block in range(num_blocks)])

    if plot:
        from EIANN.reinforcement_learning.crossing_plots import plot_crossing_summary  # keeps matplotlib off workers
        label = os.path.splitext(os.path.basename(data_file_path))[0]
        save_prefix = None if context.plot_dir is None else f"{context.plot_dir}/{label}"
        plot_crossing_summary(rl_history, context.num_episodes_reward_window, title=f'{seed}_{data_seed}',
                              save_prefix=save_prefix, plot_format=context.plot_format)

    if export:
        config_dict = {'layer_config': context.layer_config,
                       'projection_config': context.projection_config,
                       'training_kwargs': context.training_kwargs,
                       'algorithm': context.algorithm,
                       'algorithm_kwargs': context.algorithm_kwargs}
        write_to_yaml(context.export_network_config_file_path, config_dict, convert_scalars=True)
        utils.save_network(network, path=data_file_path, disp=False)
        if context.disp:
            print('nested_optimize_EIANN_Crossing_CL: pid: %i exported network config to %s and network history to %s' %
                  (os.getpid(), context.export_network_config_file_path, data_file_path))

    if not context.interactive:
        del network
        gc.collect()
    else:
        context.update(locals())

    return results


def filter_features(primitives, current_features, model_id=None, export=False, plot=False):
    # choice accuracy is NaN in a block where no tile was reached; it is averaged over the instances that have it, and
    # stays NaN when none has it (the key must still be there: nested fixes the feature names from the first model that
    # succeeds). Any other NaN or inf (e.g. a block without validation steps) fails the evaluation
    features = {}
    for instance_features in primitives:
        for key, val in instance_features.items():
            if key not in features:
                features[key] = []
            if np.isnan(val) or np.isinf(val):
                if '_choice_' in key and np.isnan(val):
                    continue
                return dict()
            features[key].append(val)
    for key, val in features.items():
        features[key] = np.mean(val) if val else np.nan

    return features


def get_objectives(features, model_id=None, export=False, plot=False):
    # one objective per block, like the per-phase accuracies of MNIST CL; everything else is a feature only
    objectives = {f'phase{block}_reward': context.max_reward - features[f'phase{block}_reward']
                  for block in range(context.num_blocks)}
    return features, objectives
