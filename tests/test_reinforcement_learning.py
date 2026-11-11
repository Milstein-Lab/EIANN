import os

import gymnasium
import numpy as np
import pytest
import torch

import EIANN
import EIANN.network as nt
from gymnasium.utils.env_checker import check_env
from EIANN.recurrent_network import RecurrentNetwork
from EIANN.utils import read_from_yaml
from EIANN.reinforcement_learning import RLAlgorithm, QLearning, OnlineQLearning, DPPQL, OnlineDPPQL
from EIANN.reinforcement_learning.algorithms.rl_algorithm import get_val_episodes
from EIANN.reinforcement_learning.algorithms.q_learning import masked_target
from EIANN.reinforcement_learning.network_updates import (apply_learning_rules, compute_learning_rule_updates,
                                                          step_learning_rules)
from EIANN.reinforcement_learning.environments import (make_env, TaskSchedule, SequentialSchedule,
                                                       AlternatingSchedule, RandomSchedule, ScheduledEnv,
                                                       CueTreadmillEnv, make_2ACDC_tracks, CrossingChoiceEnv,
                                                       make_crossing_tasks)
from EIANN.reinforcement_learning.crossing_metrics import block_outcome_rates, episode_positions, cl_summary
from EIANN.reinforcement_learning.continual import EpisodeBuffer, RecordEpisodes, estimate_importance
from EIANN.reinforcement_learning.treadmill_metrics import (stack_laps, cross_correlation, region_cross_correlation,
                                                            licks_by_zone, block_rewards, seen_track_rewards)


@pytest.fixture
def q_network():
    layer_config = {
        'Input': {'E': {'size': 147}},
        'H1': {'E': {'size': 32, 'activation': 'relu'}},
        'Output': {'E': {'size': 7, 'activation': 'linear'}}}
    projection_config = {
        'H1': {'E': {'Input': {'E': {'weight_init': 'scaled_kaiming', 'weight_init_args': (1.,), 'direction': 'F',
                                     'learning_rule': 'Backprop'}}}},
        'Output': {'E': {'H1': {'E': {'weight_init': 'scaled_kaiming', 'weight_init_args': (1.,), 'direction': 'F',
                                      'learning_rule': 'Backprop'}}}}}
    return nt.Network(layer_config, projection_config, learning_rate=0.001, optimizer='SGD', seed=66049)


def test_make_env():
    env = make_env('MiniGrid-Empty-5x5-v0', seed=0)
    obs, info = env.reset()
    assert obs.shape == (147,)
    assert env.action_space.n == 7


def test_masked_target_updates_only_chosen_action(q_network):
    env = make_env('MiniGrid-Empty-5x5-v0', seed=0)
    obs, info = env.reset()
    obs = torch.as_tensor(obs, dtype=torch.float32)
    action = 2
    output_weight_before = q_network.projections['OutputE_H1E'].weight.detach().clone()

    # one QLearning update: the target equals the output except at the action taken
    q = q_network.forward(obs)
    target = q.detach().clone()
    target[action] += 1.
    apply_learning_rules(q_network, q, target)

    changed_rows = torch.any(q_network.projections['OutputE_H1E'].weight.detach() != output_weight_before, dim=1)
    assert changed_rows[action]
    assert not torch.any(changed_rows[np.arange(7) != action])


def test_masked_target_batch():
    q = torch.arange(6.).view(2, 3)
    target = masked_target(q, torch.tensor([2, 0]), torch.tensor([10., 20.]))
    assert torch.equal(target, torch.tensor([[0., 1., 10.], [20., 4., 5.]]))
    assert torch.equal(masked_target(q[0], torch.tensor(1), torch.tensor(7.)), torch.tensor([0., 7., 2.]))


@pytest.mark.parametrize('algo_class, losses_per_step', [(QLearning, False), (OnlineQLearning, True)])
def test_q_learning_learn_and_evaluate(q_network, algo_class, losses_per_step):
    env = make_env('MiniGrid-Empty-5x5-v0', seed=0)
    algo = algo_class(q_network, seed=0)
    algo.learn(env, episodes=3)
    assert len(algo.episode_returns) == 3
    assert len(algo.episode_lengths) == 3
    # QLearning updates once per episode, OnlineQLearning once per step
    assert len(algo.loss_history) == (sum(algo.episode_lengths) if losses_per_step else 3)
    assert isinstance(algo.evaluate(env, episodes=1), float)


def test_output_size_must_match_actions(q_network):
    env = make_env('CartPole-v1', seed=0)
    with pytest.raises(AssertionError):
        QLearning(q_network).learn(env, episodes=1)


def test_schedules():
    envs = ['A', 'B', 'C']
    assert list(SequentialSchedule(envs)) == [0, 1, 2]
    assert list(AlternatingSchedule(envs, num_cycles=2)) == [0, 1, 2, 0, 1, 2]
    assert list(AlternatingSchedule(envs[:2], num_cycles=2, episodes_per_task=2)) == [0, 0, 1, 1, 0, 0, 1, 1]
    assert list(TaskSchedule(envs, order=[0, 2, 0])) == [0, 2, 0]


def test_alternating_schedule_blocks_in_scheduled_env():
    envs = [CueTreadmillEnv(cue=0, rewarded_zone=0), CueTreadmillEnv(cue=1, rewarded_zone=1)]
    scheduled_env = ScheduledEnv(AlternatingSchedule(envs, num_cycles=1, episodes_per_task=3))
    # a finite schedule restarts when it runs out, so the blocks keep alternating past num_cycles
    assert [scheduled_env.reset()[1]['task'] for _ in range(9)] == [0, 0, 0, 1, 1, 1, 0, 0, 0]


def run_lap(env, actions):
    """Run one lap with actions[position] at each cell; return the observations seen and the total reward."""
    obs, info = env.reset()
    observations, total_reward, terminated = [obs], 0., False
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(actions[env.position])
        observations.append(obs)
        total_reward += reward
    return np.array(observations), total_reward


def test_treadmill_env():
    check_env(CueTreadmillEnv())
    assert make_env('EIANN/CueTreadmill-v0', cue=1, rewarded_zone=1).reset()[0].shape == (5,)


def test_treadmill_observations():
    env = CueTreadmillEnv(cue=1)
    observations, _ = run_lap(env, np.zeros(env.length, dtype=int))
    assert len(observations) == env.length + 1  # one per cell, then the terminal observation
    cell_index = np.argmax(observations[:-1], axis=1)
    assert np.all(cell_index[8:11] == 1 + 1)  # cue 1
    assert np.all(cell_index[14:16] == 1 + 2 + 0) and np.all(cell_index[18:20] == 1 + 2 + 1)  # zones 0 and 1
    assert cell_index[0] == 0 and cell_index[16] == 0 and cell_index[-1] == 0
    assert np.all(observations[-1] == 0)


def test_treadmill_rewards():
    env = CueTreadmillEnv()
    lick_at = lambda *positions: np.isin(np.arange(env.length), positions).astype(int)
    assert run_lap(env, lick_at(15))[1] == pytest.approx(1.)
    assert run_lap(env, np.ones(env.length, dtype=int))[1] == pytest.approx(1. - 0.1 * (env.length - 1))
    assert run_lap(env, lick_at())[1] == 0.  # ignoring at the rewarded cell is not penalized
    assert run_lap(env, lick_at(19))[1] == pytest.approx(-0.1)
    assert run_lap(CueTreadmillEnv(cue=1, rewarded_zone=1), lick_at(19))[1] == pytest.approx(1.)


def test_random_schedule_and_scheduled_env():
    envs = [CueTreadmillEnv(cue=0, rewarded_zone=0), CueTreadmillEnv(cue=1, rewarded_zone=1)]
    order = list(RandomSchedule(envs, num_phases=20, seed=0))
    assert order == list(RandomSchedule(envs, num_phases=20, seed=0))
    assert set(order) == {0, 1}

    scheduled_env = ScheduledEnv(RandomSchedule(envs, seed=0))
    for task in order:
        obs, info = scheduled_env.reset()
        assert info['task'] == task and scheduled_env.env is envs[task]

    # a finite schedule restarts when it runs out
    scheduled_env = ScheduledEnv(SequentialSchedule(envs))
    assert [scheduled_env.reset()[1]['task'] for _ in range(4)] == [0, 1, 0, 1]


def test_evaluate_scheduled_env():
    envs = [CueTreadmillEnv(cue=0, rewarded_zone=0), CueTreadmillEnv(cue=1, rewarded_zone=1)]
    network = nt.Network({'Input': {'E': {'size': 5}}, 'Output': {'E': {'size': 2}}},
                         {'Output': {'E': {'Input': {'E': {'direction': 'F', 'learning_rule': 'Backprop'}}}}},
                         learning_rate=0.01, seed=0)
    algo = QLearning(network)
    scheduled_env = ScheduledEnv(RandomSchedule(envs, seed=0))
    assert algo.evaluate(scheduled_env) == pytest.approx(np.mean([algo.evaluate(env) for env in envs]))
    assert scheduled_env.task is None  # evaluation does not advance the schedule


def test_online_q_learning_treadmill():
    # with one cue and one-cell reward zones, the rewarded cell is identifiable from its own observation
    env = CueTreadmillEnv(num_cues=1, reward_positions=(15,), reward_length=1)
    proj = {'weight_init': 'scaled_kaiming', 'weight_init_args': (1.,), 'direction': 'F', 'learning_rule': 'Backprop'}
    network = nt.Network({'Input': {'E': {'size': 3}}, 'H1': {'E': {'size': 16, 'activation': 'relu'}},
                          'Output': {'E': {'size': 2, 'activation': 'linear'}}},
                         {'H1': {'E': {'Input': {'E': dict(proj)}}}, 'Output': {'E': {'H1': {'E': dict(proj)}}}},
                         learning_rate=0.05, optimizer='SGD', seed=0)
    algo = OnlineQLearning(network, gamma=0.9, epsilon_decay=0.97, seed=0).learn(env, episodes=150)
    assert algo.evaluate(env) == pytest.approx(1.)


@pytest.fixture
def recurrent_network():
    proj = lambda direction: {'weight_init': 'scaled_kaiming', 'weight_init_args': (1.,), 'direction': direction,
                              'learning_rule': 'Backprop'}
    return nt.Network({'Input': {'E': {'size': 5}}, 'H1': {'E': {'size': 16, 'activation': 'relu'}},
                       'Output': {'E': {'size': 2, 'activation': 'linear'}}},
                      {'H1': {'E': {'Input': {'E': proj('F')}, 'H1': {'E': proj('R')}}},
                       'Output': {'E': {'H1': {'E': proj('F')}}}},
                      learning_rate=0.1, optimizer='SGD', seed=0)


def lap_observations(env):
    return [torch.as_tensor(obs) for obs in env.cell_observations]


@pytest.mark.parametrize('persistent_state', [False, True])
def test_recurrent_network_carries_cue(recurrent_network, persistent_state):
    model = RecurrentNetwork(recurrent_network, persistent_state=persistent_state)
    outputs = []
    for cue in (0, 1):
        model.reset()
        outputs.append(model.run(lap_observations(CueTreadmillEnv(cue=cue)), no_grad=True))
    assert outputs[0].shape == (23, 2)
    # cell 15 shows the same observation on both tracks; only a network with persistent state remembers the cue
    assert torch.equal(outputs[0][15], outputs[1][15]) != persistent_state


def test_recurrent_network_peek(recurrent_network):
    model = RecurrentNetwork(recurrent_network, persistent_state=True)
    observations = lap_observations(CueTreadmillEnv())
    model.reset()
    for obs in observations[:5]:
        model.step(obs)
    saved = {name: (pop.activity, pop.prev_activity, getattr(pop, 'state', None))
             for name, pop in recurrent_network.populations.items()}
    peeked = model.peek(observations[5])
    for name, pop in recurrent_network.populations.items():
        assert (pop.activity, pop.prev_activity, getattr(pop, 'state', None)) == saved[name]
    assert torch.allclose(peeked, model.step(observations[5], no_grad=True))


def test_recurrent_network_gradient_through_steps(recurrent_network):
    model = RecurrentNetwork(recurrent_network, persistent_state=True)
    observations = lap_observations(CueTreadmillEnv())
    model.reset()
    model.step(observations[0])
    hidden = recurrent_network.populations['H1E'].activity
    q = model.step(observations[1])
    assert torch.autograd.grad(q.sum(), hidden, retain_graph=True)[0].abs().sum() > 0
    # detach cuts the graph: the next step's output no longer depends on earlier steps
    model.detach()
    q = model.step(observations[2])
    assert torch.autograd.grad(q.sum(), hidden, allow_unused=True)[0] is None


def build_2ACDC_network(model, date='20261109', seed=0):
    config = read_from_yaml(os.path.join(os.path.dirname(EIANN.__file__), 'optimize', 'network_config', '2ACDC',
                                         f'{date}_EIANN_2_hidden_2ACDC_{model}_config.yaml'))
    return nt.Network(config['layer_config'], config['projection_config'], seed=seed, **config['training_kwargs'])


@pytest.mark.parametrize('store_dynamics', [False, True])
def test_substep_clock_matches_step_clock(store_dynamics):
    # one lap of the bpDale network (E/I loops, forward_steps 5, backward_steps 5): running each step as 5 one-substep
    # calls of network.forward gives the same outputs, histories and BPTT gradients through the whole lap
    observations = lap_observations(CueTreadmillEnv())[:8]
    results = {}
    for trace_clock in RecurrentNetwork.TRACE_CLOCKS:
        network = build_2ACDC_network('bpDale_relu_SGD')
        assert network.forward_steps == 5
        model = RecurrentNetwork(network, persistent_state=True, trace_clock=trace_clock)
        model.reset()
        outputs = torch.stack([model.step(obs, store_history=True, store_dynamics=store_dynamics)
                               for obs in observations])
        assert (network.forward_steps, network.backward_steps) == (5, 5)
        weights = [projection.weight for projection in network.projections.values()]
        grads = torch.autograd.grad((outputs ** 2).sum(), weights)
        history = {name: pop.get_attribute_history('activity') for name, pop in network.populations.items()}
        results[trace_clock] = outputs.detach(), grads, history
    (outputs, grads, history), (sub_outputs, sub_grads, sub_history) = results['step'], results['substep']
    assert torch.allclose(outputs, sub_outputs, atol=1e-6)
    for grad, sub_grad in zip(grads, sub_grads):
        assert torch.allclose(grad, sub_grad, atol=1e-5)
    for name in history:
        assert history[name].shape == sub_history[name].shape
        assert torch.allclose(history[name], sub_history[name], atol=1e-6)


def test_substep_clock_ticks_traces_every_substep():
    network = build_2ACDC_network('eprop_bpDale_relu_SGD')
    model = RecurrentNetwork(network, persistent_state=True, trace_clock='substep')
    assert network.trace_clock == 'substep'
    rule = network.projections['H1E_InputE'].learning_rule
    ticks = []
    rule.forward = lambda forward=rule.forward: ticks.append(1) or forward()
    model.reset()
    model.step(lap_observations(CueTreadmillEnv())[0])
    assert len(ticks) == network.forward_steps == 5


def test_trace_clock_must_be_valid(recurrent_network):
    with pytest.raises(ValueError):
        RecurrentNetwork(recurrent_network, trace_clock='episode')


@pytest.mark.parametrize('algo_class', [QLearning, OnlineQLearning])
def test_q_learning_persistent_state(recurrent_network, algo_class):
    tracks = [CueTreadmillEnv(cue=0, rewarded_zone=0), CueTreadmillEnv(cue=1, rewarded_zone=1)]
    algo = algo_class(recurrent_network, persistent_state=True, seed=0)
    algo.learn(ScheduledEnv(RandomSchedule(tracks, seed=0)), episodes=3)
    assert algo.episode_lengths == [23, 23, 23]
    assert isinstance(algo.evaluate(ScheduledEnv(RandomSchedule(tracks, seed=0))), float)


@pytest.mark.parametrize('algo_class', [QLearning, OnlineQLearning])
def test_q_learning_numpy_gamma(recurrent_network, algo_class):
    # nested passes parameters as numpy scalars, which must not change the dtype of the TD targets
    algo = algo_class(recurrent_network, gamma=np.float64(0.9), persistent_state=True, seed=0)
    algo.learn(CueTreadmillEnv(), episodes=1)
    assert algo.episode_lengths == [23]


@pytest.mark.parametrize('val_interval, val_episodes', [((0, -1, 4), {0, 1, 5, 9}), ((-5, -1, 2), {5, 7, 9}),
                                                        ((-20, -1, 4), {0, 1, 5, 9})])
def test_get_val_episodes(val_interval, val_episodes):
    assert get_val_episodes(10, val_interval) == val_episodes


def test_q_learning_val_interval():
    tracks = [CueTreadmillEnv(cue=0, rewarded_zone=0), CueTreadmillEnv(cue=1, rewarded_zone=1)]
    algo = QLearning(build_recurrent_network('Backprop'), persistent_state=True, seed=0)
    env = ScheduledEnv(RandomSchedule(tracks, seed=0))
    algo.learn(env, episodes=4, val_interval=(0, -1, 2))
    assert algo.val_episodes == [0, 1, 3]
    assert len(algo.val_rewards) == 3
    # further calls to learn continue the episode count
    algo.learn(env, episodes=2, val_interval=(-1, -1, 1))
    assert algo.val_episodes == [0, 1, 3, 5]
    assert algo.val_rewards[-1] == algo.evaluate(env)


def test_run_greedy_episode(recurrent_network):
    tracks = make_2ACDC_tracks()
    algo = QLearning(recurrent_network, persistent_state=True, seed=0)
    lap = algo.run_greedy_episode(tracks[0])
    assert lap['q_values'].shape == (23, 2)
    assert lap['actions'].shape == (23,)
    assert set(lap['activity']) == {'H1E'}  # hidden populations only
    assert lap['activity']['H1E'].shape == (23, 16)
    assert np.array_equal(lap['actions'], lap['q_values'].argmax(axis=1))
    assert lap['episode_return'] == algo.evaluate(tracks[0])


@pytest.fixture
def laps_and_tracks(recurrent_network):
    tracks = make_2ACDC_tracks()
    algo = QLearning(recurrent_network, persistent_state=True, seed=0)
    algo.learn(ScheduledEnv(RandomSchedule(tracks, seed=0)), episodes=5, val_interval=(0, -1, 2))
    return algo, tracks


def test_stack_laps(laps_and_tracks):
    algo, tracks = laps_and_tracks
    assert algo.val_episodes == [0, 2, 4]
    laps = stack_laps(algo.val_laps, algo.val_episodes)
    assert laps['episodes'].tolist() == [0, 2, 4]
    assert laps['rewards'].shape == (3, 2)
    assert laps['q_values'].shape == (3, 2, 23, 2)
    assert laps['actions'].shape == (3, 2, 23)
    assert laps['activity']['H1E'].shape == (3, 2, 23, 16)
    assert np.allclose(laps['rewards'].mean(axis=1), algo.val_rewards)


def test_cross_correlation():
    rng = np.random.default_rng(0)
    track = rng.random((23, 10))
    matrix = cross_correlation(np.stack([track, track]))
    assert matrix.shape == (1, 23, 23)
    assert np.allclose(np.diagonal(matrix[0]), 1.)
    # a unit silent on both tracks is dropped, so it does not change the correlation
    with_silent_unit = np.concatenate([track, np.zeros((23, 1))], axis=1)
    assert np.allclose(cross_correlation(np.stack([with_silent_unit, with_silent_unit])), matrix)
    assert cross_correlation(np.stack([track, track, track])).shape == (3, 23, 23)


def test_region_cross_correlation():
    track = CueTreadmillEnv()
    matrix = np.arange(23. * 23.).reshape(23, 23)
    means = region_cross_correlation(matrix, track)
    assert means['pre_r2'] == matrix[17, 17]
    assert means['indicator'] == pytest.approx(matrix[8:11, 8:11].mean())
    assert means['off_diagonal'] == pytest.approx(matrix[track.get_off_diagonal_mask()].mean())


def test_treadmill_regions_match_old_branch():
    track = CueTreadmillEnv()
    assert track.get_regions() == {'initial_region': (0, 8), 'indicator': (8, 11), 'pre_r1': (13, 14),
                                   'pre_r2': (17, 18), 'end_region': (19, 23)}
    expected = np.zeros((23, 23), dtype=bool)
    stretches = [(0, 8), (11, 14), (16, 18), (20, 23)]
    for first, second in [(1, 0), (2, 0), (3, 0), (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]:
        expected[stretches[first][0]:stretches[first][1], stretches[second][0]:stretches[second][1]] = True
    assert np.array_equal(track.get_off_diagonal_mask(), expected)


def test_licks_by_zone():
    actions = np.zeros((1, 2, 23), dtype=int)
    actions[0, 0, [14, 15]] = 1  # both cells of zone 0 on track 0
    actions[0, 1, 19] = 1  # one cell of zone 1 on track 1
    assert licks_by_zone({'actions': actions}, CueTreadmillEnv()).tolist() == [[[2, 0], [0, 1]]]


@pytest.mark.parametrize('blocks', [{}, {'episodes_per_task': 2, 'num_cycles': 2, 'num_tracks': 2}])
def test_plot_2ACDC_summary(laps_and_tracks, tmp_path, blocks):
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    from EIANN.reinforcement_learning.treadmill_plots import plot_2ACDC_summary

    algo, tracks = laps_and_tracks
    # with blocks, as exported by nested_optimize_EIANN_2ACDC_CL: per-track rewards with block boundaries
    rl_history = {'val_laps': algo.val_laps, 'val_episodes': algo.val_episodes, 'treadmill_kwargs': {}, **blocks}
    plot_2ACDC_summary(rl_history, title='test', save_prefix=str(tmp_path / 'model'), plot_format='png')
    assert {path.name for path in tmp_path.iterdir()} == {
        f'model_{name}.png' for name in ('val_rewards', 'final_q_values', 'actions_over_training',
                                         'hidden_activity_H1E', 'cross_correlation_H1E', 'region_cross_correlation',
                                         'region_cross_correlation_by_population')}


def test_new_algorithm_only_needs_run_episode():
    class NeverLick(RLAlgorithm):
        def run_episode(self, env):
            env.reset()
            episode_length, terminated = 0, False
            while not terminated:
                obs, reward, terminated, truncated, info = env.step(0)
                episode_length += 1
            return 0., episode_length

    algo = NeverLick(build_recurrent_network('Backprop'))
    assert algo.learn(CueTreadmillEnv(), episodes=4, val_interval=(0, -1, 3)) is algo
    assert algo.episode_lengths == [23, 23, 23, 23]
    assert algo.val_episodes == [0, 3]
    assert len(algo.val_rewards) == 2


def test_q_learning_epsilon_decays_every_episode():
    algo = QLearning(build_recurrent_network('Backprop'), epsilon=1., epsilon_decay=0.5, epsilon_min=0.2, seed=0)
    algo.learn(CueTreadmillEnv(), episodes=1)
    assert algo.epsilon == pytest.approx(0.5)
    algo.learn(CueTreadmillEnv(), episodes=2)
    assert algo.epsilon == pytest.approx(0.2)


@pytest.mark.parametrize('task, date, model, algo_class, schedule_class, algo_kwargs', [
    ('2ACDC', '20261102', 'van_bp_relu_SGD', QLearning, RandomSchedule, {}),
    ('2ACDC', '20261102', 'eprop_relu_SGD', OnlineQLearning, RandomSchedule, {}),
    ('2ACDC', '20261109', 'bpDale_relu_SGD', QLearning, RandomSchedule, {}),
    ('2ACDC', '20261109', 'eprop_bpDale_relu_SGD', OnlineQLearning, RandomSchedule,
     {'trace_clock': 'substep', 'accumulate_updates': True}),
    ('2ACDC', '20261109', 'BTSP_Continuous', OnlineQLearning, RandomSchedule, {'trace_clock': 'substep'}),
    ('2ACDC', '20261109', 'BTSP_Eprop', OnlineQLearning, RandomSchedule, {'trace_clock': 'substep'}),
    ('2ACDC', '20261109', 'eprop_bpDale_relu_SGD', OnlineQLearning, RandomSchedule,
     {'trace_clock': 'step', 'accumulate_updates': True}),
    ('2ACDC', '20261109', 'BTSP_Continuous', OnlineQLearning, RandomSchedule, {'trace_clock': 'step'}),
    ('2ACDC', '20261109', 'BTSP_Eprop', OnlineQLearning, RandomSchedule, {'trace_clock': 'step'}),
    ('2ACDC_CL', '20261103', 'van_bp_relu_SGD', QLearning, AlternatingSchedule, {}),
    ('2ACDC_CL', '20261103', 'eprop_relu_SGD', OnlineQLearning, AlternatingSchedule, {})])
def test_2ACDC_network_config(model, algo_class, task, date, schedule_class, algo_kwargs):
    config = read_from_yaml(os.path.join(os.path.dirname(EIANN.__file__), 'optimize', 'network_config', task,
                                         f'{date}_EIANN_2_hidden_{task}_{model}_config.yaml'))
    network = nt.Network(config['layer_config'], config['projection_config'], seed=0, **config['training_kwargs'])
    assert network.persistent_state
    assert network.projections['H1E_H1E'].direction == 'R'
    assert not any(projection.post is network.H2.E and projection.pre is network.H2.E
                   for projection in network.projections.values())

    tracks = [CueTreadmillEnv(cue=0, rewarded_zone=0), CueTreadmillEnv(cue=1, rewarded_zone=1)]
    algo = algo_class(network, persistent_state=network.persistent_state, seed=0, **algo_kwargs)
    schedule = RandomSchedule(tracks, seed=0) if schedule_class is RandomSchedule else \
        AlternatingSchedule(tracks, episodes_per_task=2)
    algo.learn(ScheduledEnv(schedule), episodes=3)
    assert algo.episode_lengths == [23, 23, 23]



def test_2ACDC_bpDale_and_BTSP_share_architecture():
    # bpDale_relu_SGD, eprop_bpDale_relu_SGD, BTSP_Continuous and BTSP_Eprop are compared on 2ACDC, so their E/SomaI
    # sizes and dynamics match (tau is the x0 value: optimized for all but bpDale_relu_SGD)
    def load(model):
        return read_from_yaml(os.path.join(os.path.dirname(EIANN.__file__), 'optimize', 'network_config', '2ACDC',
                                           f'20261109_EIANN_2_hidden_2ACDC_{model}_config.yaml'))

    def architecture(config):
        sizes = {(layer, pop): kwargs['size'] for layer, pops in config['layer_config'].items()
                 for pop, kwargs in pops.items() if pop in ('E', 'SomaI')}
        dynamics = {key: config['training_kwargs'][key] for key in ('forward_steps', 'tau', 'persistent_state')}
        return sizes, dynamics

    reference = architecture(load('bpDale_relu_SGD'))
    assert reference[1]['forward_steps'] == 5
    for model in ('eprop_bpDale_relu_SGD', 'BTSP_Continuous', 'BTSP_Eprop'):
        assert architecture(load(model)) == reference


def test_2ACDC_eprop_bpDale_fixes_every_SomaI_projection():
    network = build_2ACDC_network('eprop_bpDale_relu_SGD')
    for name, projection in network.projections.items():
        rule = type(projection.learning_rule).__name__
        if projection.pre.name == 'SomaI' or projection.post.name == 'SomaI':
            assert rule == 'LearningRule', name
        else:
            assert rule == 'Eprop', name

def test_block_rewards():
    # 2 blocks of 4 episodes, validated after every 2nd episode; track 0 rewards 0.1 * episode, track 1 constant 1
    episodes = np.array([1, 3, 5, 7])
    rewards = np.stack([0.1 * episodes, np.ones(4)], axis=1)
    matrix = block_rewards(rewards, episodes, episodes_per_task=4, num_blocks=2, window=2)
    assert np.allclose(matrix, [[0.3, 1.], [0.7, 1.]])
    matrix = block_rewards(rewards, episodes, episodes_per_task=4, num_blocks=2, window=4)
    assert np.allclose(matrix, [[0.2, 1.], [0.6, 1.]])
    # a block with no validation step in its window
    assert np.isnan(block_rewards(rewards, episodes, episodes_per_task=4, num_blocks=3, window=2)[2]).all()


def test_seen_track_rewards():
    # block 0 has seen only track 0; later blocks average both tracks
    matrix = np.array([[1., -0.1], [-0.1, 1.], [1., -0.3]])
    assert np.allclose(seen_track_rewards(matrix), [1., 0.45, 0.35])


def test_eprop_matches_bptt_for_leaky_single_layer():
    # with one leaky layer, the only path from a weight to the current state is through the layer's own leak, so the
    # e-prop gradient is exact
    network = nt.Network({'Input': {'E': {'size': 3}}, 'Output': {'E': {'size': 2, 'activation': 'linear'}}},
                         {'Output': {'E': {'Input': {'E': {'direction': 'F', 'learning_rule': 'Eprop'}}}}},
                         learning_rate=0.1, optimizer='SGD', tau=2, seed=0)
    weight = network.projections['OutputE_InputE'].weight
    samples = torch.rand(5, 3, generator=torch.Generator().manual_seed(0))
    target = torch.tensor([1., -1.])
    model = RecurrentNetwork(network, persistent_state=True)

    model.reset()
    output = model.run(samples)[-1]
    bptt_grad = torch.autograd.grad(network.criterion(output, target), weight)[0]

    weight_before = weight.detach().clone()
    model.reset()
    for i, sample in enumerate(samples):
        output = model.step(sample)
        # zero error (no weight change) until the last step, so only the eligibility trace accumulates
        apply_learning_rules(network, output, target if i == len(samples) - 1 else output.detach())
        model.detach()
    assert torch.allclose(weight_before - weight.detach(), 0.1 * bptt_grad)


def build_recurrent_network(learning_rule, tau=1, optimizer='SGD', learning_rate=0.1):
    proj = lambda direction: {'weight_init': 'scaled_kaiming', 'weight_init_args': (1.,), 'direction': direction,
                              'learning_rule': learning_rule}
    return nt.Network({'Input': {'E': {'size': 5}}, 'H1': {'E': {'size': 16, 'activation': 'relu'}},
                       'Output': {'E': {'size': 2, 'activation': 'linear'}}},
                      {'H1': {'E': {'Input': {'E': proj('F')}, 'H1': {'E': proj('R')}}},
                       'Output': {'E': {'H1': {'E': proj('F')}}}},
                      learning_rate=learning_rate, optimizer=optimizer, tau=tau, seed=0)


@pytest.mark.parametrize('optimizer, learning_rate', [('SGD', 0.1), ('Adam', 0.001)])
def test_eprop_equals_truncated_backprop_when_tau_is_1(optimizer, learning_rate):
    tracks = [CueTreadmillEnv(cue=0, rewarded_zone=0), CueTreadmillEnv(cue=1, rewarded_zone=1)]
    networks = {}
    for learning_rule in ('Eprop', 'Backprop'):
        networks[learning_rule] = build_recurrent_network(learning_rule, optimizer=optimizer, learning_rate=learning_rate)
        OnlineQLearning(networks[learning_rule], persistent_state=True, seed=0).learn(
            ScheduledEnv(RandomSchedule(tracks, seed=0)), episodes=3)
    for name, projection in networks['Eprop'].projections.items():
        assert not torch.isnan(projection.weight).any()
        assert torch.allclose(projection.weight, networks['Backprop'].projections[name].weight, atol=1e-6)


def test_eprop_requires_one_update_per_step():
    with pytest.raises(AssertionError):
        QLearning(build_recurrent_network('Eprop'), persistent_state=True).learn(CueTreadmillEnv(), episodes=1)


def test_eprop_accumulated_update_matches_bptt_for_leaky_single_layer():
    network = nt.Network({'Input': {'E': {'size': 3}}, 'Output': {'E': {'size': 2, 'activation': 'linear'}}},
                         {'Output': {'E': {'Input': {'E': {'direction': 'F', 'learning_rule': 'Eprop'}}}}},
                         learning_rate=0.1, optimizer='SGD', tau=2, seed=0)
    weight = network.projections['OutputE_InputE'].weight
    samples = torch.rand(5, 3, generator=torch.Generator().manual_seed(0))
    targets = torch.rand(5, 2, generator=torch.Generator().manual_seed(1))
    model = RecurrentNetwork(network, persistent_state=True)

    # exact gradient of the loss summed over every step of the sequence
    model.reset()
    outputs = model.run(samples)
    bptt_grad = torch.autograd.grad(sum(network.criterion(outputs[i], targets[i]) for i in range(5)), weight)[0]

    weight_before = weight.detach().clone()
    model.reset()
    for i, sample in enumerate(samples):
        output = model.step(sample)
        compute_learning_rule_updates(network, output, targets[i])
        assert torch.equal(weight.detach(), weight_before)  # nothing is applied until step_learning_rules
        model.detach()
    step_learning_rules(network)
    assert torch.allclose(weight_before - weight.detach(), 0.1 * bptt_grad)


def test_eprop_substep_clock_matches_bptt_for_leaky_single_layer():
    # forward_steps 5 with the trace ticked every substep: the only path from the weight to the state is still the
    # layer's own leak, so the accumulated e-prop update over a sequence equals BPTT through every substep and step
    network = nt.Network({'Input': {'E': {'size': 3}}, 'Output': {'E': {'size': 2, 'activation': 'linear'}}},
                         {'Output': {'E': {'Input': {'E': {'direction': 'F', 'learning_rule': 'Eprop'}}}}},
                         learning_rate=0.1, optimizer='SGD', tau=3, forward_steps=5, backward_steps=5, seed=0)
    weight = network.projections['OutputE_InputE'].weight
    samples = torch.rand(4, 3, generator=torch.Generator().manual_seed(0))
    targets = torch.rand(4, 2, generator=torch.Generator().manual_seed(1))

    model = RecurrentNetwork(network, persistent_state=True)
    model.reset()
    outputs = model.run(samples)
    bptt_grad = torch.autograd.grad(sum(network.criterion(outputs[i], targets[i]) for i in range(4)), weight)[0]

    weight_before = weight.detach().clone()
    model = RecurrentNetwork(network, persistent_state=True, trace_clock='substep')
    model.reset()
    for i, sample in enumerate(samples):
        output = model.step(sample)
        compute_learning_rule_updates(network, output, targets[i])
        model.detach()
    step_learning_rules(network)
    assert torch.allclose(weight_before - weight.detach(), 0.1 * bptt_grad, atol=1e-6)


def test_eprop_forward_steps_requires_substep_clock():
    network = nt.Network({'Input': {'E': {'size': 3}}, 'Output': {'E': {'size': 2, 'activation': 'linear'}}},
                         {'Output': {'E': {'Input': {'E': {'direction': 'F', 'learning_rule': 'Eprop'}}}}},
                         learning_rate=0.1, optimizer='SGD', tau=3, forward_steps=5, backward_steps=1, seed=0)
    model = RecurrentNetwork(network, persistent_state=True)
    model.reset()
    output = model.step(torch.ones(3))
    with pytest.raises(RuntimeError, match='trace_clock'):
        compute_learning_rule_updates(network, output, output.detach() + 1.)


def test_eprop_step_clock_ticks_once_per_step_on_last_substep():
    # an explicit trace_clock='step' runs Eprop with forward_steps > 1 as an approximation: the trace ticks once per
    # step, on the activity of the last substep
    network = nt.Network({'Input': {'E': {'size': 3}}, 'Output': {'E': {'size': 2, 'activation': 'linear'}}},
                         {'Output': {'E': {'Input': {'E': {'direction': 'F', 'learning_rule': 'Eprop'}}}}},
                         learning_rate=0.1, optimizer='SGD', tau=3, forward_steps=5, backward_steps=1, seed=0)
    rule = network.projections['OutputE_InputE'].learning_rule
    model = RecurrentNetwork(network, persistent_state=True, trace_clock='step')
    assert network.trace_clock == 'step'
    model.reset()
    samples = torch.rand(3, 3, generator=torch.Generator().manual_seed(0))
    expected = None
    for sample in samples:
        output = model.step(sample)
        expected = sample / 3. if expected is None else (1. - 1. / 3.) * expected + sample / 3.
        assert torch.allclose(rule.eligibility_trace, expected)
        compute_learning_rule_updates(network, output, output.detach() + 1.)
        model.detach()
    assert rule.accumulated_grad.abs().sum() > 0


def test_trace_clock_default_runs_as_step(recurrent_network):
    model = RecurrentNetwork(recurrent_network)
    assert model.trace_clock == 'step' and recurrent_network.trace_clock is None


def build_cl_network(learning_rule='Backprop', **learning_rule_kwargs):
    """
    Small recurrent Q network whose projections all use learning_rule.
    """
    proj = lambda direction: {'weight_init': 'scaled_kaiming', 'weight_init_args': (1.,), 'direction': direction,
                              'learning_rule': learning_rule, 'learning_rule_kwargs': dict(learning_rule_kwargs)}
    return nt.Network({'Input': {'E': {'size': 5}}, 'H1': {'E': {'size': 16, 'activation': 'relu'}},
                       'Output': {'E': {'size': 2, 'activation': 'linear'}}},
                      {'H1': {'E': {'Input': {'E': proj('F')}, 'H1': {'E': proj('R')}}},
                       'Output': {'E': {'H1': {'E': proj('F')}}}},
                      learning_rate=0.1, optimizer='SGD', persistent_state=True, seed=0)


def learn_blocks(algo, tracks, num_blocks=2, episodes_per_task=3):
    # as nested_optimize_EIANN_2ACDC_CL: one learn() per block, then end_task()
    for block in range(num_blocks):
        algo.learn(ScheduledEnv(TaskSchedule(tracks, order=[block % len(tracks)])), episodes=episodes_per_task)
        algo.end_task()


def test_episode_buffer_is_a_reservoir():
    buffer = EpisodeBuffer(max_episodes=3, seed=0)
    for i in range(10):
        buffer.add({'id': i})
    assert len(buffer) == 3 and buffer.num_seen == 10
    assert len({episode['id'] for episode in buffer.episodes}) == 3
    buffer.clear()
    assert len(buffer) == 0 and buffer.num_seen == 0


def test_record_episodes():
    buffer = EpisodeBuffer()
    env = RecordEpisodes(CueTreadmillEnv(), buffer)
    obs, info = env.reset()
    done, num_steps = False, 0
    while not done:
        obs, reward, terminated, truncated, info = env.step(num_steps % 2)
        done = terminated or truncated
        num_steps += 1
    episode = buffer.episodes[0]
    assert episode['obs'].shape == (num_steps + 1, 5)
    assert episode['actions'].tolist() == [step % 2 for step in range(num_steps)]
    assert len(episode['rewards']) == len(episode['terminated']) == num_steps


@pytest.mark.parametrize('method', ['fisher', 'empirical_fisher'])
def test_estimate_importance(method):
    network = build_cl_network('Backprop_EWC', importance_source='external')
    algo = QLearning(network, persistent_state=True, seed=0)
    algo.learn(ScheduledEnv(TaskSchedule(make_2ACDC_tracks(), order=[0])), episodes=3)
    projections = algo.get_importance_projections()
    assert len(projections) == 3
    importance = estimate_importance(algo, algo.episode_buffer.episodes, projections, method=method)
    for projection, projection_importance in zip(projections, importance):
        assert projection_importance.shape == projection.weight.shape
        assert (projection_importance >= 0).all() and projection_importance.sum() > 0
    with pytest.raises(ValueError):
        estimate_importance(algo, [], projections)


def test_end_task_external_ewc():
    network = build_cl_network('Backprop_EWC', ewc_lambda=10., importance_source='external')
    algo = QLearning(network, persistent_state=True, seed=0, importance_num_episodes=2)
    learn_blocks(algo, make_2ACDC_tracks())
    for projection in network.projections.values():
        rule = projection.learning_rule
        assert rule.task_num == 2 and len(rule.fisher_list) == 2 and rule.external_importance is None
        assert all((fisher >= 0).all() for fisher in rule.fisher_list)
    # importance comes from the task's own episodes, so the buffer is reset at every task switch
    assert len(algo.episode_buffer) == 0
    # without set_importance, an external rule can't consolidate
    with pytest.raises(ValueError):
        network.update_CL_states()


def test_end_task_si():
    network = build_cl_network('Backprop_SI', si_lambda=1.)
    algo = QLearning(network, persistent_state=True, seed=0)
    learn_blocks(algo, make_2ACDC_tracks())
    assert algo.episode_buffer is None  # SI's importance is online: no episodes are recorded
    for projection in network.projections.values():
        assert projection.learning_rule.task_num == 2
        assert projection.learning_rule.omega is not None and (projection.learning_rule.omega >= 0).all()


def test_end_task_without_cl_rules():
    algo = QLearning(build_cl_network(), persistent_state=True, seed=0)
    learn_blocks(algo, make_2ACDC_tracks())
    assert algo.episode_buffer is None and len(algo.episode_returns) == 6


def test_ewc_buffer_source_unchanged():
    # the default importance source (used on MNIST) still caches samples and computes the Fisher itself
    network = nt.Network({'Input': {'E': {'size': 4}}, 'Output': {'E': {'size': 2, 'activation': 'linear'}}},
                         {'Output': {'E': {'Input': {'E': {'learning_rule': 'Backprop_EWC',
                                                           'learning_rule_kwargs': {'fisher_num_samples': 3}}}}}},
                         learning_rate=0.1, optimizer='SGD', seed=0)
    for i in range(5):
        output = network.forward(torch.rand(4))
        apply_learning_rules(network, output, torch.tensor([1., 0.]))
    rule = network.projections['OutputE_InputE'].learning_rule
    assert len(network.ewc_buffer['data']) == 3 and network.ewc_buffer['num_seen'] == 5
    network.update_CL_states()
    assert len(rule.fisher_list) == 1 and rule.fisher_list[0].sum() > 0 and len(network.ewc_buffer['data']) == 0



def test_ewc_rules_pickled_before_importance_source():
    # MNIST EWC pickles exported before importance_source existed are reloaded with --retrain=False, after which
    # nested calls update_CL_states (and backward, if a later phase retrains); they must behave as 'buffer'
    network = nt.Network({'Input': {'E': {'size': 4}}, 'Output': {'E': {'size': 2, 'activation': 'linear'}}},
                         {'Output': {'E': {'Input': {'E': {'learning_rule': 'Backprop_EWC',
                                                           'learning_rule_kwargs': {'fisher_num_samples': 3}}}}}},
                         learning_rate=0.1, optimizer='SGD', seed=0)
    rule = network.projections['OutputE_InputE'].learning_rule
    del rule.importance_source, rule.external_importance
    for i in range(2):
        for j in range(4):
            apply_learning_rules(network, network.forward(torch.rand(4)), torch.tensor([1., 0.]))
        network.update_CL_states()
    assert len(rule.fisher_list) == 2 and rule.task_num == 2

@pytest.mark.parametrize('rule', ['SI', 'EWC'])
def test_2ACDC_CL_cl_rule_network_configs(rule):
    config = read_from_yaml(os.path.join(os.path.dirname(EIANN.__file__), 'optimize', 'network_config', '2ACDC_CL',
                                         f'20261103_EIANN_2_hidden_2ACDC_CL_van_bp_{rule}_relu_SGD_config.yaml'))
    network = nt.Network(config['layer_config'], config['projection_config'], seed=0, **config['training_kwargs'])
    assert {type(projection.learning_rule).__name__ for projection in network.projections.values()} == \
        {f'Backprop_{rule}'}
    algo = QLearning(network, persistent_state=network.persistent_state, seed=0)
    learn_blocks(algo, make_2ACDC_tracks(), episodes_per_task=2)
    assert all(projection.learning_rule.task_num == 2 for projection in network.projections.values())


# Crossing_CL (CrossingChoiceEnv), with either action set
CROSSING_ACTIONS = {'egocentric': {'L': 0, 'R': 1, 'F': 2}, 'allocentric': {'N': 0, 'E': 1, 'S': 2, 'W': 3}}
# shortest paths from (0, 0) (facing south) to the rewarded tile of tasks A, B and C
CROSSING_PATHS = {'egocentric': ('FFFLFFLFFFRF', 'FLFFRFFLF', 'FFLFFFLF'),
                  'allocentric': ('SSSEENNNE', 'SEESSE', 'SSEEEN')}
# task A: onto the unrewarded tile at (3, 3); a move into the wall east of the start
CROSSING_WRONG_PATH = {'egocentric': 'FFFLFFF', 'allocentric': 'SSSEEE'}
CROSSING_INTO_WALL = {'egocentric': 'LF', 'allocentric': 'EN'}


def run_crossing(env, actions):
    """Run a string of actions; return the total reward and the last step's terminated, truncated and info."""
    env.reset()
    total_reward = 0.
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(CROSSING_ACTIONS[env.actions_type][action])
        total_reward += reward
    return total_reward, terminated, truncated, info


def crossing_obs_at(env, row, col, direction=None):
    env.reset()
    env.agent_pos = env.to_grid_pos(row, col)
    if direction is not None:
        env.agent_dir = direction
    return env.gen_obs()


def test_crossing_layouts():
    layouts = [env.layout() for env in make_crossing_tasks()]
    assert layouts == [['S # . G', '. # . g', '. # . .', '. . . g'],
                       ['S # . g', '. . . g', '. # . .', '. # . G'],
                       ['S # . g', '. # . G', '. . . .', '. # . g']]


def test_crossing_env():
    env = CrossingChoiceEnv()
    check_env(env, skip_render_check=True)
    check_env(CrossingChoiceEnv(actions='allocentric'), skip_render_check=True)
    assert env.actions_type == 'egocentric' and env.action_space.n == 3
    # egocentric, task B at the start facing south: [x, y] in the agent's frame, x left (east) to right (west), y far to
    # near; category indices (unseen, empty, wall, goal)
    obs = make_env('EIANN/Crossing-v0', gap_row=1, rewarded_tile=(3, 3)).reset()[0]
    assert obs.shape == (36,) and obs.dtype == np.float32
    # to the left the wall column at rows 2, 1 (the gap) and 0; ahead the corridor; to the right the border
    assert np.array_equal(obs.reshape(3, 3, 4).argmax(axis=-1), [[2, 1, 2], [1, 1, 1], [2, 2, 2]])
    # allocentric, task B at (0, 0): [row, col], north up; category indices (empty, wall, goal)
    obs = make_env('EIANN/Crossing-v0', gap_row=1, rewarded_tile=(3, 3), actions='allocentric').reset()[0]
    assert obs.shape == (27,) and make_crossing_tasks(actions='allocentric')[0].action_space.n == 4
    assert np.array_equal(obs.reshape(3, 3, 3).argmax(axis=-1), [[1, 1, 1], [1, 0, 1], [1, 0, 0]])
    with pytest.raises(AssertionError):
        CrossingChoiceEnv(rewarded_tile=(2, 3))  # not a goal tile


@pytest.mark.parametrize('actions', ['egocentric', 'allocentric'])
@pytest.mark.parametrize('task', [0, 1, 2])
def test_crossing_shortest_paths(task, actions):
    env = make_crossing_tasks(actions=actions)[task]
    path = CROSSING_PATHS[actions][task]
    total_reward, terminated, truncated, info = run_crossing(env, path)
    assert total_reward == pytest.approx(1.) and terminated and not truncated and info['outcome'] == 'correct'
    # stopping one action short does not end the episode
    assert not run_crossing(env, path[:-1])[1]
    # a step penalty is added to every step, including the last
    env = make_crossing_tasks(actions=actions, step_reward=-0.01)[task]
    assert run_crossing(env, path)[0] == pytest.approx(1. - 0.01 * len(path))


@pytest.mark.parametrize('actions', ['egocentric', 'allocentric'])
def test_crossing_wrong_tile_and_timeout(actions):
    env = make_crossing_tasks(actions=actions)[0]
    total_reward, terminated, truncated, info = run_crossing(env, CROSSING_WRONG_PATH[actions])
    assert total_reward == pytest.approx(-0.1) and terminated and info['outcome'] == 'wrong'
    stay = 'L' if actions == 'egocentric' else 'N'  # turning in place, or walking into the border
    total_reward, terminated, truncated, info = run_crossing(env, stay * env.max_steps)
    assert total_reward == 0. and not terminated and truncated and info['outcome'] == 'timeout'
    run_crossing(env, CROSSING_INTO_WALL[actions])
    assert tuple(env.agent_pos) == env.to_grid_pos(0, 0)


def test_crossing_cue_visibility():
    # egocentric: at the start, facing south, the wall column is in view (the gap in B and C, an unbroken wall in A)
    tasks = make_crossing_tasks()
    start_obs = [task.reset()[0] for task in tasks]
    assert not any(np.array_equal(start_obs[i], start_obs[j]) for i in range(3) for j in range(i + 1, 3))
    # from (2, 3) facing north or south the wall column is out of view: the tasks look the same
    for direction in (1, 3):
        obs = [crossing_obs_at(task, 2, 3, direction) for task in tasks]
        assert all(np.array_equal(obs[0], other) for other in obs[1:])
    # allocentric: at the start only B's gap (row 1) is in view, so A and C look the same
    tasks = make_crossing_tasks(actions='allocentric')
    start_obs = [task.reset()[0] for task in tasks]
    assert np.array_equal(start_obs[0], start_obs[2]) and not np.array_equal(start_obs[0], start_obs[1])
    # from (2, 2), next to the wall, rows 1-3 of the wall are in view: every task looks different
    obs = [crossing_obs_at(task, 2, 2) for task in tasks]
    assert not any(np.array_equal(obs[i], obs[j]) for i in range(3) for j in range(i + 1, 3))
    # from (2, 3) the wall is out of view: the tasks look the same
    obs = [crossing_obs_at(task, 2, 3) for task in tasks]
    assert all(np.array_equal(obs[0], other) for other in obs[1:])


def build_crossing_network():
    proj = lambda direction: {'weight_init': 'scaled_kaiming', 'weight_init_args': (1.,), 'direction': direction,
                              'learning_rule': 'Backprop'}
    return nt.Network({'Input': {'E': {'size': 36}}, 'H1': {'E': {'size': 16, 'activation': 'relu'}},
                       'Output': {'E': {'size': 3, 'activation': 'linear'}}},
                      {'H1': {'E': {'Input': {'E': proj('F')}, 'H1': {'E': proj('R')}}},
                       'Output': {'E': {'H1': {'E': proj('F')}}}},
                      learning_rate=0.1, optimizer='SGD', persistent_state=True, seed=0)


def test_run_greedy_episode_records_outcome():
    algo = QLearning(build_crossing_network(), persistent_state=True, seed=0)
    assert algo.run_greedy_episode(make_crossing_tasks()[0])['outcome'] in ('correct', 'wrong', 'timeout')
    assert QLearning(build_cl_network(), seed=0).run_greedy_episode(CueTreadmillEnv())['outcome'] is None


@pytest.mark.parametrize('schedule, epsilons', [('fixed', (1., 1.)), ('decay', (0.25, 0.0625)),
                                                ('reset', (0.25, 1.))])
def test_epsilon_schedules(schedule, epsilons):
    # epsilon after two episodes of the first task, then after end_task()
    algo = QLearning(build_cl_network(), epsilon=1., epsilon_decay=0.5, epsilon_min=0., epsilon_schedule=schedule,
                     persistent_state=True, seed=0)
    algo.learn(CueTreadmillEnv(), episodes=2)
    assert algo.epsilon == pytest.approx(epsilons[0])
    algo.end_task()
    if schedule == 'decay':
        algo.learn(CueTreadmillEnv(), episodes=2)
    assert algo.epsilon == pytest.approx(epsilons[1])
    with pytest.raises(ValueError):
        OnlineQLearning(build_cl_network(), epsilon_schedule='linear')


def test_block_outcome_rates():
    # 1 block of 4 episodes, validated after every episode, on 2 tasks
    outcomes = [('correct', 'timeout'), ('wrong', 'timeout'), ('correct', 'wrong'), ('correct', 'timeout')]
    val_laps = [[{'outcome': outcome} for outcome in step] for step in outcomes]
    rates = block_outcome_rates(val_laps, np.arange(4), episodes_per_task=4, num_blocks=1, window=4)
    assert np.allclose(rates['success'], [[0.75, 0.]])
    assert np.allclose(rates['navigation'], [[1., 0.25]])
    assert np.allclose(rates['choice'], [[0.75, 0.]])
    rates = block_outcome_rates([[{'outcome': 'timeout'}]], [0], episodes_per_task=1, num_blocks=1, window=1)
    assert np.isnan(rates['choice']).all()
    with pytest.raises(ValueError):
        block_outcome_rates([[{'outcome': None}]], [0], episodes_per_task=1, num_blocks=1, window=1)


def test_crossing_CL_network_config():
    config = read_from_yaml(os.path.join(os.path.dirname(EIANN.__file__), 'optimize', 'network_config', 'Crossing_CL',
                                         '20261104_EIANN_2_hidden_Crossing_CL_van_bp_relu_SGD_config.yaml'))
    network = nt.Network(config['layer_config'], config['projection_config'], seed=0, **config['training_kwargs'])
    tasks = make_crossing_tasks()
    algo = QLearning(network, persistent_state=network.persistent_state, epsilon_schedule='reset', seed=0)
    for block in range(len(tasks)):
        algo.learn(ScheduledEnv(TaskSchedule(tasks, order=[block])), episodes=2, val_interval=(0, -1, 1))
        algo.end_task()
    rates = block_outcome_rates(algo.val_laps, algo.val_episodes, episodes_per_task=2, num_blocks=3, window=2)
    assert rates['success'].shape == (3, 3)


def test_crossing_network_config():
    # Crossing: tasks interleaved at random, as nested_optimize_EIANN_Crossing; the whole run is one block
    config = read_from_yaml(os.path.join(os.path.dirname(EIANN.__file__), 'optimize', 'network_config', 'Crossing',
                                         '20261110_EIANN_2_hidden_Crossing_van_bp_relu_SGD_config.yaml'))
    network = nt.Network(config['layer_config'], config['projection_config'], seed=0, **config['training_kwargs'])
    algo = QLearning(network, persistent_state=network.persistent_state, seed=0)
    algo.learn(ScheduledEnv(RandomSchedule(make_crossing_tasks(), seed=0)), episodes=4, val_interval=(0, -1, 1))
    rates = block_outcome_rates(algo.val_laps, algo.val_episodes, episodes_per_task=4, num_blocks=1, window=4)
    assert rates['success'].shape == (1, 3)

@pytest.mark.parametrize('actions', ['egocentric', 'allocentric'])
@pytest.mark.parametrize('task', [0, 1, 2])
def test_episode_positions(task, actions):
    env = make_crossing_tasks(actions=actions)[task]
    path = CROSSING_PATHS[actions][task]
    positions, outcome = episode_positions(env, [CROSSING_ACTIONS[actions][action] for action in path])
    assert tuple(positions[0]) == env.start_pos and tuple(positions[-1]) == env.rewarded_tile
    assert len(positions) == len(path) + 1 and outcome == 'correct'
    # turning, or a move into a wall, repeats the position
    into_wall = CROSSING_INTO_WALL[actions]
    positions, outcome = episode_positions(env, [CROSSING_ACTIONS[actions][action] for action in into_wall])
    assert positions.tolist() == [[0, 0]] * (len(into_wall) + 1) and outcome is None


def test_cl_summary():
    summary = cl_summary(np.eye(3), baseline=np.zeros(3))
    assert summary == pytest.approx({'ACC': 1 / 3, 'BWT': -1., 'FWT': 0.})
    summary = cl_summary(np.ones((3, 3)), baseline=np.array([0., 0.5, 0.]))
    assert summary == pytest.approx({'ACC': 1., 'BWT': 0., 'FWT': 0.75})
    assert np.isnan(cl_summary(np.eye(3))['FWT'])


def test_crossing_plots(tmp_path):
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    from EIANN.reinforcement_learning.crossing_plots import (plot_crossing_summary, animate_learning,
                                                             animate_greedy_episodes)
    tasks = make_crossing_tasks(max_steps=10)
    algo = QLearning(build_crossing_network(), persistent_state=True, seed=0)
    for block in range(len(tasks)):
        algo.learn(ScheduledEnv(TaskSchedule(tasks, order=[block])), episodes=2, val_interval=(0, -1, 1))
        algo.end_task()
    # as exported by nested_optimize_EIANN_Crossing_CL
    rl_history = {'val_laps': algo.val_laps, 'val_episodes': algo.val_episodes, 'crossing_kwargs': {'max_steps': 10},
                  'episodes_per_task': 2, 'num_cycles': 1, 'num_tasks': 3}
    plot_crossing_summary(rl_history, 2, title='test', save_prefix=str(tmp_path / 'model'), plot_format='png')
    animate_learning(rl_history, window=2, val_steps=[0, 5], save_path=str(tmp_path / 'learning.gif'))
    animate_greedy_episodes(rl_history, val_step=-1, save_path=str(tmp_path / 'greedy.gif'))
    assert {path.name for path in tmp_path.iterdir()} == {'model_outcome_curves.png', 'model_outcome_matrices.png',
                                                          'learning.gif', 'greedy.gif'}


def test_crossing_plots_interleaved(tmp_path):
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    from EIANN.reinforcement_learning.crossing_plots import plot_crossing_summary, animate_learning
    algo = QLearning(build_crossing_network(), persistent_state=True, seed=0)
    algo.learn(ScheduledEnv(RandomSchedule(make_crossing_tasks(max_steps=10), seed=0)), episodes=4,
               val_interval=(0, -1, 1))
    # as exported by nested_optimize_EIANN_Crossing
    rl_history = {'val_laps': algo.val_laps, 'val_episodes': algo.val_episodes, 'crossing_kwargs': {'max_steps': 10},
                  'episodes_per_task': 4, 'num_cycles': 1, 'num_tasks': 3, 'interleaved': True}
    plot_crossing_summary(rl_history, 4, title='test', save_prefix=str(tmp_path / 'model'), plot_format='png')
    animate_learning(rl_history, window=2, val_steps=[0, 3], save_path=str(tmp_path / 'learning.gif'))
    assert {path.name for path in tmp_path.iterdir()} == {'model_outcome_curves.png', 'model_outcome_matrices.png',
                                                          'learning.gif'}


def build_qnext_network(input_size=16, output_size=1, learning_rule='Backprop'):
    proj = lambda: {'weight_init': 'scaled_kaiming', 'weight_init_args': (1.,), 'direction': 'F',
                    'learning_rule': learning_rule}
    return nt.Network({'Input': {'E': {'size': input_size}}, 'H1': {'E': {'size': 8, 'activation': 'relu'}},
                       'Output': {'E': {'size': output_size, 'activation': 'linear'}}},
                      {'H1': {'E': {'Input': {'E': proj()}}}, 'Output': {'E': {'H1': {'E': proj()}}}},
                      learning_rate=0.1, optimizer='SGD', seed=1)


class RecordRewards(gymnasium.Wrapper):
    def reset(self, **kwargs):
        self.rewards = []
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.rewards.append(float(reward))
        return obs, reward, terminated, truncated, info


@pytest.mark.parametrize('algo_class, learning_rule, algo_kwargs', [
    (DPPQL, 'Backprop', {}), (OnlineDPPQL, 'Backprop', {}), (OnlineDPPQL, 'Eprop', {'accumulate_updates': True})])
def test_dppql_learn_and_evaluate(algo_class, learning_rule, algo_kwargs):
    network = build_recurrent_network(learning_rule, tau=2 if learning_rule == 'Eprop' else 1)
    qnext_network = build_qnext_network()
    initial_weights = {name: projection.weight.detach().clone()
                       for name, projection in qnext_network.projections.items()}
    algo = algo_class(network, qnext_network, persistent_state=True, seed=0, **algo_kwargs)
    tracks = make_2ACDC_tracks()
    algo.learn(ScheduledEnv(RandomSchedule(tracks, seed=0)), episodes=3, val_interval=(0, -1, 2))
    assert algo.episode_lengths == [23, 23, 23]
    # DPPQL updates both networks once per episode, OnlineDPPQL once per step
    num_updates = 3 if algo_class is DPPQL else 3 * 23
    assert len(algo.loss_history) == num_updates and len(algo.qnext_loss_history) == num_updates
    assert np.isfinite(algo.loss_history).all() and np.isfinite(algo.qnext_loss_history).all()
    assert any(not torch.equal(initial_weights[name], projection.weight)
               for name, projection in qnext_network.projections.items())
    assert isinstance(algo.evaluate(tracks[0]), float)


@pytest.mark.parametrize('algo_class', [DPPQL, OnlineDPPQL])
def test_dppql_targets(algo_class):
    network, qnext_network = build_recurrent_network('Backprop'), build_qnext_network()
    algo = algo_class(network, qnext_network, gamma=0.5, epsilon=0.5, persistent_state=True, seed=0)
    q_updates, qnext_updates = [], []
    # record what each network would be updated toward, with the weights left fixed
    algo.update = lambda q, actions, td_targets, step=True: q_updates.append((q.detach(), td_targets.detach()))
    algo.update_qnext = lambda output, target, step=True: qnext_updates.append((output.detach(), target.detach()))
    env = RecordRewards(CueTreadmillEnv(cue=0, rewarded_zone=0))
    algo.run_episode(env)

    # DPPQL makes one update per episode on all steps; OnlineDPPQL one per step, in step order
    q = torch.cat([q.view(-1, 2) for q, _ in q_updates])
    td_targets = torch.cat([target.view(-1) for _, target in q_updates])
    qnext = torch.cat([output.view(-1) for output, _ in qnext_updates])
    qnext_targets = torch.cat([target.view(-1) for _, target in qnext_updates])
    rewards = torch.tensor(env.rewards)
    assert len(q) == len(qnext) == len(rewards) == 23
    # Q bootstraps from QNext(h_t), and from nothing at termination
    assert torch.allclose(td_targets[:-1], rewards[:-1] + 0.5 * qnext[:-1])
    assert td_targets[-1] == rewards[-1]
    # QNext(h_t) predicts max_a Q(s_{t+1}, a), and 0 at termination
    assert torch.allclose(qnext_targets[:-1], q[1:].max(dim=1).values)
    assert qnext_targets[-1] == 0.


def test_dppql_check_env():
    env = CueTreadmillEnv()
    with pytest.raises(AssertionError):  # QNext input must match the size of H1E (16)
        DPPQL(build_recurrent_network('Backprop'), build_qnext_network(input_size=8)).learn(env, episodes=1)
    with pytest.raises(AssertionError):  # QNext predicts one value
        DPPQL(build_recurrent_network('Backprop'), build_qnext_network(output_size=2)).learn(env, episodes=1)
    with pytest.raises(ValueError):
        DPPQL(build_recurrent_network('Backprop'), build_qnext_network(),
              qnext_input_population='H2E').learn(env, episodes=1)


@pytest.mark.parametrize('q_config, qnext_config, algo_class', [
    ('20261102_EIANN_2_hidden_2ACDC_van_bp_relu_SGD_config.yaml',
     '20261109_EIANN_1_hidden_2ACDC_DPPQL_QNext_van_bp_relu_SGD_config.yaml', DPPQL),
    ('20261102_EIANN_2_hidden_2ACDC_eprop_relu_SGD_config.yaml',
     '20261109_EIANN_1_hidden_2ACDC_DPPQL_QNext_van_bp_relu_SGD_config.yaml', OnlineDPPQL),
    ('20261109_EIANN_2_hidden_2ACDC_bpDale_relu_SGD_config.yaml',
     '20261109_EIANN_1_hidden_2ACDC_DPPQL_QNext_bpDale_relu_SGD_config.yaml', DPPQL)])
def test_2ACDC_DPPQL_network_config(q_config, qnext_config, algo_class):
    config_dir = os.path.join(os.path.dirname(EIANN.__file__), 'optimize', 'network_config', '2ACDC')
    config = read_from_yaml(os.path.join(config_dir, q_config))
    qnext_config = read_from_yaml(os.path.join(config_dir, qnext_config))
    network = nt.Network(config['layer_config'], config['projection_config'], seed=0, **config['training_kwargs'])
    qnext_network = nt.Network(qnext_config['layer_config'], qnext_config['projection_config'], seed=1,
                               **qnext_config['training_kwargs'])
    assert network.persistent_state and not qnext_network.persistent_state
    assert network.projections['H1E_H1E'].direction == 'R'
    assert not any(projection.post is network.H2.E and projection.pre is network.H2.E
                   for projection in network.projections.values())

    algo = algo_class(network, qnext_network, persistent_state=network.persistent_state, seed=0)
    algo.learn(ScheduledEnv(RandomSchedule(make_2ACDC_tracks(), seed=0)), episodes=2)
    assert algo.episode_lengths == [23, 23]
