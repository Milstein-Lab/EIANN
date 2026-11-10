import numpy as np
import torch

from EIANN.recurrent_network import RecurrentNetwork
from ..network_updates import reinit_learning_rules
from ..continual import EpisodeBuffer, RecordEpisodes, estimate_importance


def obs_to_tensor(obs, device):
    return torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device)


def get_val_episodes(episodes, val_interval):
    """
    Episodes to validate after, with the same meaning of val_interval as in Network.train.

    Parameters
    ----------
    episodes : int
    val_interval : tuple of int
        (start_index, stop_index, interval), indexing into range(episodes).

    Returns
    -------
    set of int
    """
    episode_range = np.arange(episodes)
    start, stop, interval = val_interval
    start_index = 0 if start < 0 and abs(start) > episodes else episode_range[start]
    val_episodes = set(range(episode_range[stop], start_index - 1, -interval))
    if start_index == 0:
        val_episodes.add(0)
    return val_episodes


class RLAlgorithm(object):
    """
    Base class for RL algorithms that train an EIANN Network on gymnasium environments. Subclasses implement
    run_episode(), and end_episode() if they change between episodes, and learn() takes care of the rest.
    The network's output population gives one value per action of a Discrete action space. The network is run through
    a RecurrentNetwork (self.model), which carries population states across the steps of an episode when
    persistent_state.

    Continual learning: call learn() once per task and end_task() after each, as nested_optimize_EIANN_2ACDC_CL does.
    end_task() tells the CL learning rules that the task ended (network.update_CL_states()), and first hands rules
    with importance_source='external' (e.g. Backprop_EWC) their importance, estimated from a reservoir sample of the
    task's training episodes (see reinforcement_learning.continual).
    """
    def __init__(self, network, persistent_state=False, seed=None, importance_method='fisher',
                 importance_num_episodes=100, trace_clock='step'):
        """
        Parameters
        ----------
        network : Network
        persistent_state : bool, default False
            Whether to carry population states across the steps of an episode.
        seed : int, optional
            Seeds the algorithm's random number generator (e.g. for exploration).
        importance_method : str, default 'fisher'
            How end_task() estimates importance for rules with importance_source='external' (see
            continual.estimate_importance): 'fisher' or 'empirical_fisher'.
        importance_num_episodes : int, default 100
            Training episodes of each task kept (reservoir sample) to estimate importance from.
        trace_clock : str, default 'step'
            'step' or 'substep': whether learning rules with traces tick them once per environment step or once per
            integration substep (see RecurrentNetwork). 'substep' is needed for Eprop with forward_steps > 1.
        """
        self.network = network
        self.model = RecurrentNetwork(network, persistent_state=persistent_state, trace_clock=trace_clock)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.importance_method = importance_method
        self.importance_num_episodes = int(importance_num_episodes)
        self.episode_buffer = None
        self.reset_history()

    def reset_history(self):
        self.episode_returns = []
        self.episode_lengths = []
        self.loss_history = []
        self.val_rewards = []
        self.val_episodes = []
        self.val_laps = []

    def check_env(self, env):
        num_actions = env.action_space.n
        output_size = self.network.output_pop.size
        assert output_size == num_actions, \
            f'{type(self).__name__}: output population size ({output_size}) must equal the number of actions ' \
            f'({num_actions})'

    def run_episode(self, env):
        """
        Run one episode on env, updating the network.

        Returns
        -------
        tuple of (float, int)
            Return and length of the episode.
        """
        raise NotImplementedError

    def end_episode(self):
        """
        Called after every episode of learn, before validation.
        """
        pass

    def get_importance_projections(self):
        """
        Projections whose learning rule takes its importance from outside (importance_source='external').
        """
        return [projection for projection in self.network.projections.values()
                if getattr(projection.learning_rule, 'importance_source', None) == 'external']

    def end_task(self):
        """
        End the current task of a continual-learning run: estimate the importance of the weights of rules with
        importance_source='external' from the task's training episodes and pass it to them (set_importance), then call
        network.update_CL_states() and start a new episode buffer. Does nothing to rules without CL states.
        """
        projections = self.get_importance_projections()
        if projections:
            if self.episode_buffer is None or len(self.episode_buffer) == 0:
                raise RuntimeError(f'{type(self).__name__}.end_task: no training episodes were recorded for this task '
                                   f'(call learn() first)')
            importance = estimate_importance(self, self.episode_buffer.episodes, projections,
                                             method=self.importance_method)
            for projection, projection_importance in zip(projections, importance):
                projection.learning_rule.set_importance(projection_importance)
        self.network.update_CL_states()
        if self.episode_buffer is not None:
            self.episode_buffer.clear()

    def learn(self, env, episodes, status_bar=False, val_interval=None):
        """
        Train the network on env for the given number of episodes.

        Parameters
        ----------
        env : gymnasium.Env
        episodes : int
        status_bar : bool, default False
        val_interval : tuple of int, optional
            (start_index, stop_index, interval) of the episodes after which to run a greedy episode on env (on each of
            its envs for a ScheduledEnv), stored in val_rewards, val_episodes and val_laps. Each entry of val_laps is a
            list with the run_greedy_episode record of every env.

        Returns
        -------
        self
        """
        self.check_env(env)
        reinit_learning_rules(self.network)
        val_episodes = set() if val_interval is None else get_val_episodes(episodes, val_interval)
        # training episodes are recorded only when a rule needs them for importance; validation uses env itself
        train_env = env
        if self.get_importance_projections():
            if self.episode_buffer is None:
                self.episode_buffer = EpisodeBuffer(self.importance_num_episodes, seed=self.seed)
            train_env = RecordEpisodes(env, self.episode_buffer)

        if status_bar:
            from tqdm.autonotebook import tqdm
            episode_iter = tqdm(range(episodes), desc='Episodes')
        else:
            episode_iter = range(episodes)

        for episode in episode_iter:
            episode_return, episode_length = self.run_episode(train_env)
            self.episode_returns.append(episode_return)
            self.episode_lengths.append(episode_length)
            self.end_episode()

            if episode in val_episodes:
                laps = [self.run_greedy_episode(sub_env) for sub_env in getattr(env, 'envs', [env])]
                self.val_laps.append(laps)
                self.val_rewards.append(float(np.mean([lap['episode_return'] for lap in laps])))
                self.val_episodes.append(len(self.episode_returns) - 1)

            if status_bar:
                episode_iter.set_description(f'Return: {episode_return:.3f} - Episodes')

        return self

    def predict(self, obs):
        """
        Greedy action for one observation, as the next step of the current episode (call self.model.reset() at the
        start of an episode).

        Returns
        -------
        int
        """
        output = self.model.step(obs_to_tensor(obs, self.network.device), no_grad=True)
        return int(torch.argmax(output).item())

    def run_greedy_episode(self, env):
        """
        Run one greedy episode without learning, and record what the network did at every step.

        Returns
        -------
        dict
            episode_return: float
            q_values: np.ndarray, shape (num_steps, num_actions)
            actions: np.ndarray of int, shape (num_steps,)
            activity: dict of np.ndarray, shape (num_steps, population size), for every population other than the
                input and output populations
            outcome: the env's info['outcome'] at the last step (e.g. 'correct', 'wrong' or 'timeout' for
                CrossingChoiceEnv), or None if the env does not report one
        """
        hidden_populations = {name: population for name, population in self.network.populations.items()
                              if population is not self.network.input_pop and population is not self.network.output_pop}
        q_values, actions = [], []
        activity = {name: [] for name in hidden_populations}
        self.model.reset()
        obs, info = env.reset()
        episode_return = 0.
        done = False
        while not done:
            action = self.predict(obs)
            q_values.append(self.network.output_pop.activity.detach().cpu().numpy().copy())
            actions.append(action)
            for name, population in hidden_populations.items():
                activity[name].append(population.activity.detach().cpu().numpy().copy())
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            done = terminated or truncated
        return {'episode_return': episode_return, 'q_values': np.array(q_values), 'actions': np.array(actions),
                'activity': {name: np.array(values) for name, values in activity.items()},
                'outcome': info.get('outcome')}

    def evaluate(self, env, episodes=1):
        """
        Run greedy episodes without learning. For an env made of several envs (ScheduledEnv), each of its envs is
        evaluated instead, without advancing its schedule.

        Returns
        -------
        float
            Mean return over episodes (and over envs).
        """
        if hasattr(env, 'envs'):
            return float(np.mean([self.evaluate(sub_env, episodes=episodes) for sub_env in env.envs]))
        return float(np.mean([self.run_greedy_episode(env)['episode_return'] for episode in range(episodes)]))
