import numpy as np
import torch

from EIANN.recurrent_network import RecurrentNetwork
from ..network_updates import reinit_learning_rules


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
    """
    def __init__(self, network, persistent_state=False, seed=None):
        """
        Parameters
        ----------
        network : Network
        persistent_state : bool, default False
            Whether to carry population states across the steps of an episode.
        seed : int, optional
            Seeds the algorithm's random number generator (e.g. for exploration).
        """
        self.network = network
        self.model = RecurrentNetwork(network, persistent_state=persistent_state)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
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

        if status_bar:
            from tqdm.autonotebook import tqdm
            episode_iter = tqdm(range(episodes), desc='Episodes')
        else:
            episode_iter = range(episodes)

        for episode in episode_iter:
            episode_return, episode_length = self.run_episode(env)
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
                'activity': {name: np.array(values) for name, values in activity.items()}}

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
