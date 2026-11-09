"""
Continual learning (CL) support for RL: the data and the replay that offline-importance CL rules (e.g. Backprop_EWC
with importance_source='external') need at the end of a task.

The rule consolidates importance (stores it with anchor weights and penalizes changes); what the importance is
computed from is task specific, so for RL it is estimated here from the task's own training episodes, replayed
through the network with its recurrent state, and handed to the rule by RLAlgorithm.end_task(). Rules whose importance
accumulates online (e.g. Backprop_SI) only need end_task() to mark the task switch.
"""
import gymnasium
import numpy as np
import torch


class EpisodeBuffer(object):
    """
    Uniform random sample (reservoir sampling) of the episodes of the current task: after n episodes, each one has
    been kept with equal probability max_episodes / n.
    """
    def __init__(self, max_episodes=100, seed=None):
        """
        Parameters
        ----------
        max_episodes : int, default 100
        seed : int, optional
        """
        self.max_episodes = int(max_episodes)
        self.rng = np.random.default_rng(seed)
        self.clear()

    def clear(self):
        self.episodes = []
        self.num_seen = 0

    def add(self, episode):
        """
        Parameters
        ----------
        episode : dict
            From RecordEpisodes: obs (num_steps + 1, obs size), actions, rewards and terminated (num_steps,).
        """
        if self.num_seen < self.max_episodes:
            self.episodes.append(episode)
        else:
            idx = int(self.rng.integers(self.num_seen + 1))
            if idx < self.max_episodes:
                self.episodes[idx] = episode
        self.num_seen += 1

    def __len__(self):
        return len(self.episodes)


class RecordEpisodes(gymnasium.Wrapper):
    """
    Records every complete episode run on env into an EpisodeBuffer, without changing what the algorithm sees, so any
    RLAlgorithm can be used. Only reset and step pass through: validation and evaluation use the unwrapped env.
    """
    def __init__(self, env, buffer):
        """
        Parameters
        ----------
        env : gymnasium.Env
        buffer : EpisodeBuffer
        """
        super().__init__(env)
        self.buffer = buffer
        self.episode = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # an episode left unfinished is dropped
        self.episode = {'obs': [np.asarray(obs, dtype=np.float32)], 'actions': [], 'rewards': [], 'terminated': []}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode['obs'].append(np.asarray(obs, dtype=np.float32))
        self.episode['actions'].append(int(action))
        self.episode['rewards'].append(float(reward))
        self.episode['terminated'].append(bool(terminated))
        if terminated or truncated:
            self.buffer.add({key: np.array(values) for key, values in self.episode.items()})
            self.episode = None
        return obs, reward, terminated, truncated, info


def estimate_importance(algo, episodes, projections, method='fisher'):
    """
    Diagonal per-weight importance of the task the episodes come from. Each episode is replayed through algo.model from
    a fresh state, with gradients through the recurrent state as in training, and the squared gradient of one scalar
    per step is averaged over all steps of all episodes:
        'fisher': Q(s_t, a_t). For Q values with unit-variance Gaussian noise, this is the diagonal of the Fisher
            information; it needs no targets and does not vanish when the TD errors do.
        'empirical_fisher': the step's TD loss (network.criterion on the masked target, as in training), with TD
            targets recomputed at the current weights. This is what Backprop_EWC computes on MNIST.

    Parameters
    ----------
    algo : RLAlgorithm
        Its gamma is used by 'empirical_fisher'.
    episodes : list of dict
        From EpisodeBuffer.
    projections : list of Projection
    method : str, default 'fisher'

    Returns
    -------
    list of torch.Tensor
        Importance of each projection, shaped like its weight.
    """
    from .algorithms.q_learning import masked_target  # imported here: the algorithms import this module

    if method not in ('fisher', 'empirical_fisher'):
        raise ValueError(f"estimate_importance: method must be 'fisher' or 'empirical_fisher', not {method}")
    if len(episodes) == 0:
        raise ValueError('estimate_importance: no episodes to estimate importance from')
    network, model = algo.network, algo.model
    device = network.device
    weights = [projection.weight for projection in projections]
    importance = [torch.zeros_like(weight) for weight in weights]
    num_steps = 0
    for episode in episodes:
        model.reset()
        obs = torch.as_tensor(episode['obs'], dtype=torch.float32, device=device)
        length = len(episode['actions'])
        q_values = [model.step(obs[t]) for t in range(length)]
        for t in range(length):
            q, action = q_values[t], int(episode['actions'][t])
            if method == 'fisher':
                scalar = q[action]
            else:
                if episode['terminated'][t]:
                    td_target = float(episode['rewards'][t])
                else:
                    next_q = q_values[t + 1] if t + 1 < length else model.peek(obs[length])
                    td_target = float(episode['rewards'][t]) + algo.gamma * next_q.detach().max().item()
                target = masked_target(q, torch.tensor(action, device=device), torch.tensor(td_target, device=device))
                scalar = network.criterion(q, target)
            grads = torch.autograd.grad(scalar, weights, retain_graph=True, allow_unused=True)
            for total, grad in zip(importance, grads):
                if grad is not None:
                    total += grad.detach() ** 2
            num_steps += 1
        model.detach()
    return [total / num_steps for total in importance]
