import torch

from .rl_algorithm import RLAlgorithm, obs_to_tensor
from ..network_updates import apply_learning_rules, compute_learning_rule_updates, step_learning_rules


def masked_target(q, actions, td_targets):
    """
    Target for the learning rules: the network's own output with the entries of the actions taken replaced by their
    TD targets, so the output error is non-zero only at those actions. This lets any learning rule's
    backward(network, output, target) be used unchanged.

    Parameters
    ----------
    q : torch.Tensor
        Output for one observation (num_actions,) or a sequence of observations (num_steps, num_actions).
    actions : torch.Tensor of int
    td_targets : torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    target = q.detach().clone()
    flat_target = target.view(-1, q.shape[-1])
    flat_target[torch.arange(flat_target.shape[0]), actions.reshape(-1)] = td_targets.reshape(-1)
    return target


class QLearning(RLAlgorithm):
    """
    One-step Q-learning with epsilon-greedy exploration. The network's output is Q(s, .). Each episode is run with fixed
    weights, then the weights are updated once on all of its steps. With persistent_state and
    backward_steps == forward_steps, gradients flow back through the whole episode. This suits backprop-based rules;
    see OnlineQLearning for updates after every step.
    """
    def __init__(self, network, gamma=0.99, epsilon=1., epsilon_decay=0.99, epsilon_min=0.05, persistent_state=False,
                 seed=None, importance_method='fisher', importance_num_episodes=100):
        """
        Parameters
        ----------
        network : Network
        gamma : float, default 0.99
            Discount factor.
        epsilon : float, default 1.
            Initial probability of a random action.
        epsilon_decay : float, default 0.99
            Multiplies epsilon after every episode.
        epsilon_min : float, default 0.05
        persistent_state : bool, default False
            Whether to carry population states across the steps of an episode.
        seed : int, optional
        importance_method : str, default 'fisher'
        importance_num_episodes : int, default 100
            See RLAlgorithm (continual learning with importance_source='external' rules).
        """
        super().__init__(network, persistent_state=persistent_state, seed=seed, importance_method=importance_method,
                         importance_num_episodes=importance_num_episodes)
        self.gamma = float(gamma)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def select_action(self, q):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(q.shape[-1]))
        return int(torch.argmax(q).item())

    def update(self, q, actions, td_targets, step=True):
        """
        Update weights and biases from the Q values of one step or a sequence of steps. With step=False, only compute
        the updates (rules that accumulate them, e.g. Eprop, apply them at the next step_learning_rules).
        """
        target = masked_target(q, actions, td_targets)
        q_taken = q.detach().view(-1, q.shape[-1])[torch.arange(actions.numel()), actions.reshape(-1)]
        self.loss_history.append(torch.mean((td_targets.reshape(-1) - q_taken) ** 2).item())
        if step:
            apply_learning_rules(self.network, q, target)
        else:
            compute_learning_rule_updates(self.network, q, target)

    def run_episode(self, env):
        """
        Run one episode with fixed weights, then update on all of its steps.

        Returns
        -------
        tuple of (float, int)
            Return and length of the episode.
        """
        device = self.network.device
        self.model.reset()
        obs, info = env.reset()
        q = self.model.step(obs_to_tensor(obs, device))
        q_values, actions, td_targets = [], [], []
        episode_return = 0.
        done = False
        while not done:
            action = self.select_action(q)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = obs_to_tensor(next_obs, device)
            done = terminated or truncated
            q_values.append(q)
            actions.append(action)
            episode_return += float(reward)
            if terminated:
                td_targets.append(float(reward))
            else:
                # the next step's Q values bootstrap this step's target
                next_q = self.model.peek(next_obs) if truncated else self.model.step(next_obs)
                td_targets.append(float(reward) + self.gamma * next_q.detach().max().item())
                q = next_q

        self.update(torch.stack(q_values), torch.tensor(actions, device=device),
                    torch.tensor(td_targets, device=device))
        self.model.detach()
        return episode_return, len(actions)

    def end_episode(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


class OnlineQLearning(QLearning):
    """
    One-step Q-learning that computes an update after every environment step, on that step's transition, and applies
    it at once or, with accumulate_updates, sums the updates of all steps and applies them at the end of the episode.
    With persistent_state, population states carry across steps, but gradients stop at step boundaries.

    accumulate_updates only defers rules that apply their update in step() (e.g. Eprop); Backprop updates the weights
    inside backward, at every step.
    """
    def __init__(self, network, gamma=0.99, epsilon=1., epsilon_decay=0.99, epsilon_min=0.05, persistent_state=False,
                 accumulate_updates=False, seed=None, importance_method='fisher', importance_num_episodes=100):
        """
        Parameters
        ----------
        network : Network
        gamma : float, default 0.99
        epsilon : float, default 1.
        epsilon_decay : float, default 0.99
        epsilon_min : float, default 0.05
        persistent_state : bool, default False
        accumulate_updates : bool, default False
            Whether to apply the summed updates of an episode at its end instead of after every step.
        seed : int, optional
        importance_method : str, default 'fisher'
        importance_num_episodes : int, default 100
            See RLAlgorithm.
        """
        super().__init__(network, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
                         persistent_state=persistent_state, seed=seed, importance_method=importance_method,
                         importance_num_episodes=importance_num_episodes)
        self.accumulate_updates = accumulate_updates

    def run_episode(self, env):
        device = self.network.device
        self.model.reset()
        obs, info = env.reset()
        q = self.model.step(obs_to_tensor(obs, device))
        episode_return = 0.
        episode_length = 0
        done = False
        while not done:
            action = self.select_action(q)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = obs_to_tensor(next_obs, device)
            if terminated:
                td_target = float(reward)
            else:
                # peek leaves population activities at this step for the learning rules
                td_target = float(reward) + self.gamma * self.model.peek(next_obs).max().item()
            self.update(q, torch.tensor(action, device=device), torch.tensor(td_target, device=device),
                        step=not self.accumulate_updates)
            self.model.detach()

            episode_return += float(reward)
            episode_length += 1
            done = terminated or truncated
            if not done:
                q = self.model.step(next_obs)
        if self.accumulate_updates:
            step_learning_rules(self.network)
        return episode_return, episode_length
