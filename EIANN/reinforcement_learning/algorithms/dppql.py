import torch

from EIANN.recurrent_network import RecurrentNetwork
from .q_learning import QLearning
from .rl_algorithm import obs_to_tensor
from ..network_updates import (apply_learning_rules, compute_learning_rule_updates, reinit_learning_rules,
                               step_learning_rules)


class DPPQL(QLearning):
    """
    Dual path predictive Q learning (DPPQL), with epsilon-greedy exploration and the weights updated once per episode.
    Ported from DNQL on the reinforcement_learning branch (EIANN/optimize/train/dnql.py), itself a port of basic_rl's
    dual_q_learning.

    Two paths:
    - Q path: the (recurrent) Q network gives Q(s_t, .) from the observation.
    - Predictive path: a separate feedforward QNext network reads the Q network's hidden activity h_t (the population
      qnext_input_population, detached) and predicts V(s_{t+1}) = max_a Q(s_{t+1}, a).

    Q's TD target bootstraps from the predictive path: r_t + gamma * QNext(h_t), or r_t at termination. It uses only what
    is available at step t, with no look-ahead to s_{t+1}, so the update is local in time. QNext(h_t) is trained toward
    max_a Q(s_{t+1}, a), and toward 0 at termination.

    Each episode runs with fixed weights, then both networks are updated on all of its steps, as in QLearning. With
    persistent_state and backward_steps == forward_steps, gradients flow back through the whole episode in the Q
    network. See OnlineDPPQL for updates after every step.
    """
    def __init__(self, network, qnext_network, qnext_input_population='H1E', gamma=0.99, epsilon=1.,
                 epsilon_decay=0.99, epsilon_min=0.05, epsilon_schedule='decay', persistent_state=False, seed=None,
                 importance_method='fisher', importance_num_episodes=100, trace_clock='step'):
        """
        Parameters
        ----------
        network : Network
            The Q network: one output per action.
        qnext_network : Network
            The QNext network: input size equal to the size of qnext_input_population, one output. It is run without
            persistent state.
        qnext_input_population : str, default 'H1E'
            Full name (layer + population, e.g. 'H1E') of the Q network population whose activity QNext reads.
        gamma : float, default 0.99
        epsilon : float, default 1.
        epsilon_decay : float, default 0.99
        epsilon_min : float, default 0.05
        epsilon_schedule : str, default 'decay'
            See QLearning.
        persistent_state : bool, default False
            Whether to carry the Q network's population states across the steps of an episode.
        seed : int, optional
        importance_method : str, default 'fisher'
        importance_num_episodes : int, default 100
            See RLAlgorithm.
        trace_clock : str, default 'step'
            See RLAlgorithm; applies to the Q network. QNext always uses 'step'.
        """
        self.qnext_network = qnext_network
        self.qnext_model = RecurrentNetwork(qnext_network, persistent_state=False)
        self.qnext_input_population = qnext_input_population
        super().__init__(network, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
                         epsilon_schedule=epsilon_schedule, persistent_state=persistent_state, seed=seed,
                         importance_method=importance_method, importance_num_episodes=importance_num_episodes,
                         trace_clock=trace_clock)

    def reset_history(self):
        super().reset_history()
        self.qnext_loss_history = []

    def check_env(self, env):
        super().check_env(env)
        name = type(self).__name__
        if self.qnext_input_population not in self.network.populations:
            raise ValueError(f'{name}: qnext_input_population {self.qnext_input_population!r} is not a population of '
                             f'the Q network ({list(self.network.populations)})')
        input_size = self.network.populations[self.qnext_input_population].size
        assert self.qnext_network.input_pop.size == input_size, \
            f'{name}: QNext input size ({self.qnext_network.input_pop.size}) must equal the size of ' \
            f'{self.qnext_input_population} ({input_size})'
        assert self.qnext_network.output_pop.size == 1, \
            f'{name}: QNext output size ({self.qnext_network.output_pop.size}) must be 1'

    def learn(self, env, episodes, status_bar=False, val_interval=None):
        reinit_learning_rules(self.qnext_network)
        return super().learn(env, episodes, status_bar=status_bar, val_interval=val_interval)

    def end_task(self):
        super().end_task()
        self.qnext_network.update_CL_states()

    def hidden_activity(self):
        """
        QNext's input: the current activity of qnext_input_population, detached so that no gradient flows from QNext
        into the Q network.
        """
        return self.network.populations[self.qnext_input_population].activity.detach().clone()

    def predict_next_value(self, no_grad=False):
        """
        QNext(h_t) for the current step of the Q network.

        Returns
        -------
        torch.Tensor of shape (1,)
        """
        return self.qnext_model.step(self.hidden_activity(), no_grad=no_grad)

    def update_qnext(self, qnext_output, qnext_target, step=True):
        """
        Update QNext from its predictions of one step or a sequence of steps. With step=False, only compute the updates.
        """
        qnext_target = qnext_target.reshape(qnext_output.shape).to(qnext_output.dtype)
        self.qnext_loss_history.append(torch.mean((qnext_target - qnext_output.detach()) ** 2).item())
        if step:
            apply_learning_rules(self.qnext_network, qnext_output, qnext_target)
        else:
            compute_learning_rule_updates(self.qnext_network, qnext_output, qnext_target)

    def run_episode(self, env):
        """
        Run one episode with fixed weights, then update both networks on all of its steps.

        Returns
        -------
        tuple of (float, int)
            Return and length of the episode.
        """
        device = self.network.device
        self.model.reset()
        obs, info = env.reset()
        q = self.model.step(obs_to_tensor(obs, device))
        q_values, actions, td_targets, qnext_values, qnext_targets = [], [], [], [], []
        episode_return = 0.
        done = False
        while not done:
            action = self.select_action(q)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = obs_to_tensor(next_obs, device)
            done = terminated or truncated
            # QNext reads h_t before the Q network steps on to s_{t+1}
            qnext = self.predict_next_value()
            q_values.append(q)
            actions.append(action)
            qnext_values.append(qnext)
            episode_return += float(reward)
            if terminated:
                td_targets.append(float(reward))
                qnext_targets.append(0.)
            else:
                td_targets.append(float(reward) + self.gamma * qnext.detach().item())
                # max Q(s_{t+1}) is QNext's target; when the episode goes on, it is also the next step's Q
                next_q = self.model.peek(next_obs) if truncated else self.model.step(next_obs)
                qnext_targets.append(next_q.detach().max().item())
                q = next_q

        self.update(torch.stack(q_values), torch.tensor(actions, device=device),
                    torch.tensor(td_targets, device=device))
        self.update_qnext(torch.stack(qnext_values), torch.tensor(qnext_targets, device=self.qnext_network.device))
        self.model.detach()
        return episode_return, len(actions)


class OnlineDPPQL(DPPQL):
    """
    DPPQL with an update after every environment step. At step t, after the action:
    1. QNext's prediction from the previous step, QNext(h_{t-1}), is updated toward max_a Q(s_t, a).
    2. QNext(h_t) is computed with the updated weights.
    3. Q(s_t, a_t) is updated toward r_t + gamma * QNext(h_t), or r_t at termination; at termination QNext(h_t) is
       also updated toward 0.

    Updates are applied at once or, with accumulate_updates, summed over the episode and applied at its end. With
    persistent_state, population states carry across steps, but gradients stop at step boundaries, so this suits
    rules with eligibility traces (e.g. Eprop) in the Q network. accumulate_updates only defers rules that apply their
    update in step(); Backprop updates the weights inside backward, at every step.
    """
    def __init__(self, network, qnext_network, qnext_input_population='H1E', gamma=0.99, epsilon=1.,
                 epsilon_decay=0.99, epsilon_min=0.05, epsilon_schedule='decay', persistent_state=False,
                 accumulate_updates=False, seed=None, importance_method='fisher', importance_num_episodes=100,
                 trace_clock='step'):
        """
        Parameters
        ----------
        See DPPQL, and:
        accumulate_updates : bool, default False
            Whether to apply the summed updates of an episode at its end instead of after every step.
        """
        super().__init__(network, qnext_network, qnext_input_population=qnext_input_population, gamma=gamma,
                         epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
                         epsilon_schedule=epsilon_schedule, persistent_state=persistent_state, seed=seed,
                         importance_method=importance_method, importance_num_episodes=importance_num_episodes,
                         trace_clock=trace_clock)
        self.accumulate_updates = accumulate_updates

    def run_episode(self, env):
        device = self.network.device
        qnext_device = self.qnext_network.device
        step = not self.accumulate_updates
        self.model.reset()
        obs, info = env.reset()
        q = self.model.step(obs_to_tensor(obs, device))
        prev_qnext = None
        episode_return = 0.
        episode_length = 0
        done = False
        while not done:
            action = self.select_action(q)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = obs_to_tensor(next_obs, device)
            if prev_qnext is not None:
                self.update_qnext(prev_qnext, torch.tensor(q.detach().max().item(), device=qnext_device), step=step)
            qnext = self.predict_next_value()
            if terminated:
                td_target = float(reward)
            else:
                td_target = float(reward) + self.gamma * qnext.detach().item()
            self.update(q, torch.tensor(action, device=device), torch.tensor(td_target, device=device), step=step)
            if terminated:
                self.update_qnext(qnext, torch.tensor(0., device=qnext_device), step=step)
            elif truncated:
                self.update_qnext(qnext, torch.tensor(self.model.peek(next_obs).max().item(), device=qnext_device),
                                  step=step)
            self.model.detach()

            episode_return += float(reward)
            episode_length += 1
            done = terminated or truncated
            prev_qnext = qnext
            if not done:
                q = self.model.step(next_obs)
        if self.accumulate_updates:
            step_learning_rules(self.network)
            step_learning_rules(self.qnext_network)
        return episode_return, episode_length
