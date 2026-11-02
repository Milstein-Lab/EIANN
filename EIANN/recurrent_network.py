import torch


class RecurrentNetwork(object):
    """
    Runs an EIANN Network over a sequence of samples (e.g. the steps of an episode). With persistent_state, the state
    and activity of every population carry over from one step to the next, so the network can hold information (such
    as a cue) across steps; otherwise every step starts from a fresh state.

    Where the computation graph is cut between steps (detach) is left to the caller.
    """
    state_attributes = ('state', 'activity', 'prev_activity', 'forward_steps_activity')

    def __init__(self, network, persistent_state=False):
        """
        Parameters
        ----------
        network : Network
        persistent_state : bool, default False
            Whether to carry population states across steps of a sequence.
        """
        self.network = network
        self.persistent_state = persistent_state
        self.new_sequence = True

    def reset(self):
        """
        Start a new sequence: the next step starts from a fresh state.
        """
        self.new_sequence = True

    def step(self, sample, no_grad=False, store_history=False, store_dynamics=False):
        """
        Forward pass on one sample, continuing from the previous step's state when persistent_state. Learning rules
        that keep eligibility traces (e.g. Eprop) reset them when the step starts from a fresh state, and update them
        (learning_rule.forward()) after the step.

        Returns
        -------
        torch.Tensor
            Output population activity.
        """
        persistent_state = self.persistent_state and not self.new_sequence
        if not persistent_state:
            for projection in self.network.projections.values():
                if hasattr(projection.learning_rule, 'reset_eligibility_trace'):
                    projection.learning_rule.reset_eligibility_trace()
        output = self.network.forward(sample, no_grad=no_grad, persistent_state=persistent_state,
                                      store_history=store_history, store_dynamics=store_dynamics)
        for projection in self.network.projections.values():
            if hasattr(projection.learning_rule, 'forward'):
                projection.learning_rule.forward()
        self.new_sequence = False
        return output

    def peek(self, sample):
        """
        No-grad forward pass on one sample as the next step, without changing the state or activity of any population
        (so learning rules still see the current step's activities).

        Returns
        -------
        torch.Tensor
            Output population activity.
        """
        saved = self.save_state()
        persistent_state = self.persistent_state and not self.new_sequence
        output = self.network.forward(sample, no_grad=True, persistent_state=persistent_state).detach().clone()
        self.restore_state(saved)
        return output

    def run(self, samples, no_grad=False):
        """
        Step through a sequence of samples, continuing from the current state.

        Returns
        -------
        torch.Tensor
            Stacked outputs, shape (len(samples), output size).
        """
        return torch.stack([self.step(sample, no_grad=no_grad) for sample in samples])

    def detach(self):
        """
        Cut the computation graph at the current step, so gradients do not flow into earlier steps.
        """
        for population in self.network.populations.values():
            for attr_name in ('state', 'activity', 'prev_activity'):
                value = getattr(population, attr_name, None)
                if isinstance(value, torch.Tensor):
                    setattr(population, attr_name, value.detach())

    def save_state(self):
        saved = {}
        for pop_name, population in self.network.populations.items():
            saved[pop_name] = {attr_name: getattr(population, attr_name) for attr_name in self.state_attributes
                               if hasattr(population, attr_name)}
        return saved

    def restore_state(self, saved):
        for pop_name, population in self.network.populations.items():
            for attr_name, value in saved[pop_name].items():
                setattr(population, attr_name, value)
