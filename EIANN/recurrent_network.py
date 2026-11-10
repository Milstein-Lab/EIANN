import torch


class RecurrentNetwork(object):
    """
    Runs an EIANN Network over a sequence of samples (e.g. the steps of an episode). With persistent_state, the state
    and activity of every population carry over from one step to the next, so the network can hold information (such
    as a cue) across steps; otherwise every step starts from a fresh state.

    Where the computation graph is cut between steps (detach) is left to the caller.

    Learning rules with traces (e.g. Eprop, BTSP_Continuous) tick them with learning_rule.forward(). trace_clock sets
    how often:
    - 'step': once per step, after network.forward has run all forward_steps integration substeps.
    - 'substep': once per integration substep. Each step then runs network.forward forward_steps times with one substep
      each, which computes the same dynamics: with persistent state, reinit leaves states alone, and prev_activity is
      set at the start of every substep. Rules with a set_ticks_per_step method (BTSP_Continuous and its variants) are
      told forward_steps, so their time constants stay in units of steps.
    trace_clock=None runs as 'step' but leaves network.trace_clock None, so rules that are only exact with the substep
    clock (Eprop with forward_steps > 1) can refuse to run unless 'step' was chosen explicitly.
    """
    state_attributes = ('state', 'activity', 'prev_activity', 'forward_steps_activity')
    TRACE_CLOCKS = ('step', 'substep')

    def __init__(self, network, persistent_state=False, trace_clock=None):
        """
        Parameters
        ----------
        network : Network
        persistent_state : bool, default False
            Whether to carry population states across steps of a sequence.
        trace_clock : str, optional
            'step' or 'substep': whether learning rules tick their traces once per step or once per integration
            substep (see the class docstring). None (the default) runs as 'step'. Also set as network.trace_clock
            (None stays None), so rules can check it; the last RecurrentNetwork built on a network sets its clock.
        """
        if trace_clock is not None and trace_clock not in self.TRACE_CLOCKS:
            raise ValueError(f'RecurrentNetwork: trace_clock must be one of {self.TRACE_CLOCKS}, not {trace_clock!r}')
        self.network = network
        self.persistent_state = persistent_state
        self.trace_clock = 'step' if trace_clock is None else trace_clock
        self.new_sequence = True
        network.trace_clock = trace_clock
        ticks_per_step = network.forward_steps if self.trace_clock == 'substep' else 1
        for projection in network.projections.values():
            if hasattr(projection.learning_rule, 'set_ticks_per_step'):
                projection.learning_rule.set_ticks_per_step(ticks_per_step)

    def reset(self):
        """
        Start a new sequence: the next step starts from a fresh state.
        """
        self.new_sequence = True

    def step(self, sample, no_grad=False, store_history=False, store_dynamics=False):
        """
        Forward pass on one sample, continuing from the previous step's state when persistent_state. Learning rules
        that keep eligibility traces (e.g. Eprop) reset them when the step starts from a fresh state; rules with traces
        that last a whole sequence (reset_traces, e.g. BTSP_Continuous) reset them at the start of a sequence. Both
        update them (learning_rule.forward()) after the step, or after every integration substep with
        trace_clock='substep'.

        Returns
        -------
        torch.Tensor
            Output population activity.
        """
        persistent_state = self.persistent_state and not self.new_sequence
        for projection in self.network.projections.values():
            # traces that follow population states (Eprop) reset with them; traces that last a whole sequence whether
            # or not population states persist (e.g. BTSP_Continuous) reset only at its start
            if not persistent_state and hasattr(projection.learning_rule, 'reset_eligibility_trace'):
                projection.learning_rule.reset_eligibility_trace()
            if self.new_sequence and hasattr(projection.learning_rule, 'reset_traces'):
                projection.learning_rule.reset_traces()
        if self.trace_clock == 'step':
            output = self.network.forward(sample, no_grad=no_grad, persistent_state=persistent_state,
                                          store_history=store_history, store_dynamics=store_dynamics)
            self.tick_traces()
        else:
            output = self.substep_forward(sample, no_grad=no_grad, persistent_state=persistent_state,
                                          store_history=store_history, store_dynamics=store_dynamics)
        self.new_sequence = False
        return output

    def tick_traces(self):
        for projection in self.network.projections.values():
            if hasattr(projection.learning_rule, 'forward'):
                projection.learning_rule.forward()

    def substep_forward(self, sample, no_grad=False, persistent_state=False, store_history=False,
                        store_dynamics=False):
        """
        network.forward on one sample as forward_steps calls of one integration substep each, ticking learning rule
        traces after every substep. Gradients are tracked for the last backward_steps substeps, as in network.forward.

        Returns
        -------
        torch.Tensor
            Output population activity.
        """
        network = self.network
        forward_steps, backward_steps = network.forward_steps, network.backward_steps
        steps_activity = {name: [] for name in network.populations}
        network.forward_steps = network.backward_steps = 1
        try:
            for t in range(forward_steps):
                output = network.forward(sample, no_grad=no_grad or t < forward_steps - backward_steps,
                                         persistent_state=persistent_state or t > 0,
                                         store_dynamics=store_dynamics)
                if store_dynamics:
                    for name, population in network.populations.items():
                        steps_activity[name].extend(population.forward_steps_activity)
                self.tick_traces()
        finally:
            network.forward_steps, network.backward_steps = forward_steps, backward_steps
        if store_dynamics:
            for name, population in network.populations.items():
                population.forward_steps_activity = steps_activity[name]
        if store_history:
            for population in network.populations.values():
                if store_dynamics:
                    population.append_attribute_history('activity', population.forward_steps_activity)
                else:
                    population.append_attribute_history('activity', population.activity.detach().clone())
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
