from .base_classes import LearningRule, ContinualLearningMixin
import torch
import math


class DTP_CL(ContinualLearningMixin, LearningRule):
    def __init__(self, projection, max_pop_fraction=1., stochastic=False, learning_rate=None, relu_gate=False,
                 task_incremental=False, task_classes=None):
        """
        Initialize DTP_CL learning rule: the continual-learning version of BP_like_2E.

        Output units are nudged to target. Hidden dendrites locally compute an error as the difference between
        excitation and inhibition. Weight updates are proportional to local error and (forward) presynaptic firing rate.
        If unit selection is stochastic, hidden units are selected for a weight update with a probability proportional
        to dendritic state by sampling a Bernoulli distribution. Otherwise, units are sorted by dendritic state. A fixed
        maximum fraction of hidden units are updated at each train step. Hidden units are "nudged" by dendritic state
        when selected for a weight update. Nudges to somatic state are applied instantaneously, rather than being
        subject to slow equilibration. Nudged activity is only passed top-down, but not laterally, with no
        equilibration.

        Parameters
        ----------
        projection : EIANN.Projection object (inherits from torch.nn.Linear)
            The neural projection/connection.
        max_pop_fraction : float, default=1.0
            Maximum fraction of population to update, must be in [0, 1].
        stochastic : bool, default=False
            Whether to use stochastic unit selection.
        learning_rate : float, optional
            Learning rate for weight updates.
        relu_gate : bool, default=False
            Whether to apply ReLU gating.
        task_incremental : bool, default=False
            Whether the output error is restricted to the output units of the current task's classes (see
            ContinualLearningMixin).
        task_classes : list of lists of int, optional
            The classes in each task; required when task_incremental is True.
        """
        super().__init__(projection, learning_rate)
        self.max_pop_fraction = max_pop_fraction
        self.stochastic = stochastic
        self.relu_gate = relu_gate
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
        self.init_continual_learning(task_incremental, task_classes)

    def compute_delta_weight(self):
        """
        Unscaled weight update of the rule: postsynaptic plateau times (forward) presynaptic firing rate.
        :return: tensor
        """
        if self.projection.direction in ['forward', 'F']:
            return torch.outer(self.projection.post.plateau,
                               torch.clamp(self.projection.pre.forward_activity, min=0, max=1))
        elif self.projection.direction in ['recurrent', 'R']:
            return torch.outer(self.projection.post.plateau,
                               torch.clamp(self.projection.pre.forward_prev_activity, min=0, max=1))

    def step(self):
        self.projection.weight.data += self.learning_rate * self.compute_delta_weight()

    @classmethod
    def backward_nudge_activity(cls, layer, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive a nudge.

        Parameters
        ----------
        layer : EIANN.Layer object
            The network layer to update.
        store_dynamics : bool, default=False
            Whether to store activity dynamics during the backward pass.
        """
        for post_pop in layer:
            if post_pop.backward_projections or post_pop is post_pop.network.output_pop:
                if hasattr(post_pop, 'dend_to_soma'):
                    post_pop.prev_activity = post_pop.activity
                    post_pop.activity = post_pop.activation(post_pop.state + post_pop.dend_to_soma)
                if store_dynamics:
                    post_pop.backward_steps_activity.append(post_pop.activity.detach().clone())

    @classmethod
    def backward_update_layer_dendritic_state(cls, layer):
        """
        Update dendritic state for all populations that receive projections targeting the dendritic compartment.

        Parameters
        ----------
        layer : object
            The network layer to update.
        """
        for post_pop in layer:
            if post_pop.backward_projections:
                # update dendritic state
                init_dend_state = False
                for projection in post_pop:
                    pre_pop = projection.pre
                    if projection.compartment == 'dend':
                        if not init_dend_state:
                            post_pop.dendritic_state = torch.zeros(post_pop.size, device=post_pop.network.device)
                            init_dend_state = True
                        if projection.direction in ['forward', 'F']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.activity)
                        elif projection.direction in ['recurrent', 'R']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.prev_activity)
                if init_dend_state:
                    post_pop.dendritic_state.clamp_(min=-1, max=1)

    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        Integrate top-down inputs and update dendritic state variables.

        Parameters
        ----------
        network : EIANN.Network object (inherits from torch.nn.Module)
            The neural network.
        output : torch.Tensor
            Output of the network.forward() method
        target : torch.Tensor
            Target values.
        store_history : bool, default=False
            Whether to store computation history.
        store_dynamics : bool, default=False
            Whether to store activity dynamics.
        """
        # restrict the output error to the current task (unchanged unless task_incremental)
        for projection in network.projections.values():
            if projection.learning_rule.__class__ == cls:
                target = projection.learning_rule.task_target(network, target)
                break

        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_pop = network.output_pop

        # Initialize populations that are updated during the backward phase
        for layer in reversed_layers:
            for pop in layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.backward_steps_activity = []
                # Initialize dendritic state variables
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        pop.plateau = torch.zeros(pop.size, device=network.device)
                        pop.dend_to_soma = torch.zeros(pop.size, device=network.device)
                        break

        for layer in reversed_layers:
            # Update dendritic state variables
            cls.backward_update_layer_dendritic_state(layer)
            for pop in layer:
                for projection in pop:
                    # Compute plateau events and nudge somatic state
                    if projection.learning_rule.__class__ == cls:
                        if pop is output_pop:
                            local_loss = torch.clamp(target - output_pop.activity, min=-1, max=1)
                            output_pop.dendritic_state = local_loss.detach().clone()
                            if projection.learning_rule.relu_gate:
                                local_loss[output_pop.forward_activity == 0.] = 0
                            output_pop.plateau[:] = local_loss.detach().clone()
                            output_pop.dend_to_soma[:] = local_loss.detach().clone()
                        else:
                            max_units = math.ceil(projection.learning_rule.max_pop_fraction * pop.size)
                            local_loss = pop.dendritic_state.detach().clone()
                            if projection.learning_rule.relu_gate:
                                local_loss[pop.forward_activity == 0.] = 0

                            pos_avail_indexes = (local_loss > 0.).nonzero().squeeze(1)
                            if projection.learning_rule.stochastic:
                                pos_candidate_rel_indexes = (
                                    torch.bernoulli(local_loss[pos_avail_indexes])).nonzero().squeeze(1)
                            else:
                                sorted, pos_candidate_rel_indexes = torch.sort(local_loss[pos_avail_indexes],
                                                                               descending=True, stable=True)
                            pos_event_indexes = pos_avail_indexes[pos_candidate_rel_indexes][:max_units]

                            neg_avail_indexes = (local_loss < 0.).nonzero().squeeze(1)
                            if projection.learning_rule.stochastic:
                                neg_candidate_rel_indexes = (
                                    torch.bernoulli(-local_loss[neg_avail_indexes])).nonzero().squeeze(1)
                            else:
                                sorted, neg_candidate_rel_indexes = torch.sort(local_loss[neg_avail_indexes],
                                                                               descending=True, stable=True)
                            neg_event_indexes = neg_avail_indexes[neg_candidate_rel_indexes][-max_units:]

                            pop.plateau[pos_event_indexes] = local_loss[pos_event_indexes]
                            pop.plateau[neg_event_indexes] = local_loss[neg_event_indexes]
                            pop.dend_to_soma[pos_event_indexes] = local_loss[pos_event_indexes]
                            pop.dend_to_soma[neg_event_indexes] = local_loss[neg_event_indexes]
                        break
            # Update activities
            cls.backward_nudge_activity(layer, store_dynamics=store_dynamics)

        if store_history:
            for layer in network:
                for pop in layer:
                    for projection in pop:
                        if projection.learning_rule.__class__ == cls:
                            pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                            pop.append_attribute_history('backward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                            break
                    if pop.backward_projections or pop is output_pop:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class DTP_SI(DTP_CL):
    def __init__(self, projection, max_pop_fraction=1., stochastic=False, si_lambda=1., si_xi=1e-3, learning_rate=None,
                 relu_gate=False, task_incremental=False, task_classes=None):
        """
        DTP_CL with a Synaptic Intelligence regularizer (Zenke et al., 2017, arXiv:1703.04200), adapted to a rule
        without a loss gradient. The rule's own unscaled update, delta_weight = outer(plateau, pre forward rate), takes
        the place of the negative gradient of the unregularized task loss (for the output projection it is exactly the
        negative gradient of 0.5 * (target - output) ** 2, up to clamping).
        During each task, the per-weight importance is accumulated online as the path integral
            small_omega += delta_weight * actual_delta_weight,
        where actual_delta_weight is the actual weight change of each train step (after the penalty and weight
        clamping). At the end of each task (update_CL_states), the cumulative importance
        omega = relu(omega + small_omega / (total task weight change ** 2 + si_xi)), the anchor weight is set to the
        current weight, and small_omega is reset. During subsequent tasks, each projection is updated with
            learning_rate * (delta_weight - 2 * si_lambda * omega * (weight - anchor_weight)),
        i.e. the negative gradient of the penalty si_lambda * (omega * (weight - anchor_weight) ** 2).sum() is added
        to the rule's update.
        See DTP_CL for the other parameters.
        :param si_lambda: float; strength of the penalty
        :param si_xi: float; damping term of the importance normalization
        """
        super().__init__(projection, max_pop_fraction=max_pop_fraction, stochastic=stochastic,
                         learning_rate=learning_rate, relu_gate=relu_gate, task_incremental=task_incremental,
                         task_classes=task_classes)
        self.si_lambda = si_lambda
        self.si_xi = si_xi
        self.omega = None
        self.anchor_weight = None
        self.small_omega = torch.zeros_like(projection.weight.data)
        # Weights are initialized by the network after the projection is built, so this is set on the first step
        self.task_start_weight = None
        self.unreg_delta_weight = None
        self.prev_weight = None
    
    def si_penalty_update(self):
        """
        Negative gradient of the penalty si_lambda * (omega * (weight - anchor_weight) ** 2).sum().
        :return: tensor or float
        """
        if self.omega is None:
            return 0.
        return -2. * self.si_lambda * self.omega * (self.projection.weight.data - self.anchor_weight)
    
    def step(self):
        if self.task_start_weight is None:
            self.task_start_weight = self.projection.weight.detach().clone()
        delta_weight = self.compute_delta_weight()
        self.unreg_delta_weight = delta_weight.detach().clone()
        self.prev_weight = self.projection.weight.detach().clone()
        self.projection.weight.data += self.learning_rate * (delta_weight + self.si_penalty_update())
    
    def update(self):
        # Called after step() and constrain_weights_and_biases(), so the weight change includes weight clamping.
        # unreg_delta_weight is the negative of the gradient, so the path integral -grad * delta_weight is a sum here
        if self.prev_weight is not None:
            actual_delta_weight = self.projection.weight.detach() - self.prev_weight
            self.small_omega += self.unreg_delta_weight * actual_delta_weight
            self.prev_weight = None
    
    def update_CL_states(self):
        weight = self.projection.weight.detach().clone()
        if self.task_start_weight is not None:
            omega = self.small_omega / ((weight - self.task_start_weight) ** 2 + self.si_xi)
            if self.omega is not None:
                omega = self.omega + omega
            # Clamp at zero, as in the reference implementation (ganguli-lab/pathint), so the penalty never pushes
            # weights away from their anchor
            self.omega = torch.relu(omega)
        self.anchor_weight = weight
        self.task_start_weight = weight.clone()
        self.small_omega = torch.zeros_like(weight)
        super().update_CL_states()
