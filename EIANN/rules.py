import torch
import torch.nn as nn
from .utils import get_scaled_rectified_sigmoid
import time
import math


class LearningRule(object):
    def __init__(self, projection, learning_rate=None):
        self.projection = projection
        if learning_rate is None:
            learning_rate = self.projection.post.network.learning_rate
        self.learning_rate = learning_rate

    def step(self):
        pass

    @classmethod
    def backward(cls, network, output, target, store_history=False):
        pass


class BiasLearningRule(object):
    def __init__(self, population, learning_rate=None):
        self.population = population
        if learning_rate is None:
            learning_rate = self.population.network.learning_rate
        self.learning_rate = learning_rate

    def step(self):
        pass

    @classmethod
    def backward(cls, network, output, target, store_history=False):
        pass


class Backprop(LearningRule):
    @classmethod
    def backward(cls, network, output, target, store_history=False):
        loss = network.criterion(output, target)
        network.optimizer.zero_grad()
        loss.backward()
        network.optimizer.step()


class BackpropBias(BiasLearningRule):
    backward = Backprop.backward


class Oja(LearningRule):
    def step(self):
        delta_weight = torch.outer(self.projection.post.activity, self.projection.pre.activity) - \
                       self.projection.weight * (self.projection.post.activity ** 2).unsqueeze(1)
        self.projection.weight.data += self.learning_rate * delta_weight


class BCM(LearningRule):
    def __init__(self, projection, theta_init, theta_tau, k, learning_rate=None):
        """

        :param projection:
        :param theta_init: float
        :param theta_tau: float
        :param k: float
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.theta_tau = theta_tau
        self.k = k
        self.projection.post.theta = torch.ones(projection.post.size) * theta_init

    def step(self):
        delta_weight = torch.outer(self.projection.post.activity, self.projection.pre.activity) * \
                       (self.projection.post.activity - self.projection.post.prev_theta).unsqueeze(1)
        self.projection.weight.data += self.learning_rate * delta_weight

    @classmethod
    def backward(cls, network, output, target, store_history=False):
        for i, layer in enumerate(network):
            if i > 0:
                for population in layer:
                    for projection in population:
                        if projection.learning_rule.__class__ == cls:
                            population.prev_theta = population.theta.clone()
                            delta_theta = (-population.theta + population.activity ** 2. /
                                           projection.learning_rule.k) / projection.learning_rule.theta_tau
                            population.theta += delta_theta
                            # only update theta once per population
                            break


class GjorgjievaHebb(LearningRule):
    def __init__(self, projection, sign=1, learning_rate=None):
        super().__init__(projection, learning_rate)
        self.sign = sign

    def step(self):
        delta_weight = torch.outer(self.projection.post.activity, self.projection.pre.activity)
        if self.sign > 0:
            self.projection.weight.data += self.learning_rate * delta_weight
        else:
            self.projection.weight.data -= self.learning_rate * delta_weight


class BTSP(LearningRule):
    def __init__(self, projection, pos_loss_th=2.440709E-01, neg_loss_th=-4.592181E-01, neg_loss_ET_discount=0.25,
                 dep_ratio=1., dep_th=0.01, dep_width=0.01, learning_rate=None):
        """

        :param projection: :class:'nn.Linear'
        :param pos_loss_th: float
        :param neg_loss_th: float
        :param neg_loss_ET_discount: float
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.pos_loss_th = pos_loss_th
        self.neg_loss_th = neg_loss_th
        self.neg_loss_ET_discount = neg_loss_ET_discount
        self.dep_ratio = dep_ratio
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]

    def step(self):
        plateau = self.projection.post.plateau
        discount = plateau.detach().clone()
        discount[plateau > 0] = 1.
        discount[plateau < 0] = self.neg_loss_ET_discount
        IS = torch.abs(plateau).unsqueeze(1)
        ET = torch.outer(discount, self.projection.pre.activity)
        delta_weight = IS * ((self.w_max - self.projection.weight) * ET - \
                       self.projection.weight * self.dep_ratio * self.q_dep(ET))
        self.projection.weight.data += self.learning_rate * delta_weight

    @classmethod
    def backward_update_layer_activity(cls, layer):
        """
        Update activity and dendritic state for populations in the layer that receive backward projections.
        :param layer:
        """
        pop_set = set()
        for post_pop in layer:
            pop_set.add(post_pop)
            for projection in post_pop.backward_projections:
                pop_set.add(projection.pre)

        for pop in pop_set:
            pop.prev_activity = pop.activity

        # update activity for populations that receive backward projections to the soma
        for pop in layer:
            if pop.backward_projections:
                # pop.forward_soma_state already contains bias and inputs updated during the forward phase
                delta_state = -pop.state + pop.forward_soma_state
                if hasattr(pop, 'dend_to_soma'):
                    delta_state = delta_state + pop.dend_to_soma
                for projection in pop.backward_projections:
                    pre_pop = projection.pre
                    if projection.compartment in [None, 'soma']:
                        if projection.direction in ['forward', 'F']:
                            delta_state = delta_state + projection(pre_pop.activity)
                        elif projection.direction in ['recurrent', 'R']:
                            delta_state = delta_state + projection(pre_pop.prev_activity)
                pop.state = pop.state + delta_state / pop.tau
                pop.activity = pop.activation(pop.state)

        # update dendritic state for populations that receive backward projections to the dendrite
        for pop in layer:
            init_dend_state = False
            for projection in pop.backward_projections:
                pre_pop = projection.pre
                if projection.compartment == 'dend':
                    if not init_dend_state:
                        pop.dendritic_state = torch.zeros(pop.size, device=pop.network.device)
                        init_dend_state = True
                    if projection.direction in ['forward', 'F']:
                        pop.dendritic_state = pop.dendritic_state + projection(pre_pop.activity)
                    elif projection.direction in ['recurrent', 'R']:
                        pop.dendritic_state = pop.dendritic_state + projection(pre_pop.prev_activity)
            if init_dend_state:
                pop.dendritic_state = torch.clamp(pop.dendritic_state, min=-1, max=1)


    @classmethod
    def backward(cls, network, output, target, store_history=False):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        """
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_layer = reversed_layers.pop(0)
        output_pop = network.output_pop
        for projection in output_pop:
            if projection.learning_rule.__class__ == cls:
                output_pop.dendritic_state = torch.clamp(target - output, min=-1, max=1)
                output_pop.forward_soma_state = output_pop.state.detach().clone()
                output_pop.plateau = torch.zeros(output_pop.size, device=output_pop.network.device)
                output_pop.dend_to_soma = torch.zeros(output_pop.size, device=output_pop.network.device)
                pos_indexes = (output_pop.dendritic_state > projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)
                output_pop.plateau[pos_indexes] = output_pop.dendritic_state[pos_indexes]
                output_pop.dend_to_soma[pos_indexes] = output_pop.dendritic_state[pos_indexes]
                output_pop.activity[pos_indexes] = \
                    output_pop.activation(output_pop.forward_soma_state[pos_indexes] +
                                          output_pop.dend_to_soma[pos_indexes])
                neg_indexes = (output_pop.dendritic_state < projection.learning_rule.neg_loss_th).nonzero(as_tuple=True)
                output_pop.plateau[neg_indexes] = output_pop.dendritic_state[neg_indexes]
                if store_history:
                    output_pop.plateau_history_list.append(output_pop.plateau.detach().clone())
                break
        if store_history:
            output_pop.backward_activity_history_list.append([output_pop.activity.detach().clone()])

        for layer in reversed_layers:
            # initialize cells that receive backward projections
            for pop in layer:
                if pop.backward_projections:
                    pop.forward_soma_state = pop.state.detach().clone()
                    pop.backward_steps_activity = []

            # equilibrate activites and dendritic state variables
            for t in range(network.forward_steps):
                cls.backward_update_layer_activity(layer)
                for pop in layer:
                    if pop.backward_projections:
                        for projection in pop.backward_projections:
                            if projection.learning_rule.__class__ == cls:
                                pop.plateau = torch.zeros(pop.size, device=pop.network.device)
                                pop.dend_to_soma = torch.zeros(pop.size, device=pop.network.device)
                                pos_indexes = (pop.dendritic_state > projection.learning_rule.pos_loss_th).nonzero(
                                    as_tuple=True)
                                pop.plateau[pos_indexes] = pop.dendritic_state[pos_indexes]
                                pop.dend_to_soma[pos_indexes] = pop.dendritic_state[pos_indexes]
                                break
                        if store_history:
                            pop.backward_steps_activity.append(pop.activity.detach().clone())
            for pop in layer:
                if pop.backward_projections:
                    for projection in pop.backward_projections:
                        if projection.learning_rule.__class__ == cls:
                            neg_indexes = (pop.dendritic_state < projection.learning_rule.neg_loss_th).nonzero(
                                as_tuple=True)
                            pop.plateau[neg_indexes] = pop.dendritic_state[neg_indexes]
                            if store_history:
                                pop.plateau_history_list.append(pop.plateau.detach().clone())
                            break
                    if store_history:
                        pop.backward_activity_history_list.append(pop.backward_steps_activity)


class BTSP_2(LearningRule):
    """
    In variant 2, populations with projections to the soma with update_phase in ['F', 'forward', 'A', 'all'] are first
    equilibrated during the forward pass. In the backward phase, each layer is updated separately and sequentially.
    First, projections to the dendrite with update_phase in ['B', 'backward', 'A', 'all'] are updated. Then, plateau
    events are computed. In variant 2, plateaus can only occur in a specified maximum fraction of units in each layer.
    Plateaus add a nudge to the somatic state, and then all somatic activities in the layer are re-equilibrated.
    """
    def __init__(self, projection, pos_loss_th=2.440709E-01, neg_loss_th=-4.592181E-01, neg_loss_ET_discount=0.25,
                 dep_ratio=1., dep_th=0.01, dep_width=0.01, max_pop_fraction=0.05, learning_rate=None):
        """

        :param projection: :class:'nn.Linear'
        :param pos_loss_th: float
        :param neg_loss_th: float
        :param neg_loss_ET_discount: float
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param max_pop_fraction: float in [0, 1]
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.pos_loss_th = pos_loss_th
        self.neg_loss_th = neg_loss_th
        self.neg_loss_ET_discount = neg_loss_ET_discount
        self.dep_ratio = dep_ratio
        self.max_pop_fraction = max_pop_fraction
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]

    def step(self):
        plateau = self.projection.post.plateau
        discount = plateau.detach().clone()
        discount[plateau > 0] = 1.
        discount[plateau < 0] = self.neg_loss_ET_discount
        IS = torch.abs(plateau).unsqueeze(1)
        ET = torch.outer(discount, self.projection.pre.activity)
        delta_weight = IS * ((self.w_max - self.projection.weight) * ET - \
                       self.projection.weight * self.dep_ratio * self.q_dep(ET))
        self.projection.weight.data += self.learning_rate * delta_weight

    @classmethod
    def backward_update_layer_dendritic_state(cls, layer):
        """
        Update dendritic state for populations in the layer that receive backward projections to dendrites.
        :param layer:
        """
        for pop in layer:
            init_dend_state = False
            for projection in pop.backward_projections:
                pre_pop = projection.pre
                if projection.compartment == 'dend':
                    if not init_dend_state:
                        pop.dendritic_state = torch.zeros(pop.size, device=pop.network.device)
                        init_dend_state = True
                    if projection.direction in ['forward', 'F']:
                        pop.dendritic_state = pop.dendritic_state + projection(pre_pop.activity)
                    elif projection.direction in ['recurrent', 'R']:
                        pop.dendritic_state = pop.dendritic_state + projection(pre_pop.prev_activity)
            if init_dend_state:
                pop.dendritic_state = torch.clamp(pop.dendritic_state, min=-1, max=1)

    @classmethod
    def backward_equilibrate_layer_activity(cls, layer, forward_steps, store_history=False):
        """
        Equilibrate somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer: :class:'Layer'
        :param forward_steps: int
        :param store_history: bool
        """
        for t in range(forward_steps):
            for post_pop in layer:
                post_pop.prev_activity = post_pop.activity
            for post_pop in layer:
                if post_pop.backward_projections:
                    # update somatic state and activity
                    delta_state = -post_pop.state + post_pop.bias
                    if hasattr(post_pop, 'dend_to_soma'):
                        delta_state = delta_state + post_pop.dend_to_soma
                    for projection in post_pop:
                        pre_pop = projection.pre
                        if projection.compartment in [None, 'soma']:
                            if projection.direction in ['forward', 'F']:
                                delta_state = delta_state + projection(pre_pop.activity)
                            elif projection.direction in ['recurrent', 'R']:
                                delta_state = delta_state + projection(pre_pop.prev_activity)
                    post_pop.state = post_pop.state + delta_state / post_pop.tau
                    post_pop.activity = post_pop.activation(post_pop.state)
                if store_history:
                    post_pop.backward_steps_activity.append(post_pop.activity.detach().clone())

    @classmethod
    def backward_update_layer_dendritic_state(cls, layer):
        """
        Equilibrate dendritic state for all populations that receive projections that target the dendritic
        compartment.
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
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)

    @classmethod
    def backward(cls, network, output, target, store_history=False):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        """
        for layer in network:
            for post_pop in layer:
                post_pop.backward_steps_activity = []
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_layer = reversed_layers.pop(0)
        output_pop = network.output_pop
        for projection in output_pop:
            if projection.learning_rule.__class__ == cls:
                output_pop.dendritic_state = torch.clamp(target - output, min=-1, max=1)
                output_pop.plateau = torch.zeros(output_pop.size, device=output_pop.network.device)
                output_pop.dend_to_soma = torch.zeros(output_pop.size, device=output_pop.network.device)
                pos_indexes = (output_pop.dendritic_state > projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)
                output_pop.plateau[pos_indexes] = output_pop.dendritic_state[pos_indexes]
                output_pop.dend_to_soma[pos_indexes] = output_pop.dendritic_state[pos_indexes]
                neg_indexes = (output_pop.dendritic_state < projection.learning_rule.neg_loss_th).nonzero(as_tuple=True)
                output_pop.plateau[neg_indexes] = output_pop.dendritic_state[neg_indexes]
                if store_history:
                    output_pop.plateau_history_list.append(output_pop.plateau.detach().clone())
                break
        cls.backward_equilibrate_layer_activity(output_layer, network.forward_steps, store_history)

        for layer in reversed_layers:
            # initialize cells that receive inputs to the dendritic compartment
            update_soma = False
            for post_pop in layer:
                for projection in post_pop:
                    if projection.update_phase in ['B', 'backward'] and projection.compartment in [None, 'soma']:
                        update_soma = True
            if update_soma:
                cls.backward_equilibrate_layer_activity(layer, network.forward_steps, store_history)
            cls.backward_update_layer_dendritic_state(layer)
            for post_pop in layer:
                for projection in post_pop.backward_projections:
                    if projection.learning_rule.__class__ == cls:
                        max_units = math.ceil(projection.learning_rule.max_pop_fraction * post_pop.size)
                        post_pop.plateau = torch.zeros(post_pop.size, device=post_pop.network.device)
                        post_pop.dend_to_soma = torch.zeros(post_pop.size, device=post_pop.network.device)

                        sorted, sorted_indexes = torch.sort(post_pop.dendritic_state, descending=True, stable=True)
                        pos_indexes = (post_pop.dendritic_state[sorted_indexes] >
                                       projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)[0][:max_units]
                        pos_plateau_indexes = sorted_indexes[pos_indexes]
                        post_pop.plateau[pos_plateau_indexes] = post_pop.dendritic_state[pos_plateau_indexes]
                        post_pop.dend_to_soma[pos_plateau_indexes] = post_pop.dendritic_state[pos_plateau_indexes]

                        neg_indexes = (post_pop.dendritic_state[sorted_indexes] <
                                       projection.learning_rule.neg_loss_th).nonzero(as_tuple=True)[0][-max_units:]
                        neg_plateau_indexes = sorted_indexes[neg_indexes]
                        post_pop.plateau[neg_plateau_indexes] = post_pop.dendritic_state[neg_plateau_indexes]
                        post_pop.dend_to_soma[neg_plateau_indexes] = post_pop.dendritic_state[neg_plateau_indexes]

                        if store_history:
                            post_pop.plateau_history_list.append(post_pop.plateau.detach().clone())
                        break
            cls.backward_equilibrate_layer_activity(layer, network.forward_steps, store_history)

        if store_history:
            for layer in network:
                for post_pop in layer:
                    post_pop.backward_activity_history_list.append(post_pop.backward_steps_activity)


class BTSP_3(LearningRule):
    def __init__(self, projection, pos_loss_th=2.440709E-01, neg_loss_th=-4.592181E-01, neg_loss_ET_discount=0.25,
                 dep_ratio=1., dep_th=0.01, dep_width=0.01, learning_rate=None):
        """
        Like the original BTSP class, this method includes both positive and negative modulatory events. Only positive
        modulatory events nudge the somatic activity. This variant is tolerant to the update_phase of projections
        being specified as 'A' or 'All', whereas the original class could only update projections either in the
        forward or backward phase, but not both.
        :param projection: :class:'nn.Linear'
        :param pos_loss_th: float
        :param neg_loss_th: float
        :param neg_loss_ET_discount: float
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.pos_loss_th = pos_loss_th
        self.neg_loss_th = neg_loss_th
        self.neg_loss_ET_discount = neg_loss_ET_discount
        self.dep_ratio = dep_ratio
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]

    def step(self):
        plateau = self.projection.post.plateau
        discount = plateau.detach().clone()
        discount[plateau > 0] = 1.
        discount[plateau < 0] = self.neg_loss_ET_discount
        IS = torch.abs(plateau).unsqueeze(1)
        ET = torch.outer(discount, self.projection.pre.activity)
        delta_weight = IS * ((self.w_max - self.projection.weight) * ET - \
                       self.projection.weight * self.dep_ratio * self.q_dep(ET))
        self.projection.weight.data += self.learning_rate * delta_weight

    @classmethod
    def backward_update_layer_activity(cls, layer, store_history=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_history: bool
        """
        for post_pop in layer:
            post_pop.prev_activity = post_pop.activity
        for post_pop in layer:
            if post_pop.backward_projections or post_pop is post_pop.network.output_pop:
                # update somatic state and activity
                delta_state = -post_pop.state + post_pop.bias
                if hasattr(post_pop, 'dend_to_soma'):
                    delta_state = delta_state + post_pop.dend_to_soma
                for projection in post_pop:
                    pre_pop = projection.pre
                    if projection.compartment in [None, 'soma']:
                        if projection.direction in ['forward', 'F']:
                            delta_state = delta_state + projection(pre_pop.activity)
                        elif projection.direction in ['recurrent', 'R']:
                            delta_state = delta_state + projection(pre_pop.prev_activity)
                post_pop.state = post_pop.state + delta_state / post_pop.tau
                post_pop.activity = post_pop.activation(post_pop.state)
                if store_history:
                    post_pop.backward_steps_activity.append(post_pop.activity.detach().clone())

    @classmethod
    def backward_update_layer_dendritic_state(cls, layer):
        """
        Update dendritic state for all populations that receive projections that target the dendritic
        compartment.
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
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)

    @classmethod
    def backward(cls, network, output, target, store_history=False):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        """
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_layer = reversed_layers.pop(0)
        output_pop = network.output_pop

        # initialize populations that are updated during the backward phase
        for layer in network:
            for pop in layer:
                if pop.backward_projections or pop is output_pop:
                    pop.backward_steps_activity = []

        # compute plateau events and nudge somatic state
        output_pop.dendritic_state = torch.clamp(target - output, min=-1, max=1)
        output_pop.plateau = torch.zeros(output_pop.size, device=output_pop.network.device)
        output_pop.dend_to_soma = torch.zeros(output_pop.size, device=output_pop.network.device)
        for projection in output_pop:
            if projection.learning_rule.__class__ == cls:
                pos_indexes = (output_pop.dendritic_state >
                               projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)
                output_pop.plateau[pos_indexes] = output_pop.dendritic_state[pos_indexes]
                output_pop.dend_to_soma[pos_indexes] = 1.  # output_pop.dendritic_state[pos_indexes]
                neg_indexes = (output_pop.dendritic_state <
                               projection.learning_rule.neg_loss_th).nonzero(as_tuple=True)
                output_pop.plateau[neg_indexes] = output_pop.dendritic_state[neg_indexes]
                # output_pop.dend_to_soma[neg_indexes] = output_pop.dendritic_state[neg_indexes]
                break
        # re-equilibrate soma states and activites
        for t in range(network.forward_steps):
            cls.backward_update_layer_activity(output_layer, store_history)
            # print(t, output_pop.dendritic_state, output_pop.activity)
        if store_history:
            output_pop.plateau_history_list.append(output_pop.plateau.detach().clone())

        for layer in reversed_layers:
            # equilibrate activites and dendritic state variables
            for t in range(network.forward_steps):
                cls.backward_update_layer_activity(layer, store_history)
                cls.backward_update_layer_dendritic_state(layer)
                for pop in layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            pop.plateau = torch.zeros(pop.size, device=pop.network.device)
                            pop.dend_to_soma = torch.zeros(pop.size, device=pop.network.device)
                            pos_indexes = (pop.dendritic_state > projection.learning_rule.pos_loss_th).nonzero(
                                as_tuple=True)
                            pop.plateau[pos_indexes] = pop.dendritic_state[pos_indexes]
                            pop.dend_to_soma[pos_indexes] = 1.  # pop.dendritic_state[pos_indexes]
                            break
            for pop in layer:
                for projection in pop.backward_projections:
                    if projection.learning_rule.__class__ == cls:
                        neg_indexes = (pop.dendritic_state < projection.learning_rule.neg_loss_th).nonzero(
                            as_tuple=True)
                        pop.plateau[neg_indexes] = pop.dendritic_state[neg_indexes]
                        if store_history:
                            pop.plateau_history_list.append(pop.plateau.detach().clone())
                        break

        for layer in network:
            for pop in layer:
                if pop.backward_projections or pop is output_pop:
                    pop.backward_activity_history_list.append(pop.backward_steps_activity)


class DendriticLossBias(BiasLearningRule):
    def step(self):
        # self.population.bias.data += self.learning_rate * self.population.dendritic_state
        self.population.bias.data += self.learning_rate * self.population.plateau


class DendriticLoss(LearningRule):
    """
    This is the original rule and is gated by plateaus.
    """
    def __init__(self, projection, sign=1, learning_rate=None):
        super().__init__(projection, learning_rate)
        self.sign = sign

    def step(self):
        self.projection.weight.data += self.sign * self.learning_rate * \
                                       torch.outer(self.projection.post.plateau, self.projection.pre.activity)


class DendriticLoss_2(LearningRule):
    """
    This variant 2 is gated by dendritic state. The original rule is gated by plateaus.
    """
    def __init__(self, projection, sign=1, learning_rate=None):
        super().__init__(projection, learning_rate)
        self.sign = sign

    def step(self):
        self.projection.weight.data += self.sign * self.learning_rate * \
                                       torch.outer(self.projection.post.dendritic_state, self.projection.pre.activity)


def clone_weight(projection, source=None, sign=1, scale=1, source2=None):
    """
    Force a projection to exactly copy the weights of another projection (or product of two projections).
    """
    if source is None:
        raise Exception('clone_weight: missing required weight_constraint_kwarg source')
    network = projection.post.network
    source_post_layer, source_post_pop, source_pre_layer, source_pre_pop = source.split('.')
    source_projection = \
        network.layers[source_post_layer].populations[source_post_pop].projections[source_pre_layer][source_pre_pop]
    source_weight_data = source_projection.weight.data.clone() * scale * sign
    if source2 is not None:
        source2_post_layer, source2_post_pop, source2_pre_layer, source2_pre_pop = source2.split('.')
        source2_projection = \
            network.layers[source2_post_layer].populations[source2_post_pop].projections[source2_pre_layer][
                source2_pre_pop]
        source2_weight_data = source2_projection.weight.data.clone()
        source_weight_data = source_weight_data * source2_weight_data
    projection.weight.data = source_weight_data


def normalize_weight(projection, scale, autapses=False, axis=1):
    if not autapses and projection.pre == projection.post:
        projection.weight.data.fill_diagonal_(0.)
    weight_sum = torch.sum(torch.abs(projection.weight.data), axis=axis).unsqueeze(1)
    valid_rows = torch.nonzero(weight_sum, as_tuple=True)[0]
    projection.weight.data[valid_rows,:] /= weight_sum[valid_rows,:]
    projection.weight.data *= scale
