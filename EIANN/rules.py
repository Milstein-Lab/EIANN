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

    def reinit(self):
        pass

    def update(self):
        pass

    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        pass


class BiasLearningRule(object):
    def __init__(self, population, learning_rate=None):
        self.population = population
        if learning_rate is None:
            learning_rate = self.population.network.learning_rate
        self.learning_rate = learning_rate

    def step(self):
        pass

    def reinit(self):
        pass

    def update(self):
        pass

    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        pass


class Backprop(LearningRule):
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        loss = network.criterion(output, target)
        network.optimizer.zero_grad()
        loss.backward()
        network.optimizer.step()


class BackpropBias(BiasLearningRule):
    backward = Backprop.backward


class Backprop_EWC(LearningRule):
    '''
    Implements the elastic weight consolidation (EWC) algorithm for continual learning.
    '''
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):

        ewc_penalty = 0
        for name, param in network.named_parameters():
            if name in network.phase1_params:
                _penalty = network.diag_fisher[name] * (param - network.phase1_params[name])**2
                ewc_penalty += _penalty.sum()

        # ewc_penalty = 0
        # for name, param in network.named_parameters():
        #     if name in network.phase1_params:
        #         _penalty = (param - network.phase1_params[name])**2
        #         ewc_penalty += _penalty.sum()

        network.optimizer.zero_grad()
        loss = 0.3*network.criterion(output, target) + network.ewc_lambda * ewc_penalty

        loss.backward()
        network.optimizer.step()






class Oja(LearningRule):
    def step(self):
        delta_weight = torch.outer(self.projection.post.activity, self.projection.pre.activity) - \
                       self.projection.weight * (self.projection.post.activity ** 2).unsqueeze(1)
        self.projection.weight.data += self.learning_rate * delta_weight


class BCM(LearningRule):
    def __init__(self, projection, theta_tau, k, sign=1, learning_rate=None):
        """

        :param projection:
        :param theta_tau: float
        :param k: float
        :param sign: int in {-1, 1}
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.theta_tau = theta_tau
        self.k = k
        self.projection.post.theta = torch.ones(projection.post.size, device=projection.post.network.device,
                                                requires_grad=False) * k
        self.sign = sign
        projection.post.__class__.theta_history = property(lambda self: self.get_attribute_history('theta'))

    def reinit(self):
        self.projection.post.BCM_theta_stored = False
        self.projection.post.BCM_theta_updated = False

    def update(self):
        if not self.projection.post.BCM_theta_updated:
            delta_theta = (-self.projection.post.theta + self.projection.post.activity ** 2. / self.k) / self.theta_tau
            self.projection.post.theta += delta_theta
            self.projection.post.BCM_theta_updated = True
            self.projection.post.BCM_theta_stored = False

    def step(self):
        if self.projection.direction in ['forward', 'F']:
            delta_weight = (torch.outer(self.projection.post.activity, self.projection.pre.activity) *
                            (self.projection.post.activity - self.projection.post.theta).unsqueeze(1))
        elif self.projection.direction in ['recurrent', 'R']:
            delta_weight = (torch.outer(self.projection.post.activity, self.projection.pre.prev_activity) *
                            (self.projection.post.activity - self.projection.post.theta).unsqueeze(1))
        if self.sign > 0:
            self.projection.weight.data += self.learning_rate * delta_weight
        else:
            self.projection.weight.data -= self.learning_rate * delta_weight
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        for i, layer in enumerate(network):
            if i > 0:
                for population in layer:
                    for projection in population:
                        if projection.learning_rule.__class__ == cls:
                            # only update theta once per population
                            population.BCM_theta_updated = False
                            if store_history and not population.BCM_theta_stored:
                                population.append_attribute_history('theta', population.theta.detach().clone())
                                population.BCM_theta_stored = True
                            break


class Supervised_BCM(LearningRule):
    def __init__(self, projection, theta_tau, k, sign=1, learning_rate=None):
        """
        Output units receive only positive nudges to target, subject to slow equilibration.
        :param projection:
        :param theta_tau: float
        :param k: float
        :param sign: int in {-1, 1}
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.theta_tau = theta_tau
        self.k = k
        self.projection.post.theta = torch.ones(projection.post.size, device=projection.post.network.device,
                                                requires_grad=False) * k
        self.sign = sign
        projection.post.__class__.theta_history = property(lambda self: self.get_attribute_history('theta'))
        projection.post.__class__.nudge_history = property(lambda self: self.get_attribute_history('nudge'))
        projection.post.__class__.backward_activity_history = \
            property(lambda self: self.get_attribute_history('backward_activity'))

    def reinit(self):
        self.projection.post.BCM_theta_stored = False
        self.projection.post.BCM_theta_updated = False

    def update(self):
        if not self.projection.post.BCM_theta_updated:
            delta_theta = (-self.projection.post.theta + self.projection.post.activity ** 2. / self.k) / self.theta_tau
            self.projection.post.theta += delta_theta
            self.projection.post.BCM_theta_updated = True
            self.projection.post.BCM_theta_stored = False

    def step(self):
        if self.projection.direction in ['forward', 'F']:
            delta_weight = torch.outer(self.projection.post.activity, self.projection.pre.activity) * \
                       (self.projection.post.activity - self.projection.post.theta).unsqueeze(1)
        elif self.projection.direction in ['recurrent', 'R']:
            delta_weight = torch.outer(self.projection.post.activity, self.projection.pre.prev_activity) * \
            (self.projection.post.activity - self.projection.post.theta).unsqueeze(1)
        if self.sign > 0:
            self.projection.weight.data += self.learning_rate * delta_weight
        else:
            self.projection.weight.data -= self.learning_rate * delta_weight
    
    @classmethod
    def backward_update_layer_activity(cls, layer, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for post_pop in layer:
            post_pop.prev_activity = post_pop.activity
        for post_pop in layer:
            if post_pop.backward_projections or post_pop is post_pop.network.output_pop:
                # update somatic state and activity
                delta_state = -post_pop.state + post_pop.bias
                if hasattr(post_pop, 'nudge'):
                    delta_state = delta_state + post_pop.nudge
                for projection in post_pop:
                    pre_pop = projection.pre
                    if projection.compartment in [None, 'soma']:
                        if projection.direction in ['forward', 'F']:
                            delta_state = delta_state + projection(pre_pop.activity)
                        elif projection.direction in ['recurrent', 'R']:
                            delta_state = delta_state + projection(pre_pop.prev_activity)
                post_pop.state = post_pop.state + delta_state / post_pop.tau
                post_pop.activity = post_pop.activation(post_pop.state)
                if store_dynamics:
                    post_pop.backward_steps_activity.append(post_pop.activity.detach().clone())
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):

        output_pop = network.output_pop
        output_layer = output_pop.layer

        # initialize populations that are updated during the backward phase
        if store_dynamics:
            for pop in output_layer:
                if pop.backward_projections or pop is output_pop:
                    pop.backward_steps_activity = []

        # compute nudge based on target
        for projection in output_pop:
            if projection.learning_rule.__class__ == cls:
                output_pop.nudge = torch.clamp(target - output, min=0, max=1)
                break

        # re-equilibrate soma states and activities
        for t in range(network.forward_steps):
            cls.backward_update_layer_activity(output_layer, store_dynamics=store_dynamics)
        
        if store_history:
            for pop in output_layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                    else:
                        pop.append_attribute_history('backward_activity', pop.activity.detach().clone())
            output_pop.append_attribute_history('nudge', output_pop.nudge.detach().clone())
        
        for i, layer in enumerate(network):
            if i > 0:
                for population in layer:
                    for projection in population:
                        if projection.learning_rule.__class__ == cls:
                            # only update theta once per population
                            population.BCM_theta_updated = False
                            if store_history and not population.BCM_theta_stored:
                                population.append_attribute_history('theta', population.theta.detach().clone())
                                population.BCM_theta_stored = True
                            break


class Supervised_BCM_2(LearningRule):
    def __init__(self, projection, theta_tau, k, sign=1, pos_loss_th=0.2, neg_loss_th=-0.2, max_pop_fraction=0.025,
                 learning_rate=None):
        """
        Output units are nudged to target.
        Nudges to somatic state are applied instantaneously, rather than being subject to slow equilibration.
        :param projection:
        :param theta_tau: float
        :param k: float
        :param sign: int in {-1, 1}
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.theta_tau = theta_tau
        self.k = k
        self.projection.post.theta = torch.ones(projection.post.size, device=projection.post.network.device,
                                                requires_grad=False) * k
        self.sign = sign
        self.max_pop_fraction = max_pop_fraction
        self.pos_loss_th = pos_loss_th
        self.neg_loss_th = neg_loss_th
        projection.post.__class__.theta_history = property(lambda self: self.get_attribute_history('theta'))
        projection.post.__class__.plateau_history = property(lambda self: self.get_attribute_history('plateau'))
        projection.post.__class__.backward_activity_history = \
            property(lambda self: self.get_attribute_history('backward_activity'))
        projection.post.__class__.forward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('forward_dendritic_state'))
        projection.post.__class__.backward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('backward_dendritic_state'))
    
    def reinit(self):
        self.projection.post.BCM_theta_stored = False
        self.projection.post.BCM_theta_updated = False
    
    def update(self):
        if not self.projection.post.BCM_theta_updated:
            delta_theta = (-self.projection.post.theta + self.projection.post.activity ** 2. / self.k) / self.theta_tau
            self.projection.post.theta += delta_theta
            self.projection.post.BCM_theta_updated = True
            self.projection.post.BCM_theta_stored = False
    
    def step(self):
        if self.projection.direction in ['forward', 'F']:
            delta_weight = torch.outer(self.projection.post.activity, self.projection.pre.activity) * \
                           (self.projection.post.activity - self.projection.post.theta).unsqueeze(1)
        elif self.projection.direction in ['recurrent', 'R']:
            delta_weight = torch.outer(self.projection.post.activity, self.projection.pre.prev_activity) * \
                           (self.projection.post.activity - self.projection.post.theta).unsqueeze(1)
        if self.sign > 0:
            self.projection.weight.data += self.learning_rate * delta_weight
        else:
            self.projection.weight.data -= self.learning_rate * delta_weight
    
    @classmethod
    def backward_update_activity(cls, network, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for layer in network:
            for post_pop in layer:
                post_pop.prev_activity = post_pop.activity
        for layer in network:
            for post_pop in layer:
                if post_pop.backward_projections or post_pop is post_pop.network.output_pop:
                    # update somatic state and activity
                    delta_state = -post_pop.state + post_pop.bias
                    for projection in post_pop:
                        pre_pop = projection.pre
                        if projection.compartment in [None, 'soma']:
                            if projection.direction in ['forward', 'F']:
                                delta_state = delta_state + projection(pre_pop.activity)
                            elif projection.direction in ['recurrent', 'R']:
                                delta_state = delta_state + projection(pre_pop.prev_activity)
                    post_pop.state = post_pop.state + delta_state / post_pop.tau
                    if hasattr(post_pop, 'dend_to_soma'):
                        post_pop.activity = post_pop.activation(post_pop.state + post_pop.dend_to_soma)
                    else:
                        post_pop.activity = post_pop.activation(post_pop.state)
                    if store_dynamics:
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
                            post_pop.dendritic_state = torch.zeros(post_pop.size, device=post_pop.network.device,
                                                                   requires_grad=False)
                            init_dend_state = True
                        if projection.direction in ['forward', 'F']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.activity)
                        elif projection.direction in ['recurrent', 'R']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.prev_activity)
                if init_dend_state:
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
        """
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_pop = network.output_pop
        
        # compute the forward_dendritic_state before comparing output to target
        output_pop.dendritic_state = torch.zeros(output_pop.size, device=network.device, requires_grad=False)
        for layer in reversed_layers:
            cls.backward_update_layer_dendritic_state(layer)
            # initialize populations that are updated during the backward phase
            for pop in layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.backward_steps_activity = []
                # store the forward_activity before comparing output to target
                pop.forward_activity = pop.activity.detach().clone()
                # initialize dendritic state variables
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        pop.plateau = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.dend_to_soma = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.forward_dendritic_state = pop.dendritic_state.detach().clone()
                        if store_history:
                            # store the forward_dendritic_state
                            pop.append_attribute_history('forward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break
        
        for t in range(network.forward_steps):
            for layer in reversed_layers:
                # update dendritic state variables
                cls.backward_update_layer_dendritic_state(layer)
                for pop in layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                if t == 0:
                                    output_pop.dendritic_state = (
                                        torch.clamp(target - output_pop.activity, min=-1, max=1))
                                    output_pop.plateau = output_pop.dendritic_state.detach().clone()
                                    output_pop.dend_to_soma = output_pop.dendritic_state.detach().clone()
                            else:
                                max_units = math.ceil(projection.learning_rule.max_pop_fraction * pop.size)
                                avail_indexes = (pop.plateau == 0).nonzero().squeeze(1)
                                if len(avail_indexes) > 0:
                                    pos_remaining = max_units - (pop.plateau > 0).count_nonzero()
                                    neg_remaining = max_units - (pop.plateau < 0).count_nonzero()
                                    if pos_remaining > 0 or neg_remaining > 0:
                                        sorted, sorted_indexes = torch.sort(pop.dendritic_state[avail_indexes],
                                                                            descending=True, stable=True)
                                        sorted_avail_indexes = avail_indexes[sorted_indexes]
                                    if pos_remaining > 0:
                                        pos_indexes = (pop.dendritic_state[sorted_avail_indexes] >
                                                       projection.learning_rule.pos_loss_th).nonzero().squeeze(1)
                                        pos_avail_indexes = sorted_avail_indexes[pos_indexes][:pos_remaining]
                                    else:
                                        pos_avail_indexes = []
                                    if neg_remaining > 0:
                                        neg_indexes = (pop.dendritic_state[sorted_avail_indexes] <
                                                       projection.learning_rule.neg_loss_th).nonzero().squeeze(1)
                                        neg_avail_indexes = sorted_avail_indexes[neg_indexes][-neg_remaining:]
                                    else:
                                        neg_avail_indexes = []
                                    pop.plateau[pos_avail_indexes] = pop.dendritic_state[pos_avail_indexes]
                                    pop.dend_to_soma[pos_avail_indexes] = pop.dendritic_state[pos_avail_indexes]
                                    pop.plateau[neg_avail_indexes] = pop.dendritic_state[neg_avail_indexes]
                                    pop.dend_to_soma[neg_avail_indexes] = pop.dendritic_state[neg_avail_indexes]
                            break
            # update activities
            cls.backward_update_activity(network, store_dynamics=store_dynamics)
        
        for layer in network:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        # only update theta once per population
                        pop.BCM_theta_updated = False
                        if store_history:
                            pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                            pop.append_attribute_history('backward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                            if not pop.BCM_theta_stored:
                                pop.append_attribute_history('theta', pop.theta.detach().clone())
                                pop.BCM_theta_stored = True
                        break
                if pop.backward_projections or pop is output_pop:
                    if store_history:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


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


class GjorgjievaHebb_2(LearningRule):
    def __init__(self, projection, sign=1, learning_rate=None):
        super().__init__(projection, learning_rate)
        self.sign = sign

    def step(self):
        if self.projection.direction in ['forward', 'F']:
            delta_weight = torch.outer(self.projection.post.activity, self.projection.pre.activity)
        elif self.projection.direction in ['recurrent', 'R']:
            delta_weight = torch.outer(self.projection.post.activity, self.projection.pre.prev_activity)
        if self.sign > 0:
            self.projection.weight.data += self.learning_rate * delta_weight
        else:
            self.projection.weight.data -= self.learning_rate * delta_weight


class SupervisedGjorgjievaHebb(LearningRule):
    def __init__(self, projection, sign=1, learning_rate=None):
        super().__init__(projection, learning_rate)
        self.sign = sign
        projection.post.__class__.nudge_history = property(lambda self: self.get_attribute_history('nudge'))
        projection.post.__class__.backward_activity_history = \
            property(lambda self: self.get_attribute_history('backward_activity'))

    def step(self):
        delta_weight = torch.outer(self.projection.post.activity, self.projection.pre.activity)
        if self.sign > 0:
            self.projection.weight.data += self.learning_rate * delta_weight
        else:
            self.projection.weight.data -= self.learning_rate * delta_weight

    @classmethod
    def backward_update_layer_activity(cls, layer, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for post_pop in layer:
            post_pop.prev_activity = post_pop.activity
        for post_pop in layer:
            if post_pop.backward_projections or post_pop is post_pop.network.output_pop:
                # update somatic state and activity
                delta_state = -post_pop.state + post_pop.bias
                if hasattr(post_pop, 'nudge'):
                    delta_state = delta_state + post_pop.nudge
                for projection in post_pop:
                    pre_pop = projection.pre
                    if projection.compartment in [None, 'soma']:
                        if projection.direction in ['forward', 'F']:
                            delta_state = delta_state + projection(pre_pop.activity)
                        elif projection.direction in ['recurrent', 'R']:
                            delta_state = delta_state + projection(pre_pop.prev_activity)
                post_pop.state = post_pop.state + delta_state / post_pop.tau
                post_pop.activity = post_pop.activation(post_pop.state)
                if store_dynamics:
                    post_pop.backward_steps_activity.append(post_pop.activity.detach().clone())

    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
        """
        output_pop = network.output_pop
        output_layer = output_pop.layer

        # initialize populations that are updated during the backward phase

        if store_dynamics:
            for pop in output_layer:
                if pop.backward_projections or pop is output_pop:
                    pop.backward_steps_activity = []

        # compute nudge based on target
        for projection in output_pop:
            if projection.learning_rule.__class__ == cls:
                output_pop.nudge = torch.clamp(target - output, min=0, max=1)
                break

        # re-equilibrate soma states and activities
        for t in range(network.forward_steps):
            cls.backward_update_layer_activity(output_layer, store_dynamics=store_dynamics)
        if store_history:
            for pop in output_layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                    else:
                        pop.append_attribute_history('backward_activity', pop.activity.detach().clone())
            output_pop.append_attribute_history('nudge', output_pop.nudge.detach().clone())


class SupervisedGjorgjievaHebb_2(LearningRule):
    """
    This variant 2 respects recurrent connections and consults the prev_activity to update weights.
    """
    def __init__(self, projection, sign=1, learning_rate=None):
        super().__init__(projection, learning_rate)
        self.sign = sign
        projection.post.__class__.nudge_history = property(lambda self: self.get_attribute_history('nudge'))
        projection.post.__class__.backward_activity_history = \
            property(lambda self: self.get_attribute_history('backward_activity'))

    def step(self):
        if self.projection.direction in ['forward', 'F']:
            delta_weight = torch.outer(self.projection.post.activity, self.projection.pre.activity)
        elif self.projection.direction in ['recurrent', 'R']:
            delta_weight = torch.outer(self.projection.post.activity, self.projection.pre.prev_activity)
        if self.sign > 0:
            self.projection.weight.data += self.learning_rate * delta_weight
        else:
            self.projection.weight.data -= self.learning_rate * delta_weight

    @classmethod
    def backward_update_layer_activity(cls, layer, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for post_pop in layer:
            post_pop.prev_activity = post_pop.activity
        for post_pop in layer:
            if post_pop.backward_projections or post_pop is post_pop.network.output_pop:
                # update somatic state and activity
                delta_state = -post_pop.state + post_pop.bias
                if hasattr(post_pop, 'nudge'):
                    delta_state = delta_state + post_pop.nudge
                for projection in post_pop:
                    pre_pop = projection.pre
                    if projection.compartment in [None, 'soma']:
                        if projection.direction in ['forward', 'F']:
                            delta_state = delta_state + projection(pre_pop.activity)
                        elif projection.direction in ['recurrent', 'R']:
                            delta_state = delta_state + projection(pre_pop.prev_activity)
                post_pop.state = post_pop.state + delta_state / post_pop.tau
                post_pop.activity = post_pop.activation(post_pop.state)
                if store_dynamics:
                    post_pop.backward_steps_activity.append(post_pop.activity.detach().clone())

    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
        """
        output_pop = network.output_pop
        output_layer = output_pop.layer

        # initialize populations that are updated during the backward phase
        if store_dynamics:
            for pop in output_layer:
                if pop.backward_projections or pop is output_pop:
                    pop.backward_steps_activity = []

        # compute nudge based on target
        for projection in output_pop:
            if projection.learning_rule.__class__ == cls:
                output_pop.nudge = torch.clamp(target - output, min=0, max=1)
                break

        # re-equilibrate soma states and activities
        for t in range(network.forward_steps):
            cls.backward_update_layer_activity(output_layer, store_dynamics=store_dynamics)
        if store_history:
            for pop in output_layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                    else:
                        pop.append_attribute_history('backward_activity', pop.activity.detach().clone())
            output_pop.append_attribute_history('nudge', output_pop.nudge.detach().clone())


class Hebb_WeightNorm(LearningRule):
    def __init__(self, projection, sign=1, learning_rate=None):
        super().__init__(projection, learning_rate)
        self.sign = sign

    def step(self):
        if self.projection.direction in ['forward', 'F']:
            delta_weight = torch.outer(self.projection.post.activity, self.projection.pre.activity)
        elif self.projection.direction in ['recurrent', 'R']:
            delta_weight = torch.outer(self.projection.post.activity, self.projection.pre.prev_activity)
        if self.sign > 0:
            self.projection.weight.data += self.learning_rate * delta_weight
        else:
            self.projection.weight.data -= self.learning_rate * delta_weight


class Supervised_Hebb_WeightNorm(LearningRule):
    """
    Output units are nudged to target.
    Nudges to somatic state are applied instantaneously, rather than being subject to slow equilibration.
    """
    
    def __init__(self, projection, sign=1, pos_loss_th=0.2, neg_loss_th=-0.2, max_pop_fraction=0.025,
                 learning_rate=None):
        super().__init__(projection, learning_rate)
        self.sign = sign
        self.max_pop_fraction = max_pop_fraction
        self.pos_loss_th = pos_loss_th
        self.neg_loss_th = neg_loss_th
        projection.post.__class__.plateau_history = property(lambda self: self.get_attribute_history('plateau'))
        projection.post.__class__.backward_activity_history = \
            property(lambda self: self.get_attribute_history('backward_activity'))
    
    def step(self):
        if self.projection.direction in ['forward', 'F']:
            delta_weight = torch.outer(self.projection.post.activity, self.projection.pre.activity)
        elif self.projection.direction in ['recurrent', 'R']:
            delta_weight = torch.outer(self.projection.post.activity, self.projection.pre.prev_activity)
        if self.sign > 0:
            self.projection.weight.data += self.learning_rate * delta_weight
        else:
            self.projection.weight.data -= self.learning_rate * delta_weight
    
    @classmethod
    def backward_update_activity(cls, network, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for layer in network:
            for post_pop in layer:
                post_pop.prev_activity = post_pop.activity
        for layer in network:
            for post_pop in layer:
                if post_pop.backward_projections or post_pop is post_pop.network.output_pop:
                    # update somatic state and activity
                    delta_state = -post_pop.state + post_pop.bias
                    for projection in post_pop:
                        pre_pop = projection.pre
                        if projection.compartment in [None, 'soma']:
                            if projection.direction in ['forward', 'F']:
                                delta_state = delta_state + projection(pre_pop.activity)
                            elif projection.direction in ['recurrent', 'R']:
                                delta_state = delta_state + projection(pre_pop.prev_activity)
                    post_pop.state = post_pop.state + delta_state / post_pop.tau
                    if hasattr(post_pop, 'dend_to_soma'):
                        post_pop.activity = post_pop.activation(post_pop.state + post_pop.dend_to_soma)
                    else:
                        post_pop.activity = post_pop.activation(post_pop.state)
                    if store_dynamics:
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
                            post_pop.dendritic_state = torch.zeros(post_pop.size, device=post_pop.network.device,
                                                                   requires_grad=False)
                            init_dend_state = True
                        if projection.direction in ['forward', 'F']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.activity)
                        elif projection.direction in ['recurrent', 'R']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.prev_activity)
                if init_dend_state:
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
        """
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_pop = network.output_pop
        
        # compute the forward_dendritic_state before comparing output to target
        output_pop.dendritic_state = torch.zeros(output_pop.size, device=network.device, requires_grad=False)
        for layer in reversed_layers:
            cls.backward_update_layer_dendritic_state(layer)
            # initialize populations that are updated during the backward phase
            for pop in layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.backward_steps_activity = []
                # store the forward_activity before comparing output to target
                pop.forward_activity = pop.activity.detach().clone()
                # initialize dendritic state variables
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        pop.plateau = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.dend_to_soma = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.forward_dendritic_state = pop.dendritic_state.detach().clone()
                        if store_history:
                            # store the forward_dendritic_state
                            pop.append_attribute_history('forward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break
        
        for t in range(network.forward_steps):
            for layer in reversed_layers:
                # update dendritic state variables
                cls.backward_update_layer_dendritic_state(layer)
                for pop in layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                if t == 0:
                                    output_pop.dendritic_state = (
                                        torch.clamp(target - output_pop.activity, min=-1, max=1))
                                    output_pop.plateau = output_pop.dendritic_state.detach().clone()
                                    output_pop.dend_to_soma = output_pop.dendritic_state.detach().clone()
                            else:
                                max_units = math.ceil(projection.learning_rule.max_pop_fraction * pop.size)
                                avail_indexes = (pop.plateau == 0).nonzero().squeeze(1)
                                if len(avail_indexes) > 0:
                                    pos_remaining = max_units - (pop.plateau > 0).count_nonzero()
                                    neg_remaining = max_units - (pop.plateau < 0).count_nonzero()
                                    if pos_remaining > 0 or neg_remaining > 0:
                                        sorted, sorted_indexes = torch.sort(pop.dendritic_state[avail_indexes],
                                                                            descending=True, stable=True)
                                        sorted_avail_indexes = avail_indexes[sorted_indexes]
                                    if pos_remaining > 0:
                                        pos_indexes = (pop.dendritic_state[sorted_avail_indexes] >
                                                       projection.learning_rule.pos_loss_th).nonzero().squeeze(1)
                                        pos_avail_indexes = sorted_avail_indexes[pos_indexes][:pos_remaining]
                                    else:
                                        pos_avail_indexes = []
                                    if neg_remaining > 0:
                                        neg_indexes = (pop.dendritic_state[sorted_avail_indexes] <
                                                       projection.learning_rule.neg_loss_th).nonzero().squeeze(1)
                                        neg_avail_indexes = sorted_avail_indexes[neg_indexes][-neg_remaining:]
                                    else:
                                        neg_avail_indexes = []
                                    pop.plateau[pos_avail_indexes] = pop.dendritic_state[pos_avail_indexes]
                                    pop.dend_to_soma[pos_avail_indexes] = pop.dendritic_state[pos_avail_indexes]
                                    pop.plateau[neg_avail_indexes] = pop.dendritic_state[neg_avail_indexes]
                                    pop.dend_to_soma[neg_avail_indexes] = pop.dendritic_state[neg_avail_indexes]
                            break
            # update activities
            cls.backward_update_activity(network, store_dynamics=store_dynamics)
        
        for layer in network:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        if store_history:
                            pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                            pop.append_attribute_history('backward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break
                if pop.backward_projections or pop is output_pop:
                    if store_history:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP(LearningRule):
    """
    This original rule variant is not compatible with equilibration of populations in both forward and backward phases.
    """
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
        projection.post.__class__.plateau_history = property(lambda self: self.get_attribute_history('plateau'))
        projection.post.__class__.backward_activity_history = \
            property(lambda self: self.get_attribute_history('backward_activity'))

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
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
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
                    output_pop.append_attribute_history('plateau',output_pop.plateau.detach().clone())
                break
        if store_history:
            output_pop.append_attribute_history('backward_activity', output_pop.activity.detach().clone())

        for layer in reversed_layers:
            # initialize cells that receive backward projections
            for pop in layer:
                if pop.backward_projections:
                    pop.forward_soma_state = pop.state.detach().clone()
                    if store_dynamics:
                        pop.backward_steps_activity = []

            # equilibrate activities and dendritic state variables
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
                        if store_dynamics:
                            pop.backward_steps_activity.append(pop.activity.detach().clone())
            for pop in layer:
                if pop.backward_projections:
                    for projection in pop.backward_projections:
                        if projection.learning_rule.__class__ == cls:
                            neg_indexes = (pop.dendritic_state < projection.learning_rule.neg_loss_th).nonzero(
                                as_tuple=True)
                            pop.plateau[neg_indexes] = pop.dendritic_state[neg_indexes]
                            if store_history:
                                pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                            break
                    if store_history:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_2(LearningRule):
    def __init__(self, projection, pos_loss_th=2.440709E-01, neg_loss_th=-4.592181E-01, neg_loss_ET_discount=0.25,
                 dep_ratio=1., dep_th=0.01, dep_width=0.01, max_pop_fraction=0.025, learning_rate=None):
        """
        Like the original BTSP class, this method includes both positive and negative modulatory events. Only positive
        modulatory events nudge the somatic activity. This variant is tolerant to the update_phase of projections
        being specified as 'A' or 'All', whereas the original class could only update projections either in the
        forward or backward phase, but not both. In variant 2, plateaus can only occur in a specified maximum fraction
        of units in each layer. The entire layer is re-equilibrated after plateaus are selected.
        :param projection: :class:'nn.Linear'
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
        projection.post.__class__.plateau_history = property(lambda self: self.get_attribute_history('plateau'))
        projection.post.__class__.backward_activity_history = \
            property(lambda self: self.get_attribute_history('backward_activity'))
        projection.post.__class__.forward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('forward_dendritic_state'))
        projection.post.__class__.backward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('backward_dendritic_state'))

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
    def backward_equilibrate_layer_activity(cls, layer, forward_steps, store_dynamics=False):
        """
        Equilibrate somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param forward_steps: int
        :param store_dynamics: bool
        """
        for t in range(forward_steps):
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
                if store_dynamics:
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
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
        """
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_layer = reversed_layers.pop(0)
        output_pop = network.output_pop

        if store_dynamics:
            # initialize populations that are updated during the backward phase
            for layer in network:
                for pop in layer:
                    if pop.backward_projections or pop is output_pop:
                        pop.backward_steps_activity = []

        # store the forward_activity before comparing output to target
        for layer in network:
            for pop in layer:
                pop.forward_activity = pop.activity.detach().clone()

        # compute and store the forward_dendritic_state before comparing output to target
        for layer in list(network)[:-1]:
            cls.backward_update_layer_dendritic_state(layer)
            for pop in layer:
                if hasattr(pop, 'dendritic_state'):
                    pop.forward_dendritic_state = pop.dendritic_state.detach().clone()
                    if store_history:
                        pop.append_attribute_history('forward_dendritic_state',
                                                     pop.forward_dendritic_state.detach().clone())

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

        # re-equilibrate soma states and activities
        cls.backward_equilibrate_layer_activity(output_layer, network.forward_steps, store_dynamics=store_dynamics)
        if store_history:
            output_pop.append_attribute_history('plateau', output_pop.plateau.detach().clone())

        for layer in reversed_layers:
            # equilibrate activities and dendritic state variables
            cls.backward_update_layer_dendritic_state(layer)
            for pop in layer:
                for projection in pop:
                    # compute plateau events and nudge somatic state
                    if projection.learning_rule.__class__ == cls:
                        max_units = math.ceil(projection.learning_rule.max_pop_fraction * pop.size)
                        pop.plateau = torch.zeros(pop.size, device=pop.network.device)
                        pop.dend_to_soma = torch.zeros(pop.size, device=pop.network.device)

                        sorted, sorted_indexes = torch.sort(pop.dendritic_state, descending=True, stable=True)
                        pos_indexes = (pop.dendritic_state[sorted_indexes] >
                                       projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)[0][:max_units]
                        pos_plateau_indexes = sorted_indexes[pos_indexes]
                        pop.plateau[pos_plateau_indexes] = pop.dendritic_state[pos_plateau_indexes]
                        pop.dend_to_soma[pos_plateau_indexes] = 1.  # pop.dendritic_state[pos_plateau_indexes]

                        neg_indexes = (pop.dendritic_state[sorted_indexes] <
                                       projection.learning_rule.neg_loss_th).nonzero(as_tuple=True)[0][-max_units:]
                        neg_plateau_indexes = sorted_indexes[neg_indexes]
                        pop.plateau[neg_plateau_indexes] = pop.dendritic_state[neg_plateau_indexes]
                        # pop.dend_to_soma[neg_plateau_indexes] = pop.dendritic_state[neg_plateau_indexes]

                        if store_history:
                            pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                            pop.append_attribute_history('backward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break
            cls.backward_equilibrate_layer_activity(layer, network.forward_steps, store_dynamics=store_dynamics)

        if store_history:
            for layer in network:
                for pop in layer:
                    if pop.backward_projections or pop is output_pop:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


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
        projection.post.__class__.plateau_history = property(lambda self: self.get_attribute_history('plateau'))
        projection.post.__class__.backward_activity_history = \
            property(lambda self: self.get_attribute_history('backward_activity'))

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
    def backward_update_layer_activity(cls, layer, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
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
                if store_dynamics:
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
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
        """
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_layer = reversed_layers.pop(0)
        output_pop = network.output_pop

        # store the forward_activity before comparing output to target
        for layer in network:
            for pop in layer:
                pop.forward_activity = pop.activity.detach().clone()

        # compute and store the forward_dendritic_state before comparing output to target
        for layer in list(network)[:-1]:
            cls.backward_update_layer_dendritic_state(layer)
            for pop in layer:
                if hasattr(pop, 'dendritic_state'):
                    pop.forward_dendritic_state = pop.dendritic_state.detach().clone()

        if store_dynamics:
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
        # re-equilibrate soma states and activities
        for t in range(network.forward_steps):
            cls.backward_update_layer_activity(output_layer, store_dynamics=store_dynamics)
        if store_history:
            output_pop.append_attribute_history('plateau', output_pop.plateau.detach().clone())

        for layer in reversed_layers:
            # equilibrate activities and dendritic state variables
            for t in range(network.forward_steps):
                cls.backward_update_layer_activity(layer, store_dynamics=store_dynamics)
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
                            pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                        break

        if store_history:
            for layer in network:
                for pop in layer:
                    if pop.backward_projections or pop is output_pop:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_4(LearningRule):
    def __init__(self, projection, pos_loss_th=2.440709E-01, neg_loss_th=-4.592181E-01, neg_loss_ET_discount=0.25,
                 dep_ratio=1., dep_th=0.01, dep_width=0.01, learning_rate=None):
        """
        Like the original BTSP class, this method includes both positive and negative modulatory events. Only positive
        modulatory events nudge the somatic activity. This variant is tolerant to the update_phase of projections
        being specified as 'A' or 'All', whereas the original class could only update projections either in the
        forward or backward phase, but not both. In variant 4, units are selected for positive modulatory events in
        order of largest loss, one at a time, and the entire layer is re-equilibrated after each event.
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
        projection.post.__class__.plateau_history = property(lambda self: self.get_attribute_history('plateau'))
        projection.post.__class__.backward_activity_history = \
            property(lambda self: self.get_attribute_history('backward_activity'))

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
    def backward_update_layer_activity(cls, layer, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
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
                if store_dynamics:
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
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
        """
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_layer = reversed_layers.pop(0)
        output_pop = network.output_pop

        if store_dynamics:
            # initialize populations that are updated during the backward phase
            for layer in network:
                for pop in layer:
                    if pop.backward_projections or pop is output_pop:
                        pop.backward_steps_activity = []

        # store the forward_activity before comparing output to target
        for layer in network:
            for pop in layer:
                pop.forward_activity = pop.activity.detach().clone()

        # compute and store the forward_dendritic_state before comparing output to target
        for layer in list(network)[:-1]:
            cls.backward_update_layer_dendritic_state(layer)
            for pop in layer:
                if hasattr(pop, 'dendritic_state'):
                    pop.forward_dendritic_state = pop.dendritic_state.detach().clone()

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

        # re-equilibrate soma states and activities
        for t in range(network.forward_steps):
            cls.backward_update_layer_activity(output_layer, store_dynamics=store_dynamics)
        if store_history:
            output_pop.append_attribute_history('plateau', output_pop.plateau.detach().clone())

        for layer in reversed_layers:
            # equilibrate activities and dendritic state variables
            cls.backward_update_layer_dendritic_state(layer)
            for pop in layer:
                for projection in pop:
                    # compute plateau events and nudge somatic state
                    if projection.learning_rule.__class__ == cls:
                        pos_plateau_indexes = []
                        pop.plateau = torch.zeros(pop.size, device=pop.network.device)
                        pop.dend_to_soma = torch.zeros(pop.size, device=pop.network.device)
                        # sort cells by dendritic state
                        while len(pos_plateau_indexes) < pop.size:
                            sorted, sorted_indexes = torch.sort(pop.dendritic_state, descending=True, stable=True)
                            pos_indexes = (pop.dendritic_state[sorted_indexes] >
                                           projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)[0]
                            sorted_pos_indexes = sorted_indexes[pos_indexes]
                            avail_indexes = \
                                torch.where(torch.isin(sorted_pos_indexes, torch.tensor(pos_plateau_indexes),
                                                       invert=True))[0]
                            if len(avail_indexes) > 0:
                                pos_plateau_index = sorted_pos_indexes[avail_indexes[0]]
                            else:
                                break
                            pos_plateau_indexes.append(pos_plateau_index)
                            pop.plateau[pos_plateau_index] = pop.dendritic_state[pos_plateau_index]
                            pop.dend_to_soma[pos_plateau_index] = 1.  # pop.dendritic_state[pos_indexes]
                            for t in range(network.forward_steps):
                                cls.backward_update_layer_activity(layer, store_history)
                            cls.backward_update_layer_dendritic_state(layer)
                        neg_indexes = (pop.dendritic_state < projection.learning_rule.neg_loss_th).nonzero(
                            as_tuple=True)[0]
                        avail_indexes = \
                            torch.where(torch.isin(neg_indexes, torch.tensor(pos_plateau_indexes), invert=True))[0]
                        if len(avail_indexes) > 0:
                            neg_plateau_indexes = neg_indexes[avail_indexes]
                            pop.plateau[neg_plateau_indexes] = pop.dendritic_state[neg_plateau_indexes]
                        if store_history:
                            pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                        break

        if store_history:
            for layer in network:
                for pop in layer:
                    if pop.backward_projections or pop is output_pop:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_5(LearningRule):
    def __init__(self, projection, pos_loss_th=2.440709E-01, temporal_discount=0.25, dep_ratio=1., dep_th=0.01,
                 dep_width=0.01, max_pop_fraction=0.025, learning_rate=None):
        """
        Unlike the original BTSP class, this method utilizes only positive modulatory events. Positive
        modulatory events nudge the somatic activity. However, now presynaptic eligibility and postsynaptic instruction
        persist and decay from the current sample to the next sample.
        This variant is tolerant to the update_phase of projections being specified as 'A' or 'All', whereas the
        original class could only update projections either in the forward or backward phase, but not both.
        Like variant 2, plateaus can only occur in a specified maximum fraction of units in each layer. The entire
        layer is re-equilibrated after plateaus are selected.
        :param projection: :class:'nn.Linear'
        :param pos_loss_th: float
        :param temporal_discount: float
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param max_pop_fraction: float in [0, 1]
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.pos_loss_th = pos_loss_th
        self.temporal_discount = temporal_discount
        self.dep_ratio = dep_ratio
        self.max_pop_fraction = max_pop_fraction
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]
        projection.post.__class__.plateau_history = property(lambda self: self.get_attribute_history('plateau'))
        projection.post.__class__.backward_activity_history = \
            property(lambda self: self.get_attribute_history('backward_activity'))
        projection.post.__class__.forward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('forward_dendritic_state'))
        projection.post.__class__.backward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('backward_dendritic_state'))

    def reinit(self):
        self.projection.pre.past_activity = torch.zeros(self.projection.pre.size,
                                                        device=self.projection.pre.network.device)
        self.projection.post.past_plateau = torch.zeros(self.projection.post.size,
                                                        device=self.projection.post.network.device)

    def update(self):
        self.projection.pre.past_activity = self.projection.pre.activity.detach().clone()
        self.projection.post.past_plateau = self.projection.post.plateau.detach().clone()

    def step(self):
        if torch.any(self.projection.post.plateau > 0):
            any_IS_now = True
            IS = torch.unsqueeze(self.projection.post.plateau, 1)
        else:
            any_IS_now = False

        ET = self.projection.pre.activity

        if any_IS_now:
            past_ET = self.projection.pre.past_activity * self.temporal_discount
            past_delta_weight = IS * (
                    (self.w_max - self.projection.weight) * torch.unsqueeze(past_ET, 0) -
                    self.projection.weight * self.dep_ratio * torch.unsqueeze(self.q_dep(past_ET), 0))
            self.projection.weight.data += self.learning_rate * past_delta_weight

        if torch.any(self.projection.post.past_plateau > 0):
            past_IS = torch.unsqueeze(self.projection.post.past_plateau, 1)
            future_ET = self.projection.pre.activity * self.temporal_discount
            future_delta_weight = past_IS * (
                    (self.w_max - self.projection.weight) * torch.unsqueeze(future_ET, 0) -
                    self.projection.weight * self.dep_ratio * torch.unsqueeze(self.q_dep(future_ET), 0))
            self.projection.weight.data += self.learning_rate * future_delta_weight

        if any_IS_now:
            delta_weight = IS * (
                    (self.w_max - self.projection.weight) * torch.unsqueeze(ET, 0) -
                    self.projection.weight * self.dep_ratio * torch.unsqueeze(self.q_dep(ET), 0))
            self.projection.weight.data += self.learning_rate * delta_weight

    @classmethod
    def backward_equilibrate_layer_activity(cls, layer, forward_steps, store_dynamics=False):
        """
        Equilibrate somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param forward_steps: int
        :param store_dynamics: bool
        """
        for t in range(forward_steps):
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
                if store_dynamics:
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
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
        """
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_layer = reversed_layers.pop(0)
        output_pop = network.output_pop

        # initialize populations that are updated during the backward phase
        # store the forward_activity before comparing output to target
        for layer in network:
            for pop in layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.backward_steps_activity = []
                pop.forward_activity = pop.activity.detach().clone()

        # compute and store the forward_dendritic_state before comparing output to target
        for layer in list(network)[:-1]:
            cls.backward_update_layer_dendritic_state(layer)
            for pop in layer:
                if hasattr(pop, 'dendritic_state'):
                    pop.forward_dendritic_state = pop.dendritic_state.detach().clone()
                    if store_history:
                        pop.append_attribute_history('forward_dendritic_state', pop.dendritic_state.detach().clone())

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
                break

        # re-equilibrate soma states and activities
        cls.backward_equilibrate_layer_activity(output_layer, network.forward_steps, store_dynamics=store_dynamics)
        if store_history:
            output_pop.append_attribute_history('plateau', output_pop.plateau.detach().clone())

        for layer in reversed_layers:
            # equilibrate activities and dendritic state variables
            cls.backward_update_layer_dendritic_state(layer)
            for pop in layer:
                for projection in pop:
                    # compute plateau events and nudge somatic state
                    if projection.learning_rule.__class__ == cls:
                        max_units = math.ceil(projection.learning_rule.max_pop_fraction * pop.size)
                        pop.plateau = torch.zeros(pop.size, device=pop.network.device)
                        pop.dend_to_soma = torch.zeros(pop.size, device=pop.network.device)

                        sorted, sorted_indexes = torch.sort(pop.dendritic_state, descending=True, stable=True)
                        pos_indexes = (pop.dendritic_state[sorted_indexes] >
                                       projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)[0][:max_units]
                        pos_plateau_indexes = sorted_indexes[pos_indexes]
                        pop.plateau[pos_plateau_indexes] = pop.dendritic_state[pos_plateau_indexes]
                        pop.dend_to_soma[pos_plateau_indexes] = 1.  # pop.dendritic_state[pos_plateau_indexes]

                        if store_history:
                            pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                            pop.append_attribute_history('backward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break
            cls.backward_equilibrate_layer_activity(layer, network.forward_steps, store_dynamics=store_dynamics)

        if store_history:
            for layer in network:
                for pop in layer:
                    if pop.backward_projections or pop is output_pop:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_6(LearningRule):
    def __init__(self, projection, pos_loss_th=2.440709E-01, decay_tau=2, dep_ratio=1., dep_th=0.01,
                 dep_width=0.01, max_pop_fraction=0.025, learning_rate=None):
        """
        Unlike the original BTSP class, this method utilizes only positive modulatory events. Positive
        modulatory events nudge the somatic activity. However, now presynaptic eligibility and postsynaptic instruction
        persist and decay across samples with a timescale set by a temporal_decay hyperparameter.
        Previous versions of the BTSP class considered only the ET in determining the sign of delta_weight. This
        effectively presumed that IS = 1, and ET * IS = ET. However, plateau amplitude varied with error, and was used
        to scale the learning_rate.
        In this BTSP class variant 6, ET * IS is consulted to determine change in weight. Both ET and IS saturate at 1,
        and accumulate or decay across samples. Plateau amplitude no longer separately modulates the learning_rate.
        In previous versions of the BTSP class, plateaus influenced somatic activity by slowly equilibrating the
        cell's internal state. In this variant, plateaus instantaneously increment firing rate.
        This variant is tolerant to the update_phase of projections being specified as 'A' or 'All', whereas the
        original class could only update projections either in the forward or backward phase, but not both.
        Like variant 2, plateaus can only occur in a specified maximum fraction of units in each layer. The entire
        layer is re-equilibrated after plateaus are selected.
        :param projection: :class:'nn.Linear'
        :param pos_loss_th: float
        :param decay_tau: float
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param max_pop_fraction: float in [0, 1]
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.pos_loss_th = pos_loss_th
        self.decay_tau = decay_tau
        self.dep_ratio = dep_ratio
        self.max_pop_fraction = max_pop_fraction
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]
        projection.post.__class__.plateau_history = property(lambda self: self.get_attribute_history('plateau'))
        projection.post.__class__.backward_activity_history = \
            property(lambda self: self.get_attribute_history('backward_activity'))
        projection.post.__class__.forward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('forward_dendritic_state'))
        projection.post.__class__.backward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('backward_dendritic_state'))
        projection.post.__class__.ET_history = property(lambda self: self.get_attribute_history('ET'))
        projection.post.__class__.IS_history = property(lambda self: self.get_attribute_history('IS'))

    def reinit(self):
        self.projection.pre.ET = torch.zeros(self.projection.pre.size, device=self.projection.pre.network.device,
                                             requires_grad=False)
        self.projection.post.IS = torch.zeros(self.projection.post.size, device=self.projection.post.network.device,
                                              requires_grad=False)
        self.projection.pre.ET_updated = False
        self.projection.post.IS_updated = False

    def update(self):
        self.projection.pre.ET *= (1. - 1./self.decay_tau)
        self.projection.post.IS *= (1. - 1./self.decay_tau)
        self.projection.pre.ET_updated = False
        self.projection.post.IS_updated = False

    def step(self):

        ET_IS = torch.outer(self.projection.post.IS, self.projection.pre.ET)
        delta_weight = self.learning_rate * (
                (self.w_max - self.projection.weight) * ET_IS -
                self.projection.weight * self.dep_ratio * self.q_dep(ET_IS))
        self.projection.weight.data += delta_weight

    @classmethod
    def backward_equilibrate_layer_activity(cls, layer, forward_steps, store_dynamics=False):
        """
        Equilibrate somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param forward_steps: int
        :param store_dynamics: bool
        """
        for t in range(forward_steps):
            cls.backward_update_layer_activity(layer, store_dynamics=store_dynamics)

    @classmethod
    def backward_update_layer_activity(cls, layer, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for post_pop in layer:
            post_pop.prev_activity = post_pop.activity
        for post_pop in layer:
            if post_pop.backward_projections or post_pop is post_pop.network.output_pop:
                # update somatic state and activity
                delta_state = -post_pop.state + post_pop.bias
                for projection in post_pop:
                    pre_pop = projection.pre
                    if projection.compartment in [None, 'soma']:
                        if projection.direction in ['forward', 'F']:
                            delta_state = delta_state + projection(pre_pop.activity)
                        elif projection.direction in ['recurrent', 'R']:
                            delta_state = delta_state + projection(pre_pop.prev_activity)
                post_pop.state = post_pop.state + delta_state / post_pop.tau
                post_pop.activity = post_pop.activation(post_pop.state)
                if hasattr(post_pop, 'dend_to_soma'):
                    post_pop.activity = post_pop.activity + post_pop.dend_to_soma
            if store_dynamics:
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
                            post_pop.dendritic_state = torch.zeros(post_pop.size, device=post_pop.network.device,
                                                                   requires_grad=False)
                            init_dend_state = True
                        if projection.direction in ['forward', 'F']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.activity)
                        elif projection.direction in ['recurrent', 'R']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.prev_activity)
                if init_dend_state:
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)

    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
        """
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_layer = reversed_layers.pop(0)
        output_pop = network.output_pop
        output_pop.dendritic_state = torch.zeros(output_pop.size, device=network.device, requires_grad=False)

        # initialize populations that are updated during the backward phase
        # store the forward_activity before comparing output to target
        # compute and store the forward_dendritic_state before comparing output to target
        for layer in network:
            cls.backward_update_layer_dendritic_state(layer)
            for pop in layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.backward_steps_activity = []
                pop.forward_activity = pop.activity.detach().clone()
                # initialize dendritic state variables
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        pop.plateau = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.dend_to_soma = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.forward_dendritic_state = pop.dendritic_state.detach().clone()
                        if store_history:
                            pop.append_attribute_history('forward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break

        # compute plateau events and nudge somatic state
        output_pop.dendritic_state = torch.clamp(target - output, min=-1, max=1)
        for projection in output_pop:
            if projection.learning_rule.__class__ == cls:
                pos_indexes = (output_pop.dendritic_state >
                               projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)
                output_pop.plateau[pos_indexes] = 1.  # output_pop.dendritic_state[pos_indexes]
                output_pop.dend_to_soma[pos_indexes] = 1.  # output_pop.dendritic_state[pos_indexes]
                break
        # re-equilibrate soma states and activities
        cls.backward_equilibrate_layer_activity(output_layer, network.forward_steps, store_dynamics=store_dynamics)

        for layer in reversed_layers:
            for t in range(network.forward_steps):
                # equilibrate activities and dendritic state variables
                cls.backward_update_layer_dendritic_state(layer)
                for pop in layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            max_units = math.ceil(projection.learning_rule.max_pop_fraction * pop.size)
                            if pop.plateau.count_nonzero() < max_units:
                                sorted, sorted_indexes = torch.sort(pop.dendritic_state, descending=True, stable=True)
                                candidate_indexes = \
                                    ((pop.dendritic_state[sorted_indexes] > projection.learning_rule.pos_loss_th) &
                                     (pop.plateau[sorted_indexes] == 0)).nonzero().squeeze(1)
                                if len(candidate_indexes) > 0:
                                    plateau_index = sorted_indexes[candidate_indexes][0]
                                    pop.plateau[plateau_index] = 1.
                                    pop.dend_to_soma[plateau_index] = 1.  # pop.dendritic_state[pos_plateau_indexes]
                            break
                cls.backward_update_layer_activity(layer, store_dynamics=store_dynamics)

        for layer in network:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        if not pop.IS_updated:
                            pop.IS += pop.plateau
                            pop.IS.clamp_(0., 1.)
                            pop.IS_updated = True
                            if store_history:
                                pop.append_attribute_history('IS', pop.IS.detach().clone())
                                pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                                pop.append_attribute_history('backward_dendritic_state',
                                                             pop.dendritic_state.detach().clone())
                        if not projection.pre.ET_updated:
                            projection.pre.ET += projection.pre.activity
                            projection.pre.ET.clamp_(0., 1.)
                            projection.pre.ET_updated = True
                            if store_history:
                                projection.pre.append_attribute_history('ET', projection.pre.ET.detach().clone())
                if pop.backward_projections or pop is output_pop:
                    if store_history:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_7(LearningRule):
    def __init__(self, projection, pos_loss_th=2.440709E-01, decay_tau=2, dep_ratio=1., dep_th=0.01,
                 dep_width=0.01, max_pop_fraction=0.025, learning_rate=None):
        """
        Unlike the original BTSP class, this method utilizes only positive modulatory events. Positive
        modulatory events nudge the somatic activity. However, now presynaptic eligibility and postsynaptic instruction
        persist and decay across samples with a timescale set by a temporal_decay hyperparameter.
        Previous versions of the BTSP class considered only the ET in determining the sign of delta_weight. This
        effectively presumed that IS = 1, and ET * IS = ET. However, plateau amplitude varied with error, and was used
        to scale the learning_rate.
        In this BTSP class variant 6, ET * IS is consulted to determine change in weight. Both ET and IS saturate at 1,
        and accumulate or decay across samples. Plateau amplitude no longer separately modulates the learning_rate.
        In previous versions of the BTSP class, plateaus influenced somatic activity by slowly equilibrating the
        cell's internal state. In this variant, plateaus instantaneously increment firing rate.
        This variant is tolerant to the update_phase of projections being specified as 'A' or 'All', whereas the
        original class could only update projections either in the forward or backward phase, but not both.
        Like variant 2, plateaus can only occur in a specified maximum fraction of units in each layer.
        While variants 2 and 6 equilibrates each layer sequentially and independently after plateaus were selected, this
        variant 7 selects plateaus and equilibrates activity in all layers simultaneously.
        :param projection: :class:'nn.Linear'
        :param pos_loss_th: float
        :param decay_tau: float
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param max_pop_fraction: float in [0, 1]
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.pos_loss_th = pos_loss_th
        self.decay_tau = decay_tau
        self.dep_ratio = dep_ratio
        self.max_pop_fraction = max_pop_fraction
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]
        projection.post.__class__.plateau_history = property(lambda self: self.get_attribute_history('plateau'))
        projection.post.__class__.backward_activity_history = \
            property(lambda self: self.get_attribute_history('backward_activity'))
        projection.post.__class__.forward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('forward_dendritic_state'))
        projection.post.__class__.backward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('backward_dendritic_state'))
        projection.post.__class__.ET_history = property(lambda self: self.get_attribute_history('ET'))
        projection.post.__class__.IS_history = property(lambda self: self.get_attribute_history('IS'))

    def reinit(self):
        self.projection.pre.ET = torch.zeros(self.projection.pre.size, device=self.projection.pre.network.device,
                                             requires_grad=False)
        self.projection.post.IS = torch.zeros(self.projection.post.size, device=self.projection.post.network.device,
                                              requires_grad=False)
        self.projection.pre.ET_updated = False
        self.projection.post.IS_updated = False

    def update(self):
        self.projection.pre.ET *= (1. - 1./self.decay_tau)
        self.projection.post.IS *= (1. - 1./self.decay_tau)
        self.projection.pre.ET_updated = False
        self.projection.post.IS_updated = False

    def step(self):

        ET_IS = torch.outer(self.projection.post.IS, self.projection.pre.ET)
        delta_weight = self.learning_rate * (
                (self.w_max - self.projection.weight) * ET_IS -
                self.projection.weight * self.dep_ratio * self.q_dep(ET_IS))
        self.projection.weight.data += delta_weight

    @classmethod
    def backward_update_activity(cls, network, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for layer in network:
            for post_pop in layer:
                post_pop.prev_activity = post_pop.activity
        for layer in network:
            for post_pop in layer:
                if post_pop.backward_projections or post_pop is post_pop.network.output_pop:
                    # update somatic state and activity
                    delta_state = -post_pop.state + post_pop.bias
                    for projection in post_pop:
                        pre_pop = projection.pre
                        if projection.compartment in [None, 'soma']:
                            if projection.direction in ['forward', 'F']:
                                delta_state = delta_state + projection(pre_pop.activity)
                            elif projection.direction in ['recurrent', 'R']:
                                delta_state = delta_state + projection(pre_pop.prev_activity)
                    post_pop.state = post_pop.state + delta_state / post_pop.tau
                    post_pop.activity = post_pop.activation(post_pop.state)
                    if hasattr(post_pop, 'dend_to_soma'):
                        post_pop.activity = post_pop.activity + post_pop.dend_to_soma
                    if store_dynamics:
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
                            post_pop.dendritic_state = torch.zeros(post_pop.size, device=post_pop.network.device,
                                                                   requires_grad=False)
                            init_dend_state = True
                        if projection.direction in ['forward', 'F']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.activity)
                        elif projection.direction in ['recurrent', 'R']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.prev_activity)
                if init_dend_state:
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)

    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
        """
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_pop = network.output_pop

        # compute the forward_dendritic_state before comparing output to target
        output_pop.dendritic_state = torch.zeros(output_pop.size, device=network.device, requires_grad=False)
        for layer in reversed_layers:
            cls.backward_update_layer_dendritic_state(layer)
            # initialize populations that are updated during the backward phase
            for pop in layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.backward_steps_activity = []
                # store the forward_activity before comparing output to target
                pop.forward_activity = pop.activity.detach().clone()
                # initialize dendritic state variables
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        pop.plateau = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.dend_to_soma = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.forward_dendritic_state = pop.dendritic_state.detach().clone()
                        if store_history:
                            # store the forward_dendritic_state
                            pop.append_attribute_history('forward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break

        for t in range(network.forward_steps):
            for layer in reversed_layers:
                # update dendritic state variables
                cls.backward_update_layer_dendritic_state(layer)
                for pop in layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                output_pop.dendritic_state = torch.clamp(target - output_pop.activity, min=-1, max=1)
                                pos_indexes = (output_pop.dendritic_state >
                                               projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)
                                output_pop.plateau[pos_indexes] = 1.  # output_pop.dendritic_state[pos_indexes]
                                output_pop.dend_to_soma[pos_indexes] = 1.  # output_pop.dendritic_state[pos_indexes]
                            else:
                                max_units = math.ceil(projection.learning_rule.max_pop_fraction * pop.size)
                                if pop.plateau.count_nonzero() < max_units:
                                    sorted, sorted_indexes = torch.sort(pop.dendritic_state, descending=True,
                                                                        stable=True)
                                    candidate_indexes = \
                                        ((pop.dendritic_state[sorted_indexes] > projection.learning_rule.pos_loss_th) &
                                         (pop.plateau[sorted_indexes] == 0)).nonzero().squeeze(1)
                                    if len(candidate_indexes) > 0:
                                        plateau_index = sorted_indexes[candidate_indexes][0]
                                        pop.plateau[plateau_index] = 1.
                                        pop.dend_to_soma[plateau_index] = 1.  # pop.dendritic_state[pos_plateau_indexes]
                            break
            # update activities
            cls.backward_update_activity(network, store_dynamics=store_dynamics)

        for layer in network:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        if not pop.IS_updated:
                            pop.IS += pop.plateau
                            pop.IS.clamp_(0., 1.)
                            pop.IS_updated = True
                            if store_history:
                                pop.append_attribute_history('IS', pop.IS.detach().clone())
                                pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                                pop.append_attribute_history('backward_dendritic_state',
                                                             pop.dendritic_state.detach().clone())
                        if not projection.pre.ET_updated:
                            projection.pre.ET += projection.pre.activity
                            projection.pre.ET.clamp_(0., 1.)
                            projection.pre.ET_updated = True
                            if store_history:
                                projection.pre.append_attribute_history('ET', projection.pre.ET.detach().clone())
                if pop.backward_projections or pop is output_pop:
                    if store_history:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_8(LearningRule):
    def __init__(self, projection, pos_loss_th=2.440709E-01, decay_tau=2, dep_ratio=1., dep_th=0.01,
                 dep_width=0.01, max_pop_fraction=0.025, anti_hebb_th=0.1, learning_rate=None):
        """
        Like variant 7, except now an anti-Hebbian depression occurs in cells that are active and receiving positive
        feedback within a range less than plateau threshold.
        ** There is an error in this implementation. In hidden layer E cells, the anti_hebb_th was compared to
        firing rate, when it should have been compared to dendritic_state.
        :param projection: :class:'nn.Linear'
        :param pos_loss_th: float
        :param decay_tau: float
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param max_pop_fraction: float in [0, 1]
        :param anti_hebb_th: float
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.pos_loss_th = pos_loss_th
        self.decay_tau = decay_tau
        self.dep_ratio = dep_ratio
        self.max_pop_fraction = max_pop_fraction
        self.anti_hebb_th = anti_hebb_th
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]
        projection.post.__class__.plateau_history = property(lambda self: self.get_attribute_history('plateau'))
        projection.post.__class__.backward_activity_history = \
            property(lambda self: self.get_attribute_history('backward_activity'))
        projection.post.__class__.forward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('forward_dendritic_state'))
        projection.post.__class__.backward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('backward_dendritic_state'))
        projection.post.__class__.ET_history = property(lambda self: self.get_attribute_history('ET'))
        projection.post.__class__.IS_history = property(lambda self: self.get_attribute_history('IS'))

    def reinit(self):
        self.projection.pre.ET = torch.zeros(self.projection.pre.size, device=self.projection.pre.network.device,
                                             requires_grad=False)
        self.projection.post.IS = torch.zeros(self.projection.post.size, device=self.projection.post.network.device,
                                              requires_grad=False)
        self.projection.pre.ET_updated = False
        self.projection.post.IS_updated = False

    def update(self):
        self.projection.pre.ET *= (1. - 1./self.decay_tau)
        self.projection.post.IS *= (1. - 1./self.decay_tau)
        self.projection.pre.ET_updated = False
        self.projection.post.IS_updated = False

    def step(self):
        # BTSP
        ET_IS = torch.outer(self.projection.post.IS, self.projection.pre.ET)
        delta_weight = self.learning_rate * (
                (self.w_max - self.projection.weight) * ET_IS -
                self.projection.weight * self.dep_ratio * self.q_dep(ET_IS))
        self.projection.weight.data += delta_weight

        # anti-Hebbian depression
        post_activity = torch.zeros(self.projection.post.size, device=self.projection.post.network.device,
                                    requires_grad=False)
        meets_criterion = self.projection.post.meets_BTSP_anti_hebb_criterion
        post_activity[meets_criterion] = self.projection.post.activity[meets_criterion]
        delta_weight = -self.learning_rate * torch.outer(post_activity, self.projection.pre.activity)
        self.projection.weight.data += delta_weight

    @classmethod
    def backward_update_activity(cls, network, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for layer in network:
            for post_pop in layer:
                post_pop.prev_activity = post_pop.activity
        for layer in network:
            for post_pop in layer:
                if post_pop.backward_projections or post_pop is post_pop.network.output_pop:
                    # update somatic state and activity
                    delta_state = -post_pop.state + post_pop.bias
                    for projection in post_pop:
                        pre_pop = projection.pre
                        if projection.compartment in [None, 'soma']:
                            if projection.direction in ['forward', 'F']:
                                delta_state = delta_state + projection(pre_pop.activity)
                            elif projection.direction in ['recurrent', 'R']:
                                delta_state = delta_state + projection(pre_pop.prev_activity)
                    post_pop.state = post_pop.state + delta_state / post_pop.tau
                    post_pop.activity = post_pop.activation(post_pop.state)
                    if hasattr(post_pop, 'dend_to_soma'):
                        post_pop.activity = post_pop.activity + post_pop.dend_to_soma
                    if store_dynamics:
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
                            post_pop.dendritic_state = torch.zeros(post_pop.size, device=post_pop.network.device,
                                                                   requires_grad=False)
                            init_dend_state = True
                        if projection.direction in ['forward', 'F']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.activity)
                        elif projection.direction in ['recurrent', 'R']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.prev_activity)
                if init_dend_state:
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)

    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
        """
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_pop = network.output_pop

        # compute the forward_dendritic_state before comparing output to target
        output_pop.dendritic_state = torch.zeros(output_pop.size, device=network.device, requires_grad=False)
        for layer in reversed_layers:
            cls.backward_update_layer_dendritic_state(layer)
            # initialize populations that are updated during the backward phase
            for pop in layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.backward_steps_activity = []
                # store the forward_activity before comparing output to target
                pop.forward_activity = pop.activity.detach().clone()
                # initialize dendritic state variables
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        pop.plateau = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.dend_to_soma = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.forward_dendritic_state = pop.dendritic_state.detach().clone()
                        if store_history:
                            # store the forward_dendritic_state
                            pop.append_attribute_history('forward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break

        for t in range(network.forward_steps):
            for layer in reversed_layers:
                # update dendritic state variables
                cls.backward_update_layer_dendritic_state(layer)
                for pop in layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                output_pop.dendritic_state = torch.clamp(target - output_pop.activity, min=-1, max=1)
                                pos_indexes = (output_pop.dendritic_state >
                                               projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)
                                output_pop.plateau[pos_indexes] = 1.  # output_pop.dendritic_state[pos_indexes]
                                output_pop.dend_to_soma[pos_indexes] = 1.  # output_pop.dendritic_state[pos_indexes]
                            else:
                                max_units = math.ceil(projection.learning_rule.max_pop_fraction * pop.size)
                                if pop.plateau.count_nonzero() < max_units:
                                    sorted, sorted_indexes = torch.sort(pop.dendritic_state, descending=True,
                                                                        stable=True)
                                    candidate_indexes = \
                                        ((pop.dendritic_state[sorted_indexes] > projection.learning_rule.pos_loss_th) &
                                         (pop.plateau[sorted_indexes] == 0)).nonzero().squeeze(1)
                                    if len(candidate_indexes) > 0:
                                        plateau_index = sorted_indexes[candidate_indexes][0]
                                        pop.plateau[plateau_index] = 1.
                                        pop.dend_to_soma[plateau_index] = 1.  # pop.dendritic_state[pos_plateau_indexes]
                            break
            # update activities
            cls.backward_update_activity(network, store_dynamics=store_dynamics)

        for layer in network:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        if not pop.IS_updated:
                            pop.IS += pop.plateau
                            pop.IS.clamp_(0., 1.)
                            pop.IS_updated = True
                            if pop is output_pop:
                                pop.meets_BTSP_anti_hebb_criterion = \
                                    ((pop.plateau == 0) &
                                     (pop.dendritic_state < -projection.learning_rule.anti_hebb_th))
                            else:
                                pop.meets_BTSP_anti_hebb_criterion = \
                                    ((pop.plateau == 0) &
                                     (pop.activity > projection.learning_rule.anti_hebb_th))
                            if store_history:
                                pop.append_attribute_history('IS', pop.IS.detach().clone())
                                pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                                pop.append_attribute_history('backward_dendritic_state',
                                                             pop.dendritic_state.detach().clone())
                        if not projection.pre.ET_updated:
                            projection.pre.ET += projection.pre.activity
                            projection.pre.ET.clamp_(0., 1.)
                            projection.pre.ET_updated = True
                            if store_history:
                                projection.pre.append_attribute_history('ET', projection.pre.ET.detach().clone())
                if pop.backward_projections or pop is output_pop:
                    if store_history:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_9(LearningRule):
    def __init__(self, projection, pos_loss_th=2.440709E-01, decay_tau=2, dep_ratio=1., dep_th=0.01, dep_width=0.01,
                 anti_hebb_th=0.1, learning_rate=None):
        """
        Similar to variant 8. Anti-Hebbian depression is triggered when an active unit has a dendritic state that
        crosses a threshold. This variant 9 does not impose a hard max_pop_fraction on plateaus.
        :param projection: :class:'nn.Linear'
        :param pos_loss_th: float
        :param decay_tau: float
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param anti_hebb_th: float
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.pos_loss_th = pos_loss_th
        self.decay_tau = decay_tau
        self.dep_ratio = dep_ratio
        self.anti_hebb_th = anti_hebb_th
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]
        projection.post.__class__.plateau_history = property(lambda self: self.get_attribute_history('plateau'))
        projection.post.__class__.backward_activity_history = \
            property(lambda self: self.get_attribute_history('backward_activity'))
        projection.post.__class__.forward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('forward_dendritic_state'))
        projection.post.__class__.backward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('backward_dendritic_state'))
        projection.post.__class__.ET_history = property(lambda self: self.get_attribute_history('ET'))
        projection.post.__class__.IS_history = property(lambda self: self.get_attribute_history('IS'))

    def reinit(self):
        self.projection.pre.ET = torch.zeros(self.projection.pre.size, device=self.projection.pre.network.device,
                                             requires_grad=False)
        self.projection.post.IS = torch.zeros(self.projection.post.size, device=self.projection.post.network.device,
                                              requires_grad=False)
        self.projection.pre.ET_updated = False
        self.projection.post.IS_updated = False

    def update(self):
        self.projection.pre.ET *= (1. - 1./self.decay_tau)
        self.projection.post.IS *= (1. - 1./self.decay_tau)
        self.projection.pre.ET_updated = False
        self.projection.post.IS_updated = False

    def step(self):
        # BTSP
        ET_IS = torch.outer(self.projection.post.IS, self.projection.pre.ET)
        delta_weight = self.learning_rate * (
                (self.w_max - self.projection.weight) * ET_IS -
                self.projection.weight * self.dep_ratio * self.q_dep(ET_IS))
        self.projection.weight.data += delta_weight

        # anti-Hebbian depression
        post_activity = torch.zeros(self.projection.post.size, device=self.projection.post.network.device,
                                    requires_grad=False)
        meets_criterion = self.projection.post.meets_BTSP_anti_hebb_criterion
        post_activity[meets_criterion] = self.projection.post.activity[meets_criterion]
        # TODO: Should these activities be clamped?
        delta_weight = -self.learning_rate * torch.outer(post_activity, self.projection.pre.activity)
        self.projection.weight.data += delta_weight

    @classmethod
    def backward_update_activity(cls, network, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for layer in network:
            for post_pop in layer:
                post_pop.prev_activity = post_pop.activity
        for layer in network:
            for post_pop in layer:
                if post_pop.backward_projections or post_pop is post_pop.network.output_pop:
                    # update somatic state and activity
                    delta_state = -post_pop.state + post_pop.bias
                    for projection in post_pop:
                        pre_pop = projection.pre
                        if projection.compartment in [None, 'soma']:
                            if projection.direction in ['forward', 'F']:
                                delta_state = delta_state + projection(pre_pop.activity)
                            elif projection.direction in ['recurrent', 'R']:
                                delta_state = delta_state + projection(pre_pop.prev_activity)
                    post_pop.state = post_pop.state + delta_state / post_pop.tau
                    post_pop.activity = post_pop.activation(post_pop.state)
                    if hasattr(post_pop, 'dend_to_soma'):
                        post_pop.activity = post_pop.activity + post_pop.dend_to_soma
                    if store_dynamics:
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
                            post_pop.dendritic_state = torch.zeros(post_pop.size, device=post_pop.network.device,
                                                                   requires_grad=False)
                            init_dend_state = True
                        if projection.direction in ['forward', 'F']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.activity)
                        elif projection.direction in ['recurrent', 'R']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.prev_activity)
                if init_dend_state:
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)

    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
        """
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_pop = network.output_pop

        # compute the forward_dendritic_state before comparing output to target
        output_pop.dendritic_state = torch.zeros(output_pop.size, device=network.device, requires_grad=False)
        for layer in reversed_layers:
            cls.backward_update_layer_dendritic_state(layer)
            # initialize populations that are updated during the backward phase
            for pop in layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.backward_steps_activity = []
                # store the forward_activity before comparing output to target
                pop.forward_activity = pop.activity.detach().clone()
                # initialize dendritic state variables
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        pop.plateau = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.dend_to_soma = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.forward_dendritic_state = pop.dendritic_state.detach().clone()
                        if store_history:
                            # store the forward_dendritic_state
                            pop.append_attribute_history('forward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break

        for t in range(network.forward_steps):
            for layer in reversed_layers:
                # update dendritic state variables
                cls.backward_update_layer_dendritic_state(layer)
                for pop in layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                output_pop.dendritic_state = torch.clamp(target - output_pop.activity, min=-1, max=1)
                                pos_indexes = (output_pop.dendritic_state >
                                               projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)
                                output_pop.plateau[pos_indexes] = 1.  # output_pop.dendritic_state[pos_indexes]
                                output_pop.dend_to_soma[pos_indexes] = 1.  # output_pop.dendritic_state[pos_indexes]
                            else:
                                sorted, sorted_indexes = torch.sort(pop.dendritic_state, descending=True,
                                                                    stable=True)
                                candidate_indexes = \
                                    ((pop.dendritic_state[sorted_indexes] > projection.learning_rule.pos_loss_th) &
                                     (pop.plateau[sorted_indexes] == 0)).nonzero().squeeze(1)
                                if len(candidate_indexes) > 0:
                                    plateau_index = sorted_indexes[candidate_indexes][0]
                                    pop.plateau[plateau_index] = 1.
                                    pop.dend_to_soma[plateau_index] = 1.  # pop.dendritic_state[pos_plateau_indexes]
                            break
            # update activities
            cls.backward_update_activity(network, store_dynamics=store_dynamics)

        for layer in network:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        if not pop.IS_updated:
                            pop.IS += pop.plateau
                            pop.IS.clamp_(0., 1.)
                            pop.IS_updated = True
                            if pop is output_pop:
                                pop.meets_BTSP_anti_hebb_criterion = \
                                    ((pop.plateau == 0) &
                                     (pop.dendritic_state < -projection.learning_rule.anti_hebb_th))
                            else:
                                pop.meets_BTSP_anti_hebb_criterion = \
                                    ((pop.plateau == 0) &
                                     (pop.activity > 0) &
                                     (pop.dendritic_state > projection.learning_rule.anti_hebb_th))
                            if store_history:
                                pop.append_attribute_history('IS', pop.IS.detach().clone())
                                pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                                pop.append_attribute_history('backward_dendritic_state',
                                                             pop.dendritic_state.detach().clone())
                        if not projection.pre.ET_updated:
                            projection.pre.ET += projection.pre.activity
                            projection.pre.ET.clamp_(0., 1.)
                            projection.pre.ET_updated = True
                            if store_history:
                                projection.pre.append_attribute_history('ET', projection.pre.ET.detach().clone())
                if pop.backward_projections or pop is output_pop:
                    if store_history:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_10(LearningRule):
    def __init__(self, projection, pos_loss_th=2.440709E-01, decay_tau=2, dep_ratio=1., dep_th=0.01, dep_width=0.01,
                 anti_hebb_th=(-0.1, 0.1), learning_rate=None, anti_hebb_learning_rate=None):
        """
        Similar to variant 9. Anti-Hebbian depression is triggered when an active unit has a dendritic state that
        is between a min and max bound.
        :param projection: :class:'nn.Linear'
        :param pos_loss_th: float
        :param decay_tau: float
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param anti_hebb_th: tuple of float
        :param learning_rate: float
        :param anti_hebb_learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.pos_loss_th = pos_loss_th
        self.decay_tau = decay_tau
        self.dep_ratio = dep_ratio
        self.anti_hebb_th = anti_hebb_th
        if anti_hebb_learning_rate is None:
            self.anti_hebb_learning_rate = self.learning_rate
        else:
            self.anti_hebb_learning_rate = anti_hebb_learning_rate
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]
        projection.post.__class__.plateau_history = property(lambda self: self.get_attribute_history('plateau'))
        projection.post.__class__.backward_activity_history = \
            property(lambda self: self.get_attribute_history('backward_activity'))
        projection.post.__class__.forward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('forward_dendritic_state'))
        projection.post.__class__.backward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('backward_dendritic_state'))
        projection.post.__class__.ET_history = property(lambda self: self.get_attribute_history('ET'))
        projection.post.__class__.IS_history = property(lambda self: self.get_attribute_history('IS'))

    def reinit(self):
        self.projection.pre.ET = torch.zeros(self.projection.pre.size, device=self.projection.pre.network.device,
                                             requires_grad=False)
        self.projection.post.IS = torch.zeros(self.projection.post.size, device=self.projection.post.network.device,
                                              requires_grad=False)
        self.projection.pre.ET_updated = False
        self.projection.post.IS_updated = False

    def update(self):
        self.projection.pre.ET *= (1. - 1./self.decay_tau)
        self.projection.post.IS *= (1. - 1./self.decay_tau)
        self.projection.pre.ET_updated = False
        self.projection.post.IS_updated = False

    def step(self):
        # BTSP
        ET_IS = torch.outer(self.projection.post.IS, self.projection.pre.ET)
        delta_weight = self.learning_rate * (
                (self.w_max - self.projection.weight) * ET_IS -
                self.projection.weight * self.dep_ratio * self.q_dep(ET_IS))
        self.projection.weight.data += delta_weight

        # anti-Hebbian depression
        post_activity = torch.zeros(self.projection.post.size, device=self.projection.post.network.device,
                                    requires_grad=False)
        meets_criterion = self.projection.post.meets_BTSP_anti_hebb_criterion
        post_activity[meets_criterion] = self.projection.post.activity[meets_criterion]
        # TODO: Should these activities be clamped?
        delta_weight = -self.anti_hebb_learning_rate * torch.outer(post_activity, self.projection.pre.activity)
        self.projection.weight.data += delta_weight

    @classmethod
    def backward_update_activity(cls, network, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for layer in network:
            for post_pop in layer:
                post_pop.prev_activity = post_pop.activity
        for layer in network:
            for post_pop in layer:
                if post_pop.backward_projections or post_pop is post_pop.network.output_pop:
                    # update somatic state and activity
                    delta_state = -post_pop.state + post_pop.bias
                    for projection in post_pop:
                        pre_pop = projection.pre
                        if projection.compartment in [None, 'soma']:
                            if projection.direction in ['forward', 'F']:
                                delta_state = delta_state + projection(pre_pop.activity)
                            elif projection.direction in ['recurrent', 'R']:
                                delta_state = delta_state + projection(pre_pop.prev_activity)
                    post_pop.state = post_pop.state + delta_state / post_pop.tau
                    post_pop.activity = post_pop.activation(post_pop.state)
                    if hasattr(post_pop, 'dend_to_soma'):
                        post_pop.activity = post_pop.activity + post_pop.dend_to_soma
                    if store_dynamics:
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
                            post_pop.dendritic_state = torch.zeros(post_pop.size, device=post_pop.network.device,
                                                                   requires_grad=False)
                            init_dend_state = True
                        if projection.direction in ['forward', 'F']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.activity)
                        elif projection.direction in ['recurrent', 'R']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.prev_activity)
                if init_dend_state:
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)

    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
        """
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_pop = network.output_pop

        # compute the forward_dendritic_state before comparing output to target
        output_pop.dendritic_state = torch.zeros(output_pop.size, device=network.device, requires_grad=False)
        for layer in reversed_layers:
            cls.backward_update_layer_dendritic_state(layer)
            # initialize populations that are updated during the backward phase
            for pop in layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.backward_steps_activity = []
                # store the forward_activity before comparing output to target
                pop.forward_activity = pop.activity.detach().clone()
                # initialize dendritic state variables
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        pop.plateau = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.dend_to_soma = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.forward_dendritic_state = pop.dendritic_state.detach().clone()
                        if store_history:
                            # store the forward_dendritic_state
                            pop.append_attribute_history('forward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break

        for t in range(network.forward_steps):
            for layer in reversed_layers:
                # update dendritic state variables
                cls.backward_update_layer_dendritic_state(layer)
                for pop in layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                output_pop.dendritic_state = torch.clamp(target - output_pop.activity, min=-1, max=1)
                                pos_indexes = (output_pop.dendritic_state >
                                               projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)
                                output_pop.plateau[pos_indexes] = 1.  # output_pop.dendritic_state[pos_indexes]
                                output_pop.dend_to_soma[pos_indexes] = 1.  # output_pop.dendritic_state[pos_indexes]
                            else:
                                sorted, sorted_indexes = torch.sort(pop.dendritic_state, descending=True,
                                                                    stable=True)
                                candidate_indexes = \
                                    ((pop.dendritic_state[sorted_indexes] > projection.learning_rule.pos_loss_th) &
                                     (pop.plateau[sorted_indexes] == 0)).nonzero().squeeze(1)
                                if len(candidate_indexes) > 0:
                                    plateau_index = sorted_indexes[candidate_indexes][0]
                                    pop.plateau[plateau_index] = 1.
                                    pop.dend_to_soma[plateau_index] = 1.  # pop.dendritic_state[pos_plateau_indexes]
                            break
            # update activities
            cls.backward_update_activity(network, store_dynamics=store_dynamics)

        for layer in network:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        if not pop.IS_updated:
                            pop.IS += pop.plateau
                            pop.IS.clamp_(0., 1.)
                            pop.IS_updated = True
                            if pop is output_pop:
                                pop.meets_BTSP_anti_hebb_criterion = \
                                    ((pop.plateau == 0) &
                                     (pop.dendritic_state < -projection.learning_rule.anti_hebb_th[1]))
                            else:
                                pop.meets_BTSP_anti_hebb_criterion = \
                                    ((pop.plateau == 0) &
                                     (pop.activity > 0) &
                                     (pop.dendritic_state > projection.learning_rule.anti_hebb_th[0]) &
                                     (pop.dendritic_state <= projection.learning_rule.anti_hebb_th[1]))
                            if store_history:
                                pop.append_attribute_history('IS', pop.IS.detach().clone())
                                pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                                pop.append_attribute_history('backward_dendritic_state',
                                                             pop.dendritic_state.detach().clone())
                        if not projection.pre.ET_updated:
                            projection.pre.ET += projection.pre.activity
                            projection.pre.ET.clamp_(0., 1.)
                            projection.pre.ET_updated = True
                            if store_history:
                                projection.pre.append_attribute_history('ET', projection.pre.ET.detach().clone())
                if pop.backward_projections or pop is output_pop:
                    if store_history:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_11(LearningRule):
    def __init__(self, projection, pos_loss_th=2.440709E-01, decay_tau=2, dep_ratio=1., dep_th=0.01, dep_width=0.01,
                 anti_hebb_th=(-0.1, 0.1), learning_rate=None, anti_hebb_learning_rate=None, refractory=1):
        """
        Similar to variant 10. Anti-Hebbian depression is triggered when an active unit has a dendritic state that
        is between a min and max bound. Cells cannot plateau within a refractory period.
        :param projection: :class:'nn.Linear'
        :param pos_loss_th: float
        :param decay_tau: float
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param anti_hebb_th: tuple of float
        :param learning_rate: float
        :param anti_hebb_learning_rate: float
        :param refractory: int
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.pos_loss_th = pos_loss_th
        self.decay_tau = decay_tau
        self.dep_ratio = dep_ratio
        self.anti_hebb_th = anti_hebb_th
        if anti_hebb_learning_rate is None:
            self.anti_hebb_learning_rate = self.learning_rate
        else:
            self.anti_hebb_learning_rate = anti_hebb_learning_rate
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]
        self.refractory = refractory
        projection.post.__class__.plateau_history = property(lambda self: self.get_attribute_history('plateau'))
        projection.post.__class__.backward_activity_history = \
            property(lambda self: self.get_attribute_history('backward_activity'))
        projection.post.__class__.forward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('forward_dendritic_state'))
        projection.post.__class__.backward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('backward_dendritic_state'))
        projection.post.__class__.ET_history = property(lambda self: self.get_attribute_history('ET'))
        projection.post.__class__.IS_history = property(lambda self: self.get_attribute_history('IS'))

    def reinit(self):
        self.projection.pre.ET = torch.zeros(self.projection.pre.size, device=self.projection.pre.network.device,
                                             requires_grad=False)
        self.projection.post.IS = torch.zeros(self.projection.post.size, device=self.projection.post.network.device,
                                              requires_grad=False)
        self.projection.post.plateau_refractory = (
            torch.zeros(self.projection.post.size, device=self.projection.post.network.device, requires_grad=False))
        self.projection.pre.ET_incremented = False
        self.projection.post.IS_incremented = False
        self.projection.pre.ET_decayed = False
        self.projection.post.IS_decayed = False

    def update(self):
        if not self.projection.pre.ET_decayed:
            self.projection.pre.ET *= (1. - 1./self.decay_tau)
            self.projection.pre.ET_decayed = True
            self.projection.pre.ET_incremented = False
        if not self.projection.post.IS_decayed:
            self.projection.post.IS *= (1. - 1./self.decay_tau)
            self.projection.post.IS_decayed = True
            self.projection.post.IS_incremented = False
            self.projection.post.plateau_refractory[self.projection.post.plateau == 1] += 1
            self.projection.post.plateau_refractory[self.projection.post.plateau_refractory >= self.refractory] = 0

    def step(self):
        # BTSP
        ET_IS = torch.outer(self.projection.post.IS, self.projection.pre.ET)
        delta_weight = self.learning_rate * (
                (self.w_max - self.projection.weight) * ET_IS -
                self.projection.weight * self.dep_ratio * self.q_dep(ET_IS))
        self.projection.weight.data += delta_weight

        # anti-Hebbian depression
        post_activity = torch.zeros(self.projection.post.size, device=self.projection.post.network.device,
                                    requires_grad=False)
        meets_criterion = self.projection.post.meets_BTSP_anti_hebb_criterion
        post_activity[meets_criterion] = self.projection.post.activity[meets_criterion]
        # TODO: Should these activities be clamped?
        delta_weight = -self.anti_hebb_learning_rate * torch.outer(post_activity, self.projection.pre.activity)
        self.projection.weight.data += delta_weight

    @classmethod
    def backward_update_activity(cls, network, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for layer in network:
            for post_pop in layer:
                post_pop.prev_activity = post_pop.activity
        for layer in network:
            for post_pop in layer:
                if post_pop.backward_projections or post_pop is post_pop.network.output_pop:
                    # update somatic state and activity
                    delta_state = -post_pop.state + post_pop.bias
                    for projection in post_pop:
                        pre_pop = projection.pre
                        if projection.compartment in [None, 'soma']:
                            if projection.direction in ['forward', 'F']:
                                delta_state = delta_state + projection(pre_pop.activity)
                            elif projection.direction in ['recurrent', 'R']:
                                delta_state = delta_state + projection(pre_pop.prev_activity)
                    post_pop.state = post_pop.state + delta_state / post_pop.tau
                    post_pop.activity = post_pop.activation(post_pop.state)
                    if hasattr(post_pop, 'dend_to_soma'):
                        post_pop.activity = post_pop.activity + post_pop.dend_to_soma
                    if store_dynamics:
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
                            post_pop.dendritic_state = torch.zeros(post_pop.size, device=post_pop.network.device,
                                                                   requires_grad=False)
                            init_dend_state = True
                        if projection.direction in ['forward', 'F']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.activity)
                        elif projection.direction in ['recurrent', 'R']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.prev_activity)
                if init_dend_state:
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)

    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
        """
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_pop = network.output_pop

        # compute the forward_dendritic_state before comparing output to target
        output_pop.dendritic_state = torch.zeros(output_pop.size, device=network.device, requires_grad=False)
        for layer in reversed_layers:
            cls.backward_update_layer_dendritic_state(layer)
            # initialize populations that are updated during the backward phase
            for pop in layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.backward_steps_activity = []
                # store the forward_activity before comparing output to target
                pop.forward_activity = pop.activity.detach().clone()
                # initialize dendritic state variables
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        pop.plateau = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.dend_to_soma = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.forward_dendritic_state = pop.dendritic_state.detach().clone()
                        if store_history:
                            # store the forward_dendritic_state
                            pop.append_attribute_history('forward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break

        for t in range(network.forward_steps):
            for layer in reversed_layers:
                # update dendritic state variables
                cls.backward_update_layer_dendritic_state(layer)
                for pop in layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                output_pop.dendritic_state = torch.clamp(target - output_pop.activity, min=-1, max=1)
                                pos_indexes = (output_pop.dendritic_state >
                                               projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)
                                output_pop.plateau[pos_indexes] = 1.  # output_pop.dendritic_state[pos_indexes]
                                output_pop.dend_to_soma[pos_indexes] = 1.  # output_pop.dendritic_state[pos_indexes]
                            else:
                                sorted, sorted_indexes = torch.sort(pop.dendritic_state, descending=True,
                                                                    stable=True)
                                candidate_indexes = \
                                    ((pop.dendritic_state[sorted_indexes] > projection.learning_rule.pos_loss_th) &
                                     (pop.plateau[sorted_indexes] == 0) &
                                     (pop.plateau_refractory < projection.learning_rule.refractory)).nonzero().squeeze(1)
                                if len(candidate_indexes) > 0:
                                    plateau_index = sorted_indexes[candidate_indexes][0]
                                    pop.plateau[plateau_index] = 1.
                                    pop.dend_to_soma[plateau_index] = 1.  # pop.dendritic_state[pos_plateau_indexes]
                            break
            # update activities
            cls.backward_update_activity(network, store_dynamics=store_dynamics)

        for layer in network:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        if not pop.IS_incremented:
                            pop.IS += pop.plateau
                            pop.IS.clamp_(0., 1.)
                            pop.IS_incremented = True
                            pop.IS_decayed = False
                            if pop is output_pop:
                                pop.meets_BTSP_anti_hebb_criterion = \
                                    ((pop.plateau == 0) &
                                     (pop.dendritic_state < -projection.learning_rule.anti_hebb_th[1]))
                            else:
                                pop.meets_BTSP_anti_hebb_criterion = \
                                    ((pop.plateau == 0) &
                                     (pop.activity > 0) &
                                     (pop.dendritic_state > projection.learning_rule.anti_hebb_th[0]) &
                                     (pop.dendritic_state <= projection.learning_rule.anti_hebb_th[1]))
                            if store_history:
                                pop.append_attribute_history('IS', pop.IS.detach().clone())
                                pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                                pop.append_attribute_history('backward_dendritic_state',
                                                             pop.dendritic_state.detach().clone())
                        if not projection.pre.ET_incremented:
                            projection.pre.ET += projection.pre.activity
                            projection.pre.ET.clamp_(0., 1.)
                            projection.pre.ET_incremented = True
                            projection.pre.ET_decayed = False
                            if store_history:
                                projection.pre.append_attribute_history('ET', projection.pre.ET.detach().clone())
                if pop.backward_projections or pop is output_pop:
                    if store_history:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_12_cont(LearningRule):
    def __init__(self, projection, dep_ratio=1., dep_th=0.01, dep_width=0.01, learning_rate=None):
        """
        This variant is a stripped-down version of BTSP with no thresholds for mod events. Negative mod events
        evoke a linear weight-dependent depression.
        :param projection: :class:'nn.Linear'
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.dep_ratio = dep_ratio
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]
        projection.post.__class__.plateau_history = property(lambda self: self.get_attribute_history('plateau'))
        projection.post.__class__.backward_activity_history = \
            property(lambda self: self.get_attribute_history('backward_activity'))
        projection.post.__class__.forward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('forward_dendritic_state'))
        projection.post.__class__.backward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('backward_dendritic_state'))

    def step(self):
        # BTSP
        plateau = self.projection.post.plateau
        IS = plateau.detach().clone()
        IS[plateau < 0.] = 0.
        IS = IS.unsqueeze(1)
        ET = torch.clamp(self.projection.pre.activity, 0., 1.)
        unsqueezed_ET = ET.unsqueeze(0)
        delta_weight = IS * ((self.w_max - self.projection.weight) * unsqueezed_ET -
                             self.projection.weight * self.dep_ratio * self.q_dep(ET).unsqueeze(0))
        self.projection.weight.data += self.learning_rate * delta_weight
        
        # negative dendritic state triggers linear weight-dependent depression
        neg_error = plateau.detach().clone()
        neg_error[plateau > 0.] = 0.
        neg_error = neg_error.unsqueeze(1)
        delta_weight = neg_error * self.projection.weight * unsqueezed_ET
        self.projection.weight.data += self.learning_rate * delta_weight

    @classmethod
    def backward_update_activity(cls, network, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for layer in network:
            for post_pop in layer:
                post_pop.prev_activity = post_pop.activity
        for layer in network:
            for post_pop in layer:
                if post_pop.backward_projections or post_pop is post_pop.network.output_pop:
                    # update somatic state and activity
                    delta_state = -post_pop.state + post_pop.bias
                    for projection in post_pop:
                        pre_pop = projection.pre
                        if projection.compartment in [None, 'soma']:
                            if projection.direction in ['forward', 'F']:
                                delta_state = delta_state + projection(pre_pop.activity)
                            elif projection.direction in ['recurrent', 'R']:
                                delta_state = delta_state + projection(pre_pop.prev_activity)
                    if hasattr(post_pop, 'dend_to_soma'):
                        delta_state = delta_state + post_pop.dend_to_soma
                    post_pop.state = post_pop.state + delta_state / post_pop.tau
                    post_pop.activity = post_pop.activation(post_pop.state)
                    if store_dynamics:
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
                            post_pop.dendritic_state = torch.zeros(post_pop.size, device=post_pop.network.device,
                                                                   requires_grad=False)
                            init_dend_state = True
                        if projection.direction in ['forward', 'F']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.activity)
                        elif projection.direction in ['recurrent', 'R']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.prev_activity)
                if init_dend_state:
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)

    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
        """
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_pop = network.output_pop

        # compute the forward_dendritic_state before comparing output to target
        output_pop.dendritic_state = torch.zeros(output_pop.size, device=network.device, requires_grad=False)
        for layer in reversed_layers:
            cls.backward_update_layer_dendritic_state(layer)
            # initialize populations that are updated during the backward phase
            for pop in layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.backward_steps_activity = []
                # store the forward_activity before comparing output to target
                pop.forward_activity = pop.activity.detach().clone()
                # initialize dendritic state variables
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        pop.plateau = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.dend_to_soma = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.forward_dendritic_state = pop.dendritic_state.detach().clone()
                        if store_history:
                            # store the forward_dendritic_state
                            pop.append_attribute_history('forward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break

        for t in range(network.forward_steps):
            for layer in reversed_layers:
                # update dendritic state variables
                cls.backward_update_layer_dendritic_state(layer)
                for pop in layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                if t == 0:
                                    output_pop.dendritic_state = (
                                        torch.clamp(target - output_pop.activity, min=-1, max=1))
                                    output_pop.plateau = output_pop.dendritic_state.detach().clone()
                                    output_pop.dend_to_soma = output_pop.dendritic_state.detach().clone()
                            else:
                                pop.plateau = pop.dendritic_state.detach().clone()
                                pop.dend_to_soma = pop.dendritic_state.detach().clone()
                            break
            # update activities
            cls.backward_update_activity(network, store_dynamics=store_dynamics)

        for layer in network:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        if store_history:
                            pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                            pop.append_attribute_history('backward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break
                if pop.backward_projections or pop is output_pop:
                    if store_history:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_13(LearningRule):
    def __init__(self, projection, pos_loss_th=2.440709E-01, neg_loss_th=-4.592181E-01, neg_loss_ET_discount=0.25,
                 dep_ratio=1., dep_th=0.01, dep_width=0.01, max_pop_fraction=0.025, learning_rate=None):
        """
        Like the original BTSP class, this method includes both positive and negative modulatory events. In this
        variant, both positive and negative modulatory events nudge the somatic activity. Plateaus can only occur in a
        specified maximum fraction of units in each layer. All layers are equilibrated simultaneously.
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
        projection.post.__class__.plateau_history = property(lambda self: self.get_attribute_history('plateau'))
        projection.post.__class__.backward_activity_history = \
            property(lambda self: self.get_attribute_history('backward_activity'))
        projection.post.__class__.forward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('forward_dendritic_state'))
        projection.post.__class__.backward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('backward_dendritic_state'))
    
    def step(self):
        plateau = self.projection.post.plateau
        discount = plateau.detach().clone()
        discount[plateau > 0] = 1.
        discount[plateau < 0] = self.neg_loss_ET_discount
        IS = torch.abs(plateau).unsqueeze(1)
        ET = torch.outer(discount, self.projection.pre.activity)
        delta_weight = IS * ((self.w_max - self.projection.weight) * ET -
                             self.projection.weight * self.dep_ratio * self.q_dep(ET))
        self.projection.weight.data += self.learning_rate * delta_weight
    
    @classmethod
    def backward_update_activity(cls, network, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for layer in network:
            for post_pop in layer:
                post_pop.prev_activity = post_pop.activity
        for layer in network:
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
                    if store_dynamics:
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
                            post_pop.dendritic_state = torch.zeros(post_pop.size, device=post_pop.network.device,
                                                                   requires_grad=False)
                            init_dend_state = True
                        if projection.direction in ['forward', 'F']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.activity)
                        elif projection.direction in ['recurrent', 'R']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.prev_activity)
                if init_dend_state:
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
        """
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_pop = network.output_pop
        
        # compute the forward_dendritic_state before comparing output to target
        output_pop.dendritic_state = torch.zeros(output_pop.size, device=network.device, requires_grad=False)
        for layer in reversed_layers:
            cls.backward_update_layer_dendritic_state(layer)
            # initialize populations that are updated during the backward phase
            for pop in layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.backward_steps_activity = []
                # store the forward_activity before comparing output to target
                pop.forward_activity = pop.activity.detach().clone()
                # initialize dendritic state variables
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        pop.plateau = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.dend_to_soma = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.forward_dendritic_state = pop.dendritic_state.detach().clone()
                        if store_history:
                            # store the forward_dendritic_state
                            pop.append_attribute_history('forward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break
        
        # compute plateau events and nudge somatic state
        for t in range(network.forward_steps):
            for layer in reversed_layers:
                # update dendritic state variables
                cls.backward_update_layer_dendritic_state(layer)
                for pop in layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                pop.dendritic_state = torch.clamp(target - output_pop.activity, min=-1, max=1)
                                avail_indexes = (pop.plateau == 0).nonzero().squeeze(1)
                                pos_indexes = (pop.dendritic_state[avail_indexes] >
                                               projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)
                                pos_avail_indexes = avail_indexes[pos_indexes]
                                neg_indexes = (pop.dendritic_state[avail_indexes] <
                                               projection.learning_rule.neg_loss_th).nonzero(as_tuple=True)
                                neg_avail_indexes = avail_indexes[neg_indexes]
                            else:
                                max_units = math.ceil(projection.learning_rule.max_pop_fraction * pop.size)
                                avail_indexes = (pop.plateau == 0).nonzero().squeeze(1)
                                pos_remaining = max_units - (pop.plateau > 0).count_nonzero()
                                neg_remaining = max_units - (pop.plateau < 0).count_nonzero()
                                if pos_remaining > 0 or neg_remaining > 0:
                                    sorted, sorted_indexes = torch.sort(pop.dendritic_state[avail_indexes],
                                                                        descending=True, stable=True)
                                    sorted_avail_indexes = avail_indexes[sorted_indexes]
                                if pos_remaining > 0:
                                    pos_indexes = (pop.dendritic_state[sorted_avail_indexes] >
                                                   projection.learning_rule.pos_loss_th).nonzero().squeeze(1)
                                    pos_avail_indexes = sorted_avail_indexes[pos_indexes][:pos_remaining]
                                else:
                                    pos_avail_indexes = []
                                if neg_remaining > 0:
                                    neg_indexes = (pop.dendritic_state[sorted_avail_indexes] <
                                                   projection.learning_rule.neg_loss_th).nonzero().squeeze(1)
                                    neg_avail_indexes = sorted_avail_indexes[neg_indexes][-neg_remaining:]
                                else:
                                    neg_avail_indexes = []
                            pop.plateau[pos_avail_indexes] = pop.dendritic_state[pos_avail_indexes]
                            pop.dend_to_soma[pos_avail_indexes] = pop.dendritic_state[pos_avail_indexes]
                            pop.plateau[neg_avail_indexes] = pop.dendritic_state[neg_avail_indexes]
                            pop.dend_to_soma[neg_avail_indexes] = pop.dendritic_state[neg_avail_indexes]
                            break
            # update activities
            cls.backward_update_activity(network, store_dynamics=store_dynamics)
        
        if store_history:
            for layer in network:
                for pop in layer:
                    for projection in pop:
                        if projection.learning_rule.__class__ == cls:
                            if store_history:
                                pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                                pop.append_attribute_history('backward_dendritic_state',
                                                             pop.dendritic_state.detach().clone())
                            break
                    if pop.backward_projections or pop is output_pop:
                        if store_history:
                            if store_dynamics:
                                pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                            else:
                                pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_14(LearningRule):
    def __init__(self, projection, pos_loss_th=2.440709E-01, neg_loss_th=-4.592181E-01, dep_ratio=1., dep_th=0.01,
                 dep_width=0.01, max_pop_fraction=0.025, learning_rate=None):
        """
        Like the original BTSP class, this method includes both positive and negative modulatory events. In this
        variant, both positive and negative modulatory events nudge the somatic activity. Plateaus can only occur in a
        specified maximum fraction of units in each layer. All layers are equilibrated simultaneously.
        Nudges to somatic state are applied instantaneously, rather than being subject to slow equilibration.
        Negative modulatory events result in a weight update proportional to dendritic state and presynaptic activity.
        :param projection: :class:'nn.Linear'
        :param pos_loss_th: float
        :param neg_loss_th: float
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
        self.dep_ratio = dep_ratio
        self.max_pop_fraction = max_pop_fraction
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]
        projection.post.__class__.plateau_history = property(lambda self: self.get_attribute_history('plateau'))
        projection.post.__class__.backward_activity_history = \
            property(lambda self: self.get_attribute_history('backward_activity'))
        projection.post.__class__.forward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('forward_dendritic_state'))
        projection.post.__class__.backward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('backward_dendritic_state'))
    
    def step(self):
        # pos error - BTSP weight update
        plateau = self.projection.post.plateau
        IS = plateau.detach().clone()
        IS[plateau < 0.] = 0.
        IS = IS.unsqueeze(1)
        ET = torch.clamp(self.projection.pre.activity, 0., 1.)
        delta_weight = IS * ((self.w_max - self.projection.weight) * ET.unsqueeze(0) -
                             self.projection.weight * self.dep_ratio * self.q_dep(ET).unsqueeze(0))
        self.projection.weight.data += self.learning_rate * delta_weight
        
        # neg error - weight update proportional to loss and presynaptic activity
        neg_error = plateau.detach().clone()
        neg_error[plateau > 0.] = 0.
        delta_weight = torch.outer(neg_error, ET)
        self.projection.weight.data += self.learning_rate * delta_weight
    
    @classmethod
    def backward_update_activity(cls, network, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for layer in network:
            for post_pop in layer:
                post_pop.prev_activity = post_pop.activity
        for layer in network:
            for post_pop in layer:
                if post_pop.backward_projections or post_pop is post_pop.network.output_pop:
                    # update somatic state and activity
                    delta_state = -post_pop.state + post_pop.bias
                    for projection in post_pop:
                        pre_pop = projection.pre
                        if projection.compartment in [None, 'soma']:
                            if projection.direction in ['forward', 'F']:
                                delta_state = delta_state + projection(pre_pop.activity)
                            elif projection.direction in ['recurrent', 'R']:
                                delta_state = delta_state + projection(pre_pop.prev_activity)
                    post_pop.state = post_pop.state + delta_state / post_pop.tau
                    if hasattr(post_pop, 'dend_to_soma'):
                        post_pop.activity = post_pop.activation(post_pop.state + post_pop.dend_to_soma)
                    else:
                        post_pop.activity = post_pop.activation(post_pop.state)
                    if store_dynamics:
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
                            post_pop.dendritic_state = torch.zeros(post_pop.size, device=post_pop.network.device,
                                                                   requires_grad=False)
                            init_dend_state = True
                        if projection.direction in ['forward', 'F']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.activity)
                        elif projection.direction in ['recurrent', 'R']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.prev_activity)
                if init_dend_state:
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
        """
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_pop = network.output_pop
        
        # compute the forward_dendritic_state before comparing output to target
        output_pop.dendritic_state = torch.zeros(output_pop.size, device=network.device, requires_grad=False)
        for layer in reversed_layers:
            cls.backward_update_layer_dendritic_state(layer)
            # initialize populations that are updated during the backward phase
            for pop in layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.backward_steps_activity = []
                # store the forward_activity before comparing output to target
                pop.forward_activity = pop.activity.detach().clone()
                # initialize dendritic state variables
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        pop.plateau = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.dend_to_soma = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.forward_dendritic_state = pop.dendritic_state.detach().clone()
                        if store_history:
                            # store the forward_dendritic_state
                            pop.append_attribute_history('forward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break
        
        # compute plateau events and nudge somatic state
        for t in range(network.forward_steps):
            for layer in reversed_layers:
                # update dendritic state variables
                cls.backward_update_layer_dendritic_state(layer)
                for pop in layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                if t == 0:
                                    output_pop.dendritic_state = (
                                        torch.clamp(target - output_pop.activity, min=-1, max=1))
                                    output_pop.plateau = output_pop.dendritic_state.detach().clone()
                                    output_pop.dend_to_soma = output_pop.dendritic_state.detach().clone()
                            else:
                                max_units = math.ceil(projection.learning_rule.max_pop_fraction * pop.size)
                                avail_indexes = (pop.plateau == 0).nonzero().squeeze(1)
                                if len(avail_indexes) > 0:
                                    pos_remaining = max_units - (pop.plateau > 0).count_nonzero()
                                    neg_remaining = max_units - (pop.plateau < 0).count_nonzero()
                                    if pos_remaining > 0 or neg_remaining > 0:
                                        sorted, sorted_indexes = torch.sort(pop.dendritic_state[avail_indexes],
                                                                            descending=True, stable=True)
                                        sorted_avail_indexes = avail_indexes[sorted_indexes]
                                    if pos_remaining > 0:
                                        pos_indexes = (pop.dendritic_state[sorted_avail_indexes] >
                                                       projection.learning_rule.pos_loss_th).nonzero().squeeze(1)
                                        pos_avail_indexes = sorted_avail_indexes[pos_indexes][:pos_remaining]
                                    else:
                                        pos_avail_indexes = []
                                    if neg_remaining > 0:
                                        neg_indexes = (pop.dendritic_state[sorted_avail_indexes] <
                                                       projection.learning_rule.neg_loss_th).nonzero().squeeze(1)
                                        neg_avail_indexes = sorted_avail_indexes[neg_indexes][-neg_remaining:]
                                    else:
                                        neg_avail_indexes = []
                                    pop.plateau[pos_avail_indexes] = pop.dendritic_state[pos_avail_indexes]
                                    pop.dend_to_soma[pos_avail_indexes] = pop.dendritic_state[pos_avail_indexes]
                                    pop.plateau[neg_avail_indexes] = pop.dendritic_state[neg_avail_indexes]
                                    pop.dend_to_soma[neg_avail_indexes] = pop.dendritic_state[neg_avail_indexes]
                            break
            # update activities
            cls.backward_update_activity(network, store_dynamics=store_dynamics)
        
        if store_history:
            for layer in network:
                for pop in layer:
                    for projection in pop:
                        if projection.learning_rule.__class__ == cls:
                            if store_history:
                                pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                                pop.append_attribute_history('backward_dendritic_state',
                                                             pop.dendritic_state.detach().clone())
                            break
                    if pop.backward_projections or pop is output_pop:
                        if store_history:
                            if store_dynamics:
                                pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                            else:
                                pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BP_like_1(LearningRule):
    def __init__(self, projection, learning_rate=None):
        """
        Output units are nudged to target. Hidden dendrites locally compute an error as the difference between before
        and after nudge. Weight updates are proportional to local error and presynaptic firing rate.
        :param projection: :class:'nn.Linear'
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        projection.post.__class__.plateau_history = property(lambda self: self.get_attribute_history('plateau'))
        projection.post.__class__.backward_activity_history = \
            property(lambda self: self.get_attribute_history('backward_activity'))
        projection.post.__class__.forward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('forward_dendritic_state'))
        projection.post.__class__.backward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('backward_dendritic_state'))
    
    def step(self):
        self.projection.weight.data += \
            (self.learning_rate * torch.outer(torch.clamp(self.projection.post.dendritic_state -
                                              self.projection.post.forward_dendritic_state, min=-1, max=1),
                                              torch.clamp(self.projection.pre.activity, min=0, max=1)))
        
    @classmethod
    def backward_update_activity(cls, network, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for layer in network:
            for post_pop in layer:
                post_pop.prev_activity = post_pop.activity
        for layer in network:
            for post_pop in layer:
                if post_pop.backward_projections or post_pop is post_pop.network.output_pop:
                    # update somatic state and activity
                    delta_state = -post_pop.state + post_pop.bias
                    for projection in post_pop:
                        pre_pop = projection.pre
                        if projection.compartment in [None, 'soma']:
                            if projection.direction in ['forward', 'F']:
                                delta_state = delta_state + projection(pre_pop.activity)
                            elif projection.direction in ['recurrent', 'R']:
                                delta_state = delta_state + projection(pre_pop.prev_activity)
                    if hasattr(post_pop, 'dend_to_soma'):
                        delta_state = delta_state + post_pop.dend_to_soma
                    post_pop.state = post_pop.state + delta_state / post_pop.tau
                    post_pop.activity = post_pop.activation(post_pop.state)
                    if store_dynamics:
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
                            post_pop.dendritic_state = torch.zeros(post_pop.size, device=post_pop.network.device,
                                                                   requires_grad=False)
                            init_dend_state = True
                        if projection.direction in ['forward', 'F']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.activity)
                        elif projection.direction in ['recurrent', 'R']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.prev_activity)
                if init_dend_state:
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
        """
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_pop = network.output_pop
        
        # compute the forward_dendritic_state before comparing output to target
        output_pop.dendritic_state = torch.zeros(output_pop.size, device=network.device, requires_grad=False)
        for layer in reversed_layers:
            cls.backward_update_layer_dendritic_state(layer)
            # initialize populations that are updated during the backward phase
            for pop in layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.backward_steps_activity = []
                # store the forward_activity before comparing output to target
                pop.forward_activity = pop.activity.detach().clone()
                # initialize dendritic state variables
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        pop.plateau = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.dend_to_soma = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.forward_dendritic_state = pop.dendritic_state.detach().clone()
                        if store_history:
                            # store the forward_dendritic_state
                            pop.append_attribute_history('forward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break
        
        for t in range(network.forward_steps):
            for layer in reversed_layers:
                # update dendritic state variables
                cls.backward_update_layer_dendritic_state(layer)
                for pop in layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                if t == 0:
                                    output_pop.dendritic_state = (
                                        torch.clamp(target - output_pop.activity, min=-1, max=1))
                                    output_pop.plateau = output_pop.dendritic_state.detach().clone()
                                    output_pop.dend_to_soma = output_pop.dendritic_state.detach().clone()
                                else:
                                    output_pop.dendritic_state = output_pop.plateau.detach().clone()
                            break
            # update activities
            cls.backward_update_activity(network, store_dynamics=store_dynamics)
        
        for layer in network:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        if store_history:
                            pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                            pop.append_attribute_history('backward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break
                if pop.backward_projections or pop is output_pop:
                    if store_history:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BP_like_1C(LearningRule):
    def __init__(self, projection, learning_rate=None):
        """
        Output units are nudged to target. Hidden dendrites locally compute an error as the difference between before
        and after nudge. Weight updates are proportional to local error and presynaptic firing rate.
        Nudges to somatic state are applied instantaneously, rather than being subject to slow equilibration.
        :param projection: :class:'nn.Linear'
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        projection.post.__class__.plateau_history = property(lambda self: self.get_attribute_history('plateau'))
        projection.post.__class__.backward_activity_history = \
            property(lambda self: self.get_attribute_history('backward_activity'))
        projection.post.__class__.forward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('forward_dendritic_state'))
        projection.post.__class__.backward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('backward_dendritic_state'))
    
    def step(self):
        self.projection.weight.data += \
            (self.learning_rate * torch.outer(torch.clamp(self.projection.post.dendritic_state -
                                                          self.projection.post.forward_dendritic_state, min=-1, max=1),
                                              torch.clamp(self.projection.pre.activity, min=0, max=1)))
    
    @classmethod
    def backward_update_activity(cls, network, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for layer in network:
            for post_pop in layer:
                post_pop.prev_activity = post_pop.activity
        for layer in network:
            for post_pop in layer:
                if post_pop.backward_projections or post_pop is post_pop.network.output_pop:
                    # update somatic state and activity
                    delta_state = -post_pop.state + post_pop.bias
                    for projection in post_pop:
                        pre_pop = projection.pre
                        if projection.compartment in [None, 'soma']:
                            if projection.direction in ['forward', 'F']:
                                delta_state = delta_state + projection(pre_pop.activity)
                            elif projection.direction in ['recurrent', 'R']:
                                delta_state = delta_state + projection(pre_pop.prev_activity)
                    post_pop.state = post_pop.state + delta_state / post_pop.tau
                    if hasattr(post_pop, 'dend_to_soma'):
                        post_pop.activity = post_pop.activation(post_pop.state + post_pop.dend_to_soma)
                    else:
                        post_pop.activity = post_pop.activation(post_pop.state)
                    if store_dynamics:
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
                            post_pop.dendritic_state = torch.zeros(post_pop.size, device=post_pop.network.device,
                                                                   requires_grad=False)
                            init_dend_state = True
                        if projection.direction in ['forward', 'F']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.activity)
                        elif projection.direction in ['recurrent', 'R']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.prev_activity)
                if init_dend_state:
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
        """
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_pop = network.output_pop
        
        # compute the forward_dendritic_state before comparing output to target
        output_pop.dendritic_state = torch.zeros(output_pop.size, device=network.device, requires_grad=False)
        for layer in reversed_layers:
            cls.backward_update_layer_dendritic_state(layer)
            # initialize populations that are updated during the backward phase
            for pop in layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.backward_steps_activity = []
                # store the forward_activity before comparing output to target
                pop.forward_activity = pop.activity.detach().clone()
                # initialize dendritic state variables
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        pop.plateau = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.dend_to_soma = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.forward_dendritic_state = pop.dendritic_state.detach().clone()
                        if store_history:
                            # store the forward_dendritic_state
                            pop.append_attribute_history('forward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break
        
        for t in range(network.forward_steps):
            for layer in reversed_layers:
                # update dendritic state variables
                cls.backward_update_layer_dendritic_state(layer)
                for pop in layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                if t == 0:
                                    output_pop.dendritic_state = (
                                        torch.clamp(target - output_pop.activity, min=-1, max=1))
                                    output_pop.plateau = output_pop.dendritic_state.detach().clone()
                                    output_pop.dend_to_soma = output_pop.dendritic_state.detach().clone()
                            break
            # update activities
            cls.backward_update_activity(network, store_dynamics=store_dynamics)
        
        for layer in network:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        if store_history:
                            pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                            pop.append_attribute_history('backward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break
                if pop.backward_projections or pop is output_pop:
                    if store_history:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BP_like_2(LearningRule):
    def __init__(self, projection, learning_rate=None):
        """
        Output units are nudged to target. Hidden dendrites locally compute an error as the difference between
        excitation and inhibition. Weight updates are proportional to local error and presynaptic firing rate.
        :param projection: :class:'nn.Linear'
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        projection.post.__class__.plateau_history = property(lambda self: self.get_attribute_history('plateau'))
        projection.post.__class__.backward_activity_history = \
            property(lambda self: self.get_attribute_history('backward_activity'))
        projection.post.__class__.forward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('forward_dendritic_state'))
        projection.post.__class__.backward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('backward_dendritic_state'))
    
    def step(self):
        self.projection.weight.data += \
            (self.learning_rate * torch.outer(self.projection.post.plateau,
                                              torch.clamp(self.projection.pre.activity, min=0, max=1)))
    
    @classmethod
    def backward_update_activity(cls, network, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for layer in network:
            for post_pop in layer:
                post_pop.prev_activity = post_pop.activity
        for layer in network:
            for post_pop in layer:
                if post_pop.backward_projections or post_pop is post_pop.network.output_pop:
                    # update somatic state and activity
                    delta_state = -post_pop.state + post_pop.bias
                    for projection in post_pop:
                        pre_pop = projection.pre
                        if projection.compartment in [None, 'soma']:
                            if projection.direction in ['forward', 'F']:
                                delta_state = delta_state + projection(pre_pop.activity)
                            elif projection.direction in ['recurrent', 'R']:
                                delta_state = delta_state + projection(pre_pop.prev_activity)
                    if hasattr(post_pop, 'dend_to_soma'):
                        delta_state = delta_state + post_pop.dend_to_soma
                    post_pop.state = post_pop.state + delta_state / post_pop.tau
                    post_pop.activity = post_pop.activation(post_pop.state)
                    if store_dynamics:
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
                            post_pop.dendritic_state = torch.zeros(post_pop.size, device=post_pop.network.device,
                                                                   requires_grad=False)
                            init_dend_state = True
                        if projection.direction in ['forward', 'F']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.activity)
                        elif projection.direction in ['recurrent', 'R']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.prev_activity)
                if init_dend_state:
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
        """
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_pop = network.output_pop
        
        # compute the forward_dendritic_state before comparing output to target
        output_pop.dendritic_state = torch.zeros(output_pop.size, device=network.device, requires_grad=False)
        for layer in reversed_layers:
            cls.backward_update_layer_dendritic_state(layer)
            # initialize populations that are updated during the backward phase
            for pop in layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.backward_steps_activity = []
                # store the forward_activity before comparing output to target
                pop.forward_activity = pop.activity.detach().clone()
                # initialize dendritic state variables
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        pop.plateau = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.dend_to_soma = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.forward_dendritic_state = pop.dendritic_state.detach().clone()
                        if store_history:
                            # store the forward_dendritic_state
                            pop.append_attribute_history('forward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break
        
        for t in range(network.forward_steps):
            for layer in reversed_layers:
                # update dendritic state variables
                cls.backward_update_layer_dendritic_state(layer)
                for pop in layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                if t == 0:
                                    output_pop.dendritic_state = (
                                        torch.clamp(target - output_pop.activity, min=-1, max=1))
                                    output_pop.plateau = output_pop.dendritic_state.detach().clone()
                                    output_pop.dend_to_soma = output_pop.dendritic_state.detach().clone()
                            else:
                                pop.plateau = pop.dendritic_state.detach().clone()
                            break
            # update activities
            cls.backward_update_activity(network, store_dynamics=store_dynamics)
        
        for layer in network:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        if store_history:
                            pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                            pop.append_attribute_history('backward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break
                if pop.backward_projections or pop is output_pop:
                    if store_history:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BP_like_3(LearningRule):
    def __init__(self, projection, max_pop_fraction=0.025, learning_rate=None):
        """
        Output units are nudged to target. Hidden dendrites locally compute an error as the difference between
        excitation and inhibition. Weight updates are proportional to local error and presynaptic firing rate. A fixed
        fraction of hidden units are updated at each train step.
        :param projection: :class:'nn.Linear'
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.max_pop_fraction = max_pop_fraction
        projection.post.__class__.plateau_history = property(lambda self: self.get_attribute_history('plateau'))
        projection.post.__class__.backward_activity_history = \
            property(lambda self: self.get_attribute_history('backward_activity'))
        projection.post.__class__.forward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('forward_dendritic_state'))
        projection.post.__class__.backward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('backward_dendritic_state'))
    
    def step(self):
        self.projection.weight.data += \
            (self.learning_rate * torch.outer(self.projection.post.plateau,
                                              torch.clamp(self.projection.pre.activity, min=0, max=1)))
    
    @classmethod
    def backward_update_activity(cls, network, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for layer in network:
            for post_pop in layer:
                post_pop.prev_activity = post_pop.activity
        for layer in network:
            for post_pop in layer:
                if post_pop.backward_projections or post_pop is post_pop.network.output_pop:
                    # update somatic state and activity
                    delta_state = -post_pop.state + post_pop.bias
                    for projection in post_pop:
                        pre_pop = projection.pre
                        if projection.compartment in [None, 'soma']:
                            if projection.direction in ['forward', 'F']:
                                delta_state = delta_state + projection(pre_pop.activity)
                            elif projection.direction in ['recurrent', 'R']:
                                delta_state = delta_state + projection(pre_pop.prev_activity)
                    if hasattr(post_pop, 'dend_to_soma'):
                        delta_state = delta_state + post_pop.dend_to_soma
                    post_pop.state = post_pop.state + delta_state / post_pop.tau
                    post_pop.activity = post_pop.activation(post_pop.state)
                    if store_dynamics:
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
                            post_pop.dendritic_state = torch.zeros(post_pop.size, device=post_pop.network.device,
                                                                   requires_grad=False)
                            init_dend_state = True
                        if projection.direction in ['forward', 'F']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.activity)
                        elif projection.direction in ['recurrent', 'R']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.prev_activity)
                if init_dend_state:
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
        """
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_pop = network.output_pop
        
        # compute the forward_dendritic_state before comparing output to target
        output_pop.dendritic_state = torch.zeros(output_pop.size, device=network.device, requires_grad=False)
        for layer in reversed_layers:
            cls.backward_update_layer_dendritic_state(layer)
            # initialize populations that are updated during the backward phase
            for pop in layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.backward_steps_activity = []
                # store the forward_activity before comparing output to target
                pop.forward_activity = pop.activity.detach().clone()
                # initialize dendritic state variables
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        pop.plateau = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.dend_to_soma = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.forward_dendritic_state = pop.dendritic_state.detach().clone()
                        if store_history:
                            # store the forward_dendritic_state
                            pop.append_attribute_history('forward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break
        
        for t in range(network.forward_steps):
            for layer in reversed_layers:
                # update dendritic state variables
                cls.backward_update_layer_dendritic_state(layer)
                for pop in layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                if t == 0:
                                    output_pop.dendritic_state = (
                                        torch.clamp(target - output_pop.activity, min=-1, max=1))
                                    output_pop.plateau = output_pop.dendritic_state.detach().clone()
                                    output_pop.dend_to_soma = output_pop.dendritic_state.detach().clone()
                            else:
                                if t == network.forward_steps - 1:
                                    max_units = math.ceil(projection.learning_rule.max_pop_fraction * pop.size)
                                    sorted, sorted_indexes = torch.sort(pop.dendritic_state, descending=True,
                                                                        stable=True)
                                    pos_indexes = sorted_indexes[:max_units]
                                    pop.plateau[pos_indexes] = pop.dendritic_state[pos_indexes]
                                    neg_indexes = sorted_indexes[-max_units:]
                                    pop.plateau[neg_indexes] = pop.dendritic_state[neg_indexes]
                            break
            # update activities
            cls.backward_update_activity(network, store_dynamics=store_dynamics)
        
        for layer in network:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        if store_history:
                            pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                            pop.append_attribute_history('backward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break
                if pop.backward_projections or pop is output_pop:
                    if store_history:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BP_like_4(LearningRule):
    def __init__(self, projection, pos_loss_th=0.2, neg_loss_th=-0.2, max_pop_fraction=0.025, learning_rate=None):
        """
        Output units are nudged to target. Hidden dendrites locally compute an error as the difference between
        excitation and inhibition. Weight updates are proportional to local error and presynaptic firing rate. A fixed
        fraction of hidden units are updated at each train step. Only hidden units with error beyond a threshold are
        updated. Hidden units are "nudged" by dendritic state when beyond threshold.
        :param projection: :class:'nn.Linear'
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.max_pop_fraction = max_pop_fraction
        self.pos_loss_th = pos_loss_th
        self.neg_loss_th = neg_loss_th
        projection.post.__class__.plateau_history = property(lambda self: self.get_attribute_history('plateau'))
        projection.post.__class__.backward_activity_history = \
            property(lambda self: self.get_attribute_history('backward_activity'))
        projection.post.__class__.forward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('forward_dendritic_state'))
        projection.post.__class__.backward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('backward_dendritic_state'))
    
    def step(self):
        self.projection.weight.data += \
            (self.learning_rate * torch.outer(self.projection.post.plateau,
                                              torch.clamp(self.projection.pre.activity, min=0, max=1)))
    
    @classmethod
    def backward_update_activity(cls, network, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for layer in network:
            for post_pop in layer:
                post_pop.prev_activity = post_pop.activity
        for layer in network:
            for post_pop in layer:
                if post_pop.backward_projections or post_pop is post_pop.network.output_pop:
                    # update somatic state and activity
                    delta_state = -post_pop.state + post_pop.bias
                    for projection in post_pop:
                        pre_pop = projection.pre
                        if projection.compartment in [None, 'soma']:
                            if projection.direction in ['forward', 'F']:
                                delta_state = delta_state + projection(pre_pop.activity)
                            elif projection.direction in ['recurrent', 'R']:
                                delta_state = delta_state + projection(pre_pop.prev_activity)
                    if hasattr(post_pop, 'dend_to_soma'):
                        delta_state = delta_state + post_pop.dend_to_soma
                    post_pop.state = post_pop.state + delta_state / post_pop.tau
                    post_pop.activity = post_pop.activation(post_pop.state)
                    if store_dynamics:
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
                            post_pop.dendritic_state = torch.zeros(post_pop.size, device=post_pop.network.device,
                                                                   requires_grad=False)
                            init_dend_state = True
                        if projection.direction in ['forward', 'F']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.activity)
                        elif projection.direction in ['recurrent', 'R']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.prev_activity)
                if init_dend_state:
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
        """
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_pop = network.output_pop
        
        # compute the forward_dendritic_state before comparing output to target
        output_pop.dendritic_state = torch.zeros(output_pop.size, device=network.device, requires_grad=False)
        for layer in reversed_layers:
            cls.backward_update_layer_dendritic_state(layer)
            # initialize populations that are updated during the backward phase
            for pop in layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.backward_steps_activity = []
                # store the forward_activity before comparing output to target
                pop.forward_activity = pop.activity.detach().clone()
                # initialize dendritic state variables
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        pop.plateau = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.dend_to_soma = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.forward_dendritic_state = pop.dendritic_state.detach().clone()
                        if store_history:
                            # store the forward_dendritic_state
                            pop.append_attribute_history('forward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break
        
        for t in range(network.forward_steps):
            for layer in reversed_layers:
                # update dendritic state variables
                cls.backward_update_layer_dendritic_state(layer)
                for pop in layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                if t == 0:
                                    output_pop.dendritic_state = (
                                        torch.clamp(target - output_pop.activity, min=-1, max=1))
                                    output_pop.plateau = output_pop.dendritic_state.detach().clone()
                                    output_pop.dend_to_soma = output_pop.dendritic_state.detach().clone()
                            else:
                                max_units = math.ceil(projection.learning_rule.max_pop_fraction * pop.size)
                                avail_indexes = (pop.plateau == 0).nonzero().squeeze(1)
                                pos_remaining = max_units - (pop.plateau > 0).count_nonzero()
                                neg_remaining = max_units - (pop.plateau < 0).count_nonzero()
                                if pos_remaining > 0 or neg_remaining > 0:
                                    sorted, sorted_indexes = torch.sort(pop.dendritic_state[avail_indexes],
                                                                        descending=True, stable=True)
                                    sorted_avail_indexes = avail_indexes[sorted_indexes]
                                if pos_remaining > 0:
                                    pos_indexes = (pop.dendritic_state[sorted_avail_indexes] >
                                                   projection.learning_rule.pos_loss_th).nonzero().squeeze(1)
                                    pos_avail_indexes = sorted_avail_indexes[pos_indexes][:pos_remaining]
                                else:
                                    pos_avail_indexes = []
                                if neg_remaining > 0:
                                    neg_indexes = (pop.dendritic_state[sorted_avail_indexes] <
                                                   projection.learning_rule.neg_loss_th).nonzero().squeeze(1)
                                    neg_avail_indexes = sorted_avail_indexes[neg_indexes][-neg_remaining:]
                                else:
                                    neg_avail_indexes = []
                                pop.plateau[pos_avail_indexes] = pop.dendritic_state[pos_avail_indexes]
                                pop.dend_to_soma[pos_avail_indexes] = pop.dendritic_state[pos_avail_indexes]
                                pop.plateau[neg_avail_indexes] = pop.dendritic_state[neg_avail_indexes]
                                pop.dend_to_soma[neg_avail_indexes] = pop.dendritic_state[neg_avail_indexes]
                            break
            # update activities
            cls.backward_update_activity(network, store_dynamics=store_dynamics)
        
        for layer in network:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        if store_history:
                            pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                            pop.append_attribute_history('backward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break
                if pop.backward_projections or pop is output_pop:
                    if store_history:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BP_like_5(LearningRule):
    def __init__(self, projection, pos_loss_th=0.2, neg_loss_th=-0.2, max_pop_fraction=0.025, nudge=True,
                 learning_rate=None):
        """
        Output units are nudged to target. Hidden dendrites locally compute an error as the difference between
        excitation and inhibition. Weight updates are proportional to local error and presynaptic firing rate. A fixed
        fraction of hidden units are updated at each train step. Only hidden units with error beyond a threshold are
        updated. Hidden units are by default "nudged" by dendritic state when beyond threshold.
        Nudges to somatic state are applied instantaneously, rather than being subject to slow equilibration.
        :param projection: :class:'nn.Linear'
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.pos_loss_th = pos_loss_th
        self.neg_loss_th = neg_loss_th
        self.max_pop_fraction = max_pop_fraction
        self.nudge = nudge
        projection.post.__class__.plateau_history = property(lambda self: self.get_attribute_history('plateau'))
        projection.post.__class__.backward_activity_history = \
            property(lambda self: self.get_attribute_history('backward_activity'))
        projection.post.__class__.forward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('forward_dendritic_state'))
        projection.post.__class__.backward_dendritic_state_history = \
            property(lambda self: self.get_attribute_history('backward_dendritic_state'))
    
    def step(self):
        self.projection.weight.data += \
            (self.learning_rate * torch.outer(self.projection.post.plateau,
                                              torch.clamp(self.projection.pre.activity, min=0, max=1)))
    
    @classmethod
    def backward_update_activity(cls, network, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for layer in network:
            for post_pop in layer:
                post_pop.prev_activity = post_pop.activity
        for layer in network:
            for post_pop in layer:
                if post_pop.backward_projections or post_pop is post_pop.network.output_pop:
                    # update somatic state and activity
                    delta_state = -post_pop.state + post_pop.bias
                    for projection in post_pop:
                        pre_pop = projection.pre
                        if projection.compartment in [None, 'soma']:
                            if projection.direction in ['forward', 'F']:
                                delta_state = delta_state + projection(pre_pop.activity)
                            elif projection.direction in ['recurrent', 'R']:
                                delta_state = delta_state + projection(pre_pop.prev_activity)
                    post_pop.state = post_pop.state + delta_state / post_pop.tau
                    if hasattr(post_pop, 'dend_to_soma'):
                        post_pop.activity = post_pop.activation(post_pop.state + post_pop.dend_to_soma)
                    else:
                        post_pop.activity = post_pop.activation(post_pop.state)
                    if store_dynamics:
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
                            post_pop.dendritic_state = torch.zeros(post_pop.size, device=post_pop.network.device,
                                                                   requires_grad=False)
                            init_dend_state = True
                        if projection.direction in ['forward', 'F']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.activity)
                        elif projection.direction in ['recurrent', 'R']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.prev_activity)
                if init_dend_state:
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
        """
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_pop = network.output_pop
        
        # compute the forward_dendritic_state before comparing output to target
        output_pop.dendritic_state = torch.zeros(output_pop.size, device=network.device, requires_grad=False)
        for layer in reversed_layers:
            cls.backward_update_layer_dendritic_state(layer)
            # initialize populations that are updated during the backward phase
            for pop in layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.backward_steps_activity = []
                # store the forward_activity before comparing output to target
                pop.forward_activity = pop.activity.detach().clone()
                # initialize dendritic state variables
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        pop.plateau = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.dend_to_soma = torch.zeros(pop.size, device=pop.network.device, requires_grad=False)
                        pop.forward_dendritic_state = pop.dendritic_state.detach().clone()
                        if store_history:
                            # store the forward_dendritic_state
                            pop.append_attribute_history('forward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break
        
        for t in range(network.forward_steps):
            for layer in reversed_layers:
                # update dendritic state variables
                cls.backward_update_layer_dendritic_state(layer)
                for pop in layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                if t == 0:
                                    output_pop.dendritic_state = (
                                        torch.clamp(target - output_pop.activity, min=-1, max=1))
                                    output_pop.plateau = output_pop.dendritic_state.detach().clone()
                                    if projection.learning_rule.nudge:
                                        output_pop.dend_to_soma = output_pop.dendritic_state.detach().clone()
                            else:
                                max_units = math.ceil(projection.learning_rule.max_pop_fraction * pop.size)
                                avail_indexes = (pop.plateau == 0).nonzero().squeeze(1)
                                if len(avail_indexes) > 0:
                                    pos_remaining = max_units - (pop.plateau > 0).count_nonzero()
                                    neg_remaining = max_units - (pop.plateau < 0).count_nonzero()
                                    if pos_remaining > 0 or neg_remaining > 0:
                                        sorted, sorted_indexes = torch.sort(pop.dendritic_state[avail_indexes],
                                                                            descending=True, stable=True)
                                        sorted_avail_indexes = avail_indexes[sorted_indexes]
                                    if pos_remaining > 0:
                                        pos_indexes = (pop.dendritic_state[sorted_avail_indexes] >
                                                       projection.learning_rule.pos_loss_th).nonzero().squeeze(1)
                                        pos_avail_indexes = sorted_avail_indexes[pos_indexes][:pos_remaining]
                                    else:
                                        pos_avail_indexes = []
                                    if neg_remaining > 0:
                                        neg_indexes = (pop.dendritic_state[sorted_avail_indexes] <
                                                       projection.learning_rule.neg_loss_th).nonzero().squeeze(1)
                                        neg_avail_indexes = sorted_avail_indexes[neg_indexes][-neg_remaining:]
                                    else:
                                        neg_avail_indexes = []
                                    pop.plateau[pos_avail_indexes] = pop.dendritic_state[pos_avail_indexes]
                                    pop.plateau[neg_avail_indexes] = pop.dendritic_state[neg_avail_indexes]
                                    if projection.learning_rule.nudge:
                                        pop.dend_to_soma[pos_avail_indexes] = pop.dendritic_state[pos_avail_indexes]
                                        pop.dend_to_soma[neg_avail_indexes] = pop.dendritic_state[neg_avail_indexes]
                            break
            # update activities
            cls.backward_update_activity(network, store_dynamics=store_dynamics)
        
        for layer in network:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        if store_history:
                            pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                            pop.append_attribute_history('backward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break
                if pop.backward_projections or pop is output_pop:
                    if store_history:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


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


class DendriticLoss_3(LearningRule):
    """
    This variant 3 is gated by dendritic state. Consults the initial dendritic state and initial activity obtained
    after the forward phase, before comparison to target during the backward phase. Requires BTSP_2 or equivalent rule
    to update the attributes forward_dendritic_state and forward_activity.
    """
    def __init__(self, projection, sign=1, learning_rate=None):
        super().__init__(projection, learning_rate)
        self.sign = sign

    def step(self):
        self.projection.weight.data += self.sign * self.learning_rate * \
                                       torch.outer(self.projection.post.forward_dendritic_state,
                                                   self.projection.pre.forward_activity)


class DendriticLoss_4(LearningRule):
    """
    This variant 4 is gated by dendritic state. The original rule is gated by plateaus.
    This variant returns to learning weights based on activity at the end of the backward phase.
    pre.activity is saturated at 1.
    """
    def __init__(self, projection, sign=1, learning_rate=None):
        super().__init__(projection, learning_rate)
        self.sign = sign

    def step(self):
        self.projection.weight.data += self.sign * self.learning_rate * \
                                       torch.outer(self.projection.post.dendritic_state,
                                                   torch.clamp(self.projection.pre.activity, 0, 1))


class DendriticLoss_5(LearningRule):
    """
    This variant 5 is gated by dendritic state. The original rule is gated by plateaus.
    pre.activity is saturated at 1.
    """
    def __init__(self, projection, sign=1, learning_rate=None):
        super().__init__(projection, learning_rate)
        self.sign = sign

    def step(self):
        self.projection.weight.data += self.sign * self.learning_rate * \
                                       torch.outer(self.projection.post.forward_dendritic_state,
                                                   torch.clamp(self.projection.pre.forward_activity, 0, 1))


def clone_weight(projection, source=None, sign=1, scale=1, source2=None, transpose=False):
    """
    Force a projection to exactly copy the weights of another projection (or product of two projections).
    """
    if source is None:
        raise Exception('clone_weight: missing required weight_constraint_kwarg: source')
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
    if transpose:
        source_weight_data = source_weight_data.T
    if source_weight_data.shape != projection.weight.data.shape:
        raise Exception('clone_weight: projection shapes do not match; target: %s, %s; source: %s, %s' %
                        (projection.name, str(projection.weight.data.shape), source_projection.name,
                         str(source_weight_data.shape)))
    projection.weight.data = source_weight_data


def normalize_weight(projection, scale, autapses=False, axis=1):
    if not autapses and projection.pre is projection.post:
        projection.weight.data.fill_diagonal_(0.)
    weight_sum = torch.sum(torch.abs(projection.weight.data), axis=axis).unsqueeze(1)
    valid_rows = torch.nonzero(weight_sum, as_tuple=True)[0]
    projection.weight.data[valid_rows,:] /= weight_sum[valid_rows,:]
    projection.weight.data *= scale


def no_autapses(projection):
    if projection.pre is projection.post:
        projection.weight.data.fill_diagonal_(0.)



class almost_backprop(LearningRule):
    def __init__(self, projection, learning_rate=None):
        super().__init__(projection, learning_rate)

        self.w_max = 1.
        self.k_dep = 0.5
        self.dep_sigmoid = get_scaled_rectified_sigmoid(0.01, 0.02)

    def step(self):
        # Update the weights
        
        # ETxIS = torch.outer(self.projection.post.IS, self.projection.pre.ET)
        # delta_weight = ETxIS #*(self.w_max-self.projection.weight) - self.projection.weight * self.k_dep * self.dep_sigmoid(ETxIS)
        
        # ~BTSP weight update
        ETxIS = torch.outer(self.projection.post.dendritic_state, self.projection.pre.activity)
        weight_dependence = self.w_max - (self.projection.weight)  #*self.projection.weight.sign())
        delta_weight = ETxIS * weight_dependence - self.projection.weight*self.k_dep*self.dep_sigmoid(ETxIS)

        # # ~BTSP weight update 2
        # P = self.projection.post.dendritic_state
        # e = self.projection.pre.activity
        # w = self.projection.weight
        # delta_weight = P.unsqueeze(1) * (e*(self.w_max-w) - w*self.k_dep*self.dep_sigmoid(e))

        # # ~Backprop weight update
        # delta_weight = torch.outer(self.projection.post.dendritic_state, self.projection.pre.activity)

        self.projection.weight.data += self.learning_rate * delta_weight

    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
        """

        # Compute Output loss & set dendritic state
        global_error = target - output

        # Set dendritic state with the local loss in each layer
        reversed_layers = list(network)[::-1]
        network.output_pop.dendritic_state = global_error
        for i,layer in enumerate(reversed_layers[:-1]): # Iterate over populations in reverse order starting from the output layer
            layer.E.backward_activity = layer.E.activity + layer.E.dendritic_state # nudge somatic state

            pre_layer = reversed_layers[i+1]
            forward_weight_transpose = layer.E.incoming_projections[f"{layer.E.fullname}_{pre_layer.E.fullname}"].weight.T

            inhibition = forward_weight_transpose @ layer.E.activity
            pre_layer.E.dendritic_state = forward_weight_transpose @ layer.E.backward_activity - inhibition

            # Keep the top x% of gradients and set the rest to 0
            percentage_to_keep = 0.2
            flat_error_vector = pre_layer.E.dendritic_state.flatten()
            n = round(flat_error_vector.numel()*(1-percentage_to_keep))
            indices = flat_error_vector.abs().argsort()[:n] # Find the indices of the n smallest values
            flat_error_vector[indices] = 0
            pre_layer.E.dendritic_state = flat_error_vector.view(pre_layer.E.dendritic_state.shape)


