from .base_classes import LearningRule, BiasLearningRule
from .backprop_like import BP_like_2L

import torch
import math


class Oja(LearningRule):
    def step(self):
        delta_weight = torch.outer(self.projection.post.activity, self.projection.pre.activity) - \
                       self.projection.weight * (self.projection.post.activity ** 2).unsqueeze(1)
        self.projection.weight.data += self.learning_rate * delta_weight


class BCM(LearningRule):
    def __init__(self, projection, theta_tau, k, sign=1, learning_rate=None):
        """
        Weight updates are proportional to local error and presynaptic firing rate (after
        backward phase equilibration).
        :param projection:
        :param theta_tau: float
        :param k: float
        :param sign: int in {-1, 1}
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.theta_tau = theta_tau
        self.k = k
        self.projection.post.theta = torch.ones(projection.post.size, device=projection.post.network.device) * k
        self.sign = sign
        projection.post.register_attribute_history('theta')

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


class BCM_4(LearningRule):
    def __init__(self, projection, theta_tau, k, sign=1, learning_rate=None):
        """
        Weight updates are proportional to local error and presynaptic firing rate (after
        backward phase equilibration).
        :param projection:
        :param theta_tau: float
        :param k: float
        :param sign: int in {-1, 1}
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.theta_tau = theta_tau
        self.k = k
        self.projection.post.theta = torch.ones(projection.post.size, device=projection.post.network.device) * k
        self.sign = sign
        projection.post.register_attribute_history('theta')
    
    def reinit(self):
        self.projection.post.BCM_theta_stored = False
        self.projection.post.BCM_theta_updated = False
    
    def update(self):
        if not self.projection.post.BCM_theta_updated:
            delta_theta = ((-self.projection.post.theta + self.projection.post.activity ** 2. / self.k) /
                           self.theta_tau).detach().clone()
            self.projection.post.theta += delta_theta
            self.projection.post.BCM_theta_updated = True
            self.projection.post.BCM_theta_stored = False
    
    def step(self):
        # post_activity = torch.clamp(self.projection.post.activity.detach().clone(), min=0, max=1)
        post_activity = self.projection.post.activity.detach().clone()
        if self.projection.direction in ['forward', 'F']:
            delta_weight = (torch.outer(post_activity, torch.clamp(self.projection.pre.activity, min=0, max=1)) *
                            (post_activity - self.projection.post.theta).unsqueeze(1)).detach().clone()
        elif self.projection.direction in ['recurrent', 'R']:
            delta_weight = (torch.outer(post_activity, torch.clamp(self.projection.pre.prev_activity, min=0, max=1)) *
                            (post_activity - self.projection.post.theta).unsqueeze(1)).detach().clone()
        if self.sign > 0:
            self.projection.weight.data += self.learning_rate * delta_weight
        else:
            self.projection.weight.data -= self.learning_rate * delta_weight
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        for layer in network:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        # only update theta once per population
                        pop.BCM_theta_updated = False
                        if store_history and not pop.BCM_theta_stored:
                            pop.append_attribute_history('theta', pop.theta.detach().clone())
                            pop.BCM_theta_stored = True
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
        self.projection.post.theta = torch.ones(projection.post.size, device=projection.post.network.device) * k
        self.sign = sign
        projection.post.register_attribute_history('theta')
        projection.post.register_attribute_history('nudge')
        projection.post.register_attribute_history('backward_activity')

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
        self.projection.post.theta = torch.ones(projection.post.size, device=projection.post.network.device) * k
        self.sign = sign
        self.max_pop_fraction = max_pop_fraction
        self.pos_loss_th = pos_loss_th
        self.neg_loss_th = neg_loss_th
        projection.post.register_attribute_history('theta')
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
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
        output_pop = network.output_pop
        
        # initialize populations that are updated during the backward phase
        for layer in reversed_layers:
            for pop in layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.backward_steps_activity = []
                # initialize dendritic state variables
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        pop.plateau = torch.zeros(pop.size, device=network.device)
                        pop.dend_to_soma = torch.zeros(pop.size, device=network.device)
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


class Supervised_BCM_3(LearningRule):
    """
    Output units are instantaneously nudged to target, and the Output layer is allowed to equilibrate.
    No top-down feedback in hidden layers.
    """
    
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
        self.projection.post.theta = torch.ones(projection.post.size, device=projection.post.network.device) * k
        self.sign = sign
        projection.post.register_attribute_history('theta')
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
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
        
        for t in range(network.forward_steps):
            for projection in output_pop:
                if projection.learning_rule.__class__ == cls:
                    if t == 0:
                        output_pop.dendritic_state = torch.clamp(target - output_pop.activity, min=-1, max=1)
                        output_pop.plateau = output_pop.dendritic_state.detach().clone()
                        output_pop.dend_to_soma = output_pop.dendritic_state.detach().clone()
                    break
            cls.backward_update_layer_activity(output_layer, store_dynamics=store_dynamics)
        
        for projection in output_pop:
            if projection.learning_rule.__class__ == cls:
                # only update theta once per population
                output_pop.BCM_theta_updated = False
                
                if store_history:
                    output_pop.append_attribute_history('plateau', output_pop.plateau.detach().clone())
                    output_pop.append_attribute_history('backward_dendritic_state',
                                                        output_pop.dendritic_state.detach().clone())
                    if not output_pop.BCM_theta_stored:
                        output_pop.append_attribute_history('theta', output_pop.theta.detach().clone())
                        output_pop.BCM_theta_stored = True
                break
        
        if store_history:
            for pop in output_layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                    else:
                        pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class Top_Layer_Supervised_BCM_4(LearningRule):
    """
    Output units are instantaneously nudged to target, and the Output layer is allowed to equilibrate.
    No top-down feedback in hidden layers. Weight updates are proportional to postsynaptic firing rate and presynaptic
    firing rate (after backward phase equilibration).
    """
    def __init__(self, projection, theta_tau, k, sign=1, learning_rate=None, relu_gate=False):
        """

        :param projection:
        :param theta_tau: float
        :param k: float
        :param sign: int in {-1, 1}
        :param learning_rate: float
        :param relu_gate: bool
        """
        super().__init__(projection, learning_rate)
        self.theta_tau = theta_tau
        self.k = k
        self.projection.post.theta = torch.ones(projection.post.size, device=projection.post.network.device) * k
        self.sign = sign
        self.relu_gate = relu_gate
        projection.post.register_attribute_history('theta')
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
    def reinit(self):
        self.projection.post.BCM_theta_stored = False
        self.projection.post.BCM_theta_updated = False
    
    def update(self):
        if not self.projection.post.BCM_theta_updated:
            delta_theta = ((-self.projection.post.theta + self.projection.post.activity ** 2. / self.k) /
                           self.theta_tau).detach().clone()
            self.projection.post.theta += delta_theta
            self.projection.post.BCM_theta_updated = True
            self.projection.post.BCM_theta_stored = False
    
    def step(self):
        # post_activity = torch.clamp(self.projection.post.activity.detach().clone(), min=0, max=1)
        post_activity = self.projection.post.activity.detach().clone()
        if self.projection.direction in ['forward', 'F']:
            delta_weight = (torch.outer(post_activity, torch.clamp(self.projection.pre.activity, min=0, max=1)) *
                            (post_activity - self.projection.post.theta).unsqueeze(1)).detach().clone()
        elif self.projection.direction in ['recurrent', 'R']:
            delta_weight = (torch.outer(post_activity, torch.clamp(self.projection.pre.prev_activity, min=0, max=1)) *
                            (post_activity - self.projection.post.theta).unsqueeze(1)).detach().clone()
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
        for pop in output_layer:
            if pop.backward_projections or pop is output_pop:
                if store_dynamics:
                    pop.backward_steps_activity = []
                # initialize dendritic state variables
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        pop.plateau = torch.zeros(pop.size, device=network.device)
                        pop.dend_to_soma = torch.zeros(pop.size, device=network.device)
                        break
        
        for t in range(network.forward_steps):
            if t == 0:
                for pop in output_layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                local_loss = torch.clamp(target - output_pop.activity, min=-1, max=1)
                                output_pop.dendritic_state = local_loss.detach().clone()
                                if projection.learning_rule.relu_gate:
                                    local_loss[output_pop.forward_activity == 0.] = 0
                                output_pop.plateau[:] = local_loss.detach().clone()
                                output_pop.dend_to_soma[:] = local_loss.detach().clone()
                            break
            cls.backward_update_layer_activity(output_layer, store_dynamics=store_dynamics)
        
        for pop in output_layer:
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
            
            if store_history:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                    else:
                        pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class Top_Layer_Supervised_BCM_5(LearningRule):
    """
    Output units are instantaneously nudged to target, and the Output layer is allowed to equilibrate.
    No top-down feedback in hidden layers. Weight updates are proportional to postsynaptic firing rate and presynaptic
    firing rate (after backward phase equilibration).
    """
    
    def __init__(self, projection, theta_tau, k, sign=1, learning_rate=None, relu_gate=False):
        """

        :param projection:
        :param theta_tau: float
        :param k: float
        :param sign: int in {-1, 1}
        :param learning_rate: float
        :param relu_gate: bool
        """
        super().__init__(projection, learning_rate)
        self.theta_tau = theta_tau
        self.k = k
        self.projection.post.theta = torch.ones(projection.post.size, device=projection.post.network.device) * k
        self.sign = sign
        self.relu_gate = relu_gate
        projection.post.register_attribute_history('theta')
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
    def reinit(self):
        self.projection.post.BCM_theta_stored = False
        self.projection.post.BCM_theta_updated = False
    
    def update(self):
        if not self.projection.post.BCM_theta_updated:
            delta_theta = ((-self.projection.post.theta + self.projection.post.activity ** 2. / self.k) /
                           self.theta_tau).detach().clone()
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
                if hasattr(post_pop, 'dend_to_soma'):
                    post_pop.activity = post_pop.activation(post_pop.state + post_pop.dend_to_soma)
                else:
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
        for pop in output_layer:
            if pop.backward_projections or pop is output_pop:
                if store_dynamics:
                    pop.backward_steps_activity = []
                # initialize dendritic state variables
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        pop.plateau = torch.zeros(pop.size, device=network.device)
                        pop.dend_to_soma = torch.zeros(pop.size, device=network.device)
                        break
        
        for t in range(network.forward_steps):
            if t == 0:
                for pop in output_layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                local_loss = torch.clamp(target - output_pop.activity, min=-1, max=1)
                                output_pop.dendritic_state = local_loss.detach().clone()
                                if projection.learning_rule.relu_gate:
                                    local_loss[output_pop.forward_activity == 0.] = 0
                                output_pop.plateau[:] = local_loss.detach().clone()
                                output_pop.dend_to_soma[:] = local_loss.detach().clone()
                            break
            cls.backward_update_layer_activity(output_layer, store_dynamics=store_dynamics)
        
        for pop in output_layer:
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
            
            if store_history:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                    else:
                        pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class Supervised_BCM_4(LearningRule):
    def __init__(self, projection, theta_tau, k, sign=1, max_pop_fraction=0.025, stochastic=False, learning_rate=None,
                 relu_gate=False):
        """
        Output units are nudged to target. Hidden dendrites locally compute an error as the difference between
        excitation and inhibition. Weight updates are proportional to postsynaptic firing rate and presynaptic firing
        rate (after backward phase equilibration).
        If unit selection is stochastic, hidden units are selected for a weight update with a probability proportional
        to dendritic state by sampling a Bernoulli distribution. Otherwise, units are sorted by dendritic state. A fixed
        maximum fraction of hidden units are updated at each train step. Hidden units are "nudged" by dendritic state
        when selected for a weight update. Nudges to somatic state are applied instantaneously, rather than being
        subject to slow equilibration. During nudging, activities are re-equilibrated across all layers.
        :param projection: :class:'nn.Linear'
        :param theta_tau: float
        :param k: float
        :param sign: int in {-1, 1}
        :param max_pop_fraction: float in [0, 1]
        :param stochastic: bool
        :param learning_rate: float
        :param relu_gate: bool
        """
        super().__init__(projection, learning_rate)
        self.theta_tau = theta_tau
        self.k = k
        self.projection.post.theta = torch.ones(projection.post.size, device=projection.post.network.device) * k
        self.sign = sign
        self.max_pop_fraction = max_pop_fraction
        self.stochastic = stochastic
        self.relu_gate = relu_gate
        projection.post.register_attribute_history('theta')
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
    def reinit(self):
        self.projection.post.BCM_theta_stored = False
        self.projection.post.BCM_theta_updated = False
    
    def update(self):
        if not self.projection.post.BCM_theta_updated:
            delta_theta = ((-self.projection.post.theta + self.projection.post.activity ** 2. / self.k) /
                           self.theta_tau).detach().clone()
            self.projection.post.theta += delta_theta
            self.projection.post.BCM_theta_updated = True
            self.projection.post.BCM_theta_stored = False
    
    def step(self):
        # post_activity = torch.clamp(self.projection.post.activity.detach().clone(), min=0, max=1)
        post_activity = self.projection.post.activity.detach().clone()
        if self.projection.direction in ['forward', 'F']:
            delta_weight = (torch.outer(post_activity, torch.clamp(self.projection.pre.activity, min=0, max=1)) *
                            (post_activity - self.projection.post.theta).unsqueeze(1)).detach().clone()
        elif self.projection.direction in ['recurrent', 'R']:
            delta_weight = (torch.outer(post_activity, torch.clamp(self.projection.pre.prev_activity, min=0, max=1)) *
                            (post_activity - self.projection.post.theta).unsqueeze(1)).detach().clone()
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
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
        """
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_pop = network.output_pop
        
        # initialize populations that are updated during the backward phase
        for layer in reversed_layers:
            for pop in layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.backward_steps_activity = []
                # initialize dendritic state variables
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        pop.plateau = torch.zeros(pop.size, device=network.device)
                        pop.dend_to_soma = torch.zeros(pop.size, device=network.device)
                        break
        
        for t in range(network.forward_steps):
            for layer in reversed_layers:
                # update dendritic state variables
                cls.backward_update_layer_dendritic_state(layer)
                if t == 0:
                    for pop in layer:
                        for projection in pop:
                            # compute plateau events and nudge somatic state
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
                # update activities
                cls.backward_update_layer_activity(layer, store_dynamics=store_dynamics)
        
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
                                pop.append_attribute_history('theta', output_pop.theta.detach().clone())
                                pop.BCM_theta_stored = True
                        break
                
                if store_history:
                    if pop.backward_projections or pop is output_pop:
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
        projection.post.register_attribute_history('nudge')
        projection.post.register_attribute_history('backward_activity')

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
        projection.post.register_attribute_history('nudge')
        projection.post.register_attribute_history('backward_activity')

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
    def __init__(self, projection, sign=1, learning_rate=None, forward_only=False):
        super().__init__(projection, learning_rate)
        self.sign = sign
        self.forward_only = forward_only

    def step(self):
        if self.projection.direction in ['forward', 'F']:
            if self.forward_only:
                delta_weight = torch.outer(self.projection.post.forward_activity, self.projection.pre.forward_activity)
            else:
                delta_weight = torch.outer(self.projection.post.activity, self.projection.pre.activity)
        elif self.projection.direction in ['recurrent', 'R']:
            if self.forward_only:
                delta_weight = torch.outer(self.projection.post.forward_activity,
                                           self.projection.pre.forward_prev_activity)
            else:
                delta_weight = torch.outer(self.projection.post.activity, self.projection.pre.prev_activity)
        if self.sign > 0:
            self.projection.weight.data += self.learning_rate * delta_weight
        else:
            self.projection.weight.data -= self.learning_rate * delta_weight


class Hebb_WeightNorm_4(LearningRule):
    def __init__(self, projection, sign=1, learning_rate=None, forward_only=False):
        """
        Weight updates are proportional to postsynaptic firing rate and presynaptic firing rate.
        :param projection:
        :param sign:
        :param learning_rate:
        :param forward_only:
        """
        super().__init__(projection, learning_rate)
        self.sign = sign
        self.forward_only = forward_only

    def step(self):
        if self.forward_only:
            if self.projection.direction in ['forward', 'F']:
                delta_weight = (
                    torch.outer(torch.clamp(self.projection.post.forward_activity, min=0, max=1),
                                torch.clamp(self.projection.pre.forward_activity, min=0, max=1)).detach().clone())
            elif self.projection.direction in ['recurrent', 'R']:
                delta_weight = (
                    torch.outer(torch.clamp(self.projection.post.forward_activity, min=0, max=1),
                                torch.clamp(self.projection.pre.forward_prev_activity, min=0, max=1)).detach().clone())
        else:
            if self.projection.direction in ['forward', 'F']:
                delta_weight = torch.outer(torch.clamp(self.projection.post.activity, min=0, max=1),
                                           torch.clamp(self.projection.pre.activity, min=0, max=1)).detach().clone()
            elif self.projection.direction in ['recurrent', 'R']:
                delta_weight = torch.outer(torch.clamp(self.projection.post.activity, min=0, max=1),
                                           torch.clamp(self.projection.pre.prev_activity, min=0,
                                                       max=1)).detach().clone()
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
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
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
        output_pop = network.output_pop
        
        # initialize populations that are updated during the backward phase
        for layer in reversed_layers:
            for pop in layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.backward_steps_activity = []
                # initialize dendritic state variables
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        pop.plateau = torch.zeros(pop.size, device=network.device)
                        pop.dend_to_soma = torch.zeros(pop.size, device=network.device)
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


class Supervised_Hebb_WeightNorm_2(LearningRule):
    """
    Output units are instantaneously nudged to target, and the Output layer is allowed to equilibrate.
    No top-down feedback in hidden layers.
    """
    
    def __init__(self, projection, sign=1, pos_loss_th=0.2, neg_loss_th=-0.2, max_pop_fraction=0.025,
                 learning_rate=None):
        super().__init__(projection, learning_rate)
        self.sign = sign
        self.max_pop_fraction = max_pop_fraction
        self.pos_loss_th = pos_loss_th
        self.neg_loss_th = neg_loss_th
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
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
        for pop in output_layer:
            if pop.backward_projections or pop is output_pop:
                if store_dynamics:
                    pop.backward_steps_activity = []
                
        for t in range(network.forward_steps):
            for projection in output_pop:
                if projection.learning_rule.__class__ == cls:
                    if t == 0:
                        output_pop.dendritic_state = torch.clamp(target - output_pop.activity, min=-1, max=1)
                        output_pop.plateau = output_pop.dendritic_state.detach().clone()
                        output_pop.dend_to_soma = output_pop.dendritic_state.detach().clone()
                    break
            cls.backward_update_layer_activity(output_layer, store_dynamics=store_dynamics)
        
        if store_history:
            for projection in output_pop:
                if projection.learning_rule.__class__ == cls:
                    output_pop.append_attribute_history('plateau', output_pop.plateau.detach().clone())
                    output_pop.append_attribute_history('backward_dendritic_state',
                                                        output_pop.dendritic_state.detach().clone())
                    break
            for pop in output_layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                    else:
                        pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class Top_Layer_Supervised_Hebb_WeightNorm_4(LearningRule):
    """
    Output units are instantaneously nudged to target, and the Output layer is allowed to equilibrate.
    No top-down feedback in hidden layers. Weight updates are proportional to postsynaptic firing rate and presynaptic
    firing rate (after backward phase equilibration).
    """
    
    def __init__(self, projection, sign=1, learning_rate=None, relu_gate=False):
        """

        :param projection:
        :param sign:
        :param learning_rate:
        :param relu_gate:
        """
        super().__init__(projection, learning_rate)
        self.sign = sign
        self.relu_gate = relu_gate
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
    def step(self):
        if self.projection.direction in ['forward', 'F']:
            delta_weight = torch.outer(torch.clamp(self.projection.post.activity, min=0, max=1),
                                       torch.clamp(self.projection.pre.activity, min=0, max=1)).detach().clone()
        elif self.projection.direction in ['recurrent', 'R']:
            delta_weight = torch.outer(torch.clamp(self.projection.post.activity, min=0, max=1),
                                       torch.clamp(self.projection.pre.prev_activity, min=0, max=1)).detach().clone()
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
        for pop in output_layer:
            if pop.backward_projections or pop is output_pop:
                if store_dynamics:
                    pop.backward_steps_activity = []
                # initialize dendritic state variables
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        pop.plateau = torch.zeros(pop.size, device=network.device)
                        pop.dend_to_soma = torch.zeros(pop.size, device=network.device)
                        break
        
        for t in range(network.forward_steps):
            if t == 0:
                for pop in output_layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                local_loss = torch.clamp(target - output_pop.activity, min=-1, max=1)
                                output_pop.dendritic_state = local_loss.detach().clone()
                                if projection.learning_rule.relu_gate:
                                    local_loss[output_pop.forward_activity == 0.] = 0
                                output_pop.plateau[:] = local_loss.detach().clone()
                                output_pop.dend_to_soma[:] = local_loss.detach().clone()
                            break
            cls.backward_update_layer_activity(output_layer, store_dynamics=store_dynamics)
        
        if store_history:
            for pop in output_layer:
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


class Top_Layer_Supervised_Hebb_WeightNorm_5(LearningRule):
    """
    Output units are instantaneously nudged to target, and the Output layer is allowed to equilibrate.
    No top-down feedback in hidden layers. Weight updates are proportional to postsynaptic firing rate and presynaptic
    firing rate (after backward phase equilibration).
    """
    
    def __init__(self, projection, sign=1, learning_rate=None, relu_gate=False):
        """

        :param projection:
        :param sign:
        :param learning_rate:
        :param relu_gate:
        """
        super().__init__(projection, learning_rate)
        self.sign = sign
        self.relu_gate = relu_gate
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
    def step(self):
        if self.projection.direction in ['forward', 'F']:
            delta_weight = torch.outer(self.projection.post.activity, self.projection.pre.activity).detach().clone()
        elif self.projection.direction in ['recurrent', 'R']:
            delta_weight = torch.outer(self.projection.post.activity,
                                       self.projection.pre.prev_activity).detach().clone()
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
        for pop in output_layer:
            if pop.backward_projections or pop is output_pop:
                if store_dynamics:
                    pop.backward_steps_activity = []
                # initialize dendritic state variables
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        pop.plateau = torch.zeros(pop.size, device=network.device)
                        pop.dend_to_soma = torch.zeros(pop.size, device=network.device)
                        break
        
        for t in range(network.forward_steps):
            if t == 0:
                for pop in output_layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                local_loss = torch.clamp(target - output_pop.activity, min=-1, max=1)
                                output_pop.dendritic_state = local_loss.detach().clone()
                                if projection.learning_rule.relu_gate:
                                    local_loss[output_pop.forward_activity == 0.] = 0
                                output_pop.plateau[:] = local_loss.detach().clone()
                                output_pop.dend_to_soma[:] = local_loss.detach().clone()
                            break
            cls.backward_update_layer_activity(output_layer, store_dynamics=store_dynamics)
        
        if store_history:
            for pop in output_layer:
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


class Top_Layer_Supervised_Hebb_WeightNorm_6(LearningRule):
    """
    Output units are instantaneously nudged to target, and the Output layer is allowed to equilibrate.
    No top-down feedback in hidden layers. Weight updates are proportional to postsynaptic firing rate and presynaptic
    firing rate (after backward phase equilibration). Negative weight updates occur when somatic state is less than
    zero, and are proportional to somatic state and presynaptic activity.
    """
    
    def __init__(self, projection, sign=1, learning_rate=None, relu_gate=False):
        """

        :param projection:
        :param sign:
        :param learning_rate:
        :param relu_gate:
        """
        super().__init__(projection, learning_rate)
        self.sign = sign
        self.relu_gate = relu_gate
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
    def step(self):
        if self.projection.direction in ['forward', 'F']:
            pre_act = torch.clamp(self.projection.pre.activity, min=0, max=1)
            delta_weight = torch.outer(torch.clamp(self.projection.post.activity, min=0, max=1),
                                       pre_act).detach().clone()
        elif self.projection.direction in ['recurrent', 'R']:
            pre_act = torch.clamp(self.projection.pre.prev_activity, min=0, max=1)
            delta_weight = torch.outer(torch.clamp(self.projection.post.activity, min=0, max=1),
                                       pre_act).detach().clone()
        sign = self.sign > 0
        self.projection.weight.data += sign * self.learning_rate * delta_weight
        
        soma_state = torch.clamp(self.projection.post.state, min=-1, max=0).detach().clone()
        if self.relu_gate:
            soma_state[self.projection.post.forward_activity == 0.] = 0.
        delta_weight = torch.outer(soma_state, pre_act).detach().clone()
        self.projection.weight.data += sign * self.learning_rate * delta_weight
    
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
                if hasattr(post_pop, 'dend_to_soma'):
                    post_pop.activity = post_pop.activation(post_pop.state + post_pop.dend_to_soma)
                else:
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
        for pop in output_layer:
            if pop.backward_projections or pop is output_pop:
                if store_dynamics:
                    pop.backward_steps_activity = []
                # initialize dendritic state variables
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        pop.plateau = torch.zeros(pop.size, device=network.device)
                        pop.dend_to_soma = torch.zeros(pop.size, device=network.device)
                        break
        
        for t in range(network.forward_steps):
            if t == 0:
                for pop in output_layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                local_loss = torch.clamp(target - output_pop.activity, min=-1, max=1)
                                output_pop.dendritic_state = local_loss.detach().clone()
                                if projection.learning_rule.relu_gate:
                                    local_loss[output_pop.forward_activity == 0.] = 0
                                output_pop.plateau[:] = local_loss.detach().clone()
                                output_pop.dend_to_soma[:] = local_loss.detach().clone()
                            break
            cls.backward_update_layer_activity(output_layer, store_dynamics=store_dynamics)
        
        if store_history:
            for pop in output_layer:
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


class Supervised_Hebb_WeightNorm_4(LearningRule):
    def __init__(self, projection, sign=1, max_pop_fraction=0.025, stochastic=True, learning_rate=None,
                 relu_gate=False):
        """
        Output units are nudged to target. Hidden dendrites locally compute an error as the difference between
        excitation and inhibition. Weight updates are proportional to postsynaptic firing rate and presynaptic firing
        rate (after backward phase equilibration).
        If unit selection is stochastic, hidden units are selected for a weight update with a probability proportional
        to dendritic state by sampling a Bernoulli distribution. Otherwise, units are sorted by dendritic state. A fixed
        maximum fraction of hidden units are updated at each train step. Hidden units are "nudged" by dendritic state
        when selected for a weight update. Nudges to somatic state are applied instantaneously, rather than being
        subject to slow equilibration. During nudging, activities are re-equilibrated across all layers.
        :param projection: :class:'nn.Linear'
        :param max_pop_fraction: float in [0, 1]
        :param stochastic: bool
        :param learning_rate: float
        :param relu_gate: bool
        """
        super().__init__(projection, learning_rate)
        self.sign = sign
        self.max_pop_fraction = max_pop_fraction
        self.stochastic = stochastic
        self.relu_gate = relu_gate
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
    def step(self):
        if self.projection.direction in ['forward', 'F']:
            delta_weight = torch.outer(torch.clamp(self.projection.post.activity, min=0, max=1),
                                       torch.clamp(self.projection.pre.activity, min=0, max=1)).detach().clone()
        elif self.projection.direction in ['recurrent', 'R']:
            delta_weight = torch.outer(torch.clamp(self.projection.post.activity, min=0, max=1),
                                       torch.clamp(self.projection.pre.prev_activity, min=0, max=1)).detach().clone()
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
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
        """
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_pop = network.output_pop
        
        # initialize populations that are updated during the backward phase
        for layer in reversed_layers:
            for pop in layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.backward_steps_activity = []
                # initialize dendritic state variables
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        pop.plateau = torch.zeros(pop.size, device=network.device)
                        pop.dend_to_soma = torch.zeros(pop.size, device=network.device)
                        break
        
        for t in range(network.forward_steps):
            for layer in reversed_layers:
                # update dendritic state variables
                cls.backward_update_layer_dendritic_state(layer)
                if t == 0:
                    for pop in layer:
                        for projection in pop:
                            # compute plateau events and nudge somatic state
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
                # update activities
                cls.backward_update_layer_activity(layer, store_dynamics=store_dynamics)
        
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


class Supervised_Hebb_WeightNorm_6(LearningRule):
    def __init__(self, projection, sign=1, max_pop_fraction=0.025, stochastic=True, learning_rate=None,
                 relu_gate=False):
        """
        Output units are nudged to target. Hidden dendrites locally compute an error as the difference between
        excitation and inhibition. Weight updates are proportional to postsynaptic firing rate and presynaptic firing
        rate (after backward phase equilibration).
        If unit selection is stochastic, hidden units are selected for a weight update with a probability proportional
        to dendritic state by sampling a Bernoulli distribution. Otherwise, units are sorted by dendritic state. A fixed
        maximum fraction of hidden units are updated at each train step. Hidden units are "nudged" by dendritic state
        when selected for a weight update. Nudges to somatic state are applied instantaneously, rather than being
        subject to slow equilibration. During nudging, activities are re-equilibrated across all layers.
        Negative weight updates occur when somatic state is less than zero, and are proportional to somatic state and
        presynaptic activity.
        :param projection: :class:'nn.Linear'
        :param max_pop_fraction: float in [0, 1]
        :param stochastic: bool
        :param learning_rate: float
        :param relu_gate: bool
        """
        super().__init__(projection, learning_rate)
        self.sign = sign
        self.max_pop_fraction = max_pop_fraction
        self.stochastic = stochastic
        self.relu_gate = relu_gate
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
    def step(self):
        if self.projection.direction in ['forward', 'F']:
            pre_act = torch.clamp(self.projection.pre.activity, min=0, max=1)
            delta_weight = torch.outer(torch.clamp(self.projection.post.activity, min=0, max=1),
                                       pre_act).detach().clone()
        elif self.projection.direction in ['recurrent', 'R']:
            pre_act = torch.clamp(self.projection.pre.prev_activity, min=0, max=1)
            delta_weight = torch.outer(torch.clamp(self.projection.post.activity, min=0, max=1),
                                       pre_act).detach().clone()
        sign = self.sign > 0
        self.projection.weight.data += sign * self.learning_rate * delta_weight
        
        soma_state = torch.clamp(self.projection.post.state, min=-1, max=0).detach().clone()
        if self.relu_gate:
            soma_state[self.projection.post.forward_activity == 0.] = 0.
        delta_weight = torch.outer(soma_state, pre_act).detach().clone()
        self.projection.weight.data += sign * self.learning_rate * delta_weight
    
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
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
        """
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_pop = network.output_pop
        
        # initialize populations that are updated during the backward phase
        for layer in reversed_layers:
            for pop in layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.backward_steps_activity = []
                # initialize dendritic state variables
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        pop.plateau = torch.zeros(pop.size, device=network.device)
                        pop.dend_to_soma = torch.zeros(pop.size, device=network.device)
                        break
        
        for t in range(network.forward_steps):
            for layer in reversed_layers:
                # update dendritic state variables
                cls.backward_update_layer_dendritic_state(layer)
                if t == 0:
                    for pop in layer:
                        for projection in pop:
                            # compute plateau events and nudge somatic state
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
                # update activities
                cls.backward_update_layer_activity(layer, store_dynamics=store_dynamics)
        
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
                            
                            
class Hebbian_Temporal_Contrast(BP_like_2L):
    def __init__(self, projection, max_pop_fraction=1., stochastic=False, learning_rate=None, relu_gate=True):
        """
        Output units are nudged to target. Hidden dendrites locally compute an error as the difference between
        excitation and inhibition.
        Weight updates are computed using the Contrastive Hebbian Learning approach of Xie & Seung, 2003.
        If unit selection is stochastic, hidden units are selected for a weight update with a probability proportional
        to dendritic state by sampling a Bernoulli distribution. Otherwise, units are sorted by dendritic state. A fixed
        maximum fraction of hidden units are updated at each train step. Hidden units are "nudged" by dendritic state
        when selected for a weight update. Nudges to somatic state are applied instantaneously, rather than being
        subject to slow equilibration. During nudging, activities are re-equilibrated across all layers.
        :param projection: :class:'nn.Linear'
        :param max_pop_fraction: float in [0, 1]
        :param stochastic: bool
        :param learning_rate: float
        :param relu_gate: bool
        """
        super().__init__(projection, max_pop_fraction, stochastic, learning_rate, relu_gate)
    
    def step(self):
        forward_post_activity = torch.clamp(self.projection.post.forward_activity, min=0, max=1)
        backward_post_activity = torch.clamp(self.projection.post.activity, min=0, max=1).detach().clone()
        if self.relu_gate:
            backward_post_activity[forward_post_activity == 0.] = 0.
        if self.projection.direction in ['forward', 'F']:
            forward_pre_activity = torch.clamp(self.projection.pre.forward_activity, min=0, max=1)
            backward_pre_activity = torch.clamp(self.projection.pre.activity, min=0, max=1)
        elif self.projection.direction in ['recurrent', 'R']:
            forward_pre_activity = torch.clamp(self.projection.pre.prev_forward_activity, min=0, max=1)
            backward_pre_activity = torch.clamp(self.projection.pre.prev_activity, min=0, max=1)
        delta_weight = (torch.outer(backward_post_activity, backward_pre_activity) -
                        torch.outer(forward_post_activity, forward_pre_activity)).detach().clone()
        self.projection.weight.data += self.learning_rate * delta_weight


class Top_Down_Hebbian_Temporal_Contrast_1(LearningRule):
    def __init__(self, projection, learning_rate=None, forward_only=False):
        """
        Assumes another learning rule has updated activity during a backward phase.
        Weight updates are computed using a temporally contrastive Hebbian learning approach similar to
        Xie & Seung, 2003.
        :param projection: :class:'nn.Linear'
        :param learning_rate: float
        :param forward_only: bool; whether to consult forward_activity in weight update
        """
        super().__init__(projection, learning_rate)
        self.forward_only = forward_only
    
    def step(self):
        forward_lower_layer_post_activity = torch.clamp(self.projection.post.forward_activity, min=0, max=1)
        if not self.forward_only:
            backward_lower_layer_post_activity = torch.clamp(self.projection.post.activity, min=0, max=1)
        if self.projection.direction in ['forward', 'F']:
            forward_upper_layer_pre_activity = self.projection.pre.forward_activity
            backward_upper_layer_pre_activity = self.projection.pre.activity
        elif self.projection.direction in ['recurrent', 'R']:
            forward_upper_layer_pre_activity = self.projection.pre.prev_forward_activity
            backward_upper_layer_pre_activity = self.projection.pre.prev_activity
        if self.forward_only:
            delta_weight = (torch.outer(forward_lower_layer_post_activity,
                                        torch.clamp(backward_upper_layer_pre_activity -
                                                    forward_upper_layer_pre_activity, min=-1, max=1))).detach().clone()
        else:
            delta_weight = (torch.outer(backward_lower_layer_post_activity,
                                        torch.clamp(backward_upper_layer_pre_activity, min=0, max=1)) -
                            torch.outer(forward_lower_layer_post_activity,
                                        torch.clamp(forward_upper_layer_pre_activity, min=0, max=1))).detach().clone()
        self.projection.weight.data += self.learning_rate * delta_weight


class Top_Down_Hebbian_Temporal_Contrast_3(LearningRule):
    def __init__(self, projection, learning_rate=None):
        """
        Assumes another learning rule has updated activity during a backward phase.
        Weight updates are computed using a temporally contrastive Hebbian learning approach similar to
        Xie & Seung, 2003.
        Consults presynaptic activity equilibrated during the backward phase.
        :param projection: :class:'nn.Linear'
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
    
    def step(self):
        backward_lower_layer_post_activity = torch.clamp(self.projection.post.activity, min=0, max=1)
        if self.projection.direction in ['forward', 'F']:
            forward_upper_layer_pre_activity = self.projection.pre.forward_activity
            backward_upper_layer_pre_activity = self.projection.pre.activity
        elif self.projection.direction in ['recurrent', 'R']:
            forward_upper_layer_pre_activity = self.projection.pre.prev_forward_activity
            backward_upper_layer_pre_activity = self.projection.pre.prev_activity
        delta_weight = (torch.outer(backward_lower_layer_post_activity,
                                    torch.clamp(backward_upper_layer_pre_activity -
                                                forward_upper_layer_pre_activity, min=-1, max=1))).detach().clone()
        self.projection.weight.data += self.learning_rate * delta_weight


class Top_Down_Direct(LearningRule):
    def __init__(self, projection, learning_rate=None):
        """
        Assumes another learning rule has updated activity during a backward phase.
        Weight updates are computed using the Contrastive Hebbian Learning approach of Xie & Seung, 2003.
        Consults forward (bottom up) activity and (top down) plateau.
        :param projection: :class:'nn.Linear'
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
    
    def step(self):
        lower_layer_post_forward_activity = torch.clamp(self.projection.post.forward_activity, min=0, max=1)
        upper_layer_pre_plateau = self.projection.pre.plateau
        delta_weight = torch.outer(lower_layer_post_forward_activity, upper_layer_pre_plateau).detach().clone()
        self.projection.weight.data += self.learning_rate * delta_weight
