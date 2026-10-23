from .base_classes import LearningRule, BiasLearningRule, ContinualLearningMixin
import torch
from EIANN.utils import get_scaled_rectified_sigmoid, pwlin
import math


class BTSP_19(LearningRule):
    def __init__(self, projection, neg_rate_th=None, dep_ratio=1., dep_th=0.01, dep_width=0.01, max_pop_fraction=1.,
                 temporal_discount=0.25, stochastic=False, learning_rate=None, relu_gate=True):
        """
        This method includes both positive and negative modulatory events. Both positive and negative modulatory events
        nudge the somatic activity.
        Plateaus increment an instructive signal (IS) that decays to zero in two samples. Presynaptic activity
        increments an eligibility trace (ET) that decays to zero in two samples. Positive modulatory events
        result in a BTSP weight update that depends on IS, current weight, and ET.
        Negative modulatory events result in a weight update proportional to plateaus and presynaptic activity.
        If unit selection is stochastic, hidden units are selected for a weight update with a probability proportional
        to dendritic state by sampling a Bernoulli distribution. Otherwise, units are sorted by dendritic state. A fixed
        maximum fraction of hidden units are updated at each train step. Hidden units are "nudged" by dendritic state
        when selected for a weight update. Nudges to somatic state are applied instantaneously, rather than being
        subject to slow equilibration. During nudging, activities are re-equilibrated across all layers.
        :param projection: :class:'nn.Linear'
        :param neg_rate_th: float
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param max_pop_fraction: float in [0, 1]
        :param temporal_discount: float in [0, 1]
        :param stochastic: bool
        :param learning_rate: float
        :param relu_gate: bool
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.neg_rate_th = neg_rate_th
        self.dep_ratio = dep_ratio
        self.max_pop_fraction = max_pop_fraction
        self.temporal_discount = temporal_discount
        self.stochastic = stochastic
        self.relu_gate = relu_gate
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
    def reinit(self):
        self.projection.pre.past_activity = torch.zeros(self.projection.pre.size, device=self.projection.pre.network.device)
        self.projection.post.past_plateau = torch.zeros(self.projection.post.size,
                                                   device=self.projection.post.network.device)
    
    def update(self):
        if not self.projection.pre.past_activity_updated:
            if self.projection.direction in ['forward', 'F']:
                self.projection.pre.past_activity = self.projection.pre.activity.detach().clone()
            elif self.projection.direction in ['recurrent', 'R']:
                self.projection.pre.past_activity = self.projection.pre.prev_activity.detach().clone()
            self.projection.pre.past_activity_updated = True
        
        if not self.projection.post.past_plateau_updated:
            self.projection.post.past_plateau = self.projection.post.plateau.detach().clone()
            self.projection.post.past_plateau_updated = True
    
    def step(self):
        # pos error - BTSP weight update
        plateau_prob = self.projection.post.plateau.detach().clone()
        neg_indexes = (self.projection.post.plateau < 0.).nonzero(as_tuple=True)
        plateau_prob[neg_indexes] = 0.
        plateau_prob = plateau_prob.unsqueeze(1)
        
        IS_val = 1.
        if self.projection.direction in ['forward', 'F']:
            ET = torch.clamp(self.projection.pre.activity, 0., 1.)
        elif self.projection.direction in ['recurrent', 'R']:
            ET = torch.clamp(self.projection.pre.prev_activity, 0., 1.)
            
        # pre activity and post plateau for current sample
        delta_weight = (plateau_prob *
                        ((self.w_max - self.projection.weight) * ET.unsqueeze(0) * IS_val -
                         self.projection.weight * self.dep_ratio *
                         self.q_dep(ET * IS_val).unsqueeze(0))).detach().clone()
        
        # pre activity for prev sample and post plateau for current sample
        past_ET = torch.clamp(self.projection.pre.past_activity, 0., 1.) * self.temporal_discount
        delta_weight += (plateau_prob * ((self.w_max - self.projection.weight) * past_ET.unsqueeze(0) * IS_val -
                                         self.projection.weight * self.dep_ratio *
                                         self.q_dep(past_ET * IS_val).unsqueeze(0))).detach().clone()
        
        # pre activity for current sample and post plateau for prev sample
        past_plateau_prob = self.projection.post.past_plateau.detach().clone()
        neg_indexes = (self.projection.post.past_plateau < 0.).nonzero(as_tuple=True)
        past_plateau_prob[neg_indexes] = 0.
        past_plateau_prob = past_plateau_prob.unsqueeze(1)
        
        past_IS_val = self.temporal_discount
        delta_weight += (past_plateau_prob * ((self.w_max - self.projection.weight) * ET.unsqueeze(0) * past_IS_val -
                                    self.projection.weight * self.dep_ratio *
                                    self.q_dep(ET * past_IS_val).unsqueeze(0))).detach().clone()
        
        # neg error - weight update proportional to loss and presynaptic activity
        neg_error = self.projection.post.plateau.detach().clone()
        neg_error[self.projection.post.plateau > 0.] = 0.
        delta_weight += ET.unsqueeze(0) * neg_error.unsqueeze(1)
        
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
                                    
                                    if projection.learning_rule.neg_rate_th is not None:
                                        local_loss[pop.forward_activity <= projection.learning_rule.neg_rate_th] = 0.
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
        
        pop_history_stored = []
        for layer in network:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        if store_history and pop not in pop_history_stored:
                            pop_history_stored.append(pop)
                            pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                            pop.append_attribute_history('backward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        pop.past_plateau_updated = False
                        projection.pre.past_activity_updated = False
                if store_history:
                    if pop.backward_projections or pop is output_pop:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_20(LearningRule):
    def __init__(self, projection, neg_rate_th=None, dep_ratio=1., dep_th=0.01, dep_width=0.01, max_pop_fraction=1.,
                 stochastic=False, learning_rate=None, relu_gate=True):
        """
        Like BTSP_19 but with no cross-sample temporal interaction.
        This method includes both positive and negative modulatory events. Both positive and negative modulatory events
        nudge the somatic activity.
        Plateaus activate an instructive signal (IS). Presynaptic activity activates an eligibility trace (ET).
        Positive modulatory events result in a BTSP weight update that depends on IS, current weight, and ET.
        Negative modulatory events result in a weight update proportional to plateaus and presynaptic activity.
        If unit selection is stochastic, hidden units are selected for a weight update with a probability proportional
        to dendritic state by sampling a Bernoulli distribution. Otherwise, units are sorted by dendritic state. A fixed
        maximum fraction of hidden units are updated at each train step. Hidden units are "nudged" by dendritic state
        when selected for a weight update. Nudges to somatic state are applied instantaneously, rather than being
        subject to slow equilibration. During nudging, activities are re-equilibrated across all layers.
        :param projection: :class:'nn.Linear'
        :param neg_rate_th: float
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param max_pop_fraction: float in [0, 1]
        :param stochastic: bool
        :param learning_rate: float
        :param relu_gate: bool
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.neg_rate_th = neg_rate_th
        self.dep_ratio = dep_ratio
        self.max_pop_fraction = max_pop_fraction
        self.stochastic = stochastic
        self.relu_gate = relu_gate
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
    def step(self):
        # pos error - BTSP weight update
        plateau_prob = self.projection.post.plateau.detach().clone()
        neg_indexes = (self.projection.post.plateau < 0.).nonzero(as_tuple=True)
        plateau_prob[neg_indexes] = 0.
        plateau_prob = plateau_prob.unsqueeze(1)
        
        if self.projection.direction in ['forward', 'F']:
            ET = torch.clamp(self.projection.pre.activity, 0., 1.)
        elif self.projection.direction in ['recurrent', 'R']:
            ET = torch.clamp(self.projection.pre.prev_activity, 0., 1.)
        
        # pre activity and post plateau for current sample
        delta_weight = (plateau_prob *
                        ((self.w_max - self.projection.weight) * ET.unsqueeze(0) -
                         self.projection.weight * self.dep_ratio *
                         self.q_dep(ET).unsqueeze(0))).detach().clone()
        
        # neg error - weight update proportional to loss and presynaptic activity
        neg_error = self.projection.post.plateau.detach().clone()
        neg_error[self.projection.post.plateau > 0.] = 0.
        delta_weight += ET.unsqueeze(0) * neg_error.unsqueeze(1)
        
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
                                    
                                    if projection.learning_rule.neg_rate_th is not None:
                                        local_loss[pop.forward_activity <= projection.learning_rule.neg_rate_th] = 0.
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
        
        pop_history_stored = []
        for layer in network:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        if store_history and pop not in pop_history_stored:
                            pop_history_stored.append(pop)
                            pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                            pop.append_attribute_history('backward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                if store_history:
                    if pop.backward_projections or pop is output_pop:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_ELR_1(LearningRule):
    def __init__(self, projection, dep_ratio=1., dep_th=0.01, dep_width=0.01, elr_threshold=0., elr_width=0.5,
                 elr_scale=0.1, elr_min=0.01, learning_rate=None):
        """
        Like BTSP_19 but with no cross-sample temporal interaction, and with an elastic learning rate that depends on
        distance from w to w_eq.
        This method includes both positive and negative modulatory events. Both positive and negative modulatory events
        nudge the somatic activity.
        Plateaus activate an instructive signal (IS). Presynaptic activity activates an eligibility trace (ET).
        Positive modulatory events result in a BTSP weight update that depends on IS, current weight, and ET.
        Positive modulatory events results in a modulation of learning rate.
        Negative modulatory events result in a weight update proportional to plateaus and presynaptic activity.
        Nudges to somatic state are applied instantaneously, rather than being subject to slow equilibration. During
        nudging, activities are re-equilibrated across all layers.
        :param projection: :class:'nn.Linear'
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param elr_threshold: float
        :param elr_width: float
        :param elr_scale: float
        :param elr_min: float
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.dep_ratio = dep_ratio
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]
        self.elr_threshold = elr_threshold
        self.elr_width = elr_width
        self.elr_scale = elr_scale
        self.elr_min = elr_min
        self.lr_mod = torch.ones_like(projection.weight.data)
        self.lr_mod_func = pwlin(-self.elr_scale, self.elr_scale, self.elr_threshold,
                                 self.elr_threshold + self.elr_width)
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
    def step(self):
        # pos error - BTSP weight update
        plateau_prob = self.projection.post.plateau.detach().clone()
        neg_indexes = (self.projection.post.plateau < 0.).nonzero(as_tuple=True)
        plateau_prob[neg_indexes] = 0.
        plateau_prob = plateau_prob.unsqueeze(1)
        
        if self.projection.direction in ['forward', 'F']:
            ET = torch.clamp(self.projection.pre.activity.detach().clone(), 0., 1.)
        elif self.projection.direction in ['recurrent', 'R']:
            ET = torch.clamp(self.projection.pre.prev_activity.detach().clone(), 0., 1.)
        
        # pre activity and post plateau for current sample
        BTSP_delta_weight_direction = ((self.w_max - self.projection.weight) * ET.unsqueeze(0) -
                                       self.projection.weight * self.dep_ratio * self.q_dep(ET).unsqueeze(0))
        delta_weight = plateau_prob * BTSP_delta_weight_direction
        
        # neg error - weight update proportional to loss and presynaptic activity
        neg_error = self.projection.post.plateau.detach().clone()
        neg_error[self.projection.post.plateau > 0.] = 0.
        delta_weight += ET.unsqueeze(0) * neg_error.unsqueeze(1)
        
        self.projection.weight.data += self.learning_rate * self.lr_mod * delta_weight
        
        # efficient shortcut to compute distance_to_w_eq
        denom = ET + self.dep_ratio * self.q_dep(ET)
        valid_indexes = (ET > 0.).nonzero()
        distance_to_w_eq = -self.projection.weight.data.clone()
        distance_to_w_eq[:, valid_indexes] = torch.abs(BTSP_delta_weight_direction[:, valid_indexes] /
                                                       denom.unsqueeze(0)[:, valid_indexes])
        delta_lr_mod = plateau_prob * ET.unsqueeze(0) * self.lr_mod_func(distance_to_w_eq)
        
        # elastic learning rate modulation
        self.lr_mod += delta_lr_mod
        self.lr_mod.clamp_(min=self.elr_min, max=1.)
    
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
                                    # relu gate
                                    local_loss[output_pop.forward_activity == 0.] = 0
                                    output_pop.plateau[:] = local_loss.detach().clone()
                                    output_pop.dend_to_soma[:] = local_loss.detach().clone()
                                else:
                                    local_loss = pop.dendritic_state.detach().clone()
                                    # relu gate
                                    local_loss[pop.forward_activity == 0.] = 0
                                    
                                    pos_avail_indexes = (local_loss > 0.).nonzero().squeeze(1)
                                    sorted, pos_candidate_rel_indexes = torch.sort(local_loss[pos_avail_indexes],
                                                                                   descending=True, stable=True)
                                    pos_event_indexes = pos_avail_indexes[pos_candidate_rel_indexes]
                                    
                                    neg_avail_indexes = (local_loss < 0.).nonzero().squeeze(1)
                                    sorted, neg_candidate_rel_indexes = torch.sort(local_loss[neg_avail_indexes],
                                                                                   descending=True, stable=True)
                                    neg_event_indexes = neg_avail_indexes[neg_candidate_rel_indexes]
                                    
                                    pop.plateau[pos_event_indexes] = local_loss[pos_event_indexes]
                                    pop.plateau[neg_event_indexes] = local_loss[neg_event_indexes]
                                    pop.dend_to_soma[pos_event_indexes] = local_loss[pos_event_indexes]
                                    pop.dend_to_soma[neg_event_indexes] = local_loss[neg_event_indexes]
                                break
                # update activities
                cls.backward_update_layer_activity(layer, store_dynamics=store_dynamics)
        
        pop_history_stored = []
        for layer in network:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        if store_history and pop not in pop_history_stored:
                            pop_history_stored.append(pop)
                            pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                            pop.append_attribute_history('backward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                if store_history:
                    if pop.backward_projections or pop is output_pop:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_CL(ContinualLearningMixin, LearningRule):
    def __init__(self, projection, neg_rate_th=None, dep_ratio=1., dep_th=0.01, dep_width=0.01, max_pop_fraction=1.,
                 stochastic=False, learning_rate=None, relu_gate=True, task_incremental=False, task_classes=None):
        """
        Continual-learning version of BTSP_20 (see ContinualLearningMixin): with task_incremental=True, the output
        error is restricted to the output units of the current task's classes. Otherwise identical to BTSP_20:
        Like BTSP_19 but with no cross-sample temporal interaction.
        This method includes both positive and negative modulatory events. Both positive and negative modulatory events
        nudge the somatic activity.
        Plateaus activate an instructive signal (IS). Presynaptic activity activates an eligibility trace (ET).
        Positive modulatory events result in a BTSP weight update that depends on IS, current weight, and ET.
        Negative modulatory events result in a weight update proportional to plateaus and presynaptic activity.
        If unit selection is stochastic, hidden units are selected for a weight update with a probability proportional
        to dendritic state by sampling a Bernoulli distribution. Otherwise, units are sorted by dendritic state. A fixed
        maximum fraction of hidden units are updated at each train step. Hidden units are "nudged" by dendritic state
        when selected for a weight update. Nudges to somatic state are applied instantaneously, rather than being
        subject to slow equilibration. During nudging, activities are re-equilibrated across all layers.
        :param projection: :class:'nn.Linear'
        :param neg_rate_th: float
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param max_pop_fraction: float in [0, 1]
        :param stochastic: bool
        :param learning_rate: float
        :param relu_gate: bool
        :param task_incremental: bool
        :param task_classes: list of lists of int; the classes in each task (required if task_incremental)
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.neg_rate_th = neg_rate_th
        self.dep_ratio = dep_ratio
        self.max_pop_fraction = max_pop_fraction
        self.stochastic = stochastic
        self.relu_gate = relu_gate
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
        self.init_continual_learning(task_incremental, task_classes)

    def compute_delta_weight(self):
        """
        Unscaled weight update of the rule: a BTSP update for positive plateaus, plus a weight update proportional to
        negative plateaus and presynaptic activity.
        :return: tensor
        """
        # pos error - BTSP weight update
        plateau_prob = self.projection.post.plateau.detach().clone()
        neg_indexes = (self.projection.post.plateau < 0.).nonzero(as_tuple=True)
        plateau_prob[neg_indexes] = 0.
        plateau_prob = plateau_prob.unsqueeze(1)

        if self.projection.direction in ['forward', 'F']:
            ET = torch.clamp(self.projection.pre.activity, 0., 1.)
        elif self.projection.direction in ['recurrent', 'R']:
            ET = torch.clamp(self.projection.pre.prev_activity, 0., 1.)

        # pre activity and post plateau for current sample
        delta_weight = (plateau_prob *
                        ((self.w_max - self.projection.weight) * ET.unsqueeze(0) -
                         self.projection.weight * self.dep_ratio *
                         self.q_dep(ET).unsqueeze(0))).detach().clone()

        # neg error - weight update proportional to loss and presynaptic activity
        neg_error = self.projection.post.plateau.detach().clone()
        neg_error[self.projection.post.plateau > 0.] = 0.
        delta_weight += ET.unsqueeze(0) * neg_error.unsqueeze(1)
        return delta_weight

    def step(self):
        self.projection.weight.data += self.learning_rate * self.compute_delta_weight()

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
        # restrict the output error to the current task (unchanged unless task_incremental)
        for projection in network.projections.values():
            if projection.learning_rule.__class__ == cls:
                target = projection.learning_rule.task_target(network, target)
                break
        
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
                                    
                                    if projection.learning_rule.neg_rate_th is not None:
                                        local_loss[pop.forward_activity <= projection.learning_rule.neg_rate_th] = 0.
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
        
        pop_history_stored = []
        for layer in network:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        if store_history and pop not in pop_history_stored:
                            pop_history_stored.append(pop)
                            pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                            pop.append_attribute_history('backward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                if store_history:
                    if pop.backward_projections or pop is output_pop:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_SI(BTSP_CL):
    def __init__(self, projection, si_lambda=1., si_xi=1e-3, **kwargs):
        """
        BTSP_CL with a Synaptic Intelligence regularizer (Zenke et al., 2017, arXiv:1703.04200), adapted to a rule
        without a loss gradient, as in DTP_SI. The rule's own unscaled update, delta_weight (see
        BTSP_CL.compute_delta_weight), takes the place of the negative gradient of the unregularized task loss. Unlike
        DTP_SI, delta_weight depends on the current weight for positive plateaus (potentiation toward w_max and
        depression via q_dep), so it is the rule's own drive rather than the gradient of any loss.
        During each task, the per-weight importance is accumulated online as the path integral
            small_omega += delta_weight * actual_delta_weight,
        where actual_delta_weight is the actual weight change of each train step (after the penalty and weight
        clamping). At the end of each task (update_CL_states), the cumulative importance
        omega = relu(omega + small_omega / (total task weight change ** 2 + si_xi)), the anchor weight is set to the
        current weight, and small_omega is reset. During subsequent tasks, each projection is updated with
            learning_rate * (delta_weight - 2 * si_lambda * omega * (weight - anchor_weight)),
        i.e. the negative gradient of the penalty si_lambda * (omega * (weight - anchor_weight) ** 2).sum() is added
        to the rule's update. Do not combine with BTSP_CL or BTSP_ELR_CL projections in the same network, since they
        share BTSP_CL.backward.
        :param projection: :class:'nn.Linear'
        :param si_lambda: float; strength of the penalty
        :param si_xi: float; damping term of the importance normalization
        :param kwargs: passed to BTSP_CL (e.g. dep_ratio, dep_th, dep_width, max_pop_fraction, learning_rate,
            task_incremental, task_classes)
        """
        super().__init__(projection, **kwargs)
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


class BTSP_SI_HD_v1(BTSP_CL):
    # lr_mod is updated during training, not set at the task switch (used by scripts/get_cl_metrics.py)
    continuous_lr_mod = True

    def __init__(self, projection, hd_gamma=1e-3, hd_beta=1., si_xi=1e-3, lr_min=0., lr_max=None, **kwargs):
        """
        BTSP_CL with Synaptic Intelligence (Zenke et al., 2017, arXiv:1703.04200) implemented as a per-weight learning
        rate adapted by hypergradient descent on the SI regularizer alone, as in Backprop_SI_HD_v1, instead of a loss
        penalty. As in BTSP_SI, the rule's own unscaled update, delta_weight (see BTSP_CL.compute_delta_weight), takes
        the place of the negative gradient of the unregularized task loss, and the per-weight importance omega and the
        anchor weight (the weight at the end of the previous task) are computed as in BTSP_SI. Weights are updated with
        a per-weight learning rate lr = learning_rate * lr_mod:
            W_p[t+1] = W_p[t] + lr_p[t] * delta_weight_p[t]
        so dW_p[t] / dlr_p[t-1] = delta_weight_p[t-1], and before every train step lr descends the hypergradient of
        hd_beta / 2 * sum_p (omega_p * (W_p - W_anchor_p) ** 2).sum():
            lr_p[t] = lr_p[t-1] - hd_gamma * delta_weight_p[t-1] * hd_beta * omega_p * (W_p[t] - W_anchor_p)
            lr_p[t] = clamp(lr_p[t], lr_min, lr_max)
        hd_gamma and hd_beta only enter as their product, so only one of them needs to be optimized. lr starts at
        learning_rate. During the first task omega is not defined, so lr stays at learning_rate. lr is not reset at task
        boundaries. The hypergradient ignores weight clamping (it uses delta_weight_p[t-1] even when the weight was
        clamped). Do not combine with other BTSP_CL projections in the same network, since they share
        BTSP_CL.backward.
        :param projection: :class:'nn.Linear'
        :param hd_gamma: float; hypergradient step size
        :param hd_beta: float; strength of the SI regularizer
        :param si_xi: float; damping term of the importance normalization
        :param lr_min: float; lower bound of the per-weight learning rate
        :param lr_max: float or None; upper bound of the per-weight learning rate
        :param kwargs: passed to BTSP_CL (e.g. dep_ratio, dep_th, dep_width, max_pop_fraction, learning_rate,
            task_incremental, task_classes)
        """
        super().__init__(projection, **kwargs)
        self.hd_gamma = hd_gamma
        self.hd_beta = hd_beta
        self.si_xi = si_xi
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.omega = None
        self.small_omega = torch.zeros_like(projection.weight.data)
        # Weights are initialized by the network after the projection is built, so this is set on the first step
        self.task_start_weight = None
        self.unreg_delta_weight = None
        self.prev_weight = None
        self.lr_mod = torch.ones_like(projection.weight.data)

    def update_lr(self):
        """
        Hypergradient step on the per-weight learning rate, using the rule's update of the previous train step and the
        gradient of the SI regularizer at the current weights.
        """
        if self.unreg_delta_weight is None or self.omega is None:
            return
        hypergrad = self.hd_beta * self.omega * (self.projection.weight.detach() - self.task_start_weight)
        lr = self.learning_rate * self.lr_mod - self.hd_gamma * self.unreg_delta_weight * hypergrad
        lr.clamp_(min=self.lr_min, max=self.lr_max)
        self.lr_mod = lr / self.learning_rate

    def step(self):
        if self.task_start_weight is None:
            self.task_start_weight = self.projection.weight.detach().clone()
        delta_weight = self.compute_delta_weight()
        # unreg_delta_weight still holds the update of the previous train step
        self.update_lr()
        self.unreg_delta_weight = delta_weight.detach().clone()
        self.prev_weight = self.projection.weight.detach().clone()
        self.projection.weight.data += self.learning_rate * self.lr_mod * delta_weight

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
            # Clamp at zero, as in the reference implementation (ganguli-lab/pathint)
            self.omega = torch.relu(omega)
        self.task_start_weight = weight
        self.small_omega = torch.zeros_like(weight)
        super().update_CL_states()


class BTSP_ELR_CL(BTSP_CL):
    def __init__(self, projection, elr_threshold=0., elr_width=0.5, elr_scale=0.1, elr_min=0.01, **kwargs):
        """
        BTSP_CL with an elastic learning rate (as in BTSP_ELR_1) that depends on distance from w to w_eq. Positive
        modulatory events modulate the learning rate of each weight. The elastic learning rate is a continual mechanism
        and needs no update at task boundaries. Do not combine with BTSP_CL projections in the same network, since
        both share BTSP_CL.backward. With BTSP_CL's default kwargs (max_pop_fraction=1., stochastic=False,
        relu_gate=True, neg_rate_th=None), this is equivalent to BTSP_ELR_1.
        :param projection: :class:'nn.Linear'
        :param elr_threshold: float
        :param elr_width: float
        :param elr_scale: float
        :param elr_min: float
        :param kwargs: passed to BTSP_CL (e.g. dep_ratio, dep_th, dep_width, learning_rate, task_incremental,
            task_classes)
        """
        super().__init__(projection, **kwargs)
        self.elr_threshold = elr_threshold
        self.elr_width = elr_width
        self.elr_scale = elr_scale
        self.elr_min = elr_min
        self.lr_mod = torch.ones_like(projection.weight.data)
        self.lr_mod_func = pwlin(-self.elr_scale, self.elr_scale, self.elr_threshold,
                                 self.elr_threshold + self.elr_width)
    
    def step(self):
        # pos error - BTSP weight update
        plateau_prob = self.projection.post.plateau.detach().clone()
        neg_indexes = (self.projection.post.plateau < 0.).nonzero(as_tuple=True)
        plateau_prob[neg_indexes] = 0.
        plateau_prob = plateau_prob.unsqueeze(1)
        
        if self.projection.direction in ['forward', 'F']:
            ET = torch.clamp(self.projection.pre.activity.detach().clone(), 0., 1.)
        elif self.projection.direction in ['recurrent', 'R']:
            ET = torch.clamp(self.projection.pre.prev_activity.detach().clone(), 0., 1.)
        
        # pre activity and post plateau for current sample
        BTSP_delta_weight_direction = ((self.w_max - self.projection.weight) * ET.unsqueeze(0) -
                                       self.projection.weight * self.dep_ratio * self.q_dep(ET).unsqueeze(0))
        delta_weight = plateau_prob * BTSP_delta_weight_direction
        
        # neg error - weight update proportional to loss and presynaptic activity
        neg_error = self.projection.post.plateau.detach().clone()
        neg_error[self.projection.post.plateau > 0.] = 0.
        delta_weight += ET.unsqueeze(0) * neg_error.unsqueeze(1)
        
        self.projection.weight.data += self.learning_rate * self.lr_mod * delta_weight
        
        # efficient shortcut to compute distance_to_w_eq
        denom = ET + self.dep_ratio * self.q_dep(ET)
        valid_indexes = (ET > 0.).nonzero()
        distance_to_w_eq = -self.projection.weight.data.clone()
        distance_to_w_eq[:, valid_indexes] = torch.abs(BTSP_delta_weight_direction[:, valid_indexes] /
                                                       denom.unsqueeze(0)[:, valid_indexes])
        delta_lr_mod = plateau_prob * ET.unsqueeze(0) * self.lr_mod_func(distance_to_w_eq)
        
        # elastic learning rate modulation
        self.lr_mod += delta_lr_mod
        self.lr_mod.clamp_(min=self.elr_min, max=1.)


class BTSP_BP_Mixin(object):
    """
    Shared backward method for BTSP rules whose plateaus come from backprop instead of dendritic top-down inputs
    (BTSP_BP_CL, BTSP_ELR_BP_CL). The plateau of each post population is the negative gradient of the loss with
    respect to its final forward step activity (-dL/da), scaled by the error_scale of the rule of the projections onto
    that population, clamped to [-1, 1] and (with relu_gate) zeroed for inactive units. No somatic nudging is applied
    and the optimizer is not stepped; weights are updated by the BTSP step() of the parent rule. Requires
    backward_steps >= 1. Do not combine with BTSP_CL-family or Backprop rules in the same network.
    """
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        Compute plateaus from the gradient of the loss with respect to post population activities.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
        """
        post_pop_rules = {}
        for projection in network.projections.values():
            if isinstance(projection.learning_rule, BTSP_BP_Mixin) and projection.post not in post_pop_rules:
                post_pop_rules[projection.post] = projection.learning_rule
        if not post_pop_rules:
            return
        first_rule = next(iter(post_pop_rules.values()))
        
        # restrict the output error to the current task (unchanged unless task_incremental)
        target = first_rule.task_target(network, target)
        loss = network.criterion(output, target)
        post_pops = list(post_pop_rules.keys())
        grads = torch.autograd.grad(loss, [pop.activity for pop in post_pops], allow_unused=True)
        
        for pop, grad in zip(post_pops, grads):
            if grad is None:
                grad = torch.zeros_like(pop.activity)
            local_loss = torch.clamp(-grad.detach() * post_pop_rules[pop].error_scale, min=-1, max=1)
            pop.dendritic_state = local_loss.clone()
            if post_pop_rules[pop].relu_gate:
                local_loss[pop.forward_activity == 0.] = 0
            pop.plateau = local_loss
            
            if store_history:
                pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                pop.append_attribute_history('backward_dendritic_state', pop.dendritic_state.detach().clone())
                pop.append_attribute_history('backward_activity', pop.forward_activity.detach().clone())


class BTSP_BP_CL(BTSP_BP_Mixin, BTSP_CL):
    def __init__(self, projection, error_scale=1., **kwargs):
        """
        BTSP_CL weight update with plateaus computed by backprop (see BTSP_BP_Mixin) instead of dendritic top-down
        inputs. max_pop_fraction, stochastic and neg_rate_th are ignored.
        :param projection: :class:'nn.Linear'
        :param error_scale: float; scales -dL/da before clamping to [-1, 1]
        :param kwargs: passed to BTSP_CL (e.g. dep_ratio, dep_th, dep_width, learning_rate, relu_gate,
            task_incremental, task_classes)
        """
        super().__init__(projection, **kwargs)
        self.error_scale = error_scale
        projection.weight.requires_grad = True


class BTSP_ELR_BP_CL(BTSP_BP_Mixin, BTSP_ELR_CL):
    def __init__(self, projection, error_scale=1., **kwargs):
        """
        BTSP_ELR_CL weight update (with elastic learning rate) with plateaus computed by backprop (see BTSP_BP_Mixin)
        instead of dendritic top-down inputs. max_pop_fraction, stochastic and neg_rate_th are ignored.
        :param projection: :class:'nn.Linear'
        :param error_scale: float; scales -dL/da before clamping to [-1, 1]
        :param kwargs: passed to BTSP_ELR_CL (e.g. elr_threshold, elr_width, elr_scale, elr_min, dep_ratio, dep_th,
            dep_width, learning_rate, relu_gate, task_incremental, task_classes)
        """
        super().__init__(projection, **kwargs)
        self.error_scale = error_scale
        projection.weight.requires_grad = True
