from .base_classes import LearningRule, BiasLearningRule
import torch
import math


class BP_like_1(LearningRule):
    def __init__(self, projection, learning_rate=None):
        """
        Output units are nudged to target. Hidden dendrites locally compute an error as the difference between before
        and after nudge. Weight updates are proportional to local error and presynaptic firing rate.
        :param projection: :class:'nn.Linear'
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
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
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
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
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
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
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
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
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
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
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
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


class BP_like_1E(LearningRule):
    def __init__(self, projection, max_pop_fraction=0.025, stochastic=True, learning_rate=None, relu_gate=False):
        """
        Output units are nudged to target. Hidden dendrites locally compute an error as the difference between
        forward and backward dendritic state.
        Weight updates are proportional to local error and (forward) presynaptic firing rate.
        If unit selection is stochastic, hidden units are selected for a weight update with a probability proportional
        to dendritic state by sampling a Bernoulli distribution. Otherwise, units are sorted by dendritic state. A fixed
        maximum fraction of hidden units are updated at each train step. Hidden units are "nudged" by dendritic state
        when selected for a weight update. Nudges to somatic state are applied instantaneously, rather than being
        subject to slow equilibration. Nudged activity is only passed top-down, but not laterally, with no
        equilibration.
        :param projection: :class:'nn.Linear'
        :param max_pop_fraction: float in [0, 1]
        :param stochastic: bool
        :param learning_rate: float
        :param relu_gate: bool
        """
        super().__init__(projection, learning_rate)
        self.max_pop_fraction = max_pop_fraction
        self.stochastic = stochastic
        self.relu_gate = relu_gate
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
    def step(self):
        if self.projection.direction in ['forward', 'F']:
            delta_weight = torch.outer(self.projection.post.plateau,
                                       torch.clamp(self.projection.pre.forward_activity, min=0, max=1))
        elif self.projection.direction in ['recurrent', 'R']:
            delta_weight = torch.outer(self.projection.post.plateau,
                                       torch.clamp(self.projection.pre.forward_prev_activity, min=0, max=1))
        self.projection.weight.data += self.learning_rate * delta_weight
    
    @classmethod
    def backward_nudge_activity(cls, layer, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive a nudge.
        :param layer:
        :param store_dynamics: bool
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
        
        for layer in reversed_layers:
            # update dendritic state variables
            cls.backward_update_layer_dendritic_state(layer)
            for pop in layer:
                for projection in pop:
                    # compute plateau events and nudge somatic state
                    if projection.learning_rule.__class__ == cls:
                        if pop is output_pop:
                            local_loss = torch.clamp(target - output_pop.activity, min=-1, max=1)
                            output_pop.dendritic_state = local_loss.detach().clone()
                            if projection.learning_rule.relu_gate:
                                local_loss[output_pop.forward_activity == 0.] = 0.
                            output_pop.plateau[:] = local_loss.detach().clone()
                            output_pop.dend_to_soma[:] = local_loss.detach().clone()
                        else:
                            max_units = math.ceil(projection.learning_rule.max_pop_fraction * pop.size)
                            local_loss = (pop.dendritic_state - pop.forward_dendritic_state).clamp_(-1., 1.)
                            if projection.learning_rule.relu_gate:
                                local_loss[pop.forward_activity == 0.] = 0.
                            
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
                            
                            pop.plateau[pos_event_indexes] = local_loss[pos_event_indexes].detach().clone()
                            pop.plateau[neg_event_indexes] = local_loss[neg_event_indexes].detach().clone()
                            pop.dend_to_soma[pos_event_indexes] = (
                                local_loss[pos_event_indexes].detach().clone())
                            pop.dend_to_soma[neg_event_indexes] = (
                                local_loss[neg_event_indexes].detach().clone())
                        break
            # update activities
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


class BP_like_1I(LearningRule):
    def __init__(self, projection, max_pop_fraction=0.025, stochastic=True, learning_rate=None, relu_gate=False):
        """
        Output units are nudged to target. Hidden dendrites locally compute an error as the difference between
        forward and backward dendritic state.
        Weight updates are proportional to local error and presynaptic firing rate (after backward phase equilibration).
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
        self.max_pop_fraction = max_pop_fraction
        self.stochastic = stochastic
        self.relu_gate = relu_gate
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
    def step(self):
        if self.projection.direction in ['forward', 'F']:
            delta_weight = torch.outer(self.projection.post.plateau,
                                       torch.clamp(self.projection.pre.activity, min=0, max=1))
        elif self.projection.direction in ['recurrent', 'R']:
            delta_weight = torch.outer(self.projection.post.plateau,
                                       torch.clamp(self.projection.pre.prev_activity, min=0, max=1))
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
                                        local_loss[output_pop.forward_activity == 0.] = 0.
                                    output_pop.plateau[:] = local_loss.detach().clone()
                                    output_pop.dend_to_soma[:] = local_loss.detach().clone()
                                else:
                                    max_units = math.ceil(projection.learning_rule.max_pop_fraction * pop.size)
                                    local_loss = (pop.dendritic_state - pop.forward_dendritic_state).clamp_(-1., 1.)
                                    if projection.learning_rule.relu_gate:
                                        local_loss[pop.forward_activity == 0.] = 0.
                                    
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


class BP_like_2E(LearningRule):
    def __init__(self, projection, max_pop_fraction=0.025, stochastic=True, learning_rate=None, relu_gate=False):
        """
        Output units are nudged to target. Hidden dendrites locally compute an error as the difference between
        excitation and inhibition. Weight updates are proportional to local error and (forward) presynaptic firing rate.
        If unit selection is stochastic, hidden units are selected for a weight update with a probability proportional
        to dendritic state by sampling a Bernoulli distribution. Otherwise, units are sorted by dendritic state. A fixed
        maximum fraction of hidden units are updated at each train step. Hidden units are "nudged" by dendritic state
        when selected for a weight update. Nudges to somatic state are applied instantaneously, rather than being
        subject to slow equilibration. Nudged activity is only passed top-down, but not laterally, with no
        equilibration.
        :param projection: :class:'nn.Linear'
        :param max_pop_fraction: float in [0, 1]
        :param stochastic: bool
        :param learning_rate: float
        :param relu_gate: bool
        """
        super().__init__(projection, learning_rate)
        self.max_pop_fraction = max_pop_fraction
        self.stochastic = stochastic
        self.relu_gate = relu_gate
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
    def step(self):
        if self.projection.direction in ['forward', 'F']:
            delta_weight = torch.outer(self.projection.post.plateau,
                                       torch.clamp(self.projection.pre.forward_activity, min=0, max=1))
        elif self.projection.direction in ['recurrent', 'R']:
            delta_weight = torch.outer(self.projection.post.plateau,
                                       torch.clamp(self.projection.pre.forward_prev_activity, min=0, max=1))
        self.projection.weight.data += self.learning_rate * delta_weight
    
    @classmethod
    def backward_nudge_activity(cls, layer, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive a nudge.
        :param layer:
        :param store_dynamics: bool
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
        
        for layer in reversed_layers:
            # update dendritic state variables
            cls.backward_update_layer_dendritic_state(layer)
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


class BP_like_2L(LearningRule):
    def __init__(self, projection, max_pop_fraction=0.025, stochastic=True, learning_rate=None, relu_gate=False):
        """
        Output units are nudged to target. Hidden dendrites locally compute an error as the difference between
        excitation and inhibition. Weight updates are proportional to local error and presynaptic firing rate (after
        backward phase equilibration).
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
        self.max_pop_fraction = max_pop_fraction
        self.stochastic = stochastic
        self.relu_gate = relu_gate
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
    def step(self):
        if self.projection.direction in ['forward', 'F']:
            delta_weight = torch.outer(self.projection.post.plateau,
                                       torch.clamp(self.projection.pre.activity, min=0, max=1))
        elif self.projection.direction in ['recurrent', 'R']:
            delta_weight = torch.outer(self.projection.post.plateau,
                                       torch.clamp(self.projection.pre.prev_activity, min=0, max=1))
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


class VTC_1(LearningRule):
    """
    Vectorized_Temporal_Contrast.
    This rule is used to learn top-down weights and approximate weight symmetry.
    del_W_ij ~ A_i_forward @ (A_j_backward - A_j_forward)
    """
    
    def __init__(self, projection, sign=1, learning_rate=None):
        super().__init__(projection, learning_rate)
        self.sign = sign
    
    def step(self):
        delta_weight = torch.outer(
                torch.clamp(self.projection.post.forward_activity, min=0, max=1),
                torch.clamp((self.projection.pre.activity - self.projection.pre.forward_activity),
                            min=-1, max=1))
        
        self.projection.weight.data += self.sign * self.learning_rate * delta_weight




########################################################################################################################
########################################################################################################################


class almost_backprop1(LearningRule):
    '''
    As close as possible to a manual version of backprop.
    - Errors passed in dendrites
    - Perfect cancellation of forward activity
    - Gating with derivative of activation function (only accurate for ReLU)
    - Does not factor gradients passing through recurrent connections (e.g. somaI)
    '''
    def __init__(self, projection, learning_rate=None):
        super().__init__(projection, learning_rate)

    def step(self):
        ''' Update the weights '''
        # Backprop weight update
        delta_weight = torch.outer(self.projection.post.dendritic_state, self.projection.pre.activity)
        self.projection.weight.data += self.learning_rate * delta_weight

    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        # Compute Output loss & set dendritic state
        global_error = target - output

        # Set dendritic state with the local loss in each layer
        reversed_layers = list(network)[::-1]
        network.output_pop.dendritic_state = global_error
        for i,layer in enumerate(reversed_layers[:-1]): # Iterate over populations in reverse order starting from the output layer
            d_activation = layer.E.activity > 0
            layer.E.dendritic_state = layer.E.dendritic_state * d_activation # gate updates based on derivative of activation function (in case of ReLU: = 1 if activity>0)
            layer.E.backward_activity = layer.E.activity + layer.E.dendritic_state # nudge somatic state

            pre_layer = reversed_layers[i+1]
            forward_weight_transpose = layer.E.incoming_projections[f"{layer.E.fullname}_{pre_layer.E.fullname}"].weight

            topdown_excitation =  layer.E.backward_activity @ forward_weight_transpose
            lateral_inhibition =  layer.E.activity @ forward_weight_transpose
            pre_layer.E.dendritic_state = topdown_excitation - lateral_inhibition


class almost_backprop2(LearningRule):
    '''
    - Errors passed in dendrites
    - Perfect cancellation of forward activity via dendI (with cloned weights)
    - Gating with derivative of activation function (only accurate for ReLU)
    - Does not factor gradients passing through recurrent connections (e.g. somaI)
    '''
    def __init__(self, projection, learning_rate=None):
        super().__init__(projection, learning_rate)

    def step(self):
        ''' Update the weights '''
        # Backprop weight update
        delta_weight = torch.outer(self.projection.post.dendritic_state, self.projection.pre.activity)
        self.projection.weight.data += self.learning_rate * delta_weight

    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        # Compute Output loss & set dendritic state
        global_error = target - output

        # Set dendritic state with the local loss in each layer
        reversed_layers = list(network)[::-1]
        network.output_pop.dendritic_state = global_error
        for i,layer in enumerate(reversed_layers[:-1]): # Iterate over populations in reverse order starting from the output layer (skipping the Input layer)
            d_activation = layer.E.activity > 0
            layer.E.dendritic_state = layer.E.dendritic_state * d_activation # gate updates based on derivative of activation function (in case of ReLU: = 1 if activity>0)
            layer.E.backward_activity = layer.E.activity + layer.E.dendritic_state # nudge somatic state

            # Compute the dendritic state of the lower layer
            pre_layer = reversed_layers[i+1]
            if pre_layer != network.Input:
                forward_weight_transpose = layer.E.incoming_projections[f"{layer.E.fullname}_{pre_layer.E.fullname}"].weight
                topdown_excitation =  layer.E.backward_activity @ forward_weight_transpose

                pre_layer.DendI.activity = layer.E.activity.clone()
                dendI_projection = network.module_dict[f"{pre_layer.E.fullname}_{pre_layer.DendI.fullname}"]
                lateral_inhibition =  pre_layer.DendI.activity @ dendI_projection.weight.T

                pre_layer.E.dendritic_state = topdown_excitation + lateral_inhibition

       
class almost_backprop3(LearningRule):
    '''
    - Errors passed in dendrites
    - Perfect cancellation of forward activity via dendI (with cloned weights)
    - Gating with derivative of activation function (only accurate for ReLU)
    - Does not factor gradients passing through recurrent connections (e.g. somaI)
    '''
    def __init__(self, projection, learning_rate=None):
        super().__init__(projection, learning_rate)

    def step(self):
        ''' Update the weights '''
        # Backprop weight update
        delta_weight = torch.outer(self.projection.post.backward_dendritic_state, self.projection.pre.activity)
        self.projection.weight.data += self.learning_rate * delta_weight

    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        # Compute Output loss & set dendritic state
        global_error = target - output

        # Set dendritic state with the local loss in each layer
        reversed_layers = list(network)[::-1]
        network.output_pop.backward_dendritic_state = global_error
        for i,layer in enumerate(reversed_layers[:-1]): # Iterate over populations in reverse order starting from the output layer (skipping the Input layer)
            d_activation = layer.E.activity > 0
            layer.E.backward_dendritic_state = layer.E.backward_dendritic_state * d_activation # gate updates based on derivative of activation function (in case of ReLU: = 1 if activity>0)
            layer.E.backward_activity = layer.E.activity + layer.E.backward_dendritic_state # nudge somatic state

            # Compute the dendritic state of the lower layer
            pre_layer = reversed_layers[i+1]
            if pre_layer != network.Input:
                forward_weight_transpose = layer.E.incoming_projections[f"{layer.E.fullname}_{pre_layer.E.fullname}"].weight
                topdown_excitation = layer.E.backward_activity @ forward_weight_transpose

                dendI_projection = network.module_dict[f"{pre_layer.E.fullname}_{pre_layer.DendI.fullname}"]
                lateral_inhibition = pre_layer.DendI.activity @ dendI_projection.weight.T

                pre_layer.E.backward_dendritic_state = topdown_excitation + lateral_inhibition

            # if i > 0: # Skip the output layer
            #     layer.E.dendritic_state = torch.zeros(layer.E.size, device=network.device, requires_grad=False)
            #     for projection in layer.E:
            #         if projection.update_phase in ['B', 'backward', 'A', 'all'] and projection.compartment in ['dend', 'dendrite']:
            #             if hasattr(projection.pre, 'backward_activity'):                    
            #                 layer.E.dendritic_state += projection.weight @ projection.pre.backward_activity
            #             else:
            #                 layer.E.dendritic_state += projection.weight @ projection.pre.activity            
            # d_activation = layer.E.activity > 0
            # layer.E.dendritic_state = layer.E.dendritic_state * d_activation # gate updates based on derivative of activation function (in case of ReLU: = 1 if activity>0)
            # layer.E.backward_activity = layer.E.activity + layer.E.dendritic_state # nudge somatic state


class Backprop_dendI(LearningRule):
    def __init__(self, projection, learning_rate=None):
        super().__init__(projection, learning_rate)
        assert projection.post.layer == projection.pre.layer, "Dendritic backprop is designed for recurrent dendI connections"
        if learning_rate is None:
            learning_rate = projection.post.network.learning_rate

        projection.weight.requires_grad = True

        # Create one optimizer for every relevant network.layer and add the projection parameters
        layer = projection.post.layer
        if hasattr(layer, 'optimizer'):
            layer.optimizer.add_param_group({'params': projection.parameters(), 'lr': learning_rate})
        else:
            layer.optimizer = torch.optim.SGD(projection.parameters(), lr=learning_rate)

    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):  
        reversed_layers = list(network)[::-1]
        for i,layer in enumerate(reversed_layers[:-1]): # Iterate over populations in reverse order starting from the output layer (skipping the Input layer)
            pre_layer = reversed_layers[i+1]
            if pre_layer != network.Input:
                # Compute forward dendritic state
                forward_weight_transpose = layer.E.incoming_projections[f"{layer.E.fullname}_{pre_layer.E.fullname}"].weight.T
                topdown_excitation = layer.E.activity @ forward_weight_transpose
                
                dendI_projection = network.module_dict[f"{pre_layer.E.fullname}_{pre_layer.DendI.fullname}"]
                lateral_inhibition =  pre_layer.DendI.activity @ dendI_projection.weight.T

                pre_layer.E.forward_dendritic_state = topdown_excitation + lateral_inhibition

            # Compute local loss to cancel forward dendritic state
            if hasattr(layer.E, 'forward_dendritic_state'):
                local_target = torch.zeros(layer.E.size, device=network.device)
                local_loss = network.criterion(layer.E.forward_dendritic_state, local_target)
                layer.optimizer.zero_grad()
                local_loss.backward()
                layer.optimizer.step()



# class almost_backprop(LearningRule):
#     def __init__(self, projection, learning_rate=None):
#         super().__init__(projection, learning_rate)

#         self.w_max = 1.
#         self.k_dep = 0.5
#         self.dep_sigmoid = get_scaled_rectified_sigmoid(0.01, 0.02)

#     def step(self):
#         # Update the weights
        
#         # ETxIS = torch.outer(self.projection.post.IS, self.projection.pre.ET)
#         # delta_weight = ETxIS #*(self.w_max-self.projection.weight) - self.projection.weight * self.k_dep * self.dep_sigmoid(ETxIS)
        
#         # # ~BTSP weight update
#         # ETxIS = torch.outer(self.projection.post.dendritic_state, self.projection.pre.activity)
#         # weight_dependence = self.w_max - (self.projection.weight)  #*self.projection.weight.sign())
#         # delta_weight = ETxIS * weight_dependence - self.projection.weight*self.k_dep*self.dep_sigmoid(ETxIS)

#         # # ~BTSP weight update 2
#         # P = self.projection.post.dendritic_state
#         # e = self.projection.pre.activity
#         # w = self.projection.weight
#         # delta_weight = P.unsqueeze(1) * (e*(self.w_max-w) - w*self.k_dep*self.dep_sigmoid(e))

#         # ~Backprop weight update
#         delta_weight = torch.outer(self.projection.post.dendritic_state, self.projection.pre.activity)

#         self.projection.weight.data += self.learning_rate * delta_weight

#     @classmethod
#     def backward(cls, network, output, target, store_history=False, store_dynamics=False):
#         """
#         Integrate top-down inputs and update dendritic state variables.
#         :param network:
#         :param output:
#         :param target:
#         :param store_history: bool
#         :param store_dynamics: bool
#         """

#         # Compute Output loss & set dendritic state
#         global_error = target - output

#         # Set dendritic state with the local loss in each layer
#         reversed_layers = list(network)[::-1]
#         network.output_pop.dendritic_state = global_error
#         for i,layer in enumerate(reversed_layers[:-1]): # Iterate over populations in reverse order starting from the output layer
#             layer.E.backward_activity = layer.E.activity + layer.E.dendritic_state # nudge somatic state

#             pre_layer = reversed_layers[i+1]
#             forward_weight_transpose = layer.E.incoming_projections[f"{layer.E.fullname}_{pre_layer.E.fullname}"].weight.T

#             inhibition = forward_weight_transpose @ layer.E.activity
#             pre_layer.E.dendritic_state = forward_weight_transpose @ layer.E.backward_activity - inhibition

#             # Keep the top x% of gradients and set the rest to 0
#             percentage_to_keep = 0.2
#             flat_error_vector = pre_layer.E.dendritic_state.flatten()
#             n = round(flat_error_vector.numel()*(1-percentage_to_keep))
#             indices = flat_error_vector.abs().argsort()[:n] # Find the indices of the n smallest values
#             flat_error_vector[indices] = 0
#             pre_layer.E.dendritic_state = flat_error_vector.view(pre_layer.E.dendritic_state.shape)


