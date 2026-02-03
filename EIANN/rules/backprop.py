import torch
from torch.cuda.amp import autocast
from .base_classes import LearningRule, BiasLearningRule
from EIANN.utils import pwlin


class Backprop(LearningRule):
    def __init__(self, projection, learning_rate=None):
        super().__init__(projection, learning_rate)
        projection.weight.requires_grad = True
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        if network.use_amp:
            # Use autocast for loss computation and scaled backward pass
            with autocast():
                loss = network.criterion(output, target)
            
            network.optimizer.zero_grad()
            network.scaler.scale(loss).backward()
            network.scaler.step(network.optimizer)
            network.scaler.update()
        else:
            # Standard backward pass
            loss = network.criterion(output, target)
            network.optimizer.zero_grad()
            loss.backward()
            network.optimizer.step()


class Backprop_INEL_A(LearningRule):
    # Target weight is the unit mean
    def __init__(self, projection, inel_threshold=0.5, learning_rate=None):
        super().__init__(projection, learning_rate)
        projection.weight.requires_grad = True
        self.inel_threshold = inel_threshold
        self.task_num = 0
    
    def update_CL_states(self):
        self.task_num += 1
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        
        loss = network.criterion(output, target)
        network.optimizer.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            for projection in network.projections.values():
                if projection.learning_rule.__class__ == cls:
                    if projection.learning_rule.task_num > 0:
                        unit_mean_weights = torch.mean(projection.weight.data, dim=1)
                        inel_indexes = (torch.abs(projection.weight.data - unit_mean_weights.unsqueeze(1)) <
                                        projection.learning_rule.inel_threshold).nonzero(as_tuple=True)
                        projection.weight.grad[inel_indexes] = 0.
        
        network.optimizer.step()


class Backprop_INEL_B(LearningRule):
    # Target weight is the projection mean
    def __init__(self, projection, inel_threshold=0.5, learning_rate=None):
        super().__init__(projection, learning_rate)
        projection.weight.requires_grad = True
        self.inel_threshold = inel_threshold
        self.task_num = 0
    
    def update_CL_states(self):
        self.task_num += 1
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        
        loss = network.criterion(output, target)
        network.optimizer.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            for projection in network.projections.values():
                if projection.learning_rule.__class__ == cls:
                    if projection.learning_rule.task_num > 0:
                        projection_mean_weight = torch.mean(projection.weight.data)
                        inel_indexes = (torch.abs(projection.weight.data - projection_mean_weight) <
                                        projection.learning_rule.inel_threshold).nonzero(as_tuple=True)
                        projection.weight.grad[inel_indexes] = 0.
        
        network.optimizer.step()


class Backprop_SILR_A(LearningRule):
    """
    Inspired by "Synaptic Intelligence" (Zenke et al., 2019). Parameter updates are accumulated during a task,
    and learning rates are modulated during subsequent tasks.
    """
    def __init__(self, projection, silr_threshold=0.5, silr_width=0.5, silr_min=0., learning_rate=None):
        super().__init__(projection, learning_rate)
        projection.weight.requires_grad = True
        self.lr_mod = torch.ones_like(projection.weight.data)
        self.lr_mod_func = pwlin(1., silr_min, silr_threshold, silr_threshold + silr_width)
        self.accum_delta_weight = torch.zeros_like(projection.weight.data)
        self.task_num = 0
    
    def update_CL_states(self):
        self.task_num += 1
        self.lr_mod = self.lr_mod_func(torch.abs(self.accum_delta_weight))
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        
        loss = network.criterion(output, target)
        network.optimizer.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            for projection in network.projections.values():
                if projection.learning_rule.__class__ == cls:
                    delta_weight = (-projection.learning_rule.learning_rate * projection.learning_rule.lr_mod *
                                    projection.weight.grad)
                    projection.learning_rule.accum_delta_weight += delta_weight
                    if projection.learning_rule.task_num > 0:
                        projection.weight.grad *= projection.learning_rule.lr_mod
        
        network.optimizer.step()


class Backprop_SILR_B(LearningRule):
    """
    Inspired by "Synaptic Intelligence" (Zenke et al., 2019). Parameter updates are accumulated during a task,
    and learning rates are modulated during subsequent tasks.
    """
    
    def __init__(self, projection, silr_threshold=0.5, silr_width=0.5, silr_min=0., learning_rate=None):
        super().__init__(projection, learning_rate)
        projection.weight.requires_grad = True
        self.lr_mod_func = pwlin(1., silr_min, silr_threshold, silr_threshold + silr_width)
        self.accum_delta_weight = torch.zeros_like(projection.weight.data)
        self.task_num = 0
    
    def update_CL_states(self):
        self.task_num += 1
        self.prev_task_accum_delta_weight = self.accum_delta_weight.detach().clone()
        self.prev_task_weights = self.projection.weight.data.detach().clone()
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        
        loss = network.criterion(output, target)
        network.optimizer.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            for projection in network.projections.values():
                if projection.learning_rule.__class__ == cls:
                    if projection.learning_rule.task_num > 0:
                        lr_mod = projection.learning_rule.lr_mod_func(
                            torch.abs(projection.learning_rule.prev_task_accum_delta_weight *
                                      (projection.weight.data - projection.learning_rule.prev_task_weights)))
                        projection.weight.grad *= lr_mod
                    else:
                        lr_mod = 1.
                    delta_weight = (-projection.learning_rule.learning_rate * lr_mod *
                                    projection.weight.grad)
                    projection.learning_rule.accum_delta_weight += delta_weight
        
        network.optimizer.step()


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


class BackpropBias(BiasLearningRule):
    def __init__(self, population, learning_rate=None):
        super().__init__(population, learning_rate)
        population.bias.requires_grad = True
    
    backward = Backprop.backward


class Backprop_DendriticLoss(LearningRule):
    def __init__(self, projection, source, learning_rate=None):
        """

        :param projection: :class:'nn.Linear'
        :param source: str ('layer_name.pop_name')
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        source_post_layer, source_post_pop = source.split('.')
        self.source_pop = projection.post.network.layers[source_post_layer].populations[source_post_pop]
        projection.weight.requires_grad = True
        
        # Create one optimizer the source and register the projection parameters
        if hasattr(self.source_pop, 'local_optimizer'):
            self.source_pop.local_optimizer.add_param_group({'params': projection.parameters(),
                                                             'lr': self.learning_rate})
        else:
            self.source_pop.local_optimizer = torch.optim.SGD(projection.parameters(), lr=self.learning_rate)
            
        # Add AMP scaler for local optimizer if network uses AMP
        if hasattr(self.source_pop, 'local_scaler'):
            pass  # Already has scaler
        else:
            network = projection.post.network
            if network.use_amp:
                self.source_pop.local_scaler = torch.cuda.amp.GradScaler()
            else:
                self.source_pop.local_scaler = None
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        :param network:
        :param output:
        :param target:
        :param store_history:
        :param store_dynamics:
        """
        output = output.to(network.device)
        target = target.to(network.device)

        local_optimizer_list = []
        
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        
        for layer in reversed_layers:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        source_pop = projection.learning_rule.source_pop
                        local_optimizer = source_pop.local_optimizer
                        local_scaler = getattr(source_pop, 'local_scaler', None)
                        
                        if local_optimizer not in local_optimizer_list:
                            local_optimizer_list.append(local_optimizer)
                            local_target = torch.zeros(source_pop.size, device=network.device)
                            
                            if network.use_amp and local_scaler is not None:
                                # Use AMP for local loss computation
                                with autocast():
                                    local_loss = network.criterion(source_pop.forward_dendritic_state, local_target)
                                
                                local_optimizer.zero_grad()
                                local_scaler.scale(local_loss).backward()
                                local_scaler.step(local_optimizer)
                                local_scaler.update()
                            else:
                                # Standard backward pass
                                local_loss = network.criterion(source_pop.forward_dendritic_state, local_target)
                                local_optimizer.zero_grad()
                                local_loss.backward()
                                local_optimizer.step()
