import torch
from torch.cuda.amp import autocast
from .base_classes import LearningRule, BiasLearningRule

def get_activation_derivative(x, activation_fn):
    if activation_fn == 'relu':
        return (x > 0).float()
    elif activation_fn == 'linear':
        return x


# Refer to https://www.nature.com/articles/s41467-020-17236-y for more information
# works for non-batched data
class Eprop(LearningRule):
    def __init__(self, projection, learning_rate=None):
        super().__init__(projection, learning_rate)

        self.modify_grad = True # logging

        projection.weight.requires_grad = True
        self.eligibility_trace = torch.zeros_like(projection.weight) # dh/dh
        self.dW = torch.zeros_like(projection.weight) # accumulate gradient

        self.pre = projection.pre
        self.post = projection.post

        self.diagonal_mask = None

        if self.pre is self.post:
            self.diagonal_mask = 1 - torch.eye(self.eligibility_trace.shape[0])

    @property
    def post_recurrent_projection(self):
        """
        The post population's self-to-self (recurrent) projection, or None.

        Resolved lazily because projections register on the post population
        after the learning rule is constructed, so this cannot be computed in
        __init__.
        """
        return next((proj for proj in self.post if proj.pre is self.post), None)

    def post_has_self_connection(self):
        return self.post_recurrent_projection is not None

    def proj_is_self_connection(self):
        return (self.pre is self.post)


    def update_eligibility_trace(self, reinit=False):
        
        with torch.no_grad():
            if reinit:
                self.eligibility_trace.zero_()
                self.dW.zero_()

            # Recurrent projections consume the presynaptic population's previous
            # activity in the forward pass (Population.forward uses prev_activity for
            # direction 'R'), so the eligibility term must use prev_activity too.
            if self.projection.direction in ('recurrent', 'R'):
                pre_activity = self.pre.prev_activity.detach().clone()
            else:
                pre_activity = self.pre.activity.detach().clone()
            self.post_state = self.post.state # pre_activation state of post neuron -> directly gives you dL/dh
            self.post.state.retain_grad()

            dh_dh = 0

            if self.post_has_self_connection():
                dh_dh = (1 - 1 / self.post.tau) * self.eligibility_trace

            # account for autapses for self-connections
            if self.diagonal_mask is not None:
                self.eligibility_trace = dh_dh + pre_activity.unsqueeze(-2) / self.post.tau * self.diagonal_mask
            else:
                self.eligibility_trace = dh_dh + pre_activity.unsqueeze(-2) / self.post.tau
        
        return
    
    def modify_gradient(self):
        with torch.no_grad():
            dL_dh = self.post_state.grad
            dL_dW = dL_dh.unsqueeze(-1) * self.eligibility_trace
            self.dW += dL_dW

            # Assign a copy, not self.dW itself: weight.grad and self.dW must not
            # alias, otherwise the next per-timestep loss.backward() accumulates
            # autograd's gradient in place into our e-prop accumulator (doubling it).
            self.projection.weight.grad = self.dW.clone()

    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        Accumulate gradients for the current timestep only.
        """
        if network.use_amp:
            with autocast():
                loss = network.criterion(output, target)
            network.scaler.scale(loss).backward()
        else:
            loss = network.criterion(output, target)
            loss.backward()

        
