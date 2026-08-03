import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import MSELoss, BCELoss, CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.cuda.amp import autocast, GradScaler

from copy import deepcopy
from collections import defaultdict
from functools import partial
from typing import Optional, Dict, Any, Union, Tuple, List 

import EIANN.utils as ut
import EIANN.rules as rules
import EIANN.external as external

# DNQL Algorithm
# Compute Q
# Compare Q with Q Next (MSE) 
# Compute Q Next
# Compare Q with Q Next (TD)

# moves out all the non-training logic (logging mostly) into a separate function
def prepare_for_training(q_network, qnext_network, environments, episodes, val_interval, store_history, 
                         store_history_interval, store_params, store_params_interval, status_bar):
    

    q_network.reset_history()
    qnext_network.reset_history()

    if not isinstance(environments, list):
        environments = [environments]
    
    # Define timepoints for validation
    train_step_range = torch.arange(episodes)
    if val_interval[0] < 0 and abs(val_interval[0]) > len(train_step_range):
        val_start_index = 0
    else:
        val_start_index = train_step_range[val_interval[0]]
    val_end_index = train_step_range[val_interval[1]]
    val_step_size = val_interval[2]
    val_range = torch.arange(val_end_index, val_start_index - 1, -val_step_size).flip(0)
    if val_start_index == 0 and 0 not in val_range:
        val_range = torch.cat((torch.tensor([0]), val_range))

    q_network.val_reward_history = []
    q_network.val_action_history = []
    q_network.val_cross_correlation_history = []
    q_network.val_cross_correlation_matrix_history = []
    q_network.val_history_train_steps = []

    # Store history of weights and biases
    store_history_range = None
    if store_history:
        if store_history_interval is not None:
            if store_history_interval[0] < 0 and abs(store_history_interval[0]) > len(train_step_range):
                store_history_start_index = 0
            else:
                store_history_start_index = train_step_range[store_history_interval[0]]
            store_history_end_index = train_step_range[store_history_interval[1]]
            store_history_step_size = store_history_interval[2]
            store_history_range = (
                torch.arange(store_history_end_index, store_history_start_index - 1,
                                -store_history_step_size).flip(0))
            if store_history_start_index == 0 and 0 not in store_history_range:
                store_history_range = torch.cat((torch.tensor([0]), store_history_range))

    # Store history of weights and biases
    store_params_range = None
    store_params_step_size = None
    if store_params:
        if store_params_interval is None:
            store_params_step_size = val_step_size
            store_params_range = val_range
        else:
            if store_params_interval[0] < 0 and abs(store_params_interval[0]) > len(train_step_range):
                store_params_start_index = 0
            else:
                store_params_start_index = train_step_range[store_params_interval[0]]
            store_params_end_index = train_step_range[store_params_interval[1]]
            store_params_step_size = store_params_interval[2]
            store_params_range = (
                torch.arange(store_params_end_index, store_params_start_index - 1, -store_params_step_size).flip(0))
            if store_params_start_index == 0 and 0 not in store_params_range:
                store_params_range = torch.cat((torch.tensor([0]), store_params_range))
        if (0 in store_params_range) and store_params_step_size == 1: # store initial state of the network
            q_network.param_history.append(deepcopy(q_network.state_dict()))
            q_network.param_history_steps.append(-1)

            qnext_network.param_history.append(deepcopy(qnext_network.state_dict()))
            qnext_network.param_history_steps.append(-1)

    if status_bar:
        from tqdm.autonotebook import tqdm
        episode_iter = tqdm(range(episodes), desc='Episodes')
    else:
        episode_iter = range(episodes)

    # Initialize learning rule parameters
    for post_layer in q_network:
        for post_pop in post_layer:
            if post_pop.include_bias:
                post_pop.bias_learning_rule.reinit()
            for projection in post_pop:
                projection.learning_rule.reinit()

    for post_layer in qnext_network:
        for post_pop in post_layer:
            if post_pop.include_bias:
                post_pop.bias_learning_rule.reinit()
            for projection in post_pop:
                projection.learning_rule.reinit()

    return environments, val_range, store_history_range, store_params_range, store_params_step_size, episode_iter


def compute_and_apply_gradients(network, loss, this_train_step_store_history, store_dynamics):
    # Perform backwards pass
    for backward in network.backward_methods:
        backward(network, None, None, store_history=this_train_step_store_history, store_dynamics=store_dynamics, loss=loss)
    
    # Step weights and biases
    for i, post_layer in enumerate(network):
        for post_pop in post_layer:
            if post_pop.include_bias:
                post_pop.bias_learning_rule.step()
            for projection in post_pop:
                projection.learning_rule.step()


    network.constrain_weights_and_biases()

    # Update learning rule parameters
    for i, post_layer in enumerate(network):
        for post_pop in post_layer:
            if post_pop.include_bias:
                post_pop.bias_learning_rule.update()
            for projection in post_pop:
                projection.learning_rule.update()


# Training Offline (only for Backprop-Based Methods)
def train(q_network, qnext_network, environments, epsilon, epsilon_decay, 
          gamma, episodes, val_interval, store_dynamics, store_history, 
          store_history_interval, store_params, store_params_interval, status_bar, save_to_file, meta=False, shared_population='H1E'):
    
    # shared_population: the population vector used as input to the qnext network

    environments, val_range, store_history_range, store_params_range, store_params_step_size, episode_iter = prepare_for_training(q_network, qnext_network, environments, episodes, val_interval, store_history, 
                                                                                                          store_history_interval, store_params, store_params_interval, status_bar)


    #######################################################
    #*************     Main training loop     *************
    #######################################################

    train_step = 0
    for episode in episode_iter:

        if store_history:
            if store_history_interval is None:
                this_train_step_store_history = True
            else:
                this_train_step_store_history = train_step in store_history_range
        else:
            this_train_step_store_history = False

        environment_index = q_network.rng.integers(0, len(environments))
        environment = environments[environment_index]

        # Reset Environment
        environment.reset()
        terminated = False
        qnext = None
        step_number = 0
        q_loss = 0
        qnext_loss = 0

        previous_action = 0
        previous_reward = 0

        action_one_hots = F.one_hot(torch.arange(0, len(environment.actions)))

        epsilon *= epsilon_decay

        ## COMPUTE FULL PASS OVER TREADMILL ##
        while not terminated:
            reinit = step_number == 0

            current_observation = environment.get_observation(environment.current_state)
            obs_tensor = torch.tensor(current_observation, dtype=torch.float32).unsqueeze(0)

            if meta:
                obs_tensor = torch.cat([obs_tensor, action_one_hots[previous_action].unsqueeze(0), torch.tensor([[previous_reward]])], dim=1)                    

            # Use autocast for forward pass when AMP is enabled
            # Run Q Network
            if q_network.use_amp:
                with autocast():
                    q = q_network(obs_tensor, store_history=this_train_step_store_history, store_dynamics=store_dynamics, reinit=reinit)
            else:
                q = q_network(obs_tensor, store_history=this_train_step_store_history, store_dynamics=store_dynamics, reinit=reinit)

            # Epsilon Greedy Exploration
            if q_network.rng.random() < epsilon:
                action = int(q_network.rng.choice(environment.get_action_list()))
            else:
                action = int(torch.argmax(q).item())

            # Take Action, Observe Reward
            _, reward, next_state, terminated = environment.take_action(action)

            # Pass error to previous qnext prediction
            if qnext is not None:
                qnext_loss += (qnext - q.max().item()) ** 2

            # Run Q Next Network -> always reinitialize because there is no recurrence across time
            if qnext_network.use_amp:
                with autocast():
                    qnext = qnext_network(q_network.populations[shared_population].activity.detach(), store_history=this_train_step_store_history, store_dynamics=store_dynamics, reinit=True)
            else:
                qnext = qnext_network(q_network.populations[shared_population].activity.detach(), store_history=this_train_step_store_history, store_dynamics=store_dynamics, reinit=True)

            q_loss += (gamma * qnext.max().item() + reward - torch.gather(q, 0, torch.tensor([action]))) ** 2

            step_number += 1
            previous_action = action
            previous_reward = reward

            # account for being in the last step
            if terminated:
                qnext_loss += qnext ** 2

        compute_and_apply_gradients(q_network, q_loss, this_train_step_store_history, store_dynamics)
        compute_and_apply_gradients(qnext_network, qnext_loss, this_train_step_store_history, store_dynamics)

        # Store history of weights and biases
        if store_params and train_step in store_params_range:
            q_network.param_history.append(deepcopy(q_network.state_dict()))
            q_network.param_history_steps.append(train_step)
            qnext_network.param_history.append(deepcopy(qnext_network.state_dict()))
            qnext_network.param_history_steps.append(train_step)

        q_network.loss_history.append(q_loss)
        qnext_network.loss_history.append(qnext_loss)

        if store_params and (train_step in store_params_range) and store_params_step_size > 1:
            q_network.prev_param_history.append(deepcopy(q_network.state_dict()))  # Store parameters for dW comparison
            qnext_network.prev_param_history.append(deepcopy(qnext_network.state_dict()))  # Store parameters for dW comparison
        
        # Compute validation performance over all environments
        if train_step in val_range:
            val_rewards, val_actions, val_cc = q_network.test(
                environments, return_actions=True, return_cross_correlation=True, meta=meta)
            q_network.val_reward_history.append(val_rewards)
            q_network.val_action_history.append(val_actions)
            # store full cross-correlation matrices, and their reduction to region summaries
            q_network.val_cross_correlation_matrix_history.append(val_cc)
            q_network.val_cross_correlation_history.append(
                {pop: [q_network.region_cross_correlation(mat, environments[0]) for mat in mats]
                    for pop, mats in val_cc.items()})
            q_network.val_history_train_steps.append(train_step)

            if status_bar: # Display the current loss and accuracy on the progress bar
                episode_iter.set_description(f"Validation Rewards: {q_network.val_reward_history[-1]:.4f} - Episode")
        
        train_step += 1

    q_network.loss_history = torch.tensor(q_network.loss_history)
    qnext_network.loss_history = torch.tensor(qnext_network.loss_history)


    q_network.val_reward_history = torch.tensor(q_network.val_reward_history)
    q_network.val_action_history = np.array(q_network.val_action_history)

    q_network.val_history_train_steps = torch.tensor(q_network.val_history_train_steps)

    if store_params:
        q_network.param_history_steps = torch.tensor(q_network.param_history_steps)
        qnext_network.param_history_steps = torch.tensor(qnext_network.param_history_steps)
    
    if save_to_file is not None:
        ut.save_network(q_network, path=save_to_file)
        ut.save_network(qnext_network, path=save_to_file.replace('.pkl', '_qnext.pkl'))

# Training Online (For Target Propagation-like Models)
def train_online(q_network, qnext_network, environments, epsilon, epsilon_decay, 
          gamma, episodes, val_interval, store_dynamics, store_history, 
          store_history_interval, store_params, store_params_interval, status_bar, save_to_file, meta=False, shared_population='H1E'):
    
    # shared_population: the population vector used as input to the qnext network

    environments, val_range, store_history_range, store_params_range, store_params_step_size, episode_iter = prepare_for_training(q_network, qnext_network, environments, episodes, val_interval, store_history, 
                                                                                                          store_history_interval, store_params, store_params_interval, status_bar)


    #######################################################
    #*************     Main training loop     *************
    #######################################################

    train_step = 0
    for episode in episode_iter:

        if store_history:
            if store_history_interval is None:
                this_train_step_store_history = True
            else:
                this_train_step_store_history = train_step in store_history_range
        else:
            this_train_step_store_history = False

        environment_index = q_network.rng.integers(0, len(environments))
        environment = environments[environment_index]

        # Reset Environment
        environment.reset()
        terminated = False
        qnext = None
        step_number = 0

        previous_action = 0
        previous_reward = 0

        action_one_hots = F.one_hot(torch.arange(0, len(environment.actions)))

        epsilon *= epsilon_decay

        if q_network.optimizer is not None:
            q_network.optimizer.zero_grad()

        if qnext_network.optimizer is not None:
            qnext_network.optimizer.zero_grad()

        ## COMPUTE FULL PASS OVER TREADMILL ##
        while not terminated:
            reinit = step_number == 0

            # Ensure that no gradients are propagated through time (ex. for eprop)
            for post_pop in q_network.populations.values():
                if hasattr(post_pop, 'state'):
                    post_pop.state = post_pop.state.detach()
                    post_pop.activity = post_pop.activity.detach()

            for post_pop in qnext_network.populations.values():
                if hasattr(post_pop, 'state'):
                    post_pop.state = post_pop.state.detach()
                    post_pop.activity = post_pop.activity.detach()

            current_observation = environment.get_observation(environment.current_state)
            obs_tensor = torch.tensor(current_observation, dtype=torch.float32).unsqueeze(0)

            if meta:
                obs_tensor = torch.cat([obs_tensor, action_one_hots[previous_action].unsqueeze(0), torch.tensor([[previous_reward]])], dim=1)                    

            # Use autocast for forward pass when AMP is enabled
            # Run Q Network
            if q_network.use_amp:
                with autocast():
                    q = q_network(obs_tensor, store_history=this_train_step_store_history, store_dynamics=store_dynamics, reinit=reinit)
            else:
                q = q_network(obs_tensor, store_history=this_train_step_store_history, store_dynamics=store_dynamics, reinit=reinit)
            q_network.update_forward_state(store_history=this_train_step_store_history, store_dynamics=store_dynamics)

            # Epsilon Greedy Exploration
            if q_network.rng.random() < epsilon:
                action = int(q_network.rng.choice(environment.get_action_list()))
            else:
                action = int(torch.argmax(q).item())

            # Take Action, Observe Reward
            _, reward, next_state, terminated = environment.take_action(action)

            # Pass error to previous qnext prediction
            if qnext is not None:
                qnext_loss = (qnext - q.max().item()) ** 2

            compute_and_apply_gradients(qnext_network, qnext_loss, this_train_step_store_history, store_dynamics)

            # Run Q Next Network -> always reinitialize because there is no recurrence across time
            if qnext_network.use_amp:
                with autocast():
                    qnext = qnext_network(q_network.populations[shared_population].activity.detach(), store_history=this_train_step_store_history, store_dynamics=store_dynamics, reinit=True)
            else:
                qnext = qnext_network(q_network.populations[shared_population].activity.detach(), store_history=this_train_step_store_history, store_dynamics=store_dynamics, reinit=True)
            q_network.update_forward_state(store_history=this_train_step_store_history, store_dynamics=store_dynamics)

            q_loss = (gamma * qnext.max().item() + reward - torch.gather(q, 0, torch.tensor([action]))) ** 2

            compute_and_apply_gradients(q_network, q_loss, this_train_step_store_history, store_dynamics)

            step_number += 1
            previous_action = action
            previous_reward = reward

            # account for being in the last step
            if terminated:
                qnext_loss = qnext ** 2
                compute_and_apply_gradients(qnext_network, qnext_loss, this_train_step_store_history, store_dynamics)


        # Store history of weights and biases
        if store_params and train_step in store_params_range:
            q_network.param_history.append(deepcopy(q_network.state_dict()))
            q_network.param_history_steps.append(train_step)
            qnext_network.param_history.append(deepcopy(qnext_network.state_dict()))
            qnext_network.param_history_steps.append(train_step)

        q_network.loss_history.append(q_loss)
        qnext_network.loss_history.append(qnext_loss)

        if store_params and (train_step in store_params_range) and store_params_step_size > 1:
            q_network.prev_param_history.append(deepcopy(q_network.state_dict()))  # Store parameters for dW comparison
            qnext_network.prev_param_history.append(deepcopy(qnext_network.state_dict()))  # Store parameters for dW comparison
        
        # Compute validation performance over all environments
        if train_step in val_range:
            val_rewards, val_actions, val_cc = q_network.test(
                environments, return_actions=True, return_cross_correlation=True, meta=meta)
            q_network.val_reward_history.append(val_rewards)
            q_network.val_action_history.append(val_actions)
            # store full cross-correlation matrices, and their reduction to region summaries
            q_network.val_cross_correlation_matrix_history.append(val_cc)
            q_network.val_cross_correlation_history.append(
                {pop: [q_network.region_cross_correlation(mat, environments[0]) for mat in mats]
                    for pop, mats in val_cc.items()})
            q_network.val_history_train_steps.append(train_step)

            if status_bar: # Display the current loss and accuracy on the progress bar
                episode_iter.set_description(f"Validation Rewards: {q_network.val_reward_history[-1]:.4f} - Episode")
        
        train_step += 1

    q_network.loss_history = torch.tensor(q_network.loss_history)
    qnext_network.loss_history = torch.tensor(qnext_network.loss_history)


    q_network.val_reward_history = torch.tensor(q_network.val_reward_history)
    q_network.val_action_history = np.array(q_network.val_action_history)

    q_network.val_history_train_steps = torch.tensor(q_network.val_history_train_steps)

    if store_params:
        q_network.param_history_steps = torch.tensor(q_network.param_history_steps)
        qnext_network.param_history_steps = torch.tensor(qnext_network.param_history_steps)
    
    if save_to_file is not None:
        ut.save_network(q_network, path=save_to_file)
        ut.save_network(qnext_network, path=save_to_file.replace('.pkl', '_qnext.pkl'))