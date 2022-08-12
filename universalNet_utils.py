
import torch
import itertools

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8,
                     'axes.spines.right': False,
                     'axes.spines.top': False,
                     'axes.linewidth': 1,
                     'axes.labelpad': 0,
                     'xtick.major.size': 3.5,
                     'xtick.major.width': 1,
                     'ytick.major.size': 3.5,
                     'ytick.major.width': 1,
                     'legend.frameon': False,
                     'legend.handletextpad': 0.1,
                     'figure.figsize': [14.0, 4.0],})
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs


def n_hot_patterns(n,length):
    all_permutations = torch.tensor(list(itertools.product([0., 1.], repeat=length)))
    pattern_hotness = torch.sum(all_permutations,axis=1)
    idx = torch.where(pattern_hotness == n)[0]
    n_hot_patterns = all_permutations[idx]
    return n_hot_patterns


def plot_summary(model):
    # TODO: expand plots to show individual biases, examples of input/output activities, activity dynamics
    # (replicate plots from summer project)

    mm = 1 / 25.4  # millimeters to inches
    fig = plt.figure(figsize=(180 * mm, 200 * mm), dpi=300)
    axes = gs.GridSpec(nrows=4, ncols=3,
                       left=0.05,right=0.98,
                       top = 0.95, bottom = 0.2,
                       wspace=0.4, hspace=0.8)

    # Loss history
    ax = fig.add_subplot(axes[0, 0])
    ax.plot(model.loss_history)
    ax.set_xlabel('Training steps (epochs*patterns)')
    ax.set_ylabel('Loss')
    ax.set_ylim(bottom=0)

    # ax[2].plot(sim_dict['data']['sorted_loss_history'], label='sorted')
    # ax[2].legend()

    # Weights
    ax = fig.add_subplot(axes[0, 1])
    for name,weights in model.weight_history.items():
        avg_weight = torch.mean(weights,dim=(1,2))
        ax.plot(avg_weight,label=name)
    ax.set_xlabel('Training steps (epochs*patterns)')
    ax.set_ylabel('Average weight')
    ax.legend()

    ax = fig.add_subplot(axes[0, 2])
    w = model.weight_history['layer0E_to_layer1E'].flatten(1)
    ax.plot(w)
    ax.set_xlabel('Training steps (epochs*patterns)')
    ax.set_ylabel('weight')
    ax.set_title('layer1E_to_layer2E')

    # init_weights = sim_dict['data']['weight_history'][:,:,0]
    # im = ax.imshow(init_weights, cmap='Reds')
    # plt.colorbar(im, ax=ax)
    # ax.set_title('Initial weights')
    # ax.set_xlabel('Input units')
    # ax.set_ylabel('Output units')
    #
    # final_weights = sim_dict['data']['weight_history'][:,:,-1]
    # im = ax[1].imshow(final_weights, cmap='Reds')
    # plt.colorbar(im, ax=ax[1])
    # ax[1].set_title('Final weights')
    # ax[1].set_xlabel('Input units')
    # ax[1].set_ylabel('Output units')
    #
    # mean_weight = torch.mean(sim_dict['data']['weight_history'][:,:,:], dim=(0, 1))
    # ax[3].plot(mean_weight)
    # ax[3].set_xlabel("Pattern")
    # ax[3].set_ylabel('mean weight')


    # Biases
    ax = fig.add_subplot(axes[1, 0])
    for layer in model:
        for population in layer:
            avg_bias = torch.mean(population.bias_history, dim=1)
            ax.plot(avg_bias)
    ax.set_xlabel('Training steps (epochs*patterns)')
    ax.set_ylabel('bias')

    # Activities
    # ax = fig.add_subplot(axes[3, 0])
    # initial_output_activities = []
    #

    ax = fig.add_subplot(axes[3, 1])
    output_activities = torch.zeros(model.all_patterns.shape[0],model.output_pop.size)
    for i,pattern in enumerate(model.all_patterns):
        model.output_pop.state = torch.zeros(model.output_pop.size)
        model.output_pop.activity = torch.zeros(model.output_pop.size)
        for t in range(model.num_timesteps):
            model.forward(pattern, training=False)
        output_activities[i] = model.output_pop.activity.detach()
    im = ax.imshow(output_activities)
    plt.colorbar(im, ax=ax)
    ax.set_title('Final activity (unsorted)')
    ax.set_xlabel('Input pattern')
    ax.set_ylabel('Output unit')

    # im = ax[1].imshow(sim_dict['data']['unsorted_output_history'][:,:,-1,-1])
    # plt.colorbar(im, ax=ax[1])
    # ax[1].set_title('Unsorted final activity')
    # ax[1].set_xlabel('Input pattern')
    # ax[1].set_ylabel('Output unit')
    #
    # im = ax[0].imshow(sim_dict['data']['sorted_output_history'][:,:,-1,0])
    # plt.colorbar(im, ax=ax[0])
    # ax[0].set_title('Sorted initial activity')
    # ax[0].set_xlabel('Input pattern')
    # ax[0].set_ylabel('Output unit')
    #
    # im = ax[1].imshow(sim_dict['data']['sorted_output_history'][:,:,-1,-1])
    # plt.colorbar(im, ax=ax[1])
    # ax[1].set_title('Sorted final activity')
    # ax[1].set_xlabel('Input pattern')
    # ax[1].set_ylabel('Output unit')

    # # Activity temporal dynamics
    # avg_activity = torch.mean(sim_dict['data']['sorted_output_history'][:, :, :, -1], dim=(0, 1))
    # ax[0].plot(avg_activity)
    # ax[0].set_title('Output Dynamics')
    # ax[0].set_xlabel('Time Steps')
    # ax[0].set_ylabel('Average activity')
    #
    # if 'fbi_history' in sim_dict['data']:
    #     avg_activity = torch.mean(sim_dict['data']['fbi_history'][:, :, :, -1], dim=(0, 1))
    #     ax[1].plot(avg_activity)
    #     ax[1].set_title('Inh Dynamics')
    #     ax[1].set_xlabel('Time Steps')
    #     ax[1].set_ylabel('Average activity')
    #
    # output_activity = torch.mean(sim_dict['data']['sorted_output_history'][:, :, :, -1], dim=(0, 1))
    # ax[2].plot(output_activity / torch.max(output_activity), label='output activity')
    # if 'fbi_history' in sim_dict['data']:
    #     fbi_activity = torch.mean(sim_dict['data']['fbi_history'][:, :, :, -1], dim=(0, 1))
    #     ax[2].plot(fbi_activity / torch.max(fbi_activity), label='fbi activity')
    # ax[2].set_title('Dynamics')
    # ax[2].set_xlabel('Time Steps')
    # ax[2].set_ylabel('Average activity')

    # # Learning rule parameters
    # if 'mean_theta_history' in sim_dict['data'].keys():
    #     ax[0].plot(sim_dict['data']['mean_theta_history'])
    #     ax[0].set_title('Theta History')
    #     ax[0].set_xlabel('Pattern')
    #     ax[0].set_ylabel('Theta')
    #
    #     ax[1].plot(sim_dict['data']['theta_history'].T)
    #     ax[1].set_title('Theta History')
    #     ax[1].set_xlabel('Pattern')
    #     ax[1].set_ylabel('Theta')

    plt.show()