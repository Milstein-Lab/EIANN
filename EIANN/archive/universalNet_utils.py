
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
    axes = gs.GridSpec(nrows=5, ncols=3,
                       left=0.05,right=0.98,
                       top = 0.95, bottom = 0.2,
                       wspace=0.4, hspace=0.8)

    # Loss history
    ax = fig.add_subplot(axes[0, 0])
    ax.plot(model.loss_history)
    ax.set_xlabel('Training steps (epochs*patterns)')
    ax.set_ylabel('Loss')
    ax.set_ylim(bottom=0)

    # Weights
    ax = fig.add_subplot(axes[0, 1])
    projection = model.layer1.E.layer0E
    ax.plot(projection.weight_history.flatten(1), color='black', alpha=0.05)
    avg_weight = torch.mean(projection.weight_history, dim=(1,2))
    ax.plot(avg_weight, color='red', linewidth=2, label=projection.name)
    ax.set_xlabel('Training steps (epochs*patterns)')
    ax.set_ylabel('weight')
    ax.set_title(projection.name)

    ax = fig.add_subplot(axes[1:3, 0])
    init_weights = model.layer1.E.layer0E.weight_history[0]
    im = ax.imshow(init_weights, cmap='Reds')
    plt.colorbar(im, ax=ax)
    ax.set_title('Initial weights')
    ax.set_xlabel('Input units')
    ax.set_ylabel('Output units')

    ax = fig.add_subplot(axes[1:3, 1])
    final_weights = model.layer1.E.layer0E.weight_history[-1]
    im = ax.imshow(final_weights, cmap='Reds')
    plt.colorbar(im, ax=ax)
    ax.set_title('Final weights')
    ax.set_xlabel('Input units')
    ax.set_ylabel('Output units')

    # ax = fig.add_subplot(axes[1, 2])
    # final_weights = model.layer1.I.layer1E.weight_history[-1]
    # im = ax.imshow(final_weights, cmap='Blues')
    # plt.colorbar(im, ax=ax)
    # ax.set_title('Final weights')
    # ax.set_xlabel('Input units')
    # ax.set_ylabel('Output units')

    # Biases
    ax = fig.add_subplot(axes[0, 2])
    population = model.layer1.E
    ax.plot(population.bias_history, color='black', alpha=0.2)
    avg_bias = torch.mean(population.bias_history, dim=1)
    ax.plot(avg_bias, color='red', linewidth=2)
    ax.set_xlabel('Training steps (epochs*patterns)')
    ax.set_ylabel('bias')

    # Activities
    ax = fig.add_subplot(axes[3, 0])
    im = ax.imshow(model.initial_activity[-1,:,:])
    plt.colorbar(im, ax=ax)
    ax.set_title('Initial activity (unsorted)')
    ax.set_xlabel('Input pattern')
    ax.set_ylabel('Output unit')

    ax = fig.add_subplot(axes[3, 1])
    im = ax.imshow(model.final_activity[-1,:,:])
    plt.colorbar(im, ax=ax)
    ax.set_title('Final activity (unsorted)')
    ax.set_xlabel('Input pattern')
    ax.set_ylabel('Output unit')

    # Temporal dynamics
    ax = fig.add_subplot(axes[3, 2])
    avg_out_act = torch.mean(model.final_activity, dim=[1,2])
    # avg_fbi_act = torch.mean(model.layer1.I.all_pattern_activities, dim=[1,2])
    ax.plot(avg_out_act, color='red', label='Output')
    # ax.plot(avg_fbi_act, color='blue', label='Layer1 FBI')
    ax.set_title('Activity Dynamics')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Average activity')
    ax.legend()


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