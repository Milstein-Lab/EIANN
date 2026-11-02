import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns  # registers the 'icefire' colormap with matplotlib

from EIANN.plot import clean_axes, update_plot_defaults
from .environments import make_2ACDC_tracks
from .treadmill_metrics import stack_laps, cross_correlation_history, region_cross_correlation_history

# fixed order: a population or track keeps its color whichever others are plotted with it
SERIES_COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300')
REGION_LABELS = {'initial_region': 'Initial', 'indicator': 'Indicator', 'pre_r1': 'Pre R1', 'pre_r2': 'Pre R2',
                 'end_region': 'End', 'off_diagonal': 'Off diagonal'}


def get_title(title, base):
    return base if title is None else f'{title}: {base}'


def get_population_save_path(save_path, population):
    """
    Path of the figure of one population, for plots that make one figure per population.
    """
    if save_path is None:
        return None
    base, ext = os.path.splitext(save_path)
    return f'{base}_{population}{ext}'


def save_or_show(fig, save_path, dpi=600):
    """
    Save the figure to save_path and close it, or show it without blocking.
    """
    if save_path is None:
        plt.show(block=False)
        return
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    # keep text editable in vector output
    with matplotlib.rc_context({'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def mark_track(ax, track, tracks):
    """
    Shade the cue of track in gray and the rewarded cell of every track in that track's color, with position on the
    x axis.
    """
    cue_span, reward_spans = track.get_zone_spans()
    ax.axvspan(cue_span[0] - 0.5, cue_span[1] - 0.5, color='gray', alpha=0.2, linewidth=0)
    for other_track, color in zip(tracks, SERIES_COLORS):
        ax.axvspan(other_track.rewarded_position - 0.5, other_track.rewarded_position + 0.5, color=color, alpha=0.25,
                   linewidth=0)


def get_track_legend_handles(tracks):
    return [patches.Patch(color='gray', alpha=0.2, linewidth=0, label='Cue')] + \
        [patches.Patch(color=color, alpha=0.25, linewidth=0, label=f'Track {i + 1} reward')
         for i, (track, color) in enumerate(zip(tracks, SERIES_COLORS))]


def plot_val_rewards(laps, tracks, title=None, save_path=None):
    """
    Greedy reward over training, averaged over tracks.

    Parameters
    ----------
    laps : dict
        From stack_laps.
    tracks : list of CueTreadmillEnv
    title : str, optional
    save_path : str, optional
    """
    fig, ax = plt.subplots(figsize=(2.6, 1.8), layout='constrained')
    ax.axhline(tracks[0].reward_values[1], color='gray', linestyle='--', linewidth=0.5)
    ax.plot(laps['episodes'], laps['rewards'].mean(axis=1), color=SERIES_COLORS[0], linewidth=1)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    fig.suptitle(get_title(title, 'Validation reward'))
    clean_axes(ax)
    save_or_show(fig, save_path)


def plot_final_q_values(laps, tracks, title=None, save_path=None):
    """
    Q values at every position at the last validation step, one panel per track, on a shared color scale.
    """
    q_values = laps['q_values'][-1]
    fig, axes = plt.subplots(1, len(tracks), figsize=(1.3 * len(tracks) + 0.5, 3.), sharey=True, squeeze=False,
                             layout='constrained')
    axes = axes[0]
    image = None
    for i, ax in enumerate(axes):
        image = ax.imshow(q_values[i], cmap='plasma', vmin=q_values.min(), vmax=q_values.max(), aspect='auto',
                          interpolation='nearest')
        ax.set_xticks([0, 1], ['Ignore', 'Lick'])
        ax.set_title(f'Track {i + 1}')
    axes[0].set_ylabel('Position')
    fig.colorbar(image, ax=axes, fraction=0.05, label='Q value')
    fig.suptitle(get_title(title, 'Final Q values'))
    save_or_show(fig, save_path)


def plot_actions_over_training(laps, tracks, title=None, save_path=None):
    """
    Positions of the greedy agent's licks over training, one panel per track. The rewarded cell of every track is shaded,
    which shows whether licking is gated by the cue.
    """
    fig, axes = plt.subplots(1, len(tracks), figsize=(2. * len(tracks), 2.6), sharey=True, squeeze=False,
                             layout='constrained')
    axes = axes[0]
    for i, (track, ax) in enumerate(zip(tracks, axes)):
        mark_track(ax, track, tracks)
        val_steps, positions = np.nonzero(laps['actions'][:, i] > 0)
        ax.scatter(positions, laps['episodes'][val_steps], s=2, color='#0b0b0b', linewidth=0)
        ax.set_xlim(-0.5, track.length - 0.5)
        ax.set_xlabel('Position')
        ax.set_title(f'Track {i + 1}')
    axes[0].invert_yaxis()
    axes[0].set_ylabel('Episode')
    fig.legend(handles=get_track_legend_handles(tracks), loc='outside lower center', ncols=len(tracks) + 1)
    fig.suptitle(get_title(title, 'Licks over training'))
    clean_axes(axes)
    save_or_show(fig, save_path)


def plot_hidden_activity(laps, tracks, populations=None, val_step=-1, title=None, save_path=None):
    """
    Activity by unit and position at one validation step, one figure per population with one panel per track. Units are
    sorted by their peak position on the first track, in every panel, so a preserved sequence is visible across tracks.

    Parameters
    ----------
    laps : dict
    tracks : list of CueTreadmillEnv
    populations : list of str, optional
        Defaults to every recorded population.
    val_step : int, default -1
    title : str, optional
    save_path : str, optional
        The population name is added before the extension.
    """
    for population in laps['activity'] if populations is None else populations:
        activity = laps['activity'][population][val_step]
        order = np.argsort(np.argmax(activity[0], axis=0))
        fig, axes = plt.subplots(1, len(tracks), figsize=(1.8 * len(tracks) + 0.5, 2.2), sharey=True, squeeze=False,
                                 layout='constrained')
        axes = axes[0]
        image = None
        for i, (track, ax) in enumerate(zip(tracks, axes)):
            image = ax.imshow(activity[i][:, order].T, cmap='gray_r', vmin=0., vmax=activity.max(), aspect='auto',
                              interpolation='nearest')
            mark_track(ax, track, tracks)
            ax.set_xlabel('Position')
            ax.set_title(f'Track {i + 1}')
        axes[0].set_ylabel('Unit (sorted by peak)')
        fig.colorbar(image, ax=axes, fraction=0.05, label='Activity')
        fig.suptitle(get_title(title, f'Hidden activity ({population})'))
        save_or_show(fig, get_population_save_path(save_path, population))


def draw_cross_correlation_heatmap(ax, matrix, track, label_axes=False):
    """
    Draw a cross-correlation matrix in the style of the Spruston lab OSM paper figures: icefire colormap on [-1, 1],
    a black frame, white dashed lines at zone boundaries and white boxes around the cue and reward zones.

    Returns
    -------
    The heatmap image, for a shared colorbar.
    """
    icefire = sns.color_palette('icefire', as_cmap=True)
    sns.heatmap(matrix, ax=ax, cmap=icefire, vmin=-1, vmax=1, cbar=False, rasterized=True)

    length = matrix.shape[0]
    for position in (0, length):
        ax.axhline(position, color='k', linewidth=2)
        ax.axvline(position, color='k', linewidth=2)

    cue_span, reward_spans = track.get_zone_spans()
    zones = [cue_span] + reward_spans
    for boundary in sorted({position for zone in zones for position in zone}):
        ax.axhline(boundary, linestyle=(0, (2, 5)), color='white', linewidth=0.75)
        ax.axvline(boundary, linestyle=(0, (2, 5)), color='white', linewidth=0.75)
    for start, stop in zones:
        ax.plot([start, stop, stop, start, start], [start, start, stop, stop, start], color='white', linewidth=1)

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    if label_axes:
        ax.set_xlabel('Track 2 position')
        ax.set_ylabel('Track 1 position')
    return ax.collections[0]


def plot_cross_correlation_over_training(laps, tracks, populations=None, num_timepoints=4, pair=0, title=None,
                                         save_path=None):
    """
    Cross-correlation heatmap across tracks at several points during training, from the first to the last validation
    step, one figure per population.

    Parameters
    ----------
    laps : dict
    tracks : list of CueTreadmillEnv
    populations : list of str, optional
        Defaults to every recorded population.
    num_timepoints : int, default 4
    pair : int, default 0
        Which pair of tracks to plot.
    title : str, optional
    save_path : str, optional
        The population name is added before the extension.
    """
    val_steps = np.unique(np.linspace(0, len(laps['episodes']) - 1, num_timepoints).round().astype(int))
    for population in laps['activity'] if populations is None else populations:
        matrices = cross_correlation_history(laps, population)
        fig, axes = plt.subplots(1, len(val_steps), figsize=(1.6 * len(val_steps) + 0.5, 1.9), squeeze=False,
                                 layout='constrained')
        axes = axes[0]
        image = None
        for i, (val_step, ax) in enumerate(zip(val_steps, axes)):
            image = draw_cross_correlation_heatmap(ax, matrices[val_step, pair], tracks[0], label_axes=i == 0)
            ax.set_title(f'Episode {laps["episodes"][val_step] + 1}')
        fig.colorbar(image, ax=axes, fraction=0.02, label='Correlation')
        fig.suptitle(get_title(title, f'Cross-correlation over training ({population})'))
        save_or_show(fig, get_population_save_path(save_path, population))


def plot_region_cross_correlation(laps, tracks, populations=None, regions=None, panels='region', pair=0, title=None,
                                  save_path=None):
    """
    Region means of the cross-correlation across tracks over training.

    Parameters
    ----------
    laps : dict
    tracks : list of CueTreadmillEnv
    populations : list of str, optional
        Defaults to every recorded population.
    regions : list of str, optional
        Defaults to every region with panels='region', and to Pre R1, Pre R2 and Off diagonal with panels='population'.
    panels : str, default 'region'
        'region' gives one panel per region and one line per population; 'population' the reverse.
    pair : int, default 0
        Which pair of tracks to plot.
    title : str, optional
    save_path : str, optional
    """
    assert panels in ['region', 'population'], f'plot_region_cross_correlation: invalid panels: {panels}'
    if populations is None:
        populations = list(laps['activity'])
    if regions is None:
        regions = list(REGION_LABELS) if panels == 'region' else ['pre_r1', 'pre_r2', 'off_diagonal']
    history = {population: region_cross_correlation_history(laps, population, tracks[0]) for population in populations}
    population_colors = dict(zip(laps['activity'], SERIES_COLORS))
    region_colors = dict(zip(REGION_LABELS, SERIES_COLORS))

    if panels == 'region':
        panel_names, line_names = regions, populations
        get_values = lambda region, population: history[population][region][:, pair]
        line_colors, line_labels, panel_labels = population_colors, {name: name for name in populations}, REGION_LABELS
    else:
        panel_names, line_names = populations, regions
        get_values = lambda population, region: history[population][region][:, pair]
        line_colors, line_labels, panel_labels = region_colors, REGION_LABELS, {name: name for name in populations}

    num_cols = min(3, len(panel_names))
    num_rows = int(np.ceil(len(panel_names) / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(1.7 * num_cols + 0.6, 1.5 * num_rows), sharex=True,
                             sharey=True, squeeze=False, layout='constrained')
    axes = axes.flatten()
    for ax, panel_name in zip(axes, panel_names):
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        for line_name in line_names:
            ax.plot(laps['episodes'], get_values(panel_name, line_name), color=line_colors[line_name], linewidth=1,
                    label=line_labels[line_name])
        ax.set_title(panel_labels[panel_name])
    for i, ax in enumerate(axes):
        if i >= len(panel_names):
            ax.set_visible(False)
        else:
            if i % num_cols == 0:
                ax.set_ylabel('Correlation')
            if i + num_cols >= len(panel_names):
                ax.set_xlabel('Episode')
                ax.tick_params(labelbottom=True)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside right upper')
    fig.suptitle(get_title(title, 'Region cross-correlation over training'))
    clean_axes(axes)
    save_or_show(fig, save_path)


def plot_2ACDC_summary(rl_history, title=None, save_prefix=None, plot_format='svg'):
    """
    The standard plots of a trained 2ACDC model. nested --plot and the analysis notebook both call this, so that they
    show the same thing.

    Parameters
    ----------
    rl_history : dict
        network.rl_history of an exported model, from nested_optimize_EIANN_2ACDC.
    title : str, optional
    save_prefix : str, optional
        Figures are saved to <save_prefix>_<plot name>.<plot_format> instead of shown.
    plot_format : str, default 'svg'
    """
    def get_save_path(name):
        return None if save_prefix is None else f'{save_prefix}_{name}.{plot_format}'

    update_plot_defaults()
    laps = stack_laps(rl_history['val_laps'], rl_history['val_episodes'])
    tracks = make_2ACDC_tracks(**rl_history['treadmill_kwargs'])
    plot_val_rewards(laps, tracks, title=title, save_path=get_save_path('val_rewards'))
    plot_final_q_values(laps, tracks, title=title, save_path=get_save_path('final_q_values'))
    plot_actions_over_training(laps, tracks, title=title, save_path=get_save_path('actions_over_training'))
    plot_hidden_activity(laps, tracks, title=title, save_path=get_save_path('hidden_activity'))
    plot_cross_correlation_over_training(laps, tracks, title=title, save_path=get_save_path('cross_correlation'))
    plot_region_cross_correlation(laps, tracks, title=title, save_path=get_save_path('region_cross_correlation'))
    plot_region_cross_correlation(laps, tracks, panels='population', title=title,
                                  save_path=get_save_path('region_cross_correlation_by_population'))
