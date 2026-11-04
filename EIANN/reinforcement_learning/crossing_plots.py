"""
Plots of the Crossing_CL task (CrossingChoiceEnv), from the rl_history that nested_optimize_EIANN_Crossing_CL saves
with every exported network. Everything works from the greedy episodes recorded during training, so nothing retrains.
"""
import os
import shutil

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib import animation

from EIANN.plot import clean_axes
from .environments import make_crossing_tasks
from .crossing_metrics import stack_outcomes, block_outcome_rates, episode_positions
from .treadmill_plots import SERIES_COLORS, mark_blocks, save_or_show, get_title

OUTCOME_COLORS = {'correct': '#008300', 'wrong': '#d62728', 'timeout': '#7f7f7f'}
WALL_COLOR = '#3b3b3b'
TILE_COLOR = '#f2d27a'


def task_name(task):
    return chr(ord('A') + task)


def get_tasks(rl_history):
    return make_crossing_tasks(**(rl_history.get('crossing_kwargs') or {}))


def draw_room(ax, env, frame_color=None):
    """
    Draw one task's room: walls, the goal tiles (all the same color, as the agent sees them), a star on the rewarded
    tile and the start cell, with row 0 at the top.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    env : CrossingChoiceEnv
    frame_color : str, optional
        Color of a thick frame around the room, e.g. to mark the task being trained.
    """
    size = env.size
    for row in range(size):
        for col in range(size):
            if col == env.wall_col and row != env.gap_row:
                color = WALL_COLOR
            elif (row, col) in env.goal_tiles:
                color = TILE_COLOR
            else:
                color = 'white'
            ax.add_patch(patches.Rectangle((col - 0.5, row - 0.5), 1, 1, facecolor=color, edgecolor='#d0d0d0',
                                           linewidth=0.5))
    ax.plot(env.rewarded_tile[1], env.rewarded_tile[0], marker='*', markersize=9, color='#b8860b', linestyle='none')
    ax.text(env.start_pos[1], env.start_pos[0], 'S', ha='center', va='center', fontsize=7, color='gray')
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(frame_color is not None)
        if frame_color is not None:
            spine.set_color(frame_color)
            spine.set_linewidth(2.5)


def draw_trajectory(ax, positions, outcome):
    """
    Draw a path through cell centers, darker towards its end, with its last cell marked by outcome. Moves into walls
    leave the agent in place, so such steps add nothing to the path.

    Returns
    -------
    list of matplotlib artists
    """
    artists = []
    num_steps = len(positions) - 1
    # a small offset per step separates revisits of the same cells
    offsets = np.linspace(-0.12, 0.12, num_steps + 1) if num_steps > 0 else np.zeros(1)
    xs, ys = positions[:, 1] + offsets, positions[:, 0] + offsets
    for step in range(num_steps):
        alpha = 0.25 + 0.75 * (step + 1) / num_steps
        artists += ax.plot(xs[step:step + 2], ys[step:step + 2], color='#2a2a2a', linewidth=1.2, alpha=alpha,
                           solid_capstyle='round')
    color = OUTCOME_COLORS.get(outcome, 'black')
    artists += ax.plot(positions[-1, 1], positions[-1, 0], marker='o', markersize=7, markerfacecolor='none',
                       markeredgecolor=color, markeredgewidth=2, linestyle='none')
    return artists


def get_outcome_curves(rl_history, window=None):
    """
    Success, navigation (any tile reached) and choice accuracy of every task at every validation step, optionally as a
    running mean over the last window validation steps.

    Returns
    -------
    episodes : np.ndarray, shape (num_val,)
    curves : dict of np.ndarray, each shape (num_val, num_tasks)
    """
    outcomes = stack_outcomes(rl_history['val_laps'])
    curves = {'success': (outcomes == 'correct').astype(float), 'navigation': (outcomes != 'timeout').astype(float)}
    if window is not None and window > 1:
        kernel = np.ones(window)
        counts = np.convolve(np.ones(len(outcomes)), kernel)[:len(outcomes)]
        curves = {key: np.stack([np.convolve(values[:, task], kernel)[:len(outcomes)] / counts
                                 for task in range(values.shape[1])], axis=1) for key, values in curves.items()}
    with np.errstate(invalid='ignore', divide='ignore'):
        curves['choice'] = np.where(curves['navigation'] > 0, curves['success'] / curves['navigation'], np.nan)
    return np.asarray(rl_history['val_episodes']), curves


def plot_outcome_curves(rl_history, window=10, title=None, save_path=None):
    """
    Success, navigation and choice accuracy of every task over training (running mean over window validation steps),
    with every block shaded in the color of the task it trains on.
    """
    episodes, curves = get_outcome_curves(rl_history, window)
    num_tasks = rl_history['num_tasks']
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.8), sharey=True, layout='constrained')
    for ax, (key, label) in zip(axes, (('success', 'Success'), ('navigation', 'Reached a tile'),
                                        ('choice', 'Choice accuracy'))):
        mark_blocks(ax, episodes, rl_history['episodes_per_task'], num_tasks)
        for task in range(num_tasks):
            ax.plot(episodes, curves[key][:, task], color=SERIES_COLORS[task], linewidth=1,
                    label=f'Task {task_name(task)}')
        ax.set_title(label)
        ax.set_xlabel('Episode')
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_ylabel('Fraction of greedy episodes')
    axes[-1].legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
    clean_axes(axes)
    fig.suptitle(get_title(title, f'Running mean over {window} validation steps'))
    save_or_show(fig, save_path)


def plot_outcome_matrices(rl_history, window, title=None, save_path=None):
    """
    Block x task matrices of success, navigation and choice accuracy, averaged over the last window episodes of each
    block (as the nested features).
    """
    num_tasks = rl_history['num_tasks']
    num_blocks = num_tasks * rl_history['num_cycles']
    rates = block_outcome_rates(rl_history['val_laps'], rl_history['val_episodes'], rl_history['episodes_per_task'],
                                num_blocks, window)
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 1.9), layout='constrained')
    for ax, (key, label) in zip(axes, (('success', 'Success'), ('navigation', 'Reached a tile'),
                                        ('choice', 'Choice accuracy'))):
        plot_matrix(ax, rates[key], num_tasks)
        ax.set_title(label)
    if title is not None:
        fig.suptitle(title)
    save_or_show(fig, save_path)


def plot_matrix(ax, matrix, num_tasks, vmin=0., vmax=1.):
    """
    One block x task matrix, with its values written in the cells.
    """
    image = ax.imshow(matrix, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
    num_blocks = matrix.shape[0]
    for block in range(num_blocks):
        for task in range(matrix.shape[1]):
            value = matrix[block, task]
            ax.text(task, block, '–' if np.isnan(value) else f'{value:.2f}', ha='center', va='center', fontsize=6,
                    color='white' if np.isnan(value) or value < 0.5 * (vmin + vmax) else 'black')
    ax.set_xticks(range(matrix.shape[1]), [task_name(task) for task in range(matrix.shape[1])])
    ax.set_yticks(range(num_blocks), [f'{block} ({task_name(block % num_tasks)})' for block in range(num_blocks)])
    ax.set_xlabel('Evaluated on task')
    ax.set_ylabel('Block (trained on)')
    return image


def plot_crossing_summary(rl_history, window, title=None, save_prefix=None, plot_format='svg'):
    """
    The standard plots of one exported seed: outcome curves over training and the block x task outcome matrices.

    Parameters
    ----------
    rl_history : dict
    window : int
        Episodes at the end of each block to average the matrices over (num_episodes_reward_window).
    title : str, optional
    save_prefix : str, optional
        Saves the figures to '<save_prefix>_<plot>.<plot_format>' instead of showing them.
    plot_format : str, default 'svg'
    """
    save_path = lambda name: None if save_prefix is None else f'{save_prefix}_{name}.{plot_format}'
    plot_outcome_curves(rl_history, title=title, save_path=save_path('outcome_curves'))
    plot_outcome_matrices(rl_history, window, title=title, save_path=save_path('outcome_matrices'))


def animate_learning(rl_history, window=10, val_steps=None, title=None, interval=125, save_path=None, dpi=120):
    """
    Video of learning over training: one frame per validation step, with the greedy path on every task at that step
    (end marked green if correct, red if wrong, gray on timeout), the task being trained framed in its color, and the
    success curves below with a line at the current episode.

    Parameters
    ----------
    rl_history : dict
    window : int, default 10
        Running mean of the success curves, in validation steps.
    val_steps : sequence of int, optional
        Validation steps to show (default: all).
    title : str, optional
    interval : int, default 125
        Milliseconds per frame.
    save_path : str, optional
        '.gif' (pillow) or '.mp4' (needs ffmpeg). The animation is returned either way.
    dpi : int, default 120

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    tasks = get_tasks(rl_history)
    num_tasks = len(tasks)
    episodes_per_task = rl_history['episodes_per_task']
    episodes, curves = get_outcome_curves(rl_history, window)
    if val_steps is None:
        val_steps = range(len(episodes))
    val_steps = list(val_steps)
    # paths of every task at every validation step shown, replayed from the recorded actions
    paths = {step: [episode_positions(task, rl_history['val_laps'][step][index]['actions'])
                    for index, task in enumerate(tasks)] for step in val_steps}

    fig = plt.figure(figsize=(1.6 * num_tasks + 1.6, 3.8))
    # the right margin holds the legends
    grid = fig.add_gridspec(2, num_tasks, height_ratios=(1.25, 1.), hspace=0.5, wspace=0.15, left=0.12, right=0.8,
                            top=0.86, bottom=0.12)
    room_axes = [fig.add_subplot(grid[0, task]) for task in range(num_tasks)]
    curve_ax = fig.add_subplot(grid[1, :])
    mark_blocks(curve_ax, episodes, episodes_per_task, num_tasks)
    for task in range(num_tasks):
        curve_ax.plot(episodes, curves['success'][:, task], color=SERIES_COLORS[task], linewidth=1,
                      label=f'Task {task_name(task)}')
    curve_ax.set_ylim(-0.05, 1.05)
    curve_ax.set_xlabel('Episode')
    curve_ax.set_ylabel('Success')
    curve_ax.set_title(f'Running mean over {window} validation steps', fontsize='small')
    curve_ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1))
    clean_axes(curve_ax)
    cursor = curve_ax.axvline(episodes[val_steps[0]], color='black', linewidth=0.8)
    legend_handles = [plt.Line2D([], [], marker='o', markerfacecolor='none', markeredgecolor=color,
                                 markeredgewidth=2, linestyle='none', label=outcome)
                      for outcome, color in OUTCOME_COLORS.items()]
    fig.legend(handles=legend_handles, title='Path end', loc='upper left', bbox_to_anchor=(0.81, 0.86),
               frameon=False, fontsize='small', title_fontsize='small')
    suptitle = fig.suptitle('')

    def update(frame):
        step = val_steps[frame]
        episode = episodes[step]
        trained = min(int(episode // episodes_per_task), rl_history['num_cycles'] * num_tasks - 1) % num_tasks
        artists = []
        for task, (ax, env) in enumerate(zip(room_axes, tasks)):
            ax.clear()
            draw_room(ax, env, frame_color=SERIES_COLORS[task] if task == trained else None)
            positions, outcome = paths[step][task]
            artists += draw_trajectory(ax, positions, outcome)
            ax.set_title(f'Task {task_name(task)}', color=SERIES_COLORS[task])
        cursor.set_xdata([episode, episode])
        suptitle.set_text(get_title(title, f'Episode {episode + 1}: training on task {task_name(trained)}'))
        return artists + [cursor, suptitle]

    anim = animation.FuncAnimation(fig, update, frames=len(val_steps), interval=interval, blit=False)
    if save_path is not None:
        save_animation(anim, save_path, fps=1000. / interval, dpi=dpi)
    return anim


def save_animation(anim, save_path, fps, dpi=120):
    """
    Save an animation as a GIF (pillow) or, for any other extension, with ffmpeg (e.g. .mp4).
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    if save_path.endswith('.gif'):
        writer = animation.PillowWriter(fps=fps)
    else:
        if shutil.which('ffmpeg') is None:
            raise RuntimeError('save_animation: saving as video needs ffmpeg (e.g. conda install -c conda-forge '
                               'ffmpeg); save as .gif instead')
        writer = animation.FFMpegWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=dpi)


def minigrid_frames(env, actions, tile_size=32):
    """
    MiniGrid's own rendering of a recorded episode, replayed on env: one RGB frame at the start and after every action.
    The agent is drawn as a triangle pointing the way it last moved, and every goal tile is green.

    Returns
    -------
    frames : list of np.ndarray, each (height, width, 3) uint8
    views : np.ndarray of int, shape (len(actions) + 1, 2)
        MiniGrid (x, y) of the top-left cell of the agent's 3 x 3 view in every frame (CrossingChoiceEnv.view_top_left).
    """
    env.reset()
    frames = [env.get_frame(highlight=False, tile_size=tile_size)]
    views = [env.view_top_left()]
    for action in actions:
        env.step(int(action))
        frames.append(env.get_frame(highlight=False, tile_size=tile_size))
        views.append(env.view_top_left())
    return frames, np.array(views, dtype=int)


def animate_greedy_episodes(rl_history, val_step=-1, tile_size=32, title=None, interval=250, save_path=None, dpi=120):
    """
    Video of the greedy episode on every task at one validation step, step by step, in MiniGrid's rendering. A star
    marks the rewarded tile, and a white square the 3 x 3 view the agent observes (ahead of it with egocentric actions,
    around it with allocentric ones). Tasks whose episode ended earlier hold their last frame.

    Parameters
    ----------
    rl_history : dict
    val_step : int, default -1
        Validation step (index into val_laps), e.g. -1 for the end of training.
    tile_size : int, default 32
    title : str, optional
    interval : int, default 250
        Milliseconds per frame.
    save_path : str, optional
        '.gif' (pillow) or '.mp4' (needs ffmpeg).
    dpi : int, default 120

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    tasks = get_tasks(rl_history)
    laps = rl_history['val_laps'][val_step]
    episode = np.asarray(rl_history['val_episodes'])[val_step]
    rendered = [minigrid_frames(task, lap['actions'], tile_size) for task, lap in zip(tasks, laps)]
    num_frames = max(len(frames) for frames, views in rendered)

    fig, axes = plt.subplots(1, len(tasks), figsize=(1.9 * len(tasks), 2.3), layout='constrained')
    axes = np.atleast_1d(axes)
    images, views, labels = [], [], []
    for task, (ax, env, (frames, task_views), lap) in enumerate(zip(axes, tasks, rendered, laps)):
        images.append(ax.imshow(frames[0]))
        row, col = env.rewarded_tile
        ax.plot((col + 1.5) * tile_size, (row + 1.5) * tile_size, marker='*', markersize=10, color='#f2d27a',
                markeredgecolor='black', markeredgewidth=0.5, linestyle='none')
        views.append(ax.add_patch(patches.Rectangle((0, 0), 3 * tile_size, 3 * tile_size, fill=False,
                                                    edgecolor='white', linewidth=1.5)))
        labels.append(ax.set_title(f'Task {task_name(task)}', color=SERIES_COLORS[task]))
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(get_title(title, f'Greedy episodes after episode {episode + 1}'))

    def update(frame):
        for task, (image, view, label, (frames, task_views), lap) in enumerate(zip(images, views, labels, rendered,
                                                                                     laps)):
            index = min(frame, len(frames) - 1)
            image.set_data(frames[index])
            # the observed 3 x 3 view, in pixels (MiniGrid coordinates include the border)
            view.set_xy(tuple(task_views[index] * tile_size))
            done = index == len(frames) - 1
            label.set_text(f'Task {task_name(task)}: step {index}' + (f', {lap["outcome"]}' if done else ''))
        return images + views + labels

    anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=False)
    if save_path is not None:
        save_animation(anim, save_path, fps=1000. / interval, dpi=dpi)
    return anim
