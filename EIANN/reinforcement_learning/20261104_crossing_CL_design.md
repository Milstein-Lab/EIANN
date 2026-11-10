# Crossing_CL: gridworld task for continual RL, design (2026-11-04)

Design for a small gridworld continual-learning (CL) task. It will sit next to 2ACDC_CL and the class-incremental MNIST
results in the CL paper. The task is called **Crossing_CL**. It is implemented and has passed the three checks below
(see "Status").

## Purpose

2ACDC_CL already shows that a network can use a cue to tell tasks apart. This task asks something different. The tasks
here differ in **both the route through the environment and which goal pays**, which is closer to switching between
environments. The task is compatible: a single recurrent policy can solve every task at once, because the cue (where
the gap in the wall is) is in the observation stream. So a drop in performance on an old task after training on a new
one is forgetting, not interference in the sense of Kessler et al. (2022).

Relation to Kessler et al. (AAAI 2022, arXiv:2106.02940):
- Their MiniGrid tasks are SimpleCrossingS9N1 levels with different wall and door positions and a visible goal. Their
  agent is a feedforward DQN on the current 7×7×3 view, with replay shared across tasks. Early observations look the
  same across tasks, and the shared replay holds conflicting targets for them. They call this interference.
- Our agents are recurrent and have no replay buffer. Once the agent has seen the gap, its hidden state can carry it,
  so the tasks are jointly solvable.
- Their pure "same state, different task" case is the Pendulum experiment (the same dynamics, rewarded at +90° in one
  task and −90° in the other). We do not use that setting.

## Layout

The interior is 4×4 (6×6 with MiniGrid's border wall). Rows and columns run 0–3 from the top-left. The wall is in
column 1 and has one gap. The agent starts at (0,0), facing south. Three goal tiles are marked in every task, and only
one of them pays.

```
       Task A               Task B               Task C
   gap row 3 → (0,3)    gap row 1 → (3,3)    gap row 2 → (1,3)
     S # . G              S # . g              S # . g
     . # . g              . . . g              . # . G
     . # . .              . # . .              . . . .
     . . . g              . # . G              . # . g

S start   # wall   G rewarded tile   g marked, unrewarded tile   . empty
```

- Every rewarded tile can be reached without stepping on a wrong tile.
- The pairing of gap to goal can't be solved by a simple rule. "Go to the marked tile nearest the gap" picks the wrong
  tile in A and B and is a tie in C. Any added task should also break such rules.
- Shortest paths: A 12 actions, B 9, C 8 (egocentric), or 9, 6 and 6 moves (allocentric).
- There are 13 free cells, or 52 states once the egocentric heading is included.

## Environment spec

| Item | Choice |
|---|---|
| Base | MiniGrid `MiniGridEnv` subclass, one fixed layout per instance (like `CueTreadmillEnv` = one track) |
| Actions | `actions='egocentric'` (default): turn left, turn right, forward. `actions='allocentric'`: north, east, south, west. A move into a wall leaves the agent in place |
| Observation | Egocentric: MiniGrid's forward 3×3 view (one cell to each side, two ahead), blocked by walls; one-hot of {unseen, empty, wall, goal tile} for each of the 9 cells = 36 inputs. Allocentric: the 3×3 neighborhood centered on the agent, north up; one-hot of {empty, wall, goal tile} = 27 inputs. No direction and no mission string |
| Reward | +1 for the rewarded tile; −0.1 for a wrong tile (the same as a wrong lick in 2ACDC); 0 on timeout. Flat, without MiniGrid's step decay. Optional `step_reward` added to every step: −0.01 in the Crossing_CL config, 0 by default |
| Termination | Stepping onto any goal tile ends the episode. Truncation at `max_steps` (about 50; to be tuned) |
| `info` | `outcome` ∈ {`correct`, `wrong`, `timeout`}. `ScheduledEnv` adds `task` |
| Factory | `make_crossing_tasks()` returns the list of task envs, like `make_2ACDC_tracks` |

Reasons:
- **Forced choice:** one choice per episode is the RL equivalent of one classification per test image. It also rules
  out sweeping all three tiles without using the cue.
- **Penalty size:** it has to stay well below 0.5. A random choice is then still worth more than wandering until
  timeout (1/3 − 2/3·p > 0). With a larger penalty, an uncertain agent can learn to avoid choosing, and right after a
  task switch that would look like forgetting how to navigate. After a switch, the penalty also pushes the old route's
  Q-value below the alternatives, which helps the agent find the new goal.
- **Egocentric actions:** they are standard for MiniGrid and what Kessler et al. used. They need slow exploration
  decay: with ε × 0.995 per episode, A was learned on only 1–2 of 3 seeds. With ε × 0.999, `OnlineQLearning` learned
  every task alone on every seed (5000 episodes, 3 seeds, −0.01 step penalty). The greedy policy was reliably correct
  after:

  | Task | Episodes | Frames |
  |---|---|---|
  | A | 750–1000 | 32–41k |
  | B | 650–800 | 26–32k |
  | C | 350–450 | 15–19k |

  Without the step penalty, learning was about 15–20% slower. Allocentric actions stay available: they learn about
  3× faster (all tasks by about 250 episodes).
- **3×3 view:** the gap is usually out of view at the moment of choice, so the network must carry the cue in its
  recurrent state, as in 2ACDC. With egocentric actions, facing south at the start, the wall column is in view, so
  the tasks already look different; from column 3, facing north or south, the wall is out of view. With allocentric
  actions, A and C look the same at the start.
- **One-hot input:** MiniGrid's raw object, color and state codes are integers that a network would read as ordered.
  One-hot input is also nonnegative, which suits the Dale's-law networks.

## CL protocol

Follow `nested_optimize_EIANN_2ACDC_CL.py` so the two RL tasks are reported the same way:
- Blocks of `episodes_per_task` episodes on one task, in order A, B, C. With `num_cycles=1` each task is seen once,
  matching MNIST CL. With `num_cycles>1`, relearning can be measured.
- `algo.end_task()` after each block. It estimates EWC importance from the block's episodes and calls
  `network.update_CL_states()`.
- Algorithm: `OnlineQLearning` (an update after every step) at learning rate 0.1, ε 0.999 decaying by 0.999 per
  episode, γ 0.9.
  - `QLearning` (one update per episode, backpropagating through the whole episode, as van_bp in 2ACDC_CL) needs
    learning rate about 1.0. That is because its MSE loss averages the update over the episode's steps, so the learning
    per step is the learning rate divided by the episode length. It diverges at 5.0.
  - With egocentric actions, `QLearning` failed to learn a task alone on 3 of 18 runs, and needed up to 4950 episodes
    for A. `OnlineQLearning` learned all 18.
- `episodes_per_task: 2000`, about 2× the slowest single-task learning time (task A).
- `epsilon_schedule: 'reset'` in the Crossing_CL config: exploration restarts with every task.
- Exploration across task switches is set with an `epsilon_schedule` option in `algorithm_kwargs`. It belongs to
  `QLearning` (and `OnlineQLearning`), so 2ACDC_CL gets it too:

  | `epsilon_schedule` | Behavior | Agent told about task switches? |
  |---|---|---|
  | `'fixed'` | ε stays at `epsilon` for the whole run | No |
  | `'decay'` (default) | ε decays by `epsilon_decay` per episode over the whole run, down to `epsilon_min`. This is the current 2ACDC_CL behavior | No |
  | `'reset'` | Like `'decay'`, but `end_task()` sets ε back to its initial value, so the decay restarts each block | Yes, through exploration |

  - With allocentric actions, `'decay'` learned every new task as well as `'reset'` did. Egocentric exploration is
    harder, and Crossing_CL uses `'reset'`.
  - `'reset'` gives the exploration policy the task boundary, just as `end_task()` already gives it to CL rules like
    EWC and SI. Methods must say which schedule was used, and both RL tasks should use the same one in the paper.
  - `'fixed'` with ε near 0.05 tests whether forgetting holds up without any boundary-driven exploration.
- Task-incremental variant: `info['task']` is enough for one Q head per task (the RL equivalent of `multihead: True` in
  MNIST TI, and the same as OWL's oracle).

## Metrics

- The reward matrix [block, task] and the cumulative "seen tasks" mean, from the same `block_rewards` and
  `seen_track_rewards` pattern as 2ACDC_CL. ACC/BWT/FWT are computed the same way as for MNIST.
- Evaluation is the same as in 2ACDC_CL. Every `val_interval` episodes, one greedy episode runs on each task. The
  objective `phase{b}_reward` is the greedy reward averaged over the last `num_episodes_reward_window` validation
  checkpoints of block b and over the tasks seen so far. Averaging over the window gives a graded score. With the step
  penalty, a correct return is 1 − 0.01 × the path length, so returns differ slightly between tasks. Success does not.
- Primary score: **success rate** (the fraction of validation episodes with `outcome == 'correct'`).
- Breakdown of forgetting:
  - **navigation success**: any goal tile reached, i.e. `outcome != 'timeout'`;
  - **choice accuracy**: correct choices / tiles reached. Chance is 1/3.

  Forgetting of the route and forgetting of the pairing are confounded by design; this split separates them.

## Number of tasks

Three: A, B and C. Unlike MNIST CL there is no 5-task version. A 4th task would need a gap in row 0 and a marked tile
at (2,3), and 5 tasks would need a 5×5 interior.

## Checks before comparing CL methods

1. Every rule to be compared reaches near-optimal success on each task trained alone. If a rule fails here, it can't
   learn the delay, and forgetting can't be measured for it.
2. Training on randomly interleaved tasks (`RandomSchedule`) reaches near-optimal success on all three. This shows the
   tasks are compatible and gives the upper bound for the CL plots.
   This is the nested task **Crossing** (`optimize/nested_optimize_EIANN_Crossing.py`, configs in
   `optimize/{network_config,optimize_config,optimize_params}/Crossing/`). It trains 6000 random-order episodes, the
   same total as Crossing_CL, with ε decaying over the whole run and no `end_task()`, and has one objective, the greedy
   reward averaged over tasks over the last 250 episodes. At its x0 (van_bp, learning rate 0.03, ε decay 0.9995), only 1
   of 5 seeds solved all three tasks within 6000 episodes (success A 0.38, B 0.80, C 0.80); every failure was a
   timeout, never the wrong tile. It still needs to be optimized.
3. Plain sequential backprop clearly forgets, with BWT well below 0. If it doesn't, the task is too easy to separate
   CL methods.

## Status (2026-11-04)

Model `van_bp_relu_SGD`, not optimized.

**Egocentric (current config):** `QLearning` with the loss summed over each episode's steps
(`criterion_kwargs: {reduction: sum}`), learning rate 0.02, ε 0.999 decaying by 0.999 and reset every task, γ 0.9,
step penalty −0.01, 2000 episodes per task.
1. **Each task alone:** learned within its 2000-episode block on all 5 seeds (check 3 below).
2. **Interleaved** (`RandomSchedule`, 12000 episodes, 3 seeds, ε decay 0.9996):
   - at learning rate 0.03, all three tasks were solved on every seed by about 6000 episodes;
   - at 0.02, on 2 of 3 seeds; the third solved B and C but not A.

   So the tasks are jointly solvable when backprop runs through the whole episode, which trains the memory the
   egocentric view needs:
   - The view aliases decision states across tasks. At (3,2) facing east, A and B look identical, but A needs a left
     turn and B a step forward. (2,2) is the same in A and C.
   - `OnlineQLearning` cuts gradients at every step, and it failed on A interleaved.
   - `QLearning` with the mean loss also failed. The mean gives each step of a long episode (task A) a smaller share of
     the update.
3. **Sequential** (5 seeds): every task reaches success 1.0 within its own block and drops to 0 after the next block.
   None diverged (at learning rate 0.03, 2 of 5 seeds diverged; at 0.01, A was not learned within its block). Both
   kinds of forgetting appear: earlier tasks time out, or reach a tile but the wrong one.

**Allocentric** (`actions='allocentric'`; earlier runs, `decay` and `reset` schedules):
1. **Each task alone:** `OnlineQLearning` at 0.1 learned every task in about 250 episodes. `QLearning` at 1.0 took
   about 150 episodes for C and 600 for A.
2. **Interleaved:** `OnlineQLearning` solved all three tasks on every seed by about episode 1,100 (3 seeds, 3000
   episodes); `QLearning` on 2 of 3 seeds. The tasks are compatible.
3. **Sequential:** each learned task dropped to about 0 after the next block.
   - With `QLearning` and 1000 episodes per task (5 seeds), success at the end of each task's block was A 0.40, B 0.98,
     C 1.00; BWT −0.69.
   - The navigation/choice breakdown separated two kinds of forgetting. After learning B, A no longer reached any tile
     (navigation 0.03). After learning C, B reached a tile every time but the wrong one.

Post-hoc analysis: `notebooks/Crossing_CL_analysis.ipynb`. It covers:
- the block × task matrices and ACC/BWT/FWT;
- curves over training;
- a video of learning over training (`animate_learning`);
- MiniGrid-rendered videos of the greedy episodes at the end of each block (`animate_greedy_episodes`).

## Implementation

- `environments/crossing.py`:
  - `CrossingChoiceEnv`, named so it is not confused with MiniGrid's own `CrossingEnv`;
  - `make_crossing_tasks()` and `CROSSING_TASKS`;
  - registered as `EIANN/Crossing-v0`.
- `algorithms/q_learning.py`: `epsilon_schedule`.
- `algorithms/rl_algorithm.py`: `run_greedy_episode` records `outcome`.
- `crossing_metrics.py`:
  - `block_outcome_rates`, which returns the success, navigation and choice matrices;
  - `episode_positions`, which replays a recorded episode;
  - `cl_summary`, which computes ACC/BWT/FWT.
- `crossing_plots.py`: the summary plots and the two videos. `nested.analyze --plot` makes the summary plots.
- `optimize/nested_optimize_EIANN_Crossing_CL.py`, with configs in `optimize/{network_config,optimize_config,
  optimize_params}/Crossing_CL/`. Only van_bp so far.
- `optimize/nested_optimize_EIANN_Crossing.py`: the interleaved counterpart (task Crossing), with configs in
  `optimize/{network_config,optimize_config,optimize_params}/Crossing/` and Expanse jobscripts
  `optimize/jobscripts/{optimize,export_optimized}_EIANN_expanse_cpu_Crossing.sh`. It stores `rl_history` as a single
  block with `interleaved: True`, so `crossing_metrics` and `crossing_plots` work on it unchanged, except that the
  plots draw no blocks.
- Tests in `tests/test_reinforcement_learning.py`:
  - the layouts;
  - the shortest paths, wrong tile, timeout and walls;
  - the step penalty;
  - which cells show the cue;
  - `epsilon_schedule`;
  - the outcome metrics;
  - the network config.
