# RL scaffolding on `v2dev` (2026-10-29)

Summary of the reinforcement-learning scaffolding built on 2026-10-29, for review. Everything is uncommitted. `EIANN/network.py` is untouched. Outside the new folders, the only edits are the one-line `Eprop` import in `rules/__init__.py` and two pins in `requirements.txt` (`gymnasium==1.0.0`, `minigrid==3.0.0`).

## Design principles

- **Plain `Network`.** RL drives an ordinary `EIANN.Network`; there's no forked network class like the old `network_rl.Q_Network`.
- **Gym API.** Environments follow gymnasium: `reset() -> (obs, info)` and `step(a) -> (obs, r, terminated, truncated, info)`. Stock environments come from MiniGrid.
- **Standard algorithm API.** Algorithms use Stable-Baselines-style methods (`learn` / `predict` / `evaluate`), and any rule-specific logic stays inside the learning rule.
- **One target convention for every rule.** The learning rules get a **masked target**: the network's own output, with only the chosen action's entry replaced by the TD target. Every existing `backward(network, output, target)` then works unchanged.

## Layout

```
EIANN/
  recurrent_network.py            RecurrentNetwork
  rules/eprop.py                  Eprop learning rule
  reinforcement_learning/
    __init__.py                   RLAlgorithm, QLearning, OnlineQLearning, train_continual
    network_updates.py            weight-update helpers
    continual_training.py         train_continual
    algorithms/
      rl_algorithm.py             RLAlgorithm, obs_to_tensor
      q_learning.py               QLearning, OnlineQLearning, masked_target
    environments/
      __init__.py                 make_env, registers 'EIANN/CueTreadmill-v0'
      cue_treadmill.py            CueTreadmillEnv
      task_schedules.py           TaskSchedule, SequentialSchedule, AlternatingSchedule, RandomSchedule
      scheduled_env.py            ScheduledEnv
tests/test_reinforcement_learning.py   21 tests (25 with parametrization)
```

## Files and functions

### `recurrent_network.py`: `RecurrentNetwork(network, persistent_state=False)`

It runs a network over a sequence (e.g. the steps of an episode), carrying population state across steps when `persistent_state` is on.

- `reset()` starts a new sequence.
- `step(sample, no_grad=False)` runs `forward` with `persistent_state=True` after the first step. It resets eligibility traces on a fresh start and calls `learning_rule.forward()` afterwards.
- `peek(sample)` runs a no-grad forward and then restores each population's `state`, `activity`, `prev_activity` and `forward_steps_activity`. It's used for the bootstrap Q(s′), so the learning rules still see step t.
- `run(samples)` steps through a sequence and returns the stacked outputs.
- `detach()` cuts the graph at a step boundary. The caller decides where.

### `reinforcement_learning/network_updates.py`

Helpers that update a network's weights. They copy the update sequence of `Network.train`:

- `compute_learning_rule_updates`: `update_forward_state`, then every `backward`;
- `step_learning_rules`: `step()`, `constrain_weights_and_biases()`, `update()`;
- `apply_learning_rules`: compute, then step;
- `reinit_learning_rules`.

### `algorithms/rl_algorithm.py`: `RLAlgorithm(network, persistent_state, seed)`

- It holds `self.model = RecurrentNetwork(...)` and `self.rng`.
- History: `episode_returns`, `episode_lengths`, `loss_history`.
- `check_env` asserts the output size equals the number of actions.
- `predict` returns the greedy action through `model.step`.
- `evaluate(env, episodes)` runs greedy episodes. On a `ScheduledEnv` it averages over all its envs without advancing the schedule.

### `algorithms/q_learning.py`

- **`masked_target(q, actions, td_targets)`** builds the masked target for one step or a sequence.
- **`QLearning`** takes `gamma`, `epsilon`, `epsilon_decay` (applied per episode), `epsilon_min` and `persistent_state`.
  - It runs the episode one step at a time with fixed weights, with one forward per step.
  - Each step's target is `r + γ·max q_{t+1}.detach()` (just `r` if terminal), with `peek` used for a truncated last step.
  - It then makes one update on the stacked Q-values. With `persistent_state` and `backward_steps == forward_steps`, gradients flow back through the whole lap.
  - It's meant for backprop-based rules.
- **`OnlineQLearning`** adds `accumulate_updates=False`.
  - Each step it runs `step(s_t)` → act → `peek(s_{t+1})` → compute the update → `detach()`.
  - It applies the update every step, or once at the end of the episode when `accumulate_updates=True`.
  - The accumulate option only defers rules that apply their update in `step()`, such as `Eprop`; `Backprop` steps inside `backward`.

### `continual_training.py`: `train_continual(algo, schedule, episodes_per_phase, eval_episodes=1)`

For each phase the schedule yields, it trains, evaluates every task, and calls `network.update_CL_states()`. It returns `(reward_matrix[phase, task], phase_tasks)`.

### `environments/`

- **`make_env(env_id, seed)`** calls `gymnasium.make`, and flattens MiniGrid observations (`ImgObsWrapper` + `FlattenObservation` → 147 inputs).
- **`CueTreadmillEnv`** is one fixed track.
  - Parameters: `length`, `cue_position`, `cue_length`, `num_cues`, `cue`, `reward_positions` (the last cell of each zone), `reward_length`, `rewarded_zone` and `reward_values=(0, 1, -0.1)` for ignore / correct lick / wrong lick.
  - Actions are `Discrete(2)`, ignore or lick, and the agent advances one cell per step.
  - The observation is a one-hot of `[plain, cue_k, zone_z]`, and every zone is visible on every track.
  - One lap per episode; the terminal observation is all zeros. `render_mode='ansi'` draws the track.
- **Task schedules** yield task indices:
  - `TaskSchedule(envs, order=None)`: an explicit order, or subclass with a generator `__iter__`;
  - `SequentialSchedule`: 0…n−1;
  - `AlternatingSchedule(num_cycles)`: 0…n−1, repeated;
  - `RandomSchedule(num_phases=None, seed)`: random indices, endless when `num_phases` is `None`.
- **`ScheduledEnv(schedule)`** is an env that switches to the schedule's next env on every `reset()`. It restarts finite schedules and reports `info['task']`.

How the schedules are used:

```python
tracks = [CueTreadmillEnv(cue=0, rewarded_zone=0), CueTreadmillEnv(cue=1, rewarded_zone=1)]
reversed_tracks = [CueTreadmillEnv(cue=0, rewarded_zone=1), CueTreadmillEnv(cue=1, rewarded_zone=0)]

algo.learn(ScheduledEnv(RandomSchedule(tracks, seed=0)), episodes=2000)   # interleaved, no CL state changes
train_continual(algo, SequentialSchedule([ScheduledEnv(RandomSchedule(tracks, seed=0)),
                                          ScheduledEnv(RandomSchedule(reversed_tracks, seed=1))]),
                episodes_per_phase=2000)                                  # mapping A, then its reversal
```

### `rules/eprop.py`: `Eprop` (symmetric e-prop, Bellec et al. 2020)

- Dynamics: s(t) = (1−1/τ)·s(t−1) + (b + Σ W·x(t))/τ.
- Trace: e(t) = (1−1/τ)·e(t−1) + x(t)/τ, stored as a vector over presynaptic units.
- Learning signal: L(t) = ∂L(t)/∂s(t) within the current step, from `torch.autograd.grad`.
- Gradient: Σ_t L(t)·e(t)ᵀ.
- Methods:
  - `forward()` updates the trace (called by `RecurrentNetwork.step`);
  - `backward()` adds L·eᵀ to `accumulated_grad`;
  - `step()` sets `weight.grad` and calls the network optimizer for that weight, so any torch optimizer works;
  - `reset_eligibility_trace()` and `reinit()` clear the state.
- Requirements: `forward_steps == 1`, somatic projections only, one step's output per `backward` (it raises with offline `QLearning`), and all trainable projections must be `Eprop`.

## Changes vs the old `reinforcement_learning` branch

- **No forked network.** `persistent_state` (already on `v2dev`) replaces `forward(reinit=...)`.
- **`peek` instead of a delayed update.** The old loop bootstrapped from the next step's output and then updated step t−1 while the network held step t's activities. That only worked for backprop.
- **E-prop leak fix.** The trace's leak `(1−1/τ)` was only applied when the post population had a self-connection; it now always applies. This bug was invisible at τ=1.
- **E-prop simplifications.** The autapse mask is dropped (use `no_autapses`), the trace is a vector, and the e-prop-specific calls moved out of the training loop into the rule.
- **Bug avoided:** the old `train_online` called `optimizer.step()` again at the end of the episode, so with `Backprop` the last gradient was applied twice.

## What was verified

- **Old vs new e-prop, bit for bit.** Same network, episodes and actions, with `accumulate_updates=True`: identical weights at τ=1 when the loss is summed over actions. Two expected differences:
  - With the default `reduction='mean'`, every new update is 0.5× the old one (1/number of actions). This applies to all rules, not only e-prop.
  - At τ=2 the Output projection differs because of the leak fix; the H1 projections still match exactly.
- **E-prop math.**
  - On one leaky layer (τ=2), the e-prop update equals the exact backprop-through-time gradient, both per step and accumulated.
  - At τ=1, e-prop gives the same weights as `Backprop` with both SGD and Adam.
- **Mechanics.**
  - `peek` leaves every population unchanged.
  - Persistent state carries the cue to the reward cell.
  - `detach()` cuts the graph.
  - The masked target updates only the chosen action's output weights.
  - The treadmill's observations and rewards are correct, and it passes gymnasium's `check_env`.

## Learning results (scratch scripts, not tests)

- **MiniGrid Empty-5x5:** only works with the 3 movement actions; the 4 do-nothing actions make the greedy agent get stuck. `QLearning` also needs observations ÷10 (lr 0.1), otherwise it diverges. With those changes both variants reach the optimal greedy return of 0.955.
- **1-cue treadmill:** `OnlineQLearning` reaches 1.0 with `reward_length=1`. With `reward_length=2` it gets 0.9, because the zone's two cells look identical.
- **2-cue treadmill** (`reward_length=1`; 0.9 is the most a memoryless agent can get):
  - offline `QLearning` + `Backprop` + persistent state: 1.0 on seed 0, then 0.95 / 0.85 / 0.95 on seeds 1–3, so not yet reliable;
  - online e-prop: at most 0.9 for τ ∈ {1, 2}, and τ=4 collapses to never licking.

## Open decisions

1. **Loss scale.** Use `criterion_kwargs: {reduction: sum}` in RL configs to match the old branch's learning rates, or keep the mean and rescale?
2. **E-prop default.** Should `accumulate_updates` default to `True`, to match the derivation and the old behaviour?
3. **MiniGrid observations.** Add an action-restriction wrapper and observation scaling (or one-hot) to `make_env`?
4. **Export.** Export `RecurrentNetwork` from `EIANN/__init__.py`?
5. **Continual repeats.** `update_CL_states()` runs after every phase, even when a task comes back. That's fine for class-incremental rules, but task-incremental rules would need the task index.
6. **Regression test.** Add the old-vs-new e-prop comparison as a test? It's currently a scratch script outside the repo, which won't survive a reboot.
7. **Not started:** porting the meta-RL wrapper, DNQL and `plot_rl.py`, the nested optimize module, and updating `.claude/branches/v2dev.md` and `CLAUDE.md`.
