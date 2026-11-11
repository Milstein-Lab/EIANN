# BTSP with decaying traces (BTSP_Continuous): design decisions and findings

2026-11-09, branch `v2dev`. Nothing here is committed yet.

## Why

The BTSP rules in `rules/btsp.py` have no eligibility traces (ET) or instructive signals (IS) that persist and decay over time:

- `BTSP_20`, `BTSP_CL` and their ELR/SI/BP variants pair the current sample's presynaptic activity with the current sample's plateau only.
- `BTSP_19` adds one hard-coded cross-sample term, scaled by `temporal_discount`.

RL tasks such as 2ACDC need credit to span several steps: the cue is at cells 8–10 and the reward 5–9 steps later. Memory tasks will too. The new rules make ET and IS decaying state variables, which gives BTSP's bidirectional plasticity window: a plateau potentiates synapses that were active somewhat before it or somewhat after it.

## The rule

| Decision | Choice |
|---|---|
| File | `rules/btsp_continuous.py`, kept separate from `btsp.py` |
| Classes | `BTSPTraceMixin` holds the trace logic. Concrete classes: `BTSP_Continuous` (dendritic plateaus and nudging, from `BTSP_CL`) and `BTSP_Continuous_BP` (backprop plateaus, −dL/da, from `BTSP_BP_CL`) |
| Clock | Since 2026-11-10 the 2ACDC configs use `trace_clock: substep`: one tick per integration substep (`forward_steps` ticks per `RecurrentNetwork.step`), each lasting `dt/forward_steps`, so τ is still in env steps and doesn't depend on `forward_steps`. `BTSP_Continuous` adds `x/forward_steps` to ET per tick; the normalized `BTSP_Eprop` trace needs no rescaling. Plateaus still reach IS once per env step. With the default `trace_clock: step`, one tick per env step as before. Decay is `exp(−dt/τ)` with `dt=1` by default; set `dt` to use seconds. τ = 0 means no memory. See `RecurrentNetwork` and `BTSPTraceMixin.set_ticks_per_step` |
| Trace form | Leaky integrators clamped to [0, 1]: `ET ← clamp(ET·e^(−dt/τ_ET) + clamp(x, 0, 1))` and `IS ← clamp(IS·e^(−dt/τ_IS) + max(plateau, 0))` |
| Positive update | BTSP_19 form on the trace overlap `S = IS ⊗ ET`: `ΔW = (w_max − W)·S − W·dep_ratio·q_dep(S)`, every step. The plateau amplitude is folded into IS |
| Negative update | As in BTSP_19/BTSP_20: linear, instantaneous, no trace. `ΔW += min(plateau, 0) ⊗ clamp(x_now, 0, 1)` |
| Depression threshold | Defaults `dep_th=0.2`, `dep_width=0.1`, instead of the inherited 0.01/0.01 (see below) |
| Scope | RL only, through `RecurrentNetwork` with `OnlineQLearning`/`OnlineDPPQL`. `Network.train` is not supported |

### Mechanics

- **Hooks, as in Eprop:**
  - `forward()` decays the traces and adds the presynaptic activity to ET. It runs after each step and reads forward activity, before any nudging.
  - `backward()` computes plateaus with the parent rule's backward, adds the positive ones to IS, and accumulates this step's ΔW.
  - `step()` applies the accumulated ΔW, so `accumulate_updates=True` works.
- **Trace reset:** `recurrent_network.py` now calls a new `reset_traces()` hook at the start of each episode, whatever `persistent_state` is. Eprop's `reset_eligibility_trace()` fires every step when `persistent_state=False`, which would wipe BTSP traces.
- **One `backward` per concrete class:** `Network` deduplicates backward methods by function identity, so each concrete class defines its own thin `backward`. If both shared one inherited from the mixin, only one would ever be registered.
- **Guards:** the rule raises if it's run without `RecurrentNetwork.step`, or with episode-batched `QLearning`, which passes stacked outputs.
- **Known caveat:** `BTSP_Continuous` nudges somatic state during backward. With `persistent_state=True`, the nudged state carries into the next env step. `_BP` doesn't nudge.
- **Small difference from BTSP_19:** the negative term uses presynaptic activity read right after the forward pass. BTSP_19 reads it in `step()`, after nudging.

### Why dep_th = 0.2

With the default 0.01, `q_dep` saturates once the overlap reaches about 0.02. Every step of the trace tails then adds nearly full depression, so a single pairing nets depression at W = 0.5. A sweep of the closed-form ΔW over pairing delay (both orders, τ_ET/τ_IS = 2/1 and 5/2) showed that `dep_th` ≈ 0.2–0.3 with `dep_width` ≈ 0.1 gives net potentiation. It is largest at coincidence, decays on both sides, and the pre-before-plateau side is wider. Widths of 0.01 made the equilibrium weight jump erratically between delays. Adding a `q_pot` sigmoid was considered and not needed.

### Negative plateaus: history

1. **First version:** a separate decaying `IS_neg` trace (same τ_IS), giving depression the same window as potentiation.
2. **Reverted, by choice, to the original linear same-step update** (BTSP_19/BTSP_20). BTSP_19 was already asymmetric in this way: its window applied only to potentiation.
3. **Consequence (measured):** the window now scales potentiation but not depression. At τ_IS = 1 a positive plateau acts with about 1/(1 − e^(−1)) ≈ 1.6× the weight of a negative one, against about 1.07× in BTSP_19 with the old treadmill `temporal_discount` of 0.035. τ_IS sets this balance, and its bounds [0.1, 10] reach the near-symmetric regime.

### Tests

`tests/test_btsp_continuous.py`:

- the weight change matches a hand-computed sum of IS·ET over steps, for pre-before-plateau and plateau-before-pre, with `dep_ratio` 0 and 1;
- the window decays with delay on both sides;
- τ = 0 behaves as no memory;
- a negative plateau gives `plateau × pre` on the same step and 0 across steps;
- traces last a whole sequence with `persistent_state=False` and reset with a new one;
- updates apply only in `step()`;
- the guards raise;
- a treadmill smoke test for both variants, per-step and accumulated.

The smoke-test network is feedforward: 16 purely excitatory recurrent units without inhibition made activity diverge.

## 2ACDC model

| Decision | Choice |
|---|---|
| Names | `20261109_EIANN_2_hidden_2ACDC_BTSP_Continuous_config.yaml` (network) and `20261109_nested_optimize_EIANN_2_hidden_2ACDC_BTSP_Continuous_config.yaml` (optimize), model key `BTSP_Continuous`. No `_relu_SGD` suffix, because BTSP doesn't use SGD. `test_2ACDC_network_config` now takes the full model name |
| Architecture | The old `reinforcement_learning` branch's treadmill BTSP topology: E, SomaI and DendI in each hidden layer; top-down E→E dendritic inputs as fixed transposed copies (`clone_weight`); `DendriticLoss_6` and `Hebb_WeightNorm` for DendI. Changes: 50 E units per hidden layer (was 500; 100 until 2026-11-09, now matching `bpDale_relu_SGD`), `H_I_size` 10, and **a learned recurrent H1.E→H1.E BTSP_Continuous projection** (`no_autapses`) to hold the cue, since the old model had no E→E recurrence |
| Learned projections | `BTSP_Continuous` on Input→H1, H1→H1, H1→H2 and H2→Output |
| Dynamics | `forward_steps: 5` (15 until 2026-11-09, now matching `bpDale_relu_SGD`), `backward_steps: 0`, `tau` 3, `persistent_state: True`. No `optimizer` key, since BTSP uses none |
| Algorithm | `OnlineQLearning` with `accumulate_updates: False`: weights change every step |
| Search | Full set, 44 params: the old branch's set minus `temporal_discount`, plus `H1_E_H1_E_max_weight_scale`, `H1_E_H1_E_init_weight_factor`, `tau_ET`, `tau_IS`, `dep_th`, `dep_width` and `tau` |
| Update function | `update_EIANN_config_2_hidden_2ACDC_BTSP_Continuous` in `optimize/network_rl_config_updates.py`. It is a cleaned-up port: the old branch's function didn't match its own YAML |
| x0 | Shared params from the old branch's optimize config, which was never optimized to a result. New recurrence: max scale 2.0, init factor 0.1, my guess. Traces: τ_ET 3, τ_IS 1, dep_th 0.2, dep_width 0.1. epsilon, epsilon_decay and gamma 0.15 from `eprop_relu_SGD`. BTSP learning rates were not doubled: that rule applies to loss-based rules, and BTSP plateaus come from the raw output error |
| Rewards | `(ignore, correct, wrong) = (0.1, 1.1, 0.)`, the 2ACDC values shifted by +0.1 (see below) |

### Reward shift

BTSP E weights are non-negative, so Q ≥ 0. Only relative Q values matter for the policy, but with the original rewards the target for a wrong lick, −0.1 + γ·max Q′, is negative whenever γ·max Q′ < 0.1. At γ = 0.15 that is almost everywhere. Q(lick) and Q(ignore) then both collapse onto 0, tie, and the policy becomes "always lick" or "never lick".

Shifting every reward by +0.1 makes all targets ≥ 0. Every lap is exactly 23 steps, so the shift adds the same amount to both actions at every cell and the optimal policy is unchanged.

`nested_optimize_EIANN_2ACDC.py` now reports `reward` relative to a lap with no licks: it subtracts `length × reward_values[0]`, and the objective's max is `reward_values[1] − reward_values[0]`. Shifted runs therefore report in the usual units (1.0 best, 0.9 = cue not used). Existing models have an ignore reward of 0, so nothing changes for them. `plot_2ACDC_summary` still plots raw, shifted validation rewards.

## Findings so far

All rewards are greedy laps on both tracks after 1000 episodes, in original units. With no memory, a feedforward net can reach at most **0.7**: both cells of a reward zone look the same, so it licks the whole of both zones.

1. **x0, before the reward shift:** reward 0.0 on one seed, about 220 s. Q started at about 7–12, against rewards of about 1. All errors were negative, H2 was depressed until silent, Q went to 0, and the policy was "ignore".
2. **10 random first-generation points** (nested's `RelativeBoundedStep`, step size 1), before the shift: none above 0. The runs settled on "always lick" (−1.2) or "never lick" (0), and one diverged.
3. **Sanity comparisons** (3 seeds; feedforward = Input→100 relu→2 linear):

| Condition | `IS_neg` trace | Linear negative | Linear negative + reward shift |
|---|---|---|---|
| Backprop, feedforward (reference) | 0.7 ×3 | 0.7 ×3 | 0.7 ×3 |
| BTSP_BP, feedforward, no traces, lr 0.1 | 0.4 ×3 | 0.4 ×3 | **0.7 ×3** |
| BTSP_BP, feedforward, no traces, lr 0.01 | −1.2, 0.4, −1.2 | same | **0.7, 0.7, 0.55** |
| BTSP_BP, feedforward, traces, lr 0.01 | −1.2, 0.4, −1.2 | −1.2 ×3 | −1.2 ×3 |
| BTSP_BP, feedforward, traces, lr 0.1 | −1.2, 0.0, −1.2 | −1.2 ×3 | 0.55, −1.05, −1.2 |
| BTSP_BP, feedforward, traces, unbounded weights | 0.55, 0.7, 0.55 | −1.2 ×3 | −1.05, −1.2, −1.2 |
| BTSP_BP, E/I network | −1.5, 0.3, 0.2 | −0.85, −1.1, −1.1 | 0.7, 0.3, −1.1 |
| Dendritic BTSP, E/I network | — | 0.1, 0.4, −1.1 | −1.2, −1.1, −1.1 |

The traced runs used τ_ET = 3 and τ_IS = 1.

What this shows:

- **The Q floor is real, and the reward shift fixes it.** With the shift, the untraced BTSP rule matches backprop.
- **Traces with the linear negative update tip toward potentiation,** and the policy becomes "always lick". This persists after the shift. It is the asymmetry described above.
- **Credit assignment can't be judged yet.** In the E/I network, backprop and dendritic plateaus both suffer from the asymmetry, and with 3 seeds the per-seed results vary too much to compare them.

## E-prop traces + BTSP updates: `BTSP_Eprop` / `BTSP_Eprop_BP`

Added to `rules/btsp_continuous.py` to merge the two approaches: an e-prop-style eligibility trace carries presynaptic activity across steps, and BTSP sets the weight update.

| Decision | Choice |
|---|---|
| Classes | `BTSPEpropMixin(BTSPTraceMixin)`, with `BTSP_Eprop` (dendritic plateaus, from `BTSP_CL`) and `BTSP_Eprop_BP` (−dL/da, from `BTSP_BP_CL`; for relu units with `relu_gate` this is e-prop's −dL/ds). `BTSP_Continuous*` are unchanged |
| Trace | e-prop's normalized low-pass with a free time constant, one tick per network step (or per substep with `trace_clock: substep`): `e ← d·e + (1 − d)·clamp(x, 0, 1)`, `d = exp(−dt/τ_ET)`. It has unit DC gain, so a single pulse leaves 1 − d (0.28 at τ_ET = 3) |
| Learning signal | `tau_IS` (default 0). At 0, the current step's plateau only (e-prop's L_t). Above 0, a decaying instructive signal `IS ← clamp(IS·e^(−dt/τ_IS) + plateau⁺, 0, 1)`, as in BTSP |
| Negative plateaus | Direct linear update on the step of the negative plateau only, paired with the e-prop trace: `plateau⁻ ⊗ e`. No negative trace. A `neg_IS` option (a matching `IS_neg` trace) was tested (see the table below) and removed by choice |
| Update | `S = IS ⊗ e`; `ΔW = (w_max − W)·S − W·dep_ratio·q_dep(S) + plateau⁻ ⊗ e` |
| Window | τ_IS = 0: causal only, for both signs (pre before plateau; BTSP's plateau-before-pre side is lost). τ_IS > 0: potentiation becomes bidirectional, as in BTSP; depression stays causal. τ_ET = τ_IS = 0 gives the same-step update |

Tests cover:
- the closed-form causal window at τ_IS = 0, and no credit for presynaptic activity after the plateau;
- the negative term scaling with the trace, and τ_ET = 0;
- the bidirectional window at τ_IS = 1, both orders, matching a closed-form sum and shrinking with the gap;
- a negative plateau before a pulse having no effect, even at τ_IS = 1 (no negative trace);
- the treadmill smoke tests and the `QLearning` guard.

**Sanity comparison** (shifted rewards, 3 seeds, 1000 episodes, τ_ET = 3; 0.7 is the ceiling without the cue):

| Condition | `BTSP_Continuous*` (τ_IS = 1) | `BTSP_Eprop*`, τ_IS = 0 | τ_IS = 1, with negative trace (removed) | **τ_IS = 1, current rule** |
|---|---|---|---|---|
| BP plateaus, feedforward, lr 0.01 | −1.2, −1.2, −1.2 | **0.7, 0.7, 0.55** | 0.7, 0.7, 0.55 | 0.7, 0.7, 0.55 |
| BP plateaus, feedforward, lr 0.1 | 0.55, −1.05, −1.2 | **0.7, 0.7, 0.55** | — | — |
| BP plateaus, E/I network | 0.7, 0.3, −1.1 | **0.7, 0.7, 0.7** | 0.7, 0.7, 0.7 | — |
| Dendritic plateaus, E/I network | −1.2, −1.1, −1.1 | **0.7, 0.55, 0.6** | 0.7, 0.55, 0.55 | 0.7, **0.0**, 0.7 |

The reference rows are backprop on the feedforward net (0.7 ×3) and `BTSP_Continuous_BP` with no traces (0.7 ×3).

- **Pairing both signs with one trace removes the over-licking.** Every run licks the reward zones, and the dendritic E/I model works for the first time.
- **The decaying IS is neutral on this task.** Adding BTSP's plateau-before-pre side (τ_IS = 1) changes almost nothing.
  - **Feedforward, with or without the negative trace:** identical rewards, with the same licks per seed.
  - **E/I with BP plateaus:** identical.
  - **Dendritic E/I with the negative trace:** one seed licks a cue cell (0.55 instead of 0.6).
  - **Dendritic E/I, current rule (no negative trace):** one seed stopped licking altogether (0.0), the only failure among the e-prop runs.
- **The positive-only IS doesn't over-lick, unlike `BTSP_Continuous`.** A likely reason is that the e-prop trace is normalized, at most 1 − d per pulse, so the overlap stays smaller than with `BTSP_Continuous`'s accumulating ET. This hasn't been tested.
- **No run uses the cue yet:** all are at or below 0.7, even with the learned H1 recurrence in the E/I network.
- **Cost:** E/I runs take about 140–170 s per seed with BP plateaus and about 240–270 s with dendritic plateaus.

### 2ACDC model: `BTSP_Eprop`

- **Files:** network `optimize/network_config/2ACDC/20261109_EIANN_2_hidden_2ACDC_BTSP_Eprop_config.yaml`, optimize config `optimize/optimize_config/2ACDC/20261109_nested_optimize_EIANN_2_hidden_2ACDC_BTSP_Eprop_config.yaml`, model key `BTSP_Eprop` in `20261102_2ACDC_params.yaml`.
- **Same as `BTSP_Continuous` except the rule:** the E/SomaI/DendI network with H1 recurrence and `BTSP_Eprop` (dendritic plateaus) on Input→H1, H1→H1, H1→H2 and H2→Output. Same `OnlineQLearning` (per-step), shifted rewards, 44-param search and x0.
- **Update function reused:** `update_EIANN_config_2_hidden_2ACDC_BTSP_Continuous` sets `tau_ET`, `tau_IS`, `dep_th`, `dep_width` and the rest on those projections whatever the rule.
- **`tau_IS` is searched in [0.01, 10]** (x0 1). At 0.01 the decay is e^(−100), so the low end is the causal rule (τ_IS = 0) and the search can choose between causal and bidirectional potentiation.
- **At x0:**
  - **Verifier:** static, params (all 44 reach the config), build and one tiny optimize generation pass. Learn gives **0.625**: 1 seed, mean over the last 200 episodes, original units. Log: `/private/tmp/claude-501/-Users-rahul-Documents-Milstein-lab-EIANN/8b0e69a6-468c-427e-9947-b456f115dc82/scratchpad/verify_eprop.log`.
  - **Sanity runs** of this exact rule, outside nested: 0.7, 0.0, 0.7.

## Open items

- Verifier on the shifted-reward config: static, params, build and optimize pass. Learn gives **−0.74** (1 seed, x0, original units): many wrong licks, consistent with the traced runs tipping toward "always lick". Log: `/private/tmp/claude-501/-Users-rahul-Documents-Milstein-lab-EIANN/8b0e69a6-468c-427e-9947-b456f115dc82/scratchpad/verify_shift.log`.
- The potentiation/depression imbalance from the traces. Options: rely on the search for small τ_IS; scale the negative term (`neg_ratio`, or 1/(1 − e^(−dt/τ_IS))); restore `IS_neg` as an option.
- Cue memory hasn't been tested at all: everything so far is at or below the 0.7 feedforward ceiling.
- Cost: about 220–300 s per seed locally for the E/I network at 15 forward steps and 100 E units (the results above use that architecture).
- Cluster commands, not run:
  `bash optimize/jobscripts/optimize_EIANN_expanse_cpu_2ACDC.sh optimize/optimize_config/2ACDC/20261109_nested_optimize_EIANN_2_hidden_2ACDC_BTSP_Eprop_config.yaml BTSP_Eprop`
  `bash optimize/jobscripts/optimize_EIANN_expanse_cpu_2ACDC.sh optimize/optimize_config/2ACDC/20261109_nested_optimize_EIANN_2_hidden_2ACDC_BTSP_Continuous_config.yaml BTSP_Continuous`
- Sanity scripts and raw results (scratchpad, not in the repo): `/private/tmp/claude-501/-Users-rahul-Documents-Milstein-lab-EIANN/8b0e69a6-468c-427e-9947-b456f115dc82/scratchpad/btsp_sanity/` (`sanity.py`; `results_ISneg.jsonl`, `results.jsonl`, `results_shift.jsonl`).

## Files

- New: `rules/btsp_continuous.py`, `tests/test_btsp_continuous.py`, the 2ACDC network and optimize configs for `BTSP_Continuous` and `BTSP_Eprop`, this note.
- Edited:
  - `rules/__init__.py` (import);
  - `recurrent_network.py` (`reset_traces` hook);
  - `optimize/network_rl_config_updates.py` (update function, `import math`);
  - `optimize/nested_optimize_EIANN_2ACDC.py` (reward relative to a lap without licks);
  - `optimize/optimize_params/2ACDC/20261102_2ACDC_params.yaml` (`BTSP_Continuous` and `BTSP_Eprop` entries);
  - `tests/test_reinforcement_learning.py` (`test_2ACDC_network_config` parametrization).
