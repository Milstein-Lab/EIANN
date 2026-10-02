# Continual-learning rules and the task-incremental setting: design choices, experiments and results

Branch `v2dev`, September–October 2026. This report covers the EWC and SI rules, the task-incremental (TI) setting, the continual-learning (CL) versions of the local rules (DTP, BTSP, BTSP_ELR), and the multi-head output structure. For each, it records what was built, why, how it was checked, and what the checks showed. All paths are relative to `EIANN/` (the package directory). Unless stated otherwise, the numbers come from 2-task split MNIST (task 0 = digits 0–4, task 1 = digits 5–9), network seed 66049.

---

## 1. Summary

| Piece | Where | Status |
|---|---|---|
| `ContinualLearningMixin`: shared task bookkeeping (`task_num`, `update_CL_states`, `task_target`) | `rules/base_classes.py` | done |
| `Backprop_CL`: vanilla backprop that counts tasks and masks the loss to the current task | `rules/backprop.py` | done |
| `Backprop_EWC`: elastic weight consolidation, all logic inside the rule | `rules/backprop.py` | done |
| `Backprop_SI`: synaptic intelligence, all logic inside the rule | `rules/backprop.py` | done |
| `DTP_CL`: CL copy of `BP_like_2E` | `rules/dtp.py` | done |
| `BTSP_CL` (CL copy of `BTSP_20`) and `BTSP_ELR_CL(BTSP_CL)` | `rules/btsp.py` | done |
| `task_incremental` flag: loss/error masking plus within-task evaluation | `optimize/nested_optimize_EIANN_1_hidden_CL_mnist.py`, `utils/network_utils.py` | done |
| `multihead` flag (default on with TI): block-diagonal output E/I, plus a top-down mask during training | `rules/weight_functions.py`, nested CL script | done |
| Debug report (`--debug`): val set size and class composition, class-wise output activity, per-task accuracy, output weight change per unit | `utils/network_utils.report_phase_debug` | done |
| TI and CL metrics: `--task-incremental` | `scripts/get_cl_metrics.py` | done |

**Key conclusions**
1. **Masking the loss is not enough for TI in networks whose output layer has E/I recurrence or top-down feedback.** In bpDale, the masked loss still sends gradient into the inactive heads through the shared output I cells. In the local-rule networks, inactive heads drive spurious hidden-layer errors through the top-down projection. A true multi-head structure needs three masks together:
   - the loss/error mask;
   - a block-diagonal output E/I;
   - a mask on top-down projections from inactive heads.
2. **With these masks, information flows only through the active head.** An invariance test confirms it for all six CL rules: scrambling an inactive head leaves every other tensor bitwise identical. A negative control confirms the test detects leaks when the masks are removed.
3. **The top-down mask is what lets BTSP and DTP learn in the TI setting** at their existing (class-incremental) x0. Without it, both fail (H2 collapse, DTP runaway). BTSP_ELR still doesn't learn at its x0; this is treated as a hyperparameter issue.

---

## 2. Framework constraints that shaped the design

- **`backward` is a classmethod and runs once per sample for the whole network.**
  - `Population.append_projection` registers each rule's `backward` once (deduplicated by `backward.__func__`), and `Network.train` calls each registered method once per sample with `(network, output, target)`. It is never passed a projection.
  - Rules therefore find their projections with `projection.learning_rule.__class__ == cls`.
  - Each subclass that overrides `backward` is registered separately. Two rule classes that share one `backward` function are deduplicated, which is why `BTSP_CL` and `BTSP_ELR_CL` must not be mixed in one network.
- **`step()` and `update()` are instance methods, called once per projection.**
  - `update()` runs after `optimizer.step()` and after `constrain_weights_and_biases()`, so it sees the real weight change including clamping.
- **`update_CL_states()` is an instance method, called per projection between tasks** by `network.update_CL_states()`, with no data.
  - Work that spans the whole network (EWC's Fisher replay) uses a first-caller guard.
- **`backward` never receives the input.** EWC reads it from `network.input_pop.activity`.
- **Weights are initialized after rule `__init__`** (`Network.init_weights_and_biases`), so start-of-task weights are captured lazily on the first `backward` (SI).
- **One rule for all learned projections.** Any other backprop-type `backward` (e.g. `BackpropBias` → `Backprop.backward`) would step the shared optimizer a second time. None of the configs below have learned biases.
- **Weight constraints run in two passes** (`Network.constrain_weights_and_biases`):
  1. clamp to `weight_bounds`, then every non-`clone_weight` constraint, layer by layer;
  2. every `clone_weight` constraint.

  `clone_weight` overwrites its target every step. The constraints are **not idempotent** in configs that combine `normalize_weight` with weight bounds: each extra pass shifts weights slightly (section 9.4).
- **The soma's forward update ignores `dend` projections** (`Population.forward`). Top-down projections to dendrites therefore affect only learning (dendritic states, plateaus, DendI training), never inference.
- **`network.py` was not modified** for any of this work.

---

## 3. EWC (`Backprop_EWC`)

Reference: Kirkpatrick et al. 2017, arXiv:1612.00796. The pre-existing `Backprop_EWC` relied on attributes set from outside (`network.phase1_params`, `diag_fisher`, `ewc_lambda`), anchored only one task, and hard-coded a `0.3*` factor. It was rewritten so that all logic lives in the rule.

### 3.1 Design
- **Collecting samples.** Each step, `backward` caches `(network.input_pop.activity, target)` in a reservoir sample of at most `fisher_num_samples` (default 1000; about 3 MB). The buffer lives on `network.ewc_buffer`.
- **Fisher at the end of each task.** The first EWC projection to have `update_CL_states()` called replays the cached samples one at a time at the end-of-task weights. It computes the squared per-sample gradient of the (task) loss with `torch.autograd.grad`, which leaves `.grad` and the optimizer untouched, and averages the result. This is the empirical diagonal Fisher.
- **Storage.** Each projection appends its Fisher and an anchor copy of its weights; one pair per task, kept for every past task.
- **Loss:** `task_loss + Σ_p ewc_lambda_p · Σ_t (F_pt · (W_p − W*_pt)²).sum()`.
- **Pickling.** The reservoir's RNG is stored as `generator.get_state()`, because a `torch.Generator` can't be pickled. **A bug found later:** the first version stored the generator object, so `--export` crashed. Fixed; see 9.3.
- **Hyperparameter.** `ewc_lambda`: bounds [0.1, 1e4], x0 = 100.

### 3.2 Checks
- **Synthetic 2-task data (random inputs):**
  - after phase 0 there is 1 Fisher and 1 anchor per projection, and the reservoir is capped at 100 and resets;
  - the penalty is 0 during task 0;
  - Fisher-weighted drift on the output projection during task 1 fell from 2.3e-6 at λ = 0 to 8e-9 at λ = 3000.
- **Stability.** λ = 1e6 diverged. SGD is stable roughly when `lr · 2λ · max(F) < 2`; max F was about 6e-4 at lr 0.2.
- **MNIST verifier** (2000 + 1000 steps, class-incremental): van_bp kept 34.0% of task 0, bpDale 5.8% at x0.

---

## 4. SI (`Backprop_SI`)

Reference: Zenke, Poole & Ganguli 2017, arXiv:1703.04200, checked against the reference code (`ganguli-lab/pathint`).

### 4.1 Design and how it matches the paper and code
- **Per-step importance.** `ω_k += −g_k · ΔW_k`, where:
  - `g` is the gradient of the **unregularized** task loss;
  - `ΔW` is the actual weight change. It is measured in `update()`, after the optimizer step and clamping.
- **Why the unregularized gradient.** In `pathint/optimizers.py`, `unreg_grads = tf.gradients(self.initial_loss, params)`, where `initial_loss` excludes the regularizer. The optimizer steps on the full loss, and `grads2 -= unreg_grads * deltas` (`pathint/protocols.py`). Using the full-loss gradient would make importance feed itself across tasks: the penalty pulling weights back would count as importance.
- **End of task.** `Ω = relu(Ω + ω / (ΔW_task² + ξ))`.
  - The paper's Eq. 5 is a sum over past tasks; keeping a running sum is equivalent.
  - The `relu` follows pathint (`tf.nn.relu` in the `omega` task update).
  - Then the anchor `W̃` is set to the current weights and ω is reset.
- **Anchor.** A single anchor: the weights at the end of the previous task (paper, text after Eq. 4: "θ̃_k = θ_k(t^{µ−1})").
- **Loss:** `task_loss + Σ_p si_lambda_p · (Ω_p (W_p − W̃_p)²).sum()`.
- **Hyperparameters.**
  - `si_lambda`: bounds [1e-4, 10], x0 = 0.1. Ω is much larger than the EWC Fisher (max about 27 on synthetic data), so the SGD stability limit on λ is about 1/(lr·Ω_max).
  - `si_xi`: fixed at 1e-3, the paper's split MNIST value.

### 4.2 Checks
- **ω equals the analytic value.** For plain SGD with no clamping, ω matched Σ lr·g² to a relative difference of about 3e-7.
- **State updates.** Ω is set, ω reset and the anchor equal to the weights after `update_CL_states`.
- **Clamp check.** A hand-built case with negative ω gave Ω = [0.0, 0.4995].
- **MNIST verifier, class-incremental:** van_bp kept 47.1% of task 0, bpDale 40.4%.

---

## 5. The task-incremental (TI) setting

### 5.1 Training: masking the loss or error
- **Shared bookkeeping.** `ContinualLearningMixin` holds `task_incremental`, `task_classes` and `task_num`.
- **Backprop rules:** `task_loss = criterion(output[..., task_classes[task_num]], target[..., same])`.
- **Local rules:** `task_target(network, target)` replaces off-task target entries with the unit's own current activity. That makes `clamp(target − activity)`, the only place these rules use the target, exactly 0 off-task (section 7).
- **Where the task split comes from.** `config_worker` exposes its `labels_in_tasks` as `context.labels_in_tasks`. In `compute_features`, `get_task_incremental_projection_config` injects `task_incremental` and `task_classes` into every learned projection whose rule subclasses `ContinualLearningMixin`.
  - Rules in `TARGET_FREE_RULES` (`DendriticLoss_6`, `Hebb_WeightNorm`) are left alone: they never see the target.
  - Any other learned rule (e.g. plain `Backprop` or `BTSP_20`) raises an error.
- **No update functions were edited.** `update_EIANN_config_2_hidden_van_bp_relu_SGD_G` alone is shared by about 20 configs, so injecting at the script level avoids touching them.

### 5.2 Loss scale: averaging over the task's outputs
With MSE averaged over output units, a masked loss averages over the k task units instead of 10. That multiplies the effective learning rate by 10/k: 2× for 2 tasks, 5× for 5 tasks.
- **Standard practice.** Multi-head TI computes the loss only on the active head (SI paper, split MNIST: "the categorical cross entropy loss at the readout layer was computed only for the digits present in the current task"). With cross-entropy the per-unit normalization question doesn't arise, and the per-head mean is the MSE analogue. **Kept as is.**
- **Effect at x0.** Task-0 within-task accuracy after 1000 steps:

  | model | class-incremental loss | TI, per-head mean | TI, masked but divided by 10 |
  |---|---|---|---|
  | van_bp | 95.3% | 78.6% | 96.9% |
  | bpDale | 95.1% | 96.0% | 94.8% |

- **Decision.** Keep the per-head mean and scale the TI configs' **x0 learning rates by k/10** (×0.5 for 2 tasks, ×0.2 for 5), so each TI model starts at the effective step size its class-incremental x0 was tuned for.
  - Over 3 seeds, scaled and unscaled x0 had no consistent winner in the 2-task setting. van_bp EWC: 96.7 / 89.8 / 96.6 scaled vs 96.1 / 93.4 / 79.1 unscaled. bpDale EWC: 80.2 / 96.1 / 96.2 vs 95.8 / 96.1 / 96.2.
  - The scaling mainly protects the 5-task configs from a 5× effective step.
- **Vanilla TI configs** started from the untuned template x0 (lr 0.5). They were switched to the optimized `van_bp` / `bpDale_G` values, then scaled.
- **Local rules** use a per-unit error, not an averaged loss, so their learning rates are **not** rescaled.

### 5.3 Evaluation
- **Validation stays cumulative.** `cumulative_val_set` (default True) is independent of `task_incremental`: phase i validates on tasks 0..i.
- **Scoring.** With TI, `network.train(..., store_val_output_history=True)` keeps the val outputs. Each val step is rescored with `compute_task_incremental_loss_and_accuracy`: each sample's loss and argmax use only its own task's outputs. The `final`/`best` windowing and objectives are unchanged. The `cumulative_val_set=False` path (`final_loss`/`final_accuracy`) has the same TI scoring.
- **Confirming it on a real run** (`--debug` report, `van_bp_TI` 5-task params from `optimize_params/mnist_CL/20260828_v2dev_mnist_CL_params.yaml`):
  - val set sizes 2,055 → 4,075 → 5,973 → 8,030 → 10,000, covering every class seen so far;
  - in each phase, only the current task's output units changed weight; all others changed by exactly 0;
  - task 0 within-task accuracy stayed at 99.9% across all five phases, while argmax over all outputs fell 100 → 40.9 → 19.1 → 17.2 → 4.6%;
  - after phase 4, a true "0" drives unit 0 to 0.80 vs unit 1 at 0.14 (correct within its task), but also drives units 2, 5 and 8 to about 0.95–1.0.

  High TI accuracy for vanilla backprop is therefore expected: binary within-task choices, and frozen old heads.

---

## 6. Debug instrumentation
`--debug` calls `utils.report_phase_debug` at the end of every phase. It prints:
- the val set size and samples per class;
- the mean output activity per true class;
- per-task accuracy, both within task and with argmax over all outputs;
- the summed |ΔW| of each output unit's incoming weights during the phase.

It saves a figure to `<output_dir>/debug_plots/phase{i}_{seed}[_{label}].png`. `--debug` also enables the script's other debug behaviour (timing, equilibration checks, storing every forward step), so use it for inspection only.

---

## 7. CL versions of the local rules

### 7.1 Design
- **`DTP_CL`** (`rules/dtp.py`): a self-contained copy of `BP_like_2E`, made standalone so later CL-specific changes don't touch `backprop_like.py` (13 existing YAMLs use `BP_like_2E`).
- **`BTSP_CL`** (`rules/btsp.py`): a self-contained copy of `BTSP_20`.
- **`BTSP_ELR_CL(BTSP_CL)`:** adds only the elastic learning rate, i.e. ELR's `__init__` attributes and `step()`.
  - ELR is a continual mechanism, so it needs nothing at task boundaries.
  - Its backward is inherited: with `BTSP_CL`'s defaults (`max_pop_fraction=1`, `stochastic=False`, `relu_gate=True`, `neg_rate_th=None`), `BTSP_20.backward` is exactly `BTSP_ELR_1.backward`.
- **Where the target enters.** In all three rules it enters in exactly one place: the output error `local_loss = clamp(target − output_pop.activity, −1, 1)`, which sets the output plateaus (driving `step()` and ELR's `lr_mod`) and the output nudge `dend_to_soma`. Each `backward` starts by replacing the target with `task_target`.
- **The originals are unchanged.**

### 7.2 Checks
- **Equivalence with TI off.** With x0 applied and 500 MNIST steps, each new rule reproduces its original exactly: loss history, all 23 projections' weights, and ELR's `lr_mod` (`BTSP_CL` ≡ `BTSP_20`, `DTP_CL` ≡ `BP_like_2E`, `BTSP_ELR_CL` ≡ `BTSP_ELR_1`).
- **Masking with TI on.** Over task 0, off-task output units had summed |plateau| = 0 and |dend_to_soma| = 0, and their weight rows changed by exactly 0. ELR's `lr_mod` rows stayed at 1. The reverse held in task 1.
- **ELR row drift (not a masking issue).** In ELR, masked rows still moved slightly (2.7) during task 1. This is identical with the output learning rate set to 0, and the original rule shows it too (0.85). The cause is `normalize_weight` (fixed row sums) combined with clamping to `weight_bounds` (many weights sit at the 0.44 bound), which reshuffles weights within rows every step.

---

## 8. Why masking the loss is not enough: experiments

### 8.1 Local rules failed to learn with the loss mask alone
With the loss/error mask only, task-0 within-task accuracy after 2000 verifier steps stayed near chance: BTSP 25%, DTP 20%, ELR 18%. With TI off, class-incremental BTSP reached 94.7%.

**BTSP trace** (400-step chunks, task 0):
- off-task outputs stayed at about 2.4, versus 0.05 with TI off;
- task outputs were suppressed to about 0.02, and 50% of task units went silent;
- H2 activity fell to 0%, versus 24% with TI off.

**DTP trace:**
- task units were 99% silent by step 400. Initial outputs were far too high, so task units received strong negative errors, and the output ReLU gate blocked recovery;
- off-task outputs ran away from 52 to 274, in a positive loop: off-task top-down → H2 dendritic error → stronger H2 → higher off-task outputs.

**How class-incremental training avoids this.** Non-target units get `−activity` errors, which regulate output and hidden activity. With TI masking that signal is gone, and the off-task heads become unregulated drivers of:
- (a) the shared output inhibition, and
- (b) the top-down dendritic input to H2.

### 8.2 Masking the loss does not make bpDale's output multi-head
- **Path.** bpDale tracks gradients through the last `backward_steps = 3` of 15 forward steps, and recurrent projections read the previous step's activity. So the masked task loss reaches off-task rows through task E(15) ← Output.SomaI(14) ← off-task E(13) ← W[off-task rows].
- **Single sample.** |gradient| on off-task rows was 0.65, versus 27.1 on task rows. With Output.SomaI ← Output.E zeroed, it was exactly 0.
- **Training.** Off-task rows moved by 0.52 over 1000 task-0 steps and 0.37 on the old head during task 1. Off-task activity fell from 0.78 to 0.45: backprop suppresses inactive heads to relieve the shared inhibition. The network learns (about 96%), but it isn't multi-head.

### 8.3 Masking variants tested
Variants, each added on top of the loss mask:
- **V0:** loss mask only.
- **V1:** off-task E → I columns cut during training only.
- **V2:** block-diagonal output E/I.
- **V3:** V2 plus top-down from off-task units masked during training.

Results (2000 + 1000 steps; task-0 within-task accuracy):

| model | V0 | V1 | V2 | V3 (as originally tested) |
|---|---|---|---|---|
| bpDale | 96.2%, old head moved 0.37 in task 1 | 96.2%, moved 0 | 96.2%, moved 0 | n/a (no top-down) |
| BTSP 6L | 34.7% | 21.8% | 53.9% | 41.3% |
| DTP 5J | 20.5% | 20.5% | 20.5% | 20.5% |

- **Turning off the output ReLU gate** didn't help either: BTSP was unchanged and DTP got worse (off-task runaway to about 1000).
- **The V3 test was flawed.** It zeroed the top-down weights once by hand, but those projections use `clone_weight`, which re-copies the transposed forward weights after every training step. The top-down mask was therefore never in effect, and that "V3" was really V2. The correct implementation applies the mask inside `clone_weight` (section 9). With it, BTSP and DTP learn (section 10.2). So the top-down mask is the piece that matters for the local rules, consistent with the runaway mechanism in 8.1.

---

## 9. The multi-head structure (`multihead`, default True with `task_incremental`)

### 9.1 Which units belong to which head
`get_unit_tasks` assigns each output-layer unit to a task:
- **Output E units:** by class, from `labels_in_tasks`.
- **Other output-layer populations** (e.g. Output.SomaI): split into contiguous blocks, `arange(size) · n_tasks // size`. With 10 I cells that's 5 per task (2 tasks) or 2 per task (5 tasks). If the I size isn't divisible by the number of tasks, the blocks are unequal, and with fewer I cells than tasks some heads get none.

### 9.2 The masks
1. **Loss/error mask** (section 5.1). Target information enters only through the active head.
2. **`task_block_mask`** on every projection between output-layer populations (Output.SomaI ← Output.E, Output.E ← Output.SomaI, Output.SomaI ← Output.SomaI): only connections within a head are kept.
   - It applies in training and inference.
   - Each head is then an independent E/I subnetwork on the shared hidden layers, so running all heads at once and scoring within task is exactly multi-head inference.
   - The projection's original constraint (e.g. `no_autapses`) is kept as `base_constraint` and applied first.
3. **Top-down mask** on projections from the output layer to dendrites: columns from units outside the current task are zeroed, so only the active head sends top-down signals during training.
   - For `clone_weight` projections (all current BTSP/DTP/ELR configs), it's an optional `labels_in_tasks` kwarg of `clone_weight`, applied after cloning in the clone pass. Masking from a separate constraint would be overwritten (see 8.3).
   - Other top-down projections get `task_column_mask`. It stashes the masked columns on the projection (`task_column_mask_stash`) and restores them on the next call, so an inactive head's top-down weights come back when its task starts. Zeroing them in place would lose them for good on a fixed projection. While a column is masked it stays frozen at its stashed value, and learning-rule updates to it are discarded. The base constraint sees the full, restored weights. No current config uses this path. It was tested on DTP 5J with a fixed `half_kaiming` top-down projection: the columns are restored exactly at the task switch, the stash survives a dill round trip, and the invariance test passes.
   - The current task is read from the CL rules' `task_num`.
   - Top-down projections to the soma raise an error, because they would affect inference.
   - After `update_CL_states()`, the nested script calls `network.constrain_weights_and_biases()` so the mask switches before the first step of the next task.

**Implementation details.**
- `get_task_incremental_projection_config` returns a **deep copy** of the projection config with the rule kwargs and constraints injected. The network is built and exported from that copy. `context.projection_config` itself is never modified, so wrappers can't stack across calls, and update functions keep writing to the original kwargs.
- The exported optimized YAML records `task_block_mask` and `labels_in_tasks` and rebuilds correctly with `build_EIANN_from_config`.
- `multihead: False` gives loss masking only, for comparison.

### 9.3 Invariance test: does information flow only through the active head?
**Procedure:**
1. Train a network for 100 steps of task 0 and make two exact copies (via dill, so that constraint closures are rebound to each copy's own projections).
2. In one copy, replace the **inactive** head's weights with random values in the original range: its rows of Output.E ← H2.E and its I cells' rows of Output.SomaI ← H2.E. Give both copies the same number of constraint passes.
3. Run one training step on the same sample in both.
4. Compare, bitwise, everything outside the inactive head:
   - all other weights;
   - the hidden populations' plateau, nudge, dendritic state, forward dendritic state and activity;
   - the active head's activity;
   - rule state: EWC, SI's ω/Ω/`unreg_grad`/anchor, ELR's `lr_mod`.

   Repeat in task 1 with the old head scrambled.

**Results:**

| rule | task 0 (head 1 scrambled) | task 1 (head 0 scrambled) | negative control (`multihead: False`) |
|---|---|---|---|
| Backprop_CL (bpDale G) | identical (29 tensors) | identical | leaks: task outputs differ by up to 0.04; H1/H2/output weights differ |
| Backprop_EWC (bpDale) | identical | identical | leaks |
| Backprop_SI (bpDale) | identical (35/41 tensors) | identical | leaks, including SI's ω and `unreg_grad` |
| DTP_CL (5J) | identical (52 tensors) | identical | leaks: top-down drive, H2 dendritic state, hidden weights |
| BTSP_CL (6L) | identical (52) | identical | leaks |
| BTSP_ELR_CL (A) | identical (55) | identical | leaks, including `lr_mod` |

**Pitfalls in the test harness, both fixed before these results:**
- **Unequal constraint passes.** One copy got an extra pass, and because `normalize_weight` combined with bounds isn't idempotent, differences of 1e-9 to 4.6e-2 appeared. With equal passes, everything is bitwise identical.
- **Saturated negative control.** An earlier version made the inactive head much stronger in both copies, which silenced the task outputs through the shared inhibition in the negative control. Both copies' outputs were 0, which hid the leak. Scrambling only one copy, within the original weight range, fixed it.

### 9.4 Constraint behaviour worth knowing
- `normalize_weight` combined with `weight_bounds` clamping is not idempotent, so weights shift slightly with every constraint pass. This affects the ELR configs and any analysis that calls `constrain_weights_and_biases()` extra times.
- `clone_weight` overwrites its target every step, so any manual change to a cloned projection is lost unless it's made inside the clone.

---

## 10. Results with the full TI and multi-head setup

### 10.1 Verifier on 2-task configs
2000 steps of task 0, then 1000 steps of task 1; within-task scoring; x0 as in the configs:

| config | task 0 val acc | task 1 val acc | task 0 kept |
|---|---|---|---|
| van_bp TI (G) | 22.2 → 91.9% | 92.3% | 83.8% |
| van_bp EWC TI | 22.2 → 91.9% | 90.1% | 82.1% |
| van_bp SI TI | 22.2 → 91.9% | 88.4% | 86.1% |
| bpDale TI (G) | 20.9 → 96.5% | 88.4% | 91.1% |
| bpDale EWC TI | 20.9 → 96.5% | 88.8% | 91.3% |
| bpDale SI TI | 20.9 → 96.5% | 84.7% | 94.5% |
| DTP TI (5J) | 20.9 → 93.6% | 83.1% | 92.4% |
| BTSP TI (6L) | 11.1 → 86.7% | 46.2% | 91.1% |
| BTSP_ELR TI (A) | 11.1 → 11.3% (**fails the learning check**) | 18.0% | 11.8% |

- **All 18 TI configs (2 and 5 tasks)** pass the static, params, build and nested stages.
- **Class-incremental configs are unaffected**, since multihead applies only with TI: EWC kept 34.0% of task 0, SI 47.1%, and BTSP 6L reached 94.7% and kept 13.1%, all as before.
- **Save and resume.** An exported run, resumed after phase 0 with `--retrain=False`, reproduces every feature exactly (BTSP 6L TI and bpDale EWC TI).
- **`pytest tests/`:** 3 pass. `test_load_network` fails because its saved pickle isn't on this machine; that predates this work.

### 10.2 Interpretation
- **Backprop family.** Multi-head changes little in accuracy but makes the structure exact: old heads are now frozen (0 change) in bpDale, where before they moved through the shared inhibition.
- **BTSP and DTP.** The top-down mask is what makes them learn in TI at their existing x0. Without it, inactive heads inject spurious dendritic errors (8.1).
- **BTSP_ELR.** It doesn't learn at its x0 even with multi-head. These networks are very sensitive to hyperparameters, so this is left to optimization.

---

## 11. Open issues and limitations
- **Uneven class splits.** `compute_task_incremental_loss_and_accuracy` assumes every task has the same number of classes (true for 2 and 5 splits). It would fail for 3 or 4 splits.
- **I-cell partitioning.** It is contiguous and assumes the I population size divides into tasks sensibly. `Output_I_size` is a searchable parameter in some configs; values not divisible by the number of tasks give unequal heads.
- **Mixing BTSP rules.** `BTSP_CL` and `BTSP_ELR_CL` must not be mixed in one network (they share a deduplicated `backward`).
- **Learned biases** would double-step with backprop-family CL rules (a pre-existing limitation, shared with SILR/INEL).
- **The ELR config's constraint drift** (7.2, 9.4) predates this work and is unrelated to masking.
- **EWC λ range.** Large λ diverges with SGD; the bounds were chosen with that in mind but may need tightening.

---

## 12. Files touched (this work)
- **Rules:**
  - `rules/base_classes.py` (`ContinualLearningMixin`);
  - `rules/backprop.py` (`Backprop_CL`, `Backprop_EWC`, `Backprop_SI`);
  - `rules/dtp.py` (new: `DTP_CL`);
  - `rules/btsp.py` (`BTSP_CL`, `BTSP_ELR_CL`);
  - `rules/weight_functions.py` (`clone_weight(labels_in_tasks=...)`, `task_block_mask`, `task_column_mask`, `get_unit_tasks`, `get_current_task`);
  - `rules/__init__.py`.
- **Script:** `optimize/nested_optimize_EIANN_1_hidden_CL_mnist.py`:
  - the `task_incremental`, `multihead` and `cumulative_val_set` interplay;
  - `get_task_incremental_projection_config` and `TARGET_FREE_RULES`;
  - TI scoring;
  - the debug hook;
  - re-applying constraints at task switches.
- **Utilities:** `utils/network_utils.py` (`compute_task_incremental_loss_and_accuracy`, `report_phase_debug`); `scripts/get_cl_metrics.py` (`--task-incremental`).
- **Update functions:** `optimize/network_config_updates.py` (`update_EIANN_config_2_hidden_{backprop_Dale,van_bp}_relu_SGD_CL_{EWC,SI}_A`).
- **Network YAMLs:** `optimize/network_config/MNIST_CL/20260928_*EWC*`, `20260930_*SI*`, and `20261001_*` (vanilla `Backprop_CL` for van_bp and bpDale; BTSP_CL 6L, DTP_CL 5J, BTSP_ELR_CL A).
- **Optimize configs:** `optimize/optimize_config/mnist_CL/20260928_*EWC*`, `20260930_*SI*` (class-incremental), and `20261001_*_TI_*` (18 TI configs).
- **Tooling** (under `.claude/`, likely gitignored): `skills/new-cl-model/verify_cl_model.py` (TI scoring, the multihead config builder, mask switching), `skills/new-cl-algorithm/SKILL.md`, `branches/v2dev.md`.
