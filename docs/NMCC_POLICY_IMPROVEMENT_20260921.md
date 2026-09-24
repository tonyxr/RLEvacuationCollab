# NMCC-PI: making the policy move, and move toward the better cell

## Material Passport

- Origin: Claude (Cowork), in response to "What would you change to make sure
  the policy will actually move and in the right direction. Also make sure NMCC
  is fully working, fix the design."
- Origin Mode: `design` + `implementation` + `experiment`.
- Origin Date: 2026-09-21
- Builds on: `docs/CONVERGENCE_REVIEW_20260921.md` (diagnosis and headroom).
- Model version: `RLBridge.MODEL_VERSION = 24`, architecture
  `..._nmcc_v8` (new intervention-value heads; v23 checkpoints do not load).
- Verification Status: `TARGET, BRANCHING AND LEARNING RULE VERIFIED TORCH-FREE ON
  THE CALIBRATED TESTBED; TORCH PORT STATICALLY CHECKED ONLY`.
  - 13/13 torch-free tests pass (`tests/test_nmcc_policy_improvement.py`),
    covering the target, cell selection, and branch valuation against the real
    dynamics.
  - A numpy reference learner running the same rule moved the policy from
    heuristic level to greedy-route-saving level in one or two updates (§5).
  - The RLBridge/GNN/Core port passes `flake8 --select=F`, and pyright reports no
    new errors against the saved baseline (126 now, 127 before).
  - It has **not been executed**, because torch is not installable here.
    `tests/test_nmcc_pi_torch.py` is written for the `rlevacuation` environment
    and has to pass there before the v24 curriculum is launched.
- Amendment 2026-09-21 (`nmcc_pi_v1.1`): learner audit (§9). The encoder and
  LSTM are now shared with the critic under a policy-preservation constraint.
  The M-step retries a violating epoch at a lower rate, and the v24 actor rate
  is raised to 1e-3. Readout initialization is unchanged, with the evidence
  for that choice in §9.
- Amendment 2026-09-21 (`nmcc_pi_v1.2`): signal and conversion audit on real
  testbed decision states (§10). The M-step trust region now bounds
  KL(π_new‖π_old), the direction the E-step bounds, with default cap 0.6 ≥ ε.
  An overshooting epoch is scaled back to the boundary by a parameter-space
  line search, and the learning rate no longer ratchets down across updates.
- Version Label: `nmcc_pi_v1.2`

## 1. Why the policy did not move, in one paragraph

The actor was asked the wrong question with a nearly flat instrument. The
Variant-A target `Q(s,c) − Q(s,WAIT)` measures whether deploying at all is
valuable. Deployment is mandatory, so 62% of that target's variance lies between
states rather than between cells. It arrived at the actor as a scalar on one
sampled cell per state, through a clipped ratio, with a 0.015 KL budget
(≈ 0.17 logits per update). The branch was ten minutes long, and it saw about a
third of the first deployment's value, which plays out over the hour. The prior
the policy sampled around was `relative_active`, which carries no information,
or slightly the wrong information, at t = 1, where nearly all of the regret is.
With temperature 1.5→1.0 and a prior spread of at most one logit, the behavior
policy stayed near-uniform, and the argmax stayed the heuristic.

## 2. The corrected design

NMCC's machinery is kept: keyed noise, exact snapshot/restore, `WAIT` as natural
momentum, and the dueling identity. What is computed from the branches changes.

At every training decision state `s`, before the executor installs anything:

1. **Exact, paired, full-horizon values.** The simulator is snapshotted. For
   each cell in a set `S ⊆ C_s` (every feasible cell at the first
   `nmccPiExhaustiveDecisions` decisions; otherwise the `nmccPiMaxBranches`
   most probable cells plus the executed one), the branch restores the
   snapshot, installs the cell by the shared site rule, and continues **to the
   horizon** with a fixed base policy (greedy route-time saving).
   - All cells share one independent future tape (`tape_seed`). The comparison
     is therefore common-random-number paired, but no branch sees the real
     episode's noise, so there is no hindsight bias. The target is
     implementable.
   - The live episode is restored bit-exactly afterwards (tested).
2. **Within-state advantage.**
   `A(s,c) = Q(s,c) − Σ_{c'∈S} π̃_old(c') Q(s,c')`, where π̃_old is the behavior
   distribution renormalized over `S`. `V_wait(s)` and every other quantity that
   does not depend on the cell cancel exactly. This is `Q = V_wait + D`, with the
   policy-mean constraint that makes `D` identifiable for a forced choice.
3. **KL-constrained target (MPO E-step).** `q(c|s) ∝ π_old(c|s)·exp(A(s,c)/η_s)`.
   - η_s is found by bisection so that `KL(q‖π_old) = ε` (default 0.5 nats).
   - η_s is floored at `η_min = 0.03`, which is the size of a single-tape contrast
     standard error on the testbed. Differences at the noise level therefore
     cannot become confident labels.
   - Unbranched cells keep their π_old mass.
4. **M-step.** The actor minimizes `KL(q‖π_θ)` over every decision. No entropy
   bonus, no teacher, and no ratio clipping apply, because the E-step's ε
   already sets the step.
   - The trust region is exact: `KL(π_behavior‖π_θ)` is computed over all cells,
     not estimated from the sampled action, and capped at `nmccPiKlCap`. An
     epoch that exceeds the cap is rolled back transactionally.
   - The fit has its own epoch budget (`nmccPiActorEpochs = 32`) and stops once
     `KL(q‖π_θ) ≤ 0.2 × ε_realized`. A one-epoch PPO budget could never fit a
     target half a nat away.
5. **Intervention-value ensemble (the "model" part of NMCC, now tested).**
   - `improvement_value_heads` are trained on the same within-state contrasts.
     The loss is behavior-centered over exact cells only and Poisson-bootstrap
     weighted per member, so a cell-independent level is provably outside what
     the heads can learn.
   - Each rollout's branches are scored by the ensemble *before* it trains on
     them. This held-out within-state Spearman is logged as
     `nmcc_pi_gate_rank_corr`.
   - Only when the mean over the last `nmccPiGateUpdates` rollouts reaches
     `nmccPiGateSpearman` may `nmccPiModelFill` fill unbranched cells.
   - Filled values are anchored to the exact values on the branched cells and
     penalized by ensemble spread. Fill-in is off in the v24 pilot.
6. **Prior.** `actorPrior = "route_time_saving"` makes the zero-residual policy
   the greedy route-saving rule, which measured +0.125 [0.061, 0.188] over the
   registered heuristic. The residual (±4 logits) learns the correction toward
   the rollout.

Because the base policy is fixed, the target converges to the one-step rollout
improvement of route saving. Under exact evaluation that is at least as good as
route saving (Bertsekas). The implementable rollout ceiling on the testbed is
+0.214 [0.109, 0.319] over the heuristic, versus +0.125 for route saving.

## 3. What NMCC ingredient does what now

| ingredient | before (v22/v23) | now (NMCC-PI) |
|---|---|---|
| keyed noise tape | pairs one cell with WAIT, using the live future | pairs all cells in `S`, using an independent future |
| WAIT branch | the baseline subtracted from one cell | still computed; cancels in `A` (kept as a check) |
| branch length | 10 min | to the horizon under the base policy |
| cells compared per state | 1 | all (early decisions) / 6+ (later) |
| actor signal | scalar A on the sampled cell, clipped ratio | full target distribution over cells, exact KL |
| world model | trained, never validated, 0 actor weight | within-state ranking gate on held-out rollouts; used only after it passes |
| prior | active population | route-time saving |
| encoder / LSTM | trained by the actor only | also trained by the critic, value and intervention-value losses under KL(π_ref‖π) (§9) |

## 4. Files

- `NMCCPolicyImprovement.py`: torch-free core (base policies, tapes,
  `BranchValuer`, cell selection, `improvement_target`, Spearman gate).
- `NMCCPIConfig.py`: torch-free defaults and the one table
  (`NMCC_PI_CORE_FIELDS`) that drives Core defaults, casting, RLBridge keywords,
  the recorded configuration, and the curriculum's learner keys.
- `GNN.py`:
  - `actor_prior` / `actor_prior_scale` / `actor_prior_feature_index`.
  - `improvement_value_heads`, one per ensemble member, returned as
    `NMCCPolicyOutput.improvement_value_samples`.
- `RLBridge.py`:
  - Collection before execution: `_collect_policy_improvement`.
  - Five new `Transition` fields, serialized with the rollout.
  - `_improvement_value_loss` in the critic phase.
  - Held-out gate before the critic phase.
  - Cross-entropy M-step with exact-KL rollback (`_improvement_policy_statistics`).
  - 17 `nmcc_pi_*` diagnostics.
  - Gate history and RNG state stored in checkpoints.
  - CSV schema guard: rows of a new schema never go under an old header.
  - Signature: `actor_prior` is an inference field; `nmcc_policy_improvement`
    is a training field.
- `Core.py` / `TrainingCurriculum.py`:
  - All 18 keys are configurable and recorded.
  - `actor_credit_target` becomes `nmcc_policy_improvement_exact_branch_target`
    when PI is on.
  - PI rejects staged NMCC phases and non-zero guidance, because either would
    change the distribution the target is defined against.
- `config/state_college_training_curriculum_2500_nmcc_pi_v24.json`: v23 physical
  scenario, one 64-episode stage, legacy hybrid branches off.
- `learner_flow_experiment.py`: numpy replica of the actor/critic
  parameterization with checked hand-written gradients (§9).
- `nmcc_pi_reference.py`: numpy reference learner (linear policy on 30
  features, initialized to the current actor at zero residual).
- Tests: `tests/test_nmcc_policy_improvement.py` (torch-free, 13) and
  `tests/test_nmcc_pi_torch.py` (torch; not yet run).
- Run: `runs/nmcc_pi_reference_run1_20260921.json`.

## 5. Does the rule move the policy the right way? (reference learner)

Setup:
- Testbed calibrated to State College's outcome profile (as in the review).
- Each iteration: 4 training episodes (training seeds 301+), about 15
  decisions, one tape, ε = 0.5, η_min = 0.03.
- Deterministic evaluation on 8 held-out seeds (201–208), paired.
- Reference points on those seeds: heuristic +0.100, greedy route saving +0.280.

| iter | held-out return | vs heuristic [95% CI] | vs route saving | top-1 = branch-best, before→after M-step | realized KL |
|---:|---:|---|---:|---|---:|
| 0 (current actor) | +0.096 | −0.004 | −0.184 | — | — |
| 1 | +0.247 | +0.146 [+0.000, +0.292] | −0.033 | 0.43 → 0.64 | 0.064 |
| 2 | +0.296 | +0.196 [+0.034, +0.358] | +0.016 | 0.22 → 0.33 | 0.051 |
| 3 | +0.288 | +0.188 [+0.027, +0.349] | +0.008 | 0.27 → 0.27 | 0.053 |
| 4 | +0.282 | +0.181 [+0.019, +0.343] | +0.002 | 0.33 → 0.33 | 0.023 |
| 5 | +0.278 | +0.177 [+0.020, +0.334] | −0.003 | 0.42 → 0.50 | 0.061 |
| 6 | +0.262 | +0.162 [+0.006, +0.318] | −0.018 | 0.29 → 0.36 | 0.023 |

Reading:

- **The policy moves, and it moves in the right direction.** From the actor's
  current zero-residual behavior, one update with 15 decisions of exact targets
  reached route-saving level. Every later iterate beats the heuristic, with a CI
  that excludes zero.
- **It does not yet go beyond the base policy.** The iterates oscillate around
  route saving (±0.02) instead of climbing toward the rollout ceiling. Two
  measured limits explain this:
  - The M-step reached only 0.02–0.06 nats of the 0.5 requested; it ended at its
    200-epoch limit every time.
  - A 30-feature linear policy cannot represent the state-specific corrections
    the rollout makes at t = 1.
  - The torch port addresses the first with its own epoch budget, a
    convergence-based stop and the ±4-logit residual. The GNN addresses the
    second.

  Whether the GNN clears route saving is the question the v24 pilot has to
  answer.
- The before-M-step top-1 at iteration t+1 is lower than the after-M-step value
  at t. The states are different, because the new policy visits new states.
  This is expected, and it is why the gate and the fit are always measured on
  held-out data.

## 6. Cost on State College

Each branched cell is a full continuation to the horizon. With about 18
feasible cells at the first two decisions and 6–7 at each later one, one
training episode costs roughly 45–55 continuations. That is 45–55 extra
episode-remainders, most of them shorter than a full episode because later
decisions start later. On the review's timing for a State College episode, that
is about 5–11 minutes per training episode. The 64-episode v24 stage is
therefore roughly 6–12 hours on one core, parallelizable across episodes.
Evaluation episodes do not branch.

## 7. Run commands (in the `rlevacuation` environment)

```bash
python -m unittest tests.test_nmcc_policy_improvement tests.test_nmcc_pi_torch -v
python -m unittest tests.test_rl_framework tests.test_training_curriculum
python multicity_backtest.py \
  --cities state_college_pa \
  --launch-id state_college_2500_nmcc_pi_v24 \
  --policy-replicates 1 \
  --train-episodes-per-city 64 \
  --training-curriculum config/state_college_training_curriculum_2500_nmcc_pi_v24.json \
  --train-only --no-require-convergence --no-policy-cache
python nmcc_pi_reference.py --iterations 6 --episodes 4 --eval-seeds 8   # torch-free reference
```

Watch these columns in `ppo_diagnostics.csv`:

- `nmcc_pi_fit_kl_after < nmcc_pi_fit_kl_before`: the M-step is fitting.
- `nmcc_pi_top1_after > nmcc_pi_top1_before`: the actor moves toward the
  exactly better cell.
- `nmcc_pi_eta_at_floor`: the share of states whose contrast is at the noise
  level. If this is near 1, raise `nmccPiTapes` rather than ε.
- `nmcc_pi_gate_rank_corr`: whether the world model has earned fill-in.
- `actor_representation_gradient_share` > 0 and `critic_representation_gradient_norm`
  > 0: the policy loss and the value losses both reach the encoder/LSTM.
- `representation_policy_drift_kl` ≤ 0.05 and `representation_rollback` = 0:
  the critic pass shaped features without moving the policy.

Success is judged on frozen checkpoints evaluated deterministically on fixed
held-out tapes. The paired comparison is against the heuristic, route saving
and the rollout, not training return.

## 8. What was not done

- No torch execution of the port. The first run of `tests/test_nmcc_pi_torch.py`
  is the acceptance test.
- The headroom and the reference curve are testbed numbers. State College has
  to be re-measured with a real `Core`.
- The base policy is fixed at route saving. Using the learned policy as the base
  (approximate policy iteration) is the next step once the pilot clears route
  saving. It needs a torch-side base-policy callable inside `BranchValuer`.
- `ppoRolloutEpisodes` is not a curriculum key. The pilot uses the default of 8
  episodes per update, so it makes 8 updates.

## 9. Learner audit: do gradient steps reach the policy, and the right way?

The review asked four things. Each one was checked two ways: against the v22
and v23 run logs, and with `learner_flow_experiment.py`. That experiment is a
numpy replica of the actor and critic parameterization — encoder, actor head
`B·tanh(w·gelu(W2 h))`, and value head — trained with Adam and global-norm
clipping at 0.5. Its hand-written gradients agree with finite differences to
4e-9. It was run with 3 seeds, on 8-dimensional inputs (8 rollouts) and on
32-dimensional inputs where only 4 directions carry value (16 rollouts).

### 9.1 Are the policy steps valid, and large enough?

The direction was right; the size was not. In v22, 64 episodes produced 36
accepted actor steps. The residual RMS grew linearly, at about 0.003 logits per
step, to 0.12, and the entropy stayed at 0.88–0.99. The v23 smoke run moved the
residual by 0.0023 logits in two updates.

The replica reproduces this. The v23 budget (one epoch, 0.015 KL) leaves the
residual at 0.01 after one rollout and 0.09 after eight. It closes less than 1% of a
0.5-nat target, and held-out regret goes from 0.39 to 0.36. The v24 M-step
(32 epochs, exact trust region, early stop) reaches a residual of 0.47 after one
rollout and 3.0 after eight, with regret 0.39 → 0.026. In the 32-dimensional
case the numbers are 1.02 → 0.068, versus 1.02 → 1.00 under v23.

Two changes followed:

- **Backtracking in the PI M-step.** An epoch that breaks the exact
  KL(π_behavior‖π) cap is rolled back and retried at half the rate, up to three
  consecutive times. PPO still stops at the first violation, because its
  surrogate is only trusted near π_old. The PI objective is a fixed target, so a
  violation there is a step-size failure, not a reason to give up the update.
- **v24 actor rate 1e-3.** In the replica, 1e-3 fits the first target from
  0.50 to 0.13 nats, against 0.40 at 3e-4, and halves the first-rollout regret.
  After eight rollouts the two rates are equivalent. Because the pilot has only
  8 updates, the early ones matter. The trust region, early stop and
  backtracking bound the risk.

### 9.2 Does initialization delay gradient propagation?

Yes, but by one Adam step, and it is not the binding constraint. With the zero
readout, the first actor step sends exactly zero gradient to the encoder and
LSTM. The replica measures a representation gradient share of 0.000, and v23's
first update left the residual at exactly 0.0. After that step, Adam's
per-parameter normalization makes the delay immaterial.

A small random readout (0.05-logit RMS) was implemented and tested:

- **Actor:** it reached the same regret as the zero readout, 0.026 vs 0.026 and
  0.072 vs 0.068.
- **Critic:** it ranked cells *worse* in early rollouts (0.10 vs 0.26 Spearman
  at rollout 1), because it starts from a random function instead of Adam's
  coherent first step.

It was therefore **not adopted**. The zero readout stays, and so does the exact
"zero residual = prior" contract. The GNN code now records this reasoning next
to the initialization.

### 9.3 Are useful signals disconnected from the actor?

Some were, through the representation.

- In `actor_owned` mode, the only gradient the encoder and LSTM ever see is the
  policy loss, which under v22/v23 was nearly zero.
- Three dense, low-variance supervised signals never shaped the features the
  actor reads: the exact within-state cell values (one per branched cell), the
  TD value targets, and the NMCC outcome targets.
- The momentum features enter only the LSTM. The LSTM reaches the actor
  through `temporal_actor_context`, which is zero-initialized, and the
  critic's gradient never reached it.

In `shared_phasic` mode:

- The critic pass also trains the encoder and LSTM. The intervention-value
  heads read the actor's candidate embedding, so their within-state
  regression now shapes exactly the features the actor uses to tell cells
  apart.
- The actor still owns the encoder and LSTM in its own pass.
- The ensemble's *outputs* reach the target only through the held-out ranking
  gate. That stays deliberate: an unvalidated model must not relabel cells.

### 9.4 Is the critic over-isolated?

It was. In `actor_owned` mode the critic, the NMCC world model and the ensemble
fit shallow heads on an encoder they cannot shape. Explained variance stayed
near 0 for all 64 v22 episodes (−0.006 to 0.031).

In the 32-dimensional replica, a head on a frozen representation plateaus at a
held-out within-state Spearman of 0.15. With a shared representation it reaches
0.28 by rollout 16 and is still rising. In the 8-dimensional case, where random
features are already informative, the gain is smaller (0.36 → 0.41).

`shared_phasic` mode (phasic policy gradient, Cobbe et al. 2021) works as
follows:

- The critic optimizer also owns the representation, with its own Adam
  moments. The partition is validated as three disjoint roles by
  `EvacPolicy.parameter_roles()`.
- The critic loss adds `representationCloneCoefficient · KL(π_ref‖π)`, exact over
  all cells, where π_ref is the policy that collected the rollout.
- After each critic epoch, the full-batch drift is measured. If it exceeds
  `representationKlCap` (0.05), the epoch is undone together with its optimizer
  moments.

In the replica the drift per pass was at most 2e-4, so the cap is a guarantee,
not an active constraint. Any drift that does occur is also counted inside the
actor's trust region, which is measured against the behavior policy.

`actor_owned` stays the default, so earlier configurations train exactly as
before. The v24 curriculum sets `representationMode = "shared_phasic"`.

### 9.5 New diagnostics

These are measured before gradient clipping:

- `actor_representation_gradient_norm`, `actor_head_gradient_norm` and
  `actor_representation_gradient_share`
- `critic_representation_gradient_norm` and `critic_head_gradient_norm`
- `representation_policy_drift_kl`, `representation_rollback` and
  `representation_clone_kl`
- `actor_readout_norm`, `actor_temporal_readout_norm` and `actor_rollbacks`

Evidence files: `runs/learner_flow_experiment_8d_20260921.json` and
`runs/learner_flow_experiment_32d_20260921.json`.

### 9.6 Limits

The replica has one encoder layer, not a message-passing GNN plus an LSTM, and
its "actor-owned" critic runs freeze the encoder, as the near-zero actor steps
of v22/v23 effectively did. It shows mechanisms and directions; its magnitudes
are not a prediction for State College. `tests/test_nmcc_pi_torch.py` now also
checks the following in the real model:

- the three-way ownership;
- which tensors are trainable in each pass;
- nonzero gradient to the representation from both passes;
- movement of the representation;
- drift within its cap.

## 10. Audit: is the action-differential signal weak, and is it converted into a tiny policy change?

The failure being checked is: *valid delayed rewards reach the actor, but the
action-differential signal is weak, and the optimizer converts it into an
extremely small policy change.* The audit (`nmcc_pi_signal_audit.py`) was set
up as follows:

- **States.** 43 real decision states from 12 on-policy episodes of the
  calibrated testbed (seeds 401–412).
- **Behavior policy.** The v24 zero-residual actor: softmax of route-time
  saving, temperature 1.
- **Branching.** Every feasible cell at every decision is branched to the
  horizon under 4 independent CRN tapes.
- **Scoring.** A target built from some tapes is scored against the other,
  independent tapes. Their mean is an unbiased estimate of the true
  within-state advantage, so "valid gain" Σ_c (π_new − π_old)(c)·A_true(c) is
  the return a policy change actually buys, in reward units.

### 10.1 The signal is not weak where the value is

| decision | cells | within-state signal SD | 1-tape noise SD | SNR (1 tape) |
|---:|---:|---:|---:|---:|
| t = 1 | 18.0 | 0.089 | 0.053 | 5.1 |
| t = 11 | 11.3 | 0.068 | 0.027 | 15 |
| t = 21 | 8.8 | 0.026 | 0.012 | 22 |
| t = 31 | 6.5 | 0.009 | 0.003 | 150 |
| t = 41 | 7.5 | 0.001 | 0.000 | — |

The full-horizon, all-cells, CRN-paired contrast is a strong signal at the
first two decisions, which hold nearly all of the value. Later decisions have
little value to find, but also little noise.

The earlier failure was about the *old* signal: a ten-minute, one-cell-vs-WAIT
difference, 62% of it between states. It does not describe what the actor
receives now.

### 10.2 The target it produces is valid

A one-tape E-step target (ε = 0.5, η_min = 0.03) captures part of the best
available within-state gain, and the gain is positive in almost every state:

- **t = 1:** 51% of the available gain, positive in 100% of states.
- **t = 11:** 60%, positive in 92%.
- **t = 21:** 38%, positive in 100%.

At t = 31 and t = 41 the temperature floor holds the requested move to
0.06 and 0.002 nats, so noise-level contrasts are not turned into labels.

A second tape barely changes this (50% and 61%). The share per update is set
by the step size ε, not by noise, so a second tape would double branching cost
for no gain.

### 10.3 Conversion into a policy change

The M-step was run on a replica actor (encoder plus bounded residual head,
Adam, clip 0.5), fitted to the one-tape targets of the training episodes and
scored with 4-fold leave-episodes-out validation:

| M-step | requested KL(q‖π_old) | fit | realized KL(π_old‖π) | valid gain, train | valid gain, held-out episodes | held-out top-1 = best cell |
|---|---:|---:|---:|---:|---:|---|
| v23 (1 epoch, lr 1e-4, cap 0.015) | 0.34 | 0% | 0.000 | +0.0001 of 0.081 | +0.0001 of 0.082 | 0.26 → 0.26 |
| v24 (32 epochs, lr 1e-3) | 0.34 | 60% | 0.215 | +0.029 of 0.081 | +0.022 of 0.082 | 0.26 → 0.39 |

- **v23: confirmed.** The failure was fully present: it converted a target 0.34
  nats away into no policy change and no gain.
- **v24: it no longer holds.** One update moves the policy 0.2 nats and buys
  +0.022 of return per held-out decision state, about 200 times v23. 78% of
  the in-sample gain carries over to unseen episodes.
- **Longer fitting helps little.** 96 epochs (fit 79%) raises the held-out gain
  only to +0.025, and lowers held-out top-1 to 0.30. The M-step budget
  therefore stays at 32 epochs.

### 10.4 A latent failure that was fixed

The trust region was checking the wrong KL direction:

- The E-step bounds KL(q‖π_old) = ε. The M-step cap bounded
  KL(π_old‖π_new) ≤ 0.5.
- For the real targets the two are not interchangeable. At t = 1 and t = 11,
  fitting the target exactly means KL(π_old‖π_new) = 0.68–0.71 on average, and
  up to 1.19.
- On this batch the mean over all decisions is 0.43, so the old cap did not
  bind here. It would bind on any batch dominated by early decisions, or once
  the GNN fits the targets closely.

When it bound, the old rule compounded the damage. It discarded the epoch and
halved the learning rate. Growth is only 5% per update, so three halvings
leave the actor at 1/8 of its rate for about 40 updates, five times the whole
pilot. That is exactly "the optimizer converts the signal into an extremely
small change".

Fixed as follows:

- **Same direction as the E-step.** The trust region now bounds
  KL(π_new‖π_old) with `nmccPiKlCap` = 0.6. A cap below ε is rejected, because
  the policy could then never reach its own target.
- **Line search instead of rollback.** An epoch that overshoots is scaled back
  to the boundary by bisection on the parameter step (8 steps), accepted, and
  the M-step stops there. The policy moves the maximum permitted amount
  instead of losing the epoch.
- **No learning-rate ratchet.** The rate is reset to its configured value at
  the start of every PI update. PPO mode keeps its old rule.
- **New diagnostics.** `actor_trust_region_reached`,
  `actor_line_search_fraction` and `actor_reverse_kl`.
- **New tests** in `tests/test_nmcc_pi_torch.py`: that a cap below ε is
  rejected, and that the line search ends inside the region and within 2⁻⁸ of
  the boundary.

### 10.5 What remains

Per update, the policy captures about a quarter of the available within-state
gain on held-out states. This is by design: ε = 0.5 nats per update, compounded
over updates.

With 8 updates in the 64-episode pilot, compounding is the binding limit,
not signal strength or the optimizer. The lever is more updates per episode
budget (`ppoRolloutEpisodes`, not yet a curriculum key), not a larger ε. A
larger ε would turn noise-level contrasts into confident labels.

Caveats:

- This is the synthetic, State-College-calibrated testbed, with 43 states.
- The replica is not the GNN.
- The pickled branch values are kept, so the analysis can be re-run with
  other ε, η_min or M-step settings without re-simulating.

Evidence: `runs/nmcc_pi_signal_audit_20260921.json`,
`runs/nmcc_pi_signal_audit_states_20260921.pkl`.
