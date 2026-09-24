# Audit of the model-v26 `value_lcb` implementation now in the working tree

## Material Passport

- Origin: Claude (Cowork), at the user's request: "Check the code again and
  see if the issues are fixed."
- Origin Mode: `review` + `experiment`. No file in the working tree was
  modified.
- Origin Date: 2026-09-22.
- Subject: the working-tree code as of `RLBridge.py` 19:54 UTC / 
  `nmcc_learning_backtest.py` 20:19 UTC, `MODEL_VERSION = 26`,
  `nmccPiActorObjective = "value_lcb"`. This is the concurrent implementation,
  not the proposal in `proposals/v26_fitted_policy_iteration_20260922/`.
- Reference: the four issues and the capacity confound in
  `docs/V25_PILOT_REVIEW_20260922.md`.
- Evidence:
  - a line-by-line diff of every changed file against the v25 snapshot
    (`RLBridge.py` 781 changed lines, `GNN.py` 167, `DecisionInterface.py` 94,
    `NMCCPIConfig.py` 58, `backtest.py` 85, `NMCCPolicyImprovement.py` 54,
    `TrainingCurriculum.py` 32, `RewardProcessor.py` 21);
  - torch-free suites executed against this tree;
  - two new executed experiments (below).
- New experiments:
  - `live26_gate_check.py`: the deployment gate's operating characteristics,
    replayed against the real 3-tape full-horizon audit labels, 500 trials per
    condition.
  - A direct probe of the `WAIT` branch under the new decision schedule.
- Verification Status: `TORCH-FREE PATHS EXECUTED; TORCH LEARNER READ AND
  REASONED ABOUT, NOT EXECUTED (no torch in this environment)`.
- Version Label: `v26_value_lcb_audit_v1`.

## 0. Verdict

The four issues are addressed, and the capacity confound is addressed at the
mechanism level. The design is sound and, on several points, better than what
the review asked for.

Nine defects remain. One blocks the next milestone, one is a latent crash in
the new training path, one silently changes what the `WAIT` branch means, and
one makes the deployment gate weaker than its numbers suggest.

| v25 issue | verdict |
|---|---|
| 1. Training return cannot show improvement | **Fixed**, with one residual: the convergence audit still requires 100 episodes while the registered curriculum runs 96. |
| 2. Actor moved but did not generalize | **Fixed.** The actor fit is disabled for `value_lcb`; the controller is the route-saving prior plus a gated LCB correction. |
| 3. Exact labels discarded after one update | **Fixed.** A bounded (256-episode) reservoir replay, refit from a fresh intervention model each rollout with held-out early stopping. |
| 4. Labels myopic and incomplete; the ensemble never acts | **Fixed.** Full horizon and exhaustive branching at the first two decisions; the ensemble now selects the action. |
| Capacity confound | **Fixed at the mechanism**, and evaluation now fails closed on unequal realized deployments or capacity. One residual: a long block still shifts the whole schedule later and can lose a token. |

## 1. What I verified, and how

Executed here:

- The torch-free suites against this tree: `test_nmcc_policy_improvement`
  (15), `test_counterfactual_branch` (10), `test_training_curriculum` (13),
  `test_cell_partitioning` (6). All pass.
- `flake8 --select=F` over the tree: no new F-level finding in any changed
  file. (`or_journal_experiments.py:343` has a pre-existing `F821 undefined
  name 'candidate_count'`, untouched by v26.)
- The `WAIT` branch probe and the gate operating-characteristic study
  described in sections 3 and 4.

Not executed: everything that needs torch — the replay refit, the batched
recurrent replay, the checkpoint round-trip with a populated replay, and
`tests/test_nmcc_pi_torch.py`. Torch cannot be installed in this environment.

## 2. Defects, most consequential first

### 2.1 The standard evaluation path cannot load a checkpoint this curriculum trained

- **Where.** `RLBridge.py:1426` `_inference_signature`, which now also
  compares `nmcc_policy_improvement` and `candidate_constraints`;
  `multicity_backtest.py:1703`, unchanged, which evaluates with
  `overrides=overrides_by_city[city.city_id]`.
- **Why it fails.** Those overrides are `suite.common_experiment` plus the
  city profile. `config/city_profiles.json` sets no `nmcc*`, no `actorPrior`,
  and no candidate-constraint field, so evaluation builds `Core` with
  `nmccPolicyImprovement=False`, `actorPrior="active_population"`,
  `requireCandidateOperationalBenefit=False` and
  `minimumCandidateHazardSafetyMargin=0.0`. The checkpoint was trained with
  the curriculum's values. The signatures differ, and `_load_checkpoint`
  raises "Refusing to evaluate incompatible checkpoint".
- **Severity.** Blocks the next TODO milestone (deterministic held-out
  evaluation). It also predates v26: `actor_prior` was already in the
  inference signature in v25, so a v25 checkpoint trained by its own
  curriculum could not be evaluated either. v26 widens the mismatch from one
  field to three, and the two new fields are the right ones to check —
  evaluating a mask-trained policy without its masks was the real bug.
- **Fix.** Apply the curriculum's learner and decision-interface overrides to
  every evaluated strategy, not only to training. Reading them once from the
  loaded `TrainingCurriculum` and merging them into the evaluation overrides
  is about ten lines in `multicity_backtest.main`. Applying them to the
  benchmarks too is required anyway, or the benchmarks run without the masks
  and without the retry rule that the parity claim depends on.

### 2.2 The replay refit can crash on a minibatch with no within-state contrast

- **Where.** `RLBridge.py:4879-4894`.
- **Why.** Both `_improvement_value_loss` and `_improvement_pairwise_loss`
  return `samples.new_zeros(())` when no row in the minibatch has at least two
  branched cells. `new_zeros` does not require grad, so `loss` does not
  either, and `loss.backward()` raises "element 0 of tensors does not require
  grad and does not have a grad_fn". In v25 this path was safe because the
  same zero was added to a critic loss that had other terms; here the two
  terms are the whole loss.
- **When it fires.** A decision with exactly one feasible candidate produces a
  single branched cell. `_iter_episode_minibatches` can emit a trailing batch
  holding one episode, so an episode whose labelled decisions all had one
  feasible cell crashes the refit. The v25 pilot had 11 of 64 episodes where
  the mask left too few options, including one that installed a single token,
  so this is reachable rather than theoretical. A whole-minibatch zero
  bootstrap draw reaches the same line with probability about `e^-5` per state.
- **Fix.** Skip the step when the loss carries no gradient:

  ```python
  if not loss.requires_grad:
      continue
  ```

  Better, drop rows with fewer than two branched cells when the replay arrays
  are built, so they never enter a batch.

### 2.3 The `WAIT` branch no longer waits

- **Where.** `NMCCPolicyImprovement.py` `EpisodeClock.is_decision` and
  `BranchValuer.value`, with `RLBridge.py:3563` passing
  `next_decision_time=int(simulation_time)`.
- **What happens.** Under the new retry rule a decision is due at every
  timestep at or after `next_decision_time`. The `WAIT` clock keeps
  `next_decision_time = t`, so the branch steps to `t + 1`, finds a decision
  due, and installs the base policy's shelter there.
- **Measured.** I probed the real testbed dynamics. At a decision at `t = 1`,
  the `WAIT` branch installed at timesteps `[2, 12, 22]`. A deferred-epoch
  `WAIT` would have installed at `[11, 21, 31]`. Over three seeds, mean
  `|Q(c) − Q(WAIT)|` falls from `0.159` to `0.116` (−27%) and the mean
  per-cell physical effect from `0.042` to `0.033` (−21%).
- **Why it matters.** `natural_outcome_head` is documented, and
  architecturally constrained by its population-conservation softmax, to
  predict the *no-deployment* outcome. Its target is now a trajectory that
  contains a shelter installed one tick later. The candidate-effect target
  `outcomes − wait_outcomes` shrinks accordingly. The controller's own labels
  — within-state candidate values — are unaffected, so this does not corrupt
  the decision rule. What it degrades is the world-model supervision that
  shapes the shared representation the controller reads, and it makes the two
  natural and three causal pretraining rollouts (20 of the 96 pilot episodes,
  several hours of simulation) do something other than what they are for.
- **Fix.** One line in `BranchValuer.value`: give the `WAIT` clock the next
  scheduled epoch rather than the current time.

  ```python
  wait_clock.next_decision_time = (
      clock.t + self.interval_to_next_scheduled_epoch()  # or clock.t + interval
  )
  ```

### 2.4 The gate is weaker than its structure suggests

Three compounding points.

**(a) The gate statistic and the early-stopping criterion use the same
episodes.** `_fit_improvement_replay` early-stops on the validation loss over
`validation_records`, then computes `gain_lower` on the same records and
appends it to the gate history. The epoch is selected on the data the
certificate is computed from.

**(b) "Three consecutive refits" does not buy three independent tests.** The
validation split is `episode_id % 5 == 0`, which is nearly the same set of
episodes at every refit. Consecutive statistics are strongly positively
correlated, so the joint false-open rate is much closer to the per-refit rate
than to its cube.

**(c) The gate scores a controller restricted to branched cells; deployment
is not restricted.** `_improvement_validation_metrics` takes the argmax over
`exact_mask`, but `_controller_logits` takes it over every feasible cell. At
decisions past the second, that is 6 scored cells against roughly 18 the
deployed controller may choose from.

**Measured.** I replayed the implemented rule against the real audit labels
(tape 0 as the label the gate sees, tapes 1 and 2 as truth), 500 trials per
condition. "Deployed gain" is the true gain when the controller chooses among
all feasible cells:

| model | true deployed gain | open per refit | three consecutive |
|---|---:|---:|---:|
| *3 validation episodes (the audit pool)* | | | |
| no skill | −0.011 | 0.071 | 0.000 |
| noisy oracle, σ = 2 | +0.031 | 0.422 | 0.083 |
| noisy oracle, σ = 1 | +0.050 | 0.609 | 0.170 |
| noisy oracle, σ = 0.5 | +0.061 | 0.828 | 0.556 |
| *≈19 validation episodes (where the 96-episode run ends up)* | | | |
| no skill | −0.011 | **0.265** | 0.021 |
| noisy oracle, σ = 2 | +0.031 | 0.734 | 0.394 |
| noisy oracle, σ = 1 | +0.050 | 0.876 | 0.655 |
| noisy oracle, σ = 0.5 | +0.061 | 0.980 | 0.949 |

Reading it: early in the run the gate is very conservative and will rarely
open. By the end of the run a genuinely skill-free model clears the
per-refit test 27% of the time, and the 2.1% figure for three in a row
assumes independence the design does not have. The realistic false-open rate
is between 2% and 27%, and the controller it would deploy has a *negative*
true gain of −0.011.

Early stopping on the same episodes adds a further, smaller inflation
(0.609 → 0.672 per refit at σ = 1; 0.827 → 0.902 at σ = 0.25).

**Fixes, in order of effect.**
1. Split three ways: fit, validation for early stopping, and a separate gate
   split that nothing selects on.
2. Raise `nmccPiValidationGainZ` from 1.0 (about 84% one-sided) to 2.0. On the
   same data that takes the no-skill three-in-a-row rate from 2.1% to 1.0%
   and the per-refit rate from 26.5% to 18.9%, while a model worth +0.031
   still opens 60% of refits.
3. Require the consecutive positives to come from disjoint validation
   episodes, or score the gate over all feasible cells by branching holdout
   decisions exhaustively.

### 2.5 The new learner path has no end-to-end test

`tests/test_nmcc_pi_torch.py` gained six good unit tests: the closed-gate
controller, LCB invariance to member offsets, parameter ownership, the exact
reward decomposition, and that validation scores the same composite as
deployment. None of them runs a `value_lcb` episode through
`end_episode → _optimize_policy → _append_improvement_replay →
_fit_improvement_replay`, and none saves and reloads a checkpoint with a
populated replay. `TrainingEpisodeTests` still exercises the v25
`score_ranking` path. The highest-risk new code — the refit loop, the replay
serialization, the validation metrics — is the untested part, and 2.2 lives
inside it.

### 2.6 The convergence audit still demands more episodes than the curriculum runs

`backtest.py:622` keeps `enough = episodes >= minimum_episodes`, and the
fitted-policy branch at line 668 requires it too. The default is 100;
`config/state_college_training_curriculum_2500_nmcc_value_v26.json` runs 96.
The audit will report `converged: false` regardless of the held-out gate. The
README works around it with `--no-require-convergence`. Either raise the
curriculum to 100 episodes or lower `--convergence-min-episodes` for this mode.

### 2.7 Nothing checks that the prior and the base policy agree

`_controller_logits` derives the base action from the argmax of
`prior_logits`, while the stored label's base action comes from
`NPI.BASE_POLICIES[nmcc_pi_base_policy]`. These coincide only because the
curriculum sets `actorPrior = "route_time_saving"` and
`nmccPiBasePolicy = "route_saving"`. With `actorPrior = "active_population"`
the deployed base and the labelled base would silently differ, and the gate
would certify a controller that is not the one deployed. Add a constructor
check that pairs them when the objective is `value_lcb`.

### 2.8 Checkpoint size and per-episode revalidation

The replay is stored inside the checkpoint and written on every episode.
Measured on an 8×8 grid, one stored observation frame is about 9.5 KB
(4.6 KB cell features, 2.0 KB candidate features, the rest edges and
indices); the v25 pilot diagnostics show 41 frames per episode. So the
checkpoint grows to roughly 40 MB for the 96-episode State College run and
100–150 MB at the 256-episode cap, rewritten every episode, and
`_validate_improvement_replay` re-validates every frame of every replay
episode each time a new per-episode bridge loads it. That is workable against
a 13-minute episode, but it is gigabytes of checkpoint writes per run. A
sidecar directory keyed by episode id would avoid rewriting the whole dataset
to save one episode.

### 2.9 The v25 decision schedule is no longer reproducible

`_decision_due` now always uses the retry rule; there is no flag that restores
`(t − first) % interval == 0`. Outside a blocked mask the two rules agree, so
nothing silently changes in an unblocked scenario, but a v25 experiment rerun
against this tree will differ wherever the mask ever emptied. The repository
discipline in `TODO.md` asks that older model versions stay reproducible. A
boolean in `NMCC_PI_CORE_FIELDS`, defaulting to the retry rule, would restore
that and would also let the two schedules be compared as an ablation.

## 3. Things I checked that are correct

- The decision-rule change: `_controller_logits` returns the masked prior
  until the gate opens, and then replaces the ordering, keeping the base
  level, rather than adding the full advantage to a varying prior. The
  comment explaining why is right.
- `_base_relative_improvement_lcb` differences every member against the base
  action before taking the ensemble spread. Without that, the ensemble's
  unidentified per-state offsets would be read as action uncertainty. This is
  the same correction the proposal makes, arrived at independently.
- The actor is genuinely disabled for `value_lcb`: `actor_enabled`,
  `improvement_rows` and `pi_actor` all exclude it, so there is no
  double-counted improvement loss and no PPO step.
- Replay tensors are held on CPU and moved per graph in `_graph_from_frame`,
  so the dataset does not sit on the GPU.
- The episode-level train/validation split keeps whole episodes on one side,
  so no recurrent prefix leaks.
- Reservoir sampling is a correct uniform sample, and
  `_validate_improvement_replay` checks the bound, id uniqueness, episode
  integrity and label shape on load.
- The new `improvement_*` modules give the value branch its own projection
  instead of borrowing actor-owned `candidate_hidden`, which in v25 was frozen
  during the critic pass. The wide linear path on raw interpretable features
  is a reasonable answer to a small label set.
- `RewardProcessor` now exports the weights and `GNN._outcomes_to_components`
  uses them, so changing a reward weight can no longer leave the world model
  optimizing a stale hard-coded `-3.0`. The reward's own arithmetic is
  unchanged and correct.
- Scenario descriptors are bounded, dimensionless and defined for any
  population, hazard count and panic level; the capacity-coverage feature is
  a good addition.
- `_verify_matched_interface` now fails closed on unequal realized deployment
  count and total installed capacity.
- Curriculum schema v2 with a single `learner_overrides` block, conflict
  detection against variant overrides, and rejection of non-learner keys.
- `_improvement_validation_metrics` reorders both the samples and the priors
  by the replay's returned index before use.

## 4. Recommended order of work

1. 2.2, the crash — two lines, and it is in the path every refit takes.
2. 2.1, evaluation — otherwise the run cannot be evaluated when it finishes.
3. 2.3, the `WAIT` branch — one line, and it restores what the pretraining
   phases are for.
4. 2.5, an end-to-end `value_lcb` episode test plus a checkpoint round trip
   with a populated replay. This is what would have caught 2.2.
5. 2.4, the gate. At minimum raise `nmccPiValidationGainZ` to 2.0 and record
   in the run manifest that the gate split is shared with early stopping.
   The three-way split is the real fix.
6. 2.6 through 2.9 before the confirmatory campaign.

Artifacts: `live26_gate_check.py` and
`runs/live26_value_lcb_gate_check.json` hold the gate study and are
reproducible with `python live26_gate_check.py --trials 500`.
