# Model v26: rollout-labelled fitted policy iteration

## Material Passport

- Origin: Claude (Cowork), at the user's request: "do your best at fixing
  with the four issues you have flagged, do fundamental fixes, no minimalism
  fallbacks and patch, check the reward function as well. The training
  suppose to work on any city, any number of pedestrians, any number of hazard
  instances, and panic levels … Use the full GNN and architecture".
- Origin Mode: `design` + `implementation` + `validation`.
- Origin Date: 2026-09-22.
- Inputs: `docs/V25_PILOT_REVIEW_20260922.md` (the four flagged issues and
  the capacity-parity finding); model version 25 source.
- Files changed:
  - Model: `RLBridge.py` (MODEL_VERSION 26), `GNN.py`, `DecisionInterface.py`,
    `NMCCPolicyImprovement.py`, `NMCCPIConfig.py`, `TrainingCurriculum.py`.
  - Runners: `backtest.py`, `multicity_backtest.py`.
  - New: `FittedPolicyIteration.py`, `fpi_label_workers.py`,
    `fpi_gate_validation.py`.
- New configuration: `config/scenario_general_training_curriculum_nmcc_fpi_v26.json`
  and `config/state_college_training_curriculum_nmcc_fpi_v26_pilot.json`.
- New tests: `tests/test_fitted_policy_iteration.py` (torch-free),
  `tests/test_fitted_policy_iteration_torch.py` (needs the `rlevacuation`
  environment), and `tests/test_reward_contract.py`.
- Verification Status: `TORCH-FREE CORE TESTED AND GATE VALIDATED ON REAL
  CRN LABELS; TORCH LEARNER STATICALLY CHECKED AND REVIEWED, NOT EXECUTED`.
  - This tool cannot install torch (package index outside the egress
    allowlist), so the GNN refit, replay and bridge integration have not
    been run.
  - The torch tests must pass in `rlevacuation` before any run.
- Version Label: `nmcc_fpi_v26_design_v1`.
- Integration status: `PROPOSAL, NOT MERGED`.
  - This implementation was written against the v25 device snapshot.
  - While it was being written, a different model-v26 implementation
    (`nmccPiActorObjective = "value_lcb"`) appeared in the device working
    tree and changed the same files.
  - To avoid overwriting that work, this version is delivered as a
    self-contained proposal under
    `proposals/v26_fitted_policy_iteration_20260922/`: the full files plus a
    patch against v25.

## 1. What changed, in one paragraph

NMCC exact branching turns shelter placement into supervised learning of a
within-state value function under a fixed base policy. Version 26 therefore
uses the matching estimator: rollout-labelled fitted policy iteration.

- **Labels.** Every decision's exact common-random-number (CRN) labels are
  full-horizon values of the route-saving base policy. They are kept forever
  in a persistent label store.
- **Refit.** Every rollout, the whole GNN and LSTM network is refit from
  scratch on the store with validation early stopping.
- **Decision rule.** The executed rule is the base policy plus λ times the
  lower confidence bound of the value ensemble's advantage over the base
  cell.
- **Gate.** λ and β are chosen by a safe-policy-improvement gate on
  exhaustively branched holdout episodes. Until the gate certifies an
  improvement, the base policy acts. Under a no-skill model, the gate
  certified 0.5% of splits (α = 10%).
- **Decision interface.** Every strategy carries unused capacity tokens
  forward instead of forfeiting them.
- **Scenario generality.** Scenario descriptors make the first decision
  conditional on panic susceptibility, population and hazard count.
  Per-family normalization and an orthogonal-array curriculum make one policy
  train and evaluate across cities × pedestrians × hazards × panic.

## 2. Issue-by-issue

| v25 issue (review §0) | v26 fix |
|---|---|
| 1. Training return cannot show improvement: scenario SD 2–3× any policy effect; 46% of variance from mask-driven token count; PPO-KL/100-episode gate unpassable | Progress is measured by exact CRN-paired quantities: the per-episode gain of the deployed rule over the base on its own states, and the gate's certified holdout gain. `backtest`/`multicity_backtest` audit FPI runs by `fpi_convergence` (stable certified gain across refits, non-negative bound, validation plateau). A `route_saving` benchmark strategy enables paired held-out evaluation against the base. Token catch-up removes schedule-driven token loss for every strategy. |
| 2. Actor moved but did not generalize (out-of-sample top-1 0.41 vs 0.40 prior) | The actor fit is retired. The decision rule acts on the value ensemble, the component that did generalize, and only when a held-out gate certifies it. |
| 3. Exact labels discarded after one update | A persistent, contract-bound label store with episode-level fit/validation/holdout splits. The whole network is refit on all of it every rollout (from scratch by default, the regime in which held-out gain grew with data). Labels can be produced by parallel collectors. |
| 4. Labels myopic (20 steps) and incomplete (6 of ~18 cells); value ensemble never acts | Full-horizon branches (`nmccPiBranchHorizon = 0`). Exhaustive branching at the first two decisions and at every decision of holdout episodes. Later decisions branch the base, executed and greedy cells, then optimistic cells, then random support. The ensemble acts through the gated rule. |
| Capacity parity (11/64 episodes lost tokens) | Token-bucket schedule with catch-up (`shelterDecisionCatchUp`) for all strategies and in every branch. Realized-capacity parity is reported by the matched-interface check. |

## 3. The estimator

**Labels.**

- **Definition.** At decision state `s`, for each branched cell `c`, the
  label is `Q_b(s, c)`: the simulator is restored to `s`, `c` is installed,
  and the base policy `b` (route saving) continues to the horizon on a CRN
  tape that is independent of the live future.
- **Why labels never expire.** `Q_b` does not depend on the learner. Only
  which states are visited does.
- **WAIT branch.** It supplies the natural-momentum outcome for the
  world-model auxiliary heads. Under catch-up it defers the token to the next
  scheduled epoch, exactly as before.
- **Contract.** Labels are bound to a digest of everything that makes two
  labels comparable: base policy, horizon rule, tapes, reward weights,
  decision schedule, masks, observation schema, capacity token and time
  step. A store refuses a different contract.

**Store and splits.**

- **Files.** Each episode is committed atomically as frames (`.frames.pt`),
  labels (`.labels.npz`) and a JSON record written last. Readers never see a
  partial episode.
- **Splits.** Holdout (default 20%), validation (10%) and fit are a hash of
  `(city, scenario seed)`, so no process can leak a holdout scenario into
  fitting.
- **Holdout episodes.** They branch every feasible cell at every decision and
  run the deployed rule without exploration.

**Normalization across scenarios.**

- **Why it is needed.** One 500-place shelter is worth five times more
  return at 5,000 pedestrians than at 25,000 (pinned by
  `test_one_fixed_capacity_shelter_is_worth_less_in_a_larger_population`).
  A pooled squared error would be dominated by small, severe scenarios.
- **What is normalized.** Within-state contrasts are divided by a family
  scale: the root mean within-state variance per family (city, P, H, panic),
  shrunk toward the global scale with a pseudo-count of 8.
- **Gate weighting.** The gate weights families equally.

**Refit.**

- **Loss.** A bootstrap-weighted within-state Huber loss in normalized units.
  Predictions and labels are both centered over the branched cells of the
  same state, so no cell-independent level (severity, natural momentum) can
  be learned as, or leak into, a cell value.
- **Auxiliary losses.** WAIT outcomes and per-cell physical effects, which
  shape the shared representation.
- **Architecture.** The full EvacPolicy: relational GNN, attentive and max
  pooling, episode LSTM over every frame, candidate pathway, and the
  5-member value ensemble.
- **Replay.** Episodes are replayed in lock step through packed graphs, and
  the replay reproduces the live forward pass. Bootstrap weights are a fixed
  function of the state, and a state always keeps at least one member.
- **Early stopping.** On the validation split, with patience 8 and at most
  60 epochs.

**Decision rule.**

- **The rule.** `a(s) = b(s)` if `λ = 0`, otherwise
  `argmax_c prior(s,c) + λ (μ(s,c) − β σ(s,c))`.
- **What μ and σ are.** The mean and spread across members of each member's
  value difference to the base cell. Differencing removes each member's
  unidentified per-state level, which would otherwise inflate σ.
- **Parameters.** `λ ∈ {0, 0.5, 2, 8}` and `β ∈ {0, 1}` are in normalized
  units, so one setting means the same trade-off in every family.

**Safe-policy-improvement gate.**

- **Candidates.** Every `(model ∈ {deployed, refit}, λ > 0, β)` is scored
  on the exhaustive holdout states.
- **Score.** The per-episode sum of exact paired gains `Q(s, a(s)) − Q(s, b(s))`
  in normalized units. Episodes are averaged within a family and families
  with equal weight.
- **Certification.** A candidate is certified if its lower bound is
  positive and no family with at least 3 holdout episodes is significantly
  harmed.
  - The lower bound is the minimum of an episode-cluster bootstrap and a
    cluster-robust Student-t bound.
  - It is Bonferroni-corrected over candidates, and at least 6 holdout
    episodes are required.
- **Selection.** The certified candidate with the largest bound is
  deployed. Otherwise the base policy is deployed on the refit weights.
- **What the score estimates.**
  - By the performance-difference lemma, the per-episode sum is exact for
    the first decision, whose state does not depend on the policy. The
    first-decision component is reported as `nmcc_fpi_gate_first_decision_gain`.
  - For later decisions, it is the conservative-policy-iteration surrogate
    over the deployed rule's state distribution. Its error is bounded by the
    state-distribution shift times the later-decision headroom, which the
    review measured as small (regret ≤ 0.03 after t = 1 versus 0.21 at t = 1
    on the testbed).
  - The claim of improvement is confirmed by paired held-out evaluation
    against the `route_saving` strategy.

## 4. Reward function check

The reward

`r = ΔS/P − 3ΔC/P − A/(PH) − E/(PH)`

is sound, and no defect was found. `tests/test_reward_contract.py` pins
three properties:

- **Ordering.** Per person, safe completion is never worse than remaining
  unfinished, and a casualty is always worst, for any timing and exposure.
  Weights that would violate this are refused.
- **Additivity.** Per-step scoring (branches) sums exactly to per-interval
  scoring (bridge), so labels and transitions measure the same objective.
- **Scale invariance.** The reward does not change when population and all
  counts are scaled together.

The one cross-scenario issue is not in the reward. The *value of a
fixed-capacity shelter* scales like capacity/population. The learner
therefore normalizes per scenario family and the gate weights families
equally. The objective itself is unchanged.

## 5. Working on any city, pedestrians, hazards and panic

**Scenario descriptors.** `scenarioDescriptors` appends three bounded
features, defined for any value, to the global features:

- panic susceptibility (the onset probability itself);
- `P/(P+10 000)`;
- `H/(H+3)`.

The first decision is taken before any panic onset is observable. Without
the panic feature, a policy pooled across panic levels could not condition
that decision on it. Population is otherwise visible only through
per-capita features.

**Curricula.** They sample the factorial design with a strength-2
orthogonal array: every pair of factor levels is trained together exactly
once per cycle.

- `scenario_general_…_v26` is OA(25, 5³) at the factorial levels:
  - P ∈ {5 000, 10 000, 15 000, 20 000, 25 000};
  - H ∈ {1, …, 5};
  - panic ∈ {0.1, 0.3, 0.5, 0.7, 0.9}.
- `…_v26_pilot` is OA(9, 3³) at pilot scale for one city.
- Cities come from the multicity schedule on a shared grid.

**Evaluation.** Evaluation now applies the curriculum's learner and
decision-interface contract to every strategy (`--evaluation-learner-contract`).
Without it, a v26 checkpoint is refused, and benchmarks would run without
catch-up. `--evaluation-scenarios curriculum` cycles paired evaluation
through the curriculum's scenario cells.

## 6. Validation performed

**Tests.**

- The torch-free suites all pass: 107 tests across 12 modules. The new ones
  are 27 for fitted policy iteration, 5 for the reward contract and 1 for
  the curricula.
- `flake8 --select=F` is clean. `pyright` shows no new errors against the
  v25 baseline apart from one numpy-stub false positive.

**Independent review.** A static review of the torch integration was
performed by a separate agent. It found eight defects, all fixed:

- a crash on minibatches with no differentiable loss;
- ensemble σ contaminated by unidentified per-member levels;
- collectors sharing random streams;
- catch-up combined with hybrid-NMCC credit;
- an unbounded device-memory replay cache;
- refits discarded when validation had no contrast;
- holdout episodes exploring, which made the gate more off-policy;
- evaluation run without the learner contract.

It also strengthened the torch tests: multi-decision episodes, unequal
replay lengths, and a real catch-up event.

**Gate operating characteristics** (`fpi_gate_validation.py`).

- Data: the v25 audit's 16 testbed episodes with every cell branched to the
  full horizon on 3 CRN tapes.
- Protocol: 400 random episode splits with 8 holdout episodes. The gate sees
  one tape. Truth is the two other tapes. Gains are per-episode normalized
  units, and the exhaustive-oracle headroom is 2.07.

| model | within-state ρ vs truth | certified | false certification | true gain when certified | λ = 8 without gate | λ = 8 harms |
|---|---:|---:|---:|---:|---:|---:|
| no skill | 0.00 | 0.5% | 0.5% | −0.55 | −1.02 | 97% of splits |
| truth + noise 0.25 | 0.94 | 66% | 0% | 2.06 | 2.02 | 0% |
| truth + noise 0.5 | 0.88 | 57% | 0% | 2.00 | 1.94 | 0% |
| truth + noise 1 | 0.77 | 40% | 0% | 1.85 | 1.74 | 0% |
| truth + noise 2 | 0.59 | 23% | 0% | 1.52 | 1.28 | 0% |
| replica ridge, 8 fit episodes | 0.29 | 0.5% | 0% | 0.78 | −0.17 | 60% of splits |

What the table shows:

- **Safe under the null.** The gate certifies a no-skill model in 0.5% of
  splits against a 10% nominal rate.
- **Never certified harm.** It never certified a rule whose true gain was
  negative, except in that one null split.
- **Power grows with model quality.** Certification rises with the model's
  within-state ranking quality.
- **The replica.** A weak model at this data size would have harmed in 60%
  of splits if deployed ungated. The gate kept the base policy.

The gate is deliberately conservative, and its power grows with the number
of holdout episodes. Parallel collection is how that number grows.

## 7. How to run

1. Run the torch tests in `rlevacuation`:
   `python -m unittest tests.test_fitted_policy_iteration_torch tests.test_fitted_policy_iteration tests.test_reward_contract tests.test_nmcc_pi_torch`,
   then the full suite.
2. Pilot the learner (one city):

   ```
   python multicity_backtest.py --train-only --launch-id fpi_v26_pilot --cities state_college_pa \
     --train-episodes-per-city 72 \
     --training-curriculum config/state_college_training_curriculum_nmcc_fpi_v26_pilot.json
   ```

3. Collectors, in parallel. Use the same curriculum and cities; they can
   start before the learner.

   ```
   python fpi_label_workers.py --launch-id fpi_v26_pilot --cities state_college_pa \
     --training-curriculum config/state_college_training_curriculum_nmcc_fpi_v26_pilot.json \
     --workers 6 --episodes-per-worker 12
   ```

4. Read progress from the training rows and `regional_policy.pt.fpi_gate.jsonl`:
   `nmcc_fpi_gate_lcb`, `nmcc_fpi_gate_mean_gain`, `nmcc_fpi_holdout_rank_corr`,
   and the online `nmcc_fpi_episode_greedy_gain`.
5. Evaluate on paired held-out seeds with `--strategies rl,heuristic,route_saving,…`
   and `--evaluation-scenarios curriculum`.

## 8. Cost

The label cost is mostly simulation.

- **Rate.** The v25 pilot ran about 0.28 s per simulator timestep at 2,500
  pedestrians.
- **Per episode at that scale.** A v26 training episode simulates about
  2,750 timesteps: two exhaustive decisions of ~18 cells plus WAIT to the
  full horizon, and three 8-cell decisions. That is about 13 minutes. A
  holdout episode simulates about 3,100 timesteps, about 15 minutes.
- **Scaling.** Cost scales roughly linearly with pedestrians.
- **Throughput.** Collectors scale throughput linearly with cores.
- **Refit time** is logged as `nmcc_fpi_refit_seconds`.

## 9. Compatibility

- Every new setting defaults to the historical contract: no catch-up, no
  descriptors, and v25 objectives unchanged. The GNN prior refactor is
  numerically identical.
- `MODEL_VERSION = 26` and the new signature fields (`decision_rule`,
  `token_rule`, `fitted_policy_iteration`, `scenario_descriptors`) refuse
  pre-26 checkpoints, as every prior version bump did.
- `--evaluation-learner-contract none` restores pre-v26 evaluation
  overrides.
- The policy-cache source fingerprint now includes the learner modules.

## 10. Limits and open items

- **Not executed.** The torch integration has not been executed.
- **Gate estimand.** The gate's later-decision term is a surrogate (see
  §3). Final claims need paired held-out evaluation against `route_saving`.
- **Capacity.** Catch-up removes schedule-driven token loss, but a token
  can still be uninstallable for one trajectory: every safe cell depleted,
  or operational benefit absent before the deadline relaxation. Parity is
  now reported. An ex-ante hazard-only viability proof remains on `TODO.md`.
- **Base policy.** The base stays fixed, so the fixed point is the rollout
  improvement of route saving. Relabelling with the learned rule as the new
  base, which would be a second policy-iteration generation, is not
  implemented.
