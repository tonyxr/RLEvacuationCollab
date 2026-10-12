# Engineering Review Log

## 2026-10-02 — Model v38 scalable sampled NMCC

### Diagnosis

- The corrected v37 pipeline was no longer hung, but its exact target still
  simulated `(20 feasible sites + WAIT) x 2 futures x 15 timesteps` at every
  decision. With eight installations, one episode executed up to 336 physical
  counterfactual branches. A real 5,000-person State College decision took
  157.3 seconds; PPO and continuation inference were not the bottleneck.

### Implementation

- Replaced the exhaustive action table with an exact sampled MPO E-step. The
  executed action is always included; the remaining branch budget combines
  policy-relevant and uniform-coverage actions. The production v38 curriculum
  uses two actions per state. Only physically evaluated actions are
  reweighted, while unobserved action probability remains exactly equal to the
  behavior policy. No heuristic score, learned fill, correction gate, or
  fallback action is used.
- Both branches now install equal-capacity shelters. One installed action is
  the common comparator, so its physical result replaces the separate WAIT
  branch. Each contrast still uses identical state, hazard/panic/casualty
  random tape, complete congestion and pedestrian dynamics, the full
  15-timestep decision interval, and the dedicated learned continuation value.
- Activated the existing `nmccPiExhaustiveDecisions` and
  `nmccPiMaxBranches` controls in the live collector, set the scalable defaults
  to zero exhaustive decisions and two sampled actions, updated model and run
  contracts to v38, and retained backward continuation learning across all
  eight factual installations.

### Measured result and verification

- On the identical 5,000-person State College first decision, physical NMCC
  time fell from 157.3 to 22.0 seconds (7.1x faster, 86.0% reduction) with two
  futures and four workers; continuation inference took 0.013 seconds.
- A complete 2,500-person, eight-installation action-stage episode finished in
  157.6 seconds including fresh process/map initialization; the simulator loop
  itself finished in 122 seconds and the population accounting invariant
  passed (`1,941 evacuated + 559 unfinished = 2,500`).
- Added an exact regression proving installed-site direct contrasts reproduce
  the corresponding action differences from a full WAIT-based table. The
  focused counterfactual, learner, curriculum, and GNN suite passes 137 tests;
  the complete repository suite passes 274 tests. A fresh v38 checkpoint is
  required because the actor-learning contract changed intentionally.

## 2026-10-01 — State College 2,500-person precursor launch

- Registered
  `config/state_college_training_curriculum_2500_random_centered_foresight_v37.json`
  with the same 24/48/88 E/A/F update budget, random-centred reward, two-tape
  decision-interval NMCC, stage-local schedules, density-adaptive cells, and
  `2,500 + 200` capacity contract as the current production design.
- Launched a fresh one-policy run at
  `runs/state_college_n2500_random_centered_foresight_v37_20261001/`. The
  launcher will train for 160 episodes, then deterministically evaluate RL and
  all ten comparator policies on eight matched held-out State College
  scenarios. Convergence is reported but does not block the requested full
  evaluation.
- Preflight confirmed the E/A/F totals, 270-person equal shelter tokens, and
  2,700 total capacity. Stage E completed all 24 episodes with its exploration
  and learning-rate endpoints reached exactly; stage A began with both
  schedules reset. The run was still active when this entry was written.

## 2026-10-01 — Random-site credit, surplus capacity, and stage-local schedules

### Design

- Replaced the learning reward's install-nothing baseline with the arithmetic
  mean over every feasible equal-capacity installation. The signed reward is
  `(mean feasible-site cost - chosen-site cost) / (population * horizon)`;
  common capacity value therefore cancels and a worse-than-random location is
  allowed to produce negative learning credit.
- Kept policy evaluation independent of that variance-reducing baseline.
  Frozen policies are compared using negative realised hazard-weighted time to
  safety: active/panicked person-time, danger-weighted exposure person-time,
  and capacity-valid terminal remaining-time debt. Casualty has no standalone
  coefficient, but does not disappear from time-to-safety accounting.
- Preserved NMCC's one-decision-interval physical branches and dedicated
  continuation value. Candidate advantages are now centred on the uniform
  feasible-site mean. WAIT remains only a shared common-noise control variate;
  heuristic actions and heuristic agreement do not enter a training target.

### Implementation

- Added `shelterCapacitySurplus` to configuration validation and set Seattle's
  ten equal tokens to 270/520/770/1,020 for populations
  2,500/5,000/7,500/10,000. The executor verifies zero unassigned rational
  population after the eighth dynamic installation; complete t=0 comparators
  are checked after initial routing.
- Population-matched runs now reject a detected candidate pool smaller than
  the complete ten-shelter budget and use the absolute `15,30,...,120`
  timetable. A missing candidate or a missed timestamp is a model-contract
  error rather than a silent delay, so all `population + 200` capacity is
  installed by `t = 120` or the run fails visibly.
- Added explicit stage coordinates to each curriculum episode. Exploration,
  entropy, temperature, and actor/critic cosine schedules reset inside every
  stage and reach their configured endpoints. Removed the NMCC M-step's hidden
  restoration of the unscheduled base actor learning rate.
- Added realised-objective accumulation and terminal residual reporting to
  `RLBridge`. Training CSVs retain the signed random-site reward trajectory;
  paired evaluation reads the separate action-count-invariant objective.
- Moved terminal objective evaluation ahead of Core's administrative
  `Unfinished` classification so live survivors contribute their true route or
  no-shelter residual instead of disappearing before the calculation.
- Final evaluation now cycles the variants from the curriculum's final stage,
  uses the same scenario and policy seeds for all strategies, and evaluates
  RL by deterministic masked argmax. Default comparison includes all registered
  dynamic and static heuristic arms.
- Confirmed the pedestrian hazard-speed path already recomputes from
  `maxSpeed` only when danger level changes; added an explicit regression test
  proving recovery from level 4 to level 1 without multiplicative carry-over.

### Compatibility and verification

- The reward, continuation-centering, capacity, and evaluation signatures are
  intentionally incompatible with the interrupted checkpoint. The next run
  must start from a fresh checkpoint; no fallback loading path was added.
- Python compilation and `git diff --check` pass. The focused model suite
  passes 187 tests, and the full repository suite passes 270 tests. New
  coverage includes signed random-site credit, stage start/end/reset values,
  Seattle `population + 200` capacity, the absolute decision timetable,
  safer-region speed recovery, and the realised evaluation objective.

## 2026-09-30 — Model v37 decision-interval NMCC and ten-shelter parity

### Design

- Replaced zero-step NMCC policy-improvement labels with paired physical
  branches that run from the decision state to the next installation epoch.
  Candidate and WAIT twins retain identical starting state and random hazard
  tape. Their score adds the difference of a dedicated learned continuation
  value at the two endpoints.
- Kept the approved environmental reward unchanged: it is still the exact
  zero-horizon installation-induced reduction in ordinary plus
  hazard-exposure remaining time. The longer NMCC branch is a learning target,
  not a reward redefinition.
- Defined the continuation target by a backward semi-Markov sweep over the
  complete sequence of eight realized installation rewards. This separates
  NMCC foresight from the ordinary PPO critic and from heuristic continuation.

### Implementation

- `CounterfactualBranch` accepts an endpoint-value callback and records the
  result without coupling the simulator brancher to Torch. `GNN.EvacPolicy`
  has a separately owned, zero-initialized NMCC continuation head.
- `RLBridge` evaluates branch endpoints with the same GNN/LSTM and a pure
  previous-to-endpoint momentum calculation, trains the continuation head in
  the action stage, and reports its loss, explained variance, target
  dispersion, and contribution to candidate contrasts.
- NMCC actor fitting now targets the masked actor distribution itself rather
  than the actor/uniform-exploration mixture. It uses the configured base actor
  learning rate, does not clip NMCC gradients, permits 64 fitting passes, and
  converges only at absolute `KL(q || pi) <= 0.01` with exact top-1 agreement.
- Replaced nine redundant momentum inputs with clearance, congestion relief,
  route-time relief, forecast-exposure relief, hazard-front growth, and
  hazardous-congestion relief. Added the physical congestion speed loss as a
  separate pedestrian cell feature.
- `initial_only` now predeploys the full common shelter target. Dynamic policies
  continue scheduled installation after early network clearance, and the
  backtest rejects any final shelter-count mismatch. Thus the registered setup
  is two initial plus eight sequential shelters for dynamic arms and all ten
  at `t = 0` for initial-only.
- Registered
  `config/state_college_training_curriculum_multiscale_env_action_v37.json`
  and documented the equations and diagnostics in
  `docs/DECISION_INTERVAL_NMCC_V37.md`.

### Compatibility and verification

- Incremented the checkpoint model signature to v37 because the pedestrian
  and momentum feature dimensions and auxiliary head changed. V36 checkpoints
  are intentionally incompatible; v37 training must start fresh.
- Python compilation and JSON validation pass for the changed model modules
  and v37 curriculum. The 112-test focused NMCC/branching/RL suite passes,
  including new tests for decision-interval continuation training, separate
  zero-horizon reward and decision-interval learning branchers, rejection of a
  mismatched horizon, exact eight-decision backward targets, six momentum-rate
  signs/dimensions, and initial-only t=0 budget selection. The complete suite
  passes 260 tests. The lightweight RL fixture was upgraded with an explicit
  routing contract rather than introducing a production fallback.

## 2026-09-30 — Objective-matched greedy benchmark and frozen-policy audit

### Design

- Added `objective_greedy`, a deliberately myopic comparator that minimizes
  the same quantity as the current actor target: remaining ordinary travel
  time plus forecast-danger-weighted remaining travel time for all active
  pedestrians. Every feasible candidate is compared from the identical live
  state with the same capacity, reservation, rerouting, route-cost, masking,
  and normalization rules used by the RL reward.
- The benchmark receives no physical future rollout, LSTM state, momentum
  history, learned continuation value, or later-action information. It tests
  whether RL adds useful foresight/context beyond direct one-step computation
  of its own objective; it is not an oracle for the full episode.

### Implementation

- Registered the policy in `DecisionInterface.py`, `RLBridge.py`, both
  backtest drivers, figure/report generators, and the benchmark protocol.
  `RLBridge` evaluates the full feasible action set through the existing exact
  zero-horizon counterfactual evaluator and asserts that scoring leaves the
  simulator state unchanged.
- Extended `evaluate_frozen_policy_postfix.py` with explicit strategy and
  output-stem arguments while preserving its historical default strategy set.
  Added a deterministic one-worker path for restricted environments.
- Added `analyze_objective_greedy_benchmark.py` to validate matched scenario
  identities, join frozen RL and greedy episodes, calculate paired effects and
  bootstrap intervals, compare action sequences, and generate the diagnostic
  figure and machine-readable tables.

### Results

- Sixteen matched held-out State College scenarios were evaluated: four seeds
  at each of 2,500, 5,000, 7,500, and 10,000 pedestrians. Mean full-sequence
  action agreement was 0.2031.
- Pooled RL-minus-greedy episode return was -0.10819 (bootstrap 95% interval
  [-0.15827, -0.05660]); RL won 2/16 scenarios. RL restricted mean
  time-to-safety was 3.40 minutes worse (advantage -3.3958, interval
  [-6.7519, -0.0276]); RL won 6/16.
- At 10,000 pedestrians, RL had 1.87 minutes lower mean RMTTS and won 3/4
  scenarios despite lower immediate-objective return and low action agreement.
  This is a useful hypothesis about congestion-aware continuation value, but
  its four-seed interval crosses zero and it does not identify a causal feature.
- Therefore frozen v36 does not generally beat or keep up with the direct
  myopic calculation of its own objective. Before claiming that it understands
  self-evolution, retained-state forward-ordering tests, more held-out seeds,
  and retrained temporal/forecast ablations are required.

### Compatibility and verification

- No reward, simulator, trained checkpoint, or RL action-selection behavior
  changed. Historical evaluation defaults remain unchanged.
- Unit coverage checks masked selection, exact ties, strategy dispatch, exact
  evaluator integration, and simulator non-mutation. A complete synthetic
  episode also finished with all deployments and no unfinished pedestrians.
- The first two-worker launch failed before any episode because the sandbox
  denied process-semaphore creation; its failed manifest is preserved. The
  sequential rerun completed all 16 episodes. Results and figure are under
  `runs/state_college_v36_complete_multiscale_nmcc_20260927/objective_greedy_analysis_20260930/`.

## 2026-09-29 — Independent review of the frozen v36 policy (analysis only)

Mode: `analysis`. No model, simulator, configuration or run artifact was changed.
Artifacts: `runs/state_college_v36_complete_multiscale_nmcc_20260927/independent_review_20260929/`
(`INDEPENDENT_REVIEW.md`, figures R01-R11, `v36_independent_review.py`, which needs only numpy, pandas and matplotlib).

### Verified

- The source fingerprint on disk equals training for all 32 dynamics files.
- Evaluation is argmax-deterministic.
- Training and evaluation seeds and initial digests are disjoint.
- The CRN initial digest is shared by all 7 strategies in every scenario.
- The `EXPERIMENT_REPORT.md` reward and unfinished figures reproduce exactly.

### Findings

- Learning is concentrated in action updates 1-12:
  - reward +0.0173 per update (SE 0.0049);
  - RMTTS -0.84 min per update (SE 0.27).
- Updates 13-24 are flat:
  - reward +0.0023 per update (SE 0.0055);
  - RMTTS +0.10 min per update (SE 0.26).
- Top-1 agreement with the zero-horizon label best reaches about 0.35-0.45 (chance 0.07).
- Held-out results, pooled over 16 scenarios, primary endpoint RMTTS:

  | RL vs | RMTTS difference | Exact p | Holm p (5 benchmarks) |
  |---|---|---|---|
  | risk-reduction | -6.0 min | 0.006 | 0.025 |
  | random | -7.7 min | 0.002 | 0.012 |
  | hazard-weighted | -4.0 min | 0.066 | 0.16 |
  | max-active | -4.2 min | 0.053 | 0.16 |

  - No reward difference against an engineered benchmark is distinguishable from zero.
  - No population-by-policy trend is significant (permutation p 0.13-0.87).
- Reward is a within-scenario proxy for RMTTS (Spearman -0.76).
  - It is weak for unfinished share at 2,500 (+0.02).
  - It is unrelated to casualties (+0.08).
- Correction: `congestion_person_minutes` is the congestion model's person-minute denominator (r = 0.991 with RMTTS), not congested minutes.
  - Congested minutes = denominator x `congested_person_minute_share`.
  - The earlier r ~= 0.79 reproduces (0.793); on true congested minutes it is 0.72.
- The actor's NMCC target is purely myopic (horizon 0, no continuation value). The next measurement should be held-out target attainment (label regret) of the frozen policy.
- Accessibility-deficit duplicates max-active-population (15/16 identical sequences).
- Reproducibility: the v36 code is uncommitted. The manifest `policy_design` and `learning_assessment.json` are stale relative to the checkpoint signature.

## 2026-09-28 — Session record: v36 objective, learning redesign, multiscale result, and next cycle

### User-directed model contract established during the session

- The shelter policy's primary objective was reduced to one interpretable
  principle: decrease the total expected future hazard-weighted time-to-safety
  of pedestrians still active in the network. The corresponding decision
  reward is the positive reduction in remaining ordinary travel time plus
  hazard exposure produced by installing a shelter relative to the same-state
  `WAIT` condition, normalized by initial population and episode horizon.
- Standalone casualty reward, panic reward, and terminal-residual reward were
  removed from the optimized objective. Casualty and panic remain required
  evaluation and graph outputs. An active pedestrian without a valid shelter
  receives the configured horizon-valued remaining-time cost; an assigned
  pedestrian changes destination only when the new valid reservation is
  strictly better.
- Routing and motion semantics were clarified: after every installation all
  active non-panicked pedestrians recheck whether the new shelter is a better
  valid destination; otherwise they retain the current shelter and shortest
  route. Danger-level speed reduction is relative to maximum speed and changes
  with danger level rather than compounding every timestep.
- The user rejected the earlier correction gate, fallback/base-placement
  selector, safeguard-heavy controller, and engineered-prior behavior. The
  accepted learning structure is environment-dynamics pretraining followed by
  reinforcement/action learning and NMCC policy improvement, with ordinary
  positive and negative action evidence, scheduled stochastic exploration in
  training, deterministic ranking in evaluation, and no post-policy action
  replacement. Physical feasibility and hazard safety remain MDP constraints,
  not policy fallbacks.
- Actor learning rate and exploration were changed to smooth decaying
  schedules with materially larger early values. Exploration samples from the
  feasible categorical policy during training and approaches a low plateau;
  evaluation selects the highest feasible learned score.
- The experiment contract was fixed across RL and dynamic benchmarks: two
  initial shelters, eight sequential installations at minutes 15, 30, 45, 60,
  75, 90, 105, and 120, a 180-minute horizon, and a 60-minute post-installation
  tail. Every shelter has capacity `N/10`, giving total capacity `N`, for
  populations 2,500, 5,000, 7,500, and 10,000. The initial-only comparator is
  the documented exception with two shelters and no sequential deployment.

### v36 implementation and learning evidence reviewed in this session

- The active representation is a relational regional GNN with spatial and
  current assignment-flow edges, population-magnitude-preserving route
  aggregation, candidate-conditioned attention over the whole graph, global
  load/scenario features, and episode LSTM/momentum context. Added operational
  features include assigned/unassigned population, outgoing/incoming
  assignment flow, mean route time, long-route share, shelter capacity and
  utilization, hazard forecast and source proximity, population/network load,
  and deployment progress.
- The registered v36 experiment is
  `runs/state_college_v36_complete_multiscale_nmcc_20260927/`. It contains 120
  balanced training episodes: 24 environment-dynamics episodes and 96 action-
  learning episodes, 30 total episodes at each population. Each four-episode
  action rollout contains one episode from every population; optimizer
  diagnostics are written on the fourth (10,000-person) row but the update is
  not trained on that population alone.
- Environment prediction loss fell from `0.05511` to `0.01042` (81.1%). All
  24 action updates were accepted, with no rollback or rejection. Mean action
  KL was `0.04035`, final KL was `0.04891`, full feasible-action NMCC coverage
  was 100%, early-to-late selection total-variation distance was `0.2344`, and
  mean rollout reward moved from `0.5920` to `0.7274`. These measurements show
  material policy movement; the legacy 0.015-KL convergence flag is not the
  applicable success criterion for this deliberately larger-update run.
- Frozen evaluation used four matched seeds at each population for RL,
  risk-reduction, active-population heuristic, hazard-weighted demand,
  accessibility deficit, random, and initial-only policies (112 episodes).
  Overall equal-scenario RL reward was `0.733337`, safe-completion share
  `0.849983`, and unfinished share `0.146998`. RL beat every dynamic benchmark
  on reward in 13 of 16 matched population/seed scenarios.
- Population-specific RL reward versus the best dynamic benchmark was:
  `0.6666` versus `0.7240` at 2,500; `0.7272` versus `0.7339` at 5,000;
  `0.7526` versus `0.7164` at 7,500; and `0.7869` versus `0.7059` at 10,000.
  The corresponding RL advantage therefore crossed from `-0.0574` to
  `-0.0068`, `+0.0362`, and `+0.0810`. At 10,000, RL reduced unfinished share
  by 7.9 points versus risk reduction, 10.3 versus hazard weighted, and 10.5
  versus heuristic/accessibility.
- The high-demand advantage was not produced by rerouting the most people. At
  10,000, RL rerouted about 19.5% of the population per decision versus 21.6%
  for risk reduction, but removed about 90.9 hazard-weighted minutes per
  rerouted pedestrian versus 69.1. Across the 16 matched scenarios, RL reward
  advantage over the mean dynamic benchmark was descriptively correlated
  (`r ~= 0.79`) with congestion-person-time reduction. This is a diagnostic
  association, not a causal feature attribution.

### Exact information advantage and limitations identified

- RL can condition candidate ranking on regional active/assigned/unassigned
  population, outgoing/incoming shelter-assignment flow, observed mobility
  delay, mean route time, long-route share, wellness state, current/forecast
  hazard, infrastructure/capacity state, global network load, wind/hazard
  spread, installation budget, momentum, spatial adjacency, route-flow
  topology, and recurrent history. Every engineered benchmark uses a much
  smaller fixed local formula and has no learned temporal or graph-wide
  interaction.
- This does not mean every available feature caused the observed advantage.
  No grouped feature ablation or retrained representation ablation has yet
  established actual policy dependence.
- The policy does not directly observe the logged link-density diagnostics.
  Regional `mobility_delay_fraction` combines physical congestion with other
  speed effects. `network_load_share` is an initial population-to-link-storage
  descriptor created once, so its nominal reduction-velocity momentum feature
  is zero. `population_network_density_fraction` is likewise a static scale
  descriptor.
- The exact immediate candidate evaluator uses each pedestrian's current speed
  to convert every hypothetical route distance to time. It therefore gives
  extra value to shortening the route of a currently slow pedestrian, but it
  does not recompute candidate-induced future link density, queues, or speed.
  The learned congestion benefit is consequently an emergent correlation from
  current slowdown, routes, flows, capacity, subsequent states, and recurrent
  training—not an exact per-candidate congestion forecast.
- Four evaluation seeds in one city are sufficient for mechanism diagnosis,
  not a general superiority or scale law. The 7,500 result includes one clear
  RL outlier, and its best reward did not imply the best completion share
  because the optimized quantity is hazard-weighted remaining time rather than
  terminal completion alone.

### Decision for the next development/training cycle

- Do not launch another unchanged long v36 run. First run a frozen-checkpoint
  grouped feature-use audit, then minimal retrained ablations for route flow,
  congestion/load, and temporal/attention context.
- Make physical congestion separately observable from generic speed using
  current regional congestion-speed, congested-population, and occupied-link
  load summaries. Split the existing static scenario-load descriptor from a
  genuinely dynamic load feature and remove the identically-zero momentum
  derivative. These are current-state operational observations, not future
  candidate information.
- Before changing credit, compare the present zero-horizon candidate ordering
  with equal-capacity, common-noise physical candidate branches advanced
  through one 15-minute decision interval. Adopt the congestion-aware target
  only if it improves held-out action ordering and regret at matched states.
- After those P0 measurements, run one fresh balanced engineering policy with
  32 environment episodes and 128 action episodes, followed by at least 12 new
  matched evaluation seeds per population. Only a passing engineering result
  should advance to at least three independent policy seeds and 20 untouched
  matched scenario seeds per population.
- No correction gate, conservative fallback, actor/base heuristic prior,
  perfect-information benchmark, standalone casualty reward, or terminal-
  residual reward is authorized by this plan. Detailed acceptance criteria are
  recorded in the new active section at the top of `TODO.md`.

### Change and verification boundary

- This entry and the corresponding `TODO.md` section are documentation-only.
  No simulator, policy, reward, training, configuration, or test code was
  changed, and no new training process was started.

## 2026-09-22 — Model v26 persistent fitted NMCC and wide-and-deep GNN control

### Root-cause fixes

- Replaced the v25 one-rollout actor fit with checkpointed fitted policy
  iteration over a bounded persistent exact-branch dataset. Complete episodes,
  including recurrent observation histories, exact candidate masks, paired
  advantages, bootstrap membership, and fixed base actions, survive optimizer
  updates and checkpoint resume. Training/validation assignment is
  deterministic at the episode level, preventing recurrent-prefix and
  scenario-tape leakage.
- The first two shelter decisions now branch every feasible cell through the
  true physical episode terminal. Later decisions retain bounded cellular-
  automata branches. This moves computation to the decisions where the v25
  audit found most policy regret and removes the early 20-step label censoring.
- The deployed controller no longer uses the oscillating PPO/listwise actor.
  It starts from the interpretable route-time-saving policy and activates the
  conservative intervention-value ensemble only after three consecutive
  episode-heldout paired-gain lower bounds are positive. Validation evaluates
  the exact deployed score and compares its selected branch with the fixed
  base-policy branch; PPO KL and noisy unpaired training-return stationarity
  are not used as fitted-policy convergence criteria.
- Ensemble uncertainty is computed on candidate-minus-base paired values.
  This removes arbitrary state/head offsets that are unidentified by
  within-state labels. Once the gate opens, the base prior supplies only a
  common score level and the conservative physical advantage supplies the
  ordering, so route saving is not counted twice and a perfect fit recovers
  exact branch-return ordering.
- System identification and control fitting now have explicit ownership. The
  relational road/route GNN and episode LSTM are trained by natural dynamics,
  causal outcomes, and factorized critic losses, then frozen during each
  growing-dataset control refit. The intervention ensemble is wide-and-deep:
  a direct linear path over normalized pedestrian, hazard, infrastructure,
  candidate, scenario, and momentum features reduces small-sample variance,
  while a parallel nonlinear path consumes the full GNN/LSTM embeddings.
- Temporarily empty hard-safety masks now defer and retry a deployment; the
  action interval begins at the actual installation time. The hard hazard mask
  is never relaxed. The general backtest also fails closed when matched arms
  differ in realized deployment count or total shelter capacity.

### Reward and scenario contract

- Audited the reward end to end and made its weights a single source of truth
  for exact branches, the GNN outcome-to-component map, actor/critic targets,
  full-episode evaluation, and checkpoint provenance. The objective remains
  linear and unclipped: normalized safe-completion gain minus coefficient-three
  casualties, active person-time, and hazard-exposure person-time. Since both
  time weights sum to two, casualty weight three remains strictly worse than
  the maximum artificial time saving caused by removing a person from the
  active population. No site bonus, state-occupancy reward, or action penalty
  contaminates the scientific objective; feasibility remains in masks.
- Added normalized scenario inputs for population relative to road-network
  size, hazard-instance count, configured panic susceptibility, and dynamic
  capacity coverage. They join flexible regional physical, hazard, pedestrian,
  candidate, route-flow, and momentum tensors rather than city identifiers.
- Added `config/scenario_general_training_curriculum_nmcc_v26.json` with one
  immutable learner contract and 120 episodes per city. Its factor-coverage
  stage contains the complete 3 x 3 x 3 population (1,000/2,500/5,000), hazard
  count (1/3/5), and panic (0.1/0.5/0.9) factorial twice per city; every stage
  ends on a complete five-city pooled PPO rollout boundary.
- Registered the exact route-saving continuation rule as an ordinary dynamic
  benchmark so branch generation, controller fallback, and held-out evaluation
  use one implementation. Model signature architecture is now
  `resolution_flexible_relational_route_gnn_lstm_nmcc_v10` and records the
  physical-advantage target, paired uncertainty reference, deployed gate score,
  and frozen-system wide/deep refit contract.

### Verification and bounded backtest

- At 24 v26 scenarios the learned selector already improved episode-heldout
  top-1 from the route prior's 0.40 to 0.533 and mean exact paired gain to
  `+0.01482`, but its five-episode lower bound was `-0.00798`; the three-update
  gate correctly remained closed. This is the intended safe small-data
  behavior, not convergence.
- Continued the same checkpoint on 24 new training seeds without exposing the
  twelve evaluation seeds. The lower bound became positive at episode 31 and
  passed the registered three-consecutive-update gate at episode 33. It stayed
  positive after the controller changed the visited state distribution and
  ended at `+0.009815` on 48 retained labelled episodes.
- Final deterministic evaluation on twelve untouched common-random-number
  scenarios passed all 10 engineering gates. Mean full objective return was
  `0.29101` for v26, `0.25032` for the fixed route-saving base, and `0.24156`
  for v25: v26 differences `+0.04069` and `+0.04946`, respectively. Mean
  casualties were 0.583 for v26 versus 0.833 for route saving; mean safe
  completions were 92.08 versus 90.83. Every matched arm installed identical
  shelter count and capacity. Per-seed return differences remain variable, so
  this is an engineering learning-path result, not a State College or policy-
  superiority claim.
- Artifacts: `runs/nmcc_value_backtest_v26.json`, its four-panel PNG, and
  `runs/nmcc_value_backtest_v26_artifacts/` (v25/v26 checkpoints and optimizer
  diagnostics). The continuation runner recovers completed optimizer rows
  after an interrupted report without repeating training or relabeling a
  post-action return as the full-episode objective.
- Full repository verification passed: 266 tests, zero failures, zero errors,
  and zero skips. Focused coverage includes exact physical-return ordering,
  offset-invariant paired uncertainty, full-horizon early branches, persistent
  replay, route benchmark dispatch, scenario-factorial scheduling, deployment
  retry, reward decomposition, checkpoint ownership, and capacity fail-close.

### Compatibility and remaining boundary

- Model v26 requires a fresh checkpoint relative to v25. The architecture,
  global feature width, transition/replay schema, intervention heads, action
  gate, and inference signature changed; incompatible checkpoints fail closed.
- No new State College training was started in this change. Before a
  confirmatory map-backed comparison, the existing P0 hazard-only capacity-
  viability preflight must either prove a common safe full-token schedule or
  reject/resample the scenario. The synthetic backtest made safety permissive
  specifically to test learning under exact capacity parity.

## 2026-09-22 — Completed 2,500-person State College v25 staged pilot

### Execution

- Ran the registered 64-episode State College curriculum from a fresh v25
  checkpoint with 2,500 individual pedestrians, three stochastic hazards,
  four episodes per optimizer update, five nominal 500-person capacity tokens,
  and the 2/3/2 natural/causal/controller rollout schedule followed by joint
  optimization. The map-backed run completed normally in about 6 hours 40
  minutes and finalized a 5.5 MB checkpoint.
- Before launch, the complete 255-test suite passed and map preflight confirmed
  a cached 6,973-node/25,804-edge consolidated State College walking network
  with 2,375 detected shelter candidates.
- All 64 population ledgers balanced, all 16 optimizer rollouts completed, and
  maximum absolute reward-accounting error was `1.81e-16`.

### Learning and credit-assignment findings

- Phase order was exactly 0/1/2/3. The actor remained frozen for both natural
  and causal pretraining (zero actor updates and gradients), then completed 11
  accepted controller updates and 132 actor optimizer steps with no rejected
  update or representation rollback.
- Natural loss fell from `0.04220` to `0.02090` (−50.5%). Dueling/intervention
  loss fell from `0.10576` at its first causal update to `0.01476` (−86.0%).
  Exact-target fit KL improved on every actor update; its update mean fell from
  `0.27610` before fitting to `0.20198` after fitting.
- Exact-branch top-1 improved on 7 of 11 actor updates and worsened on 4. Its
  update mean increased from `0.40833` to `0.48776` (+0.07943). This confirms a
  usable action-differential optimization signal, but not monotone decision
  convergence. The causal-head rank gate never passed and model-filled labels
  remained disabled, as registered.
- Exploration decayed from `0.35` to `0.03`; actor learning rate decayed from
  `5e-4` to `1.3724e-4`, and critic learning rate from `3e-4` to its `7.5e-5`
  floor. Actor and critic gradients stayed finite and nonzero after activation.

### Outcome, convergence, and fairness findings

- Mean training safe completions were `1921.45/2500`. Casualties were nonzero
  in 23/64 episodes (mean `4.72`, median `0`, maximum `145`), so the calibrated
  casualty term is material on severe scenarios rather than identically zero.
- Training return did **not** converge. Full-run slope was
  `-0.000726/episode`; actor-period slope was `-0.000207/episode`. Last-eight
  minus first-eight mean return was `+0.02744`, but its 95% episode-bootstrap
  interval was `[-0.11754, 0.17614]`. The registered 100-episode stationarity
  minimum was also not met. This run therefore supports learning-mechanics and
  credit-signal claims only, not return convergence or heuristic superiority.
- A production fairness failure was found: the budget was five equal tokens,
  but hard safety/feasibility masks permitted all five installations in only
  53/64 episodes. Realized installations averaged `4.625` tokens (2,312.5
  places) and ranged from 1 to 5. No heuristic was run in this train-only
  launch, so equal *realized* RL/heuristic capacity is not established. The
  fixed-token contract is necessary but insufficient when no safe action is
  available; a hazard-aware capacity-viability/deadline gate is required before
  confirmatory held-out evaluation.

### Analysis artifacts and compatibility

- Added `analyze_nmcc_score_training.py`, a v25-native audit and six-panel plot
  generator. It reports phase invariants, exact top-1, target-fit KL, staged
  losses, gradients, schedules, reward accounting, outcomes, and realized
  deployment capacity. This is intentionally separate from the legacy v22
  `analyze_staged_nmcc_training.py`, which correctly rejects non-8/8/16/32
  schedules.
- Evidence is under `runs/state_college_2500_nmcc_score_v25/`, including
  `regional_policy.pt`, both training ledgers, the generic convergence plots,
  `nmcc_score_training_audit.json`, and
  `nmcc_score_training_diagnostics.png`.
- The analyzer and documentation do not change model version, policy weights,
  optimizer state, simulator dynamics, or checkpoint compatibility.

## 2026-09-21 — Model v25: staged factored NMCC and masked candidate-score control

### Design and implementation

- Advanced the checkpoint schema to model version 25. The deployed action is
  now explicitly the highest actor score among currently feasible regional
  shelter candidates. During training only, a scheduled epsilon-greedy
  behavior mixture supplies broad early coverage and decays after controller
  updates; evaluation remains deterministic argmax.
- Completed the NMCC physical target. Every common-random-number cellular-
  automata branch now records scalar return plus normalized safe completions,
  casualties, active person-time, hazard-exposure person-time, terminal active
  population, and terminal risk mass. Candidate effects are paired against the
  same-tape `WAIT` outcome and supervise every exactly branched candidate, not
  only the factual action.
- Made system-first learning compatible with policy improvement. Natural
  pretraining updates only the action-independent outcome model; causal
  pretraining adds the all-candidate effect ensemble and intervention-value
  head; controller warm-up then enables the factorized critic and actor; joint
  optimization continues all heads. The actor cannot change during the first
  two phases.
- Added the `score_ranking` objective. It uses a listwise target and explicit
  best-versus-rival margin only within candidates evaluated from the same
  snapshot, so unbranched cells are not mislabeled as poor. The v24
  KL-constrained target remains available as a checkpoint-incompatible
  ablation.
- Added independent actor/critic warm-up and cosine learning-rate schedules.
  Schedule age does not advance while the actor is frozen. Added explicit
  epsilon start/end settings and diagnostics for the live exploration rate.
- Extended the action mask with configurable forecast danger, hazard safety,
  reroutable population, route-time saving, positive site capacity, and
  remaining-token constraints. A deployment-deadline rule relaxes only the
  optional benefit filter when delaying again would make the equal-capacity
  token schedule impossible; safety and physical feasibility remain hard.
- Kept the existing cellular-automata dynamics as the branch engine and bounded
  branch computation with `nmccPiBranchHorizon` and `nmccPiMaxBranches`.
  Learned model fill remains gated and disabled in the registered pilot.
- Registered
  `config/state_college_training_curriculum_2500_nmcc_score_v25.json`: 64
  episodes, four episodes per optimizer rollout, 2/3/2 natural/causal/
  controller rollout phases, 20-timestep CA branches, six exact candidates per
  decision, cosine learning rates, and 0.35 -> 0.03 score exploration.
- Replaced the obsolete v23 backtest implementation in
  `nmcc_learning_backtest.py` with a matched v24-v25 harness. It enforces equal
  installed capacity before comparing returns and generates JSON, checkpoint,
  diagnostic CSV, and PNG evidence.

### Verification and backtest

- Torch-free NMCC and curriculum suites: 25 tests passed.
- PyTorch NMCC-PI suite in the project environment: 14 tests passed, including
  staged phase/schedule compatibility and a complete exact-branch actor update.
- RL framework and NMCC integration regression identified and fixed two issues:
  mask monkey-patch compatibility and accidental use of epsilon-greedy
  likelihoods by legacy PPO. The focused regressions pass; the complete suites
  also pass.
- Final release gate: `python -m unittest discover -s tests` completed all 255
  tests successfully in the project simulator environment, including the hard
  safety-mask/deployment-deadline regression.
- Final matched engineering backtest: 12 training episodes per arm and eight
  deterministic held-out scenario seeds, population 120, three additional
  equal 40-person shelter tokens, eight exact candidates, and 12-timestep CA
  branches. All eight engineering gates passed. Branch-best top-1 increased
  from 0.333 to 0.667. Mean held-out return was 0.24418 for v25 versus 0.21543
  for v24 (+0.02876); mean casualties were 1.125 versus 1.25 and mean safe
  completions were 90.5 versus 88.875. Per-seed return differences remain
  variable, so this is evidence that the learning path works, not a
  convergence or State College efficacy claim.
- Evidence: `runs/nmcc_score_backtest_v25.json`,
  `runs/nmcc_score_backtest_v25.png`, and the corresponding artifact folder.

### Compatibility and next step

- v25 checkpoints are intentionally incompatible with v24 because transition
  records, model signature, action-behavior contract, and optimizer schedules
  changed. Historical v24 curricula explicitly retain `kl_target` and constant
  learning rates.
- The next authorized experiment is the 2,500-person State College v25 pilot.
  It must start from a fresh checkpoint and pass deterministic held-out,
  equal-capacity, multi-seed gates before any convergence claim.

## 2026-09-02 — Repository-wide code review

### Review baseline

- Git commit reviewed: `2fa2b51` (`main`, "updated reward").
- Scope: all 22 top-level Python modules, `RLEvacuationParameter.csv`, `requirements.txt`, and the checked-in progress logs.
- Existing user changes at review start: `.DS_Store` and `runs/.DS_Store` only. They were not modified by this review.
- Application source was reviewed but not changed. Follow-up work is tracked in `TODO.md`.

### Current execution flow

`script.py` seeds one experiment launch, trains the RL strategy over multiple replications, evaluates four deployment strategies, aggregates CSV metrics, generates plots, and exports the run artifacts. Each replication creates a `Core`, which prepares OpenStreetMap data, divides the map into cells, creates hazards, shelters, and pedestrians, runs the timestep loop, invokes `RLBridge`, and writes `progress.csv` through `TrainingLogger`.

### Verification performed

- Parsed every top-level Python file with Python's AST parser: all 22 files passed syntax parsing.
- Ran `pyflakes`: no undefined-name errors were reported; it reported unused imports/variables and one shadowed name.
- Ran a policy forward-pass smoke check: a four-cell input produced shelter logits shaped `(1, 5)` and a value shaped `(1,)`.
- Probed `RewardProcessor.rewardMode` with sentinel values and confirmed that its full-reward arguments are shifted.
- Inspected the two checked-in progress logs. One contains a single legacy-format data row; the newer launch log contains only a header. These files are not evidence of a complete end-to-end run.
- A full `script.py` run was not performed because it launches 68 map-backed simulation replications, can access external OSM services, and writes substantial run output.
- `ruff` was unavailable in the current environment; `pyflakes` was used for the available static check.

### Findings

#### R-01 — Critical: full-reward arguments are mapped to the wrong parameters

`RewardProcessor.rewardMode` calls `fullReward` positionally (`RewardProcessor.py:148-165`) but omits `immediateReroutedCount` from the argument sequence. The resulting mapping is:

- `hazardExposureDelta` becomes `immediateReroutedCount`;
- `strandedCount` becomes `hazardExposureDelta`;
- `t` becomes `strandedCount`;
- `maxEpisodeSteps` becomes `t`;
- the real `maxEpisodeSteps` remains its default of 120.

`RLBridge.step` does not pass `maxEpisodeSteps` (`RLBridge.py:507-523`), so every full-reward call currently receives `t=120` and `maxEpisodeSteps=120`. The terminal bonus is therefore applied on every timestep, hazard exposure is derived from the active-population count, and the real timestep is treated as stranded population. This invalidates the reward signal and any training or comparison based on it.

#### R-02 — High: PPO minibatch sizing is inverted

`RLBridge.end_episode` sets `mb = max(1, T // self.minibatch_size)` and then uses `mb` as the slice width (`RLBridge.py:626-640`). With the current 239-transition episode and configured minibatch size of 64, this creates slices of three samples, or roughly 80 optimizer updates per epoch, rather than four batches of at most 64 samples. For shorter episodes it can degrade to one-sample updates. This materially changes training dynamics and cost.

The permutation is also created once before all epochs (`RLBridge.py:627`) instead of being refreshed each epoch.

#### R-03 — High: the actor cannot condition a cell action on that cell's local state

`EvacPolicy` hard-codes both graph processing and attention off (`GNN.py:111-123`). Each cell is encoded independently, but the encodings are mean-pooled into one global vector (`GNN.py:237-257`) before the actor emits all cell logits (`GNN.py:259-266`). The output layer can learn a static preference for cell indices, but it cannot compare the current hazard, demand, or shelter pressure of one cell against another. That conflicts with the cell-criticality reward objective.

Setting `EVAC_ENABLE_PYG=1` only imports PyG; it does not enable `self.use_pyg`. The `force_mlp` constructor argument is also ignored.

#### R-04 — High: the initial-only shelter baseline is structurally disadvantaged

Pedestrians are initialized with routes to random map nodes (`PedestrianDatabase.py:419-431`), not to the shelters deployed before the simulation. Rerouting toward shelters is triggered when a *new* shelter is installed (`RLBridge.py:465-480`). The `initial_only` strategy never installs a new shelter after initialization (`RLBridge.py:435-440`), so its pedestrians are not deliberately routed to its shelters. Shelter utilization can therefore remain near zero for reasons unrelated to placement quality, making the strategy comparison unreliable.

Relatedly, an affected pedestrian reaching its original random destination is counted as evacuated (`PedestrianDatabase.py:631-637`), even when that destination is not a shelter.

#### R-05 — High: configured hazard behavior is disconnected from pedestrian outcomes

The CSV supplies casualty, spread, and speed-reduction distributions. `HazardDS` samples them (`HazardDatabase.py:74-77`), but the default deterministic spread path ignores the sampled spread rate, and pedestrian casualty/speed logic uses fixed tables instead (`PedestrianDatabase.py:52-55`, `PedestrianDatabase.py:557-565`). The sampled hazard casualty and speed-reduction values are not consumed by pedestrian interactions.

`CellTracker.cellUpdate` applies heat/smoke force effects only after pedestrians have moved (`Core.py:516-531`). On the next timestep, `pedestrianHazardInteraction` resets speed from `desired_speed` before movement, so the force-adjusted speed does not persist into motion.

Under the current heat/smoke formulas and normalizations, a state-5 source cell produces a danger value of about 0.214, below the first default casualty threshold of 0.40. Because multiple hazards overwrite cell heat/smoke rather than aggregate them, the default `danger_level` casualty mode is expected to produce no hazard casualties. Hazard lifetimes also have no effect because `terminateHazard` is never called by `Core`.

#### R-06 — High: experimental comparisons are not reproducibly paired or isolated

`script.py` selects one random launch seed and then lets RNG state advance across every training and evaluation replication (`script.py:518-547`). RL, initial-only, random, and heuristic strategies therefore see different pedestrian origins, destinations, and hazard sources. Strategy differences are confounded with scenario differences instead of using common random numbers.

Policies are automatically loaded and saved at `runs/<address>/policy.pt` (`RLBridge.py:114-123`, `RLBridge.py:717-721`), outside the unique launch folder. A new launch can silently resume a policy from an earlier launch, so the printed launch seed is insufficient to reproduce training.

#### R-07 — Medium: episode finalization and reported summaries disagree

After the final logged timestep, all active pedestrians are forcibly classified as `Arrival` (`Core.py:571-576`). No final CSV row is written after that classification, while `_episode_summary` reads the last CSV row (`script.py:419-436`). Console totals and exported summary tables can therefore describe different populations. Treating unfinished pedestrians as successful arrivals also hides a meaningful `stranded` outcome.

#### R-08 — Medium: route and population invariants are not enforced

`MapDS.shortestPath` appends an edge even if neither direction is found, allowing `None` into a route (`MapDatabase.py:418-431`). `PedDS.advanceFromNode` then dereferences that edge without checking it. During initialization, pedestrians whose routes fail after six attempts are silently skipped (`PedestrianDatabase.py:424-433`), but later result checks still compare totals with the configured `pedVol`. A disconnected or incomplete graph can therefore cause a crash or an unexplained population deficit.

#### R-09 — Medium: runtime paths and dependencies are incomplete

`Core` reads `RLEvacuationParameter.csv` and writes `runs/` relative to the process working directory (`Core.py:203-208`, `Core.py:433-437`), while `script.py` aggregates results relative to its own file location. Launching the script outside the repository root can split outputs or fail to find the configuration.

`script.py` and `TrainingLogger.py` require Matplotlib for several paths, but `requirements.txt` does not list it. `script.py` tolerates its absence in some plotting functions, but `_plot_training_convergence` calls `plt.subplots` without a guard (`script.py:441-469`). Dependency versions are unpinned.

#### R-10 — Low: dormant features and repository artifacts obscure the active model

Guidance classes exist, but `Core` never creates a `GuidanceDS`; `optimize_guidance` only controls configured counts, and the logged `guided` metric defaults to zero. Several reward inputs and state fields are accepted but unused. Static checking also found many unused imports/variables.

There is no `.gitignore`, while bytecode, OSM caches, run outputs, and `.DS_Store` files are tracked. This makes review and experiment provenance harder to maintain.

### Review conclusion

The code is syntactically valid and its isolated neural-network forward path runs, but the current training outputs should not be treated as scientifically reliable until R-01 through R-06 are resolved and covered by deterministic tests. The highest priority is to repair reward argument mapping and PPO batching, then make cell actions spatially conditioned and reconnect the shelter/hazard mechanics to the evaluated outcomes.

## 2026-09-02 — RL/optimization repairs for errors 1–5

### Implemented

- Replaced the ambiguous positional reward wrapper with a strict keyword-only contract and explicit terminal signaling. Removed reward arguments and delayed-shelter state that were accepted but not used by the formula.
- Rebuilt PPO rollout tensor assembly so scalar transition fields remain `(T,)`. Log-ratio construction now validates identical one-dimensional shapes, preventing the previous `(B, B)` broadcast failure.
- Removed epsilon-random action mixing and post-sampling action replacement from RL training. Gating and deployment constraints now form the final mask before the categorical policy samples, so stored log-probabilities describe the action actually executed.
- Replaced the globally pooled actor with a cell-conditioned architecture: per-cell encoders, normalized grid positions, cross-cell attention, per-cell scores, a separate no-op head, and a pooled critic.
- Corrected minibatches to use the configured size, retain the final partial batch, cover each transition once per epoch, and reshuffle for every PPO epoch.

### Verification

- Added 16 focused `unittest` cases covering reward dispatch and terminal behavior, policy shapes and direct local-state dependence, graph validation, action-mask semantics, end-to-end action/deployment consistency, masked sampling, PPO ratio shape invariants, minibatch coverage, checkpoint compatibility, and a complete synthetic PPO optimizer update.
- The focused suite passes, the modified RL modules pass `pyflakes`, all top-level Python modules parse successfully, and the default 8-by-8 policy produces finite `(1, 65)` logits with a `(1,)` value.
- A full map-backed simulator run was deliberately excluded from this repair scope. Existing policy checkpoints target the old network architecture and must not be used as evidence for the repaired policy; training should start from a fresh compatible checkpoint.
- The default shell environment cannot currently import the simulator stack because its installed Pandas requires NumPy 1.22.4 or newer while NumPy 1.21.5 is installed. This pre-existing dependency mismatch does not affect the focused RL tests, and simulator dependency changes were not made as part of errors 1–5.

### Remaining review findings

The original repository findings about shelter semantics, hazard coupling, reproducible paired seeds, checkpoint isolation, final outcome accounting, and route/population invariants remain open in `TODO.md`. They are outside errors 1–5 in this RL/optimization repair.

## 2026-09-02 — Regional MDP, optimization, and research backtest redesign

### Scientific design implemented

- Replaced the previous multi-component shaped reward with one fixed, undiscounted decision-interval objective: normalized safe completions minus three times normalized casualties minus normalized hazard-weighted evacuation person-time.
- Added a formal casualty-dominance argument: because danger is bounded in `[0,1]`, one casualty can avoid at most `2/P` normalized future exposure cost; the `3/P` casualty cost makes death strictly unattractive.
- Removed reward normalization state, terminal bonuses, cell criticality, local impact, fulfillment, rerouting bonuses, and other proxy terms from the scientific objective.
- Made reward timing causal. An installed shelter earns only outcomes and exposure observed after its execution and before the next decision or terminal boundary.
- Defined one immutable `RegionalObservation`, one feasible-cell mask, and one cell-indexed action space for RL, active-population heuristic, and random policies.
- Implemented the benchmark exactly as specified: choose the feasible cell with the largest active pedestrian population, with deterministic cell-ID tie breaking.
- Preserved the hierarchical action: the upper level selects a priority cell; the common lower level deterministically installs the maximum-capacity candidate in that cell.

### Optimization and bridge repairs

- Rebuilt `RLBridge` around decision epochs and pending action intervals rather than timestep-aligned, pre-action rewards.
- The actor now produces exactly one logit per cell; the obsolete no-op action was removed. Feasibility is enforced before sampling.
- Added fixed-scale global features to the critic and actor context, semi-Markov duration-aware GAE, correct minibatches, value clipping, entropy, gradient clipping, KL early stopping, and finite-value checks.
- Checkpoints now include an interface signature, neural and optimizer state, policy/action RNG state, minibatch RNG state, episode count, and optimizer-update count. Training resumes and evaluation reject incomplete or incompatible checkpoints.
- Initial and dynamically installed shelters are synchronized to the authoritative cell tracker. Observation capacity comes from `ShelterDS`, and pedestrian state comes directly from active `PedDS` agents.
- Initial shelters now participate in evacuation routing for every policy. Routes are selected by network distance, invalid paths do not silently substitute a different origin/destination, missing route edges reject the route, and initialization must create the requested population or fail explicitly.
- Finite-horizon survivors are recorded as `unfinished`, never forced to `arrival`. Episode output checks the full population identity.

### Reproducible study infrastructure

- Added `backtest.py` and made `script.py` its compatibility entry point.
- The original preregistered default was five independent policy-training seeds, 300 episodes per policy, and 50 held-out stochastic scenarios; the completed-rollout revision below changes the episode count to 320.
- RL and heuristic evaluations use matched scenario seeds. A first-observation SHA-256 digest verifies byte-identical observations and masks, and deployment-budget parity is checked before statistics are produced.
- Statistical output uses a complete policy-seed by scenario matrix, a two-way bootstrap over both sources of variation, and policy-level sign randomization.
- Added manifests, per-episode summaries, PPO diagnostics, consolidated CSVs, parity evidence, a paper-ready table, and plots.
- Added paper-facing design and protocol documents under `docs/`.

### Verification performed

- All changed Python modules pass bytecode compilation.
- `pyflakes` reports no issues in the changed optimization, bridge, logging, experiment, or test modules.
- Sixteen focused tests pass. Coverage includes the exact reward equation and dominance constraint, heuristic action rule, shared observation/action semantics, authoritative observations, maximum-capacity lower-level placement, neural output and gradient paths, causal interval rewards, a full PPO update and checkpoint reload, unfinished accounting, logger schema, interface parity, and paired statistics.
- The full map-backed preregistered backtest has not been launched. The academic experiment workflow requires confirmation of the exact long-running command, and the existing `evacuationModel` environment has a conflicting OpenMP runtime that must be resolved without the unsafe `KMP_DUPLICATE_LIB_OK` workaround.

## 2026-09-02 — OSM milestone visualization and map-backed smoke backtest

### Implemented

- Added a read-only `EvacuationVisualizer` reporting layer. It uses the exact
  OSM road graph already loaded by the simulator and overlays active
  pedestrians, hazard-state cells and sources, initial and installed shelters,
  and the latest upper-level priority cell.
- Added five-milestone episode panels, individual milestone images, SVG output,
  source-layer CSV exports, OSM attribution, and a hash-bearing provenance
  manifest. Explicit milestone lists are supported, while `quartiles` adapts
  to any episode horizon.
- Added selective backtest rendering: by default only the first matched
  held-out RL/heuristic pair is visualized, with RL policy replicate one, so
  figures do not alter the numerical study design or create excessive output.
- Moved `PedDS.docuStatus()` before the RL decision boundary. Outcomes from the
  current simulator transition are now committed before reward evaluation, so
  terminal-step completions and casualties are included exactly once in the
  causal post-action interval.

### Verification

- A safe copied runtime now imports NumPy 1.24.2, Pandas 1.5.3, OSMnx 2.0.1,
  GeoPandas 1.0.1, Shapely 2.0.1, PyTorch 2.2.2, and Matplotlib 3.6.3 with one
  OpenMP implementation. No duplicate-runtime bypass is used.
- Twenty-four focused tests pass, including data and figure generation for an
  offline OSM-like toy graph.
- A reduced three-episode map-backed smoke backtest completed on Reading, PA:
  one training episode and one matched RL/heuristic evaluation pair with 30
  pedestrians and a 15-step horizon. The initial observation digests and
  dynamic deployment budgets matched, population identities balanced, and
  both milestone figure packages were generated successfully.
- The smoke run is implementation evidence only. Its one-episode, one-scenario
  estimates are not suitable for scientific performance claims; as expected,
  the effectively untrained RL policy did not beat the heuristic in this run.
- Corrected the policy-level sign-randomization test to enumerate all sign
  assignments for the five-seed design. This prevents impossible p-values
  below the exact two-sided minimum of 0.0625 from being reported.
- Added restricted mean time to safety as the confirmatory time outcome and
  relegated conditional mean completion time to diagnostics. Casualties and
  unfinished pedestrians receive the horizon, preventing a low-success policy
  from appearing artificially fast. Completion timestamps were also corrected
  to the boundary at which the event actually occurs.
- Removed two major sources of avoidable backtest cost without changing the
  simulation equations: active pedestrians are indexed by cell once per
  timestep instead of rescanned for every cell, and shelter-distance queries
  use one exact reverse-graph Dijkstra search per destination rather than one
  search per pedestrian-destination pair.
- Reconnected the configured stochastic-hazard parameters to the active model.
  CSV means are percentages and variances are squared percentage points;
  sampled spread, casualty, and speed-reduction probabilities are all used.
  Hazard state divided by five is now the reward/observation danger variable,
  overlapping effects combine through complementary probabilities, speed is
  reduced before movement, heat/smoke contributions add, and hazard lifetimes
  advance. The obsolete unreachable danger-threshold casualty table and
  post-movement speed mutation were removed.
- Added a preregistered convergence gate. Each policy seed must have finite PPO
  diagnostics, stable adjacent reward windows, a flat normalized tail trend,
  and controlled target-KL violations before held-out evaluation is opened.
  Added a separate performance assessment that declares RL superiority only
  when the paired two-way-bootstrap return interval is entirely above zero.

## 2026-09-02 — Common-random-number and convergence audit hardening

- Split stochasticity into recorded initialization, hazard-evolution,
  pedestrian-hazard, and policy streams. Pedestrian casualty shocks are now
  counter-based by scenario, timestep, and person, so policy-dependent active
  population cannot advance the hazard or another pedestrian's random draw.
- Added a full hazard-trajectory SHA-256 digest to every episode and made exact
  digest equality a mandatory RL/benchmark parity check. A map-backed smoke
  run confirmed identical 320-row hazard milestone trajectories, and a later
  smoke confirmed identical full-trajectory digests.
- Made JSON artifacts standards-compliant by converting unavailable numerical
  diagnostics to `null`. One-seed/one-scenario smoke runs are now labeled
  descriptive-only instead of receiving a misleading superiority/inferiority
  classification.
- Added a reward-component, entropy, KL, and heuristic-agreement diagnostic
  figure alongside the return convergence plot.
- Replaced misleading pre-override parameter output with one validated
  effective-configuration record. Unknown overrides, off-policy epsilon
  mixing, unsafe PPO learning rates, and unsupported optimizers now fail fast;
  the project CSV explicitly configures AdamW at `3e-4` and zero epsilon.
- Thirty-one focused framework tests pass after these changes. A corrected
  two-seed, 100-episode convergence-gated pilot is in progress; its results
  remain separate from the preregistered full-scale study.

## 2026-09-03 — Batched residual PPO and held-out pilot

### Optimizer and policy redesign

- Diagnosed the episode-local PPO failure from training-only evidence. Each
  episode supplied only eight decisions, so per-episode advantage normalization
  produced high-variance updates and seed-dependent near-uniform policies.
- Replaced episode-local updates with eight-complete-episode on-policy
  rollouts. The policy is held fixed while 64 decisions are collected in the
  40-step pilot; GAE and advantage normalization span the batch while terminal
  flags prevent cross-episode return propagation.
- Added exact partial-rollout checkpointing. Resume restores observations,
  masks, executed actions, old log probabilities, values, rewards, durations,
  terminal boundaries, both RNG states, optimizer state, and rollout count.
- Expanded the checkpoint signature to include every PPO hyperparameter.
  Silent resume under a changed optimizer contract is rejected.
- Replaced the large absolute-position/attention actor with a compact shared
  spatial scorer. It uses explicit four-neighbor mean messages plus mean/max
  city context, avoiding sparse-signal dilution and static cell memorization.
- Formulated RL as a learning-augmented residual policy. A fixed relative
  active-population logit prior exactly reproduces the benchmark ranking when
  the residual is zero; PPO learns corrections from hazard, speed, capacity,
  candidate, neighboring-region, and global features. The last residual layer
  is zero-initialized, so initial deterministic behavior is independent of
  neural initialization.
- Reduced entropy regularization from `0.01` to `0.001`; it remains an
  optimizer regularizer and is not part of the paper reward.

### Verification and pilot result

- Thirty-four focused framework tests pass. New coverage verifies exact
  heuristic-prior ranking, direct local and graph-neighbor gradient paths,
  update-only-at-rollout-boundary behavior, and partial-rollout checkpoint
  resume.
- Map-backed integration smokes verified that episodes 1–7 do not update,
  episode 8 consumes the full batch, the checkpoint buffer returns to zero,
  and all diagnostics remain finite.
- A reduced two-policy pilot required a uniform extension from 104 to 160
  episodes. Both seeds then passed the preregistered convergence gate with 20
  complete PPO updates, zero tail KL violations, finite diagnostics, and no
  pending rollout data.
- The one-shot held-out evaluation used 20 matched stochastic scenarios and
  produced 80 episode rows across two RL policies, heuristic, and random.
  All 60 policy–scenario parity comparisons passed, including identical first
  observations, budgets, component seeds, and complete hazard trajectories.
- Mean RL episode return was `0.13117`, versus `0.12052` for the heuristic, a
  mean improvement of `0.01065`. RL also improved safe completions (`+0.425`),
  unfinished population (`-0.55`), restricted mean time to safety (`-0.400`),
  and normalized risk-weighted person-time (`-0.00998`), while casualties were
  `0.125` higher on average.
- The two-way-bootstrap 95% interval for return improvement was
  `[-0.02725, 0.04361]`; the machine-readable result is therefore correctly
  classified as `inconclusive`. This pilot demonstrates convergence and a
  positive point estimate, not confirmatory superiority. The planned
  five-policy, 50-scenario study remains required for a paper claim.
- Publication-style OSM milestone panels were generated for the prespecified
  first matched RL/heuristic pair, with complete plotted-source CSVs and hash
  manifests. All 806 JSON artifacts in the launch parse under strict JSON.

## 2026-09-06 — Five-city pooled transfer experiment

- Added a validated, versioned city catalog for State College, Reading,
  Spokane, Seattle, and Chicago. Scale order is fixed by 2020 Decennial Census
  municipal population; downtown-centered OSM radii increase from 1 to 3 km.
- Generalized the map layer from address-only place queries to explicit point
  and radius specifications. Roads and building/amenity features now use the
  same footprint, cache identities contain the full query, arbitrary raw-cache
  reconstruction cannot substitute another city's graph, and graph/query hashes
  are included in run provenance.
- Added two transferable observation features shared by RL and benchmarks:
  regional OSM road-node mass and the free-flow map-crossing-time share. The
  city identifier is never supplied to the policy. This changes the observation
  signature and intentionally invalidates schema-v6 checkpoints.
- Added `multicity_backtest.py`. It trains one checkpoint on a deterministic,
  exactly balanced block-randomized sequence of all selected cities; preserves
  disjoint training/evaluation seed streams; and prevents resume under a changed
  map catalog, grid, replicate design, or simulator contract.
- Added fixed-site, equal-city macro analysis. Policy seeds are resampled jointly
  across cities and held-out scenarios are resampled within city. City-specific
  results are retained, and the five purposively selected cities are not treated
  as a random sample of all cities.
- Added profile-only validation, OSM map preflight, per-city milestone-map
  controls, and a dedicated cross-city protocol. Forty-two offline framework
  tests pass, including eight new city/profile/query/schedule/analysis tests.
- Validated and snapshotted the five profiles without network access. The OSM
  preflight and full map-backed training/evaluation remain intentionally open;
  they require downloads for the four uncached city study areas and a long run.

## 2026-09-06 — Five-city execution and runtime validation

- Completed the exact OSM preflight for all five profiles. The audited graphs
  contain 5,902/17,022 nodes/edges for State College, 3,455/10,176 for Reading,
  10,849/33,472 for Spokane, 18,778/52,718 for Seattle, and 24,723/69,634 for
  Chicago. Every study area contains eligible shelter features and the artifact
  records graph/query hashes and ODbL provenance.
- Corrected OSM shelter semantics exposed by cross-city data. When a feature is
  tagged `building=yes` and `amenity=library` (or another eligible use), the
  functional amenity now consistently determines eligibility, capacity, and
  the type passed to the lower-level optimizer.
- Diagnosed the supported macOS runtime: the old x86 environment combined pip
  PyTorch's Intel OpenMP with Conda's LLVM OpenMP and also incurred Rosetta
  loader stalls. No duplicate-runtime bypass was used. Added a pinned, native
  Apple Silicon, all-Conda-forge `environment.yml` with a CPU-generic PyTorch
  build and one LLVM OpenMP runtime.
- All 47 focused tests pass in that environment. An eight-episode State College
  smoke collected 24 transitions, completed the PPO update, produced finite
  gradients/losses/KL, saved a complete checkpoint, and preserved population
  accounting.
- Completed a reduced five-city pooled smoke: 40 balanced training episodes,
  five optimizer events, and ten matched held-out RL/heuristic episodes. All
  five parity comparisons passed. The descriptive equal-city return difference
  was +0.01371, but the policy did not meet the 100-episode convergence minimum
  and one seed/one scenario per city is not inferential evidence.
- Hardened inference labels so point resampling ranges from under-replicated
  pilots can never be reported as superiority. The full five-seed, 20-scenario
  per-city confirmatory experiment remains open.
- Invalidated a 30-episode learning audit after detecting that fixed
  eight-episode rollouts do not align with five-city schedule blocks. Pooled
  rollout length is now derived from the selected city count (ten episodes for
  five cities), ensuring every PPO update contains exactly two episodes from
  every city. The partial audit is retained with an explicit invalidated status
  rather than being silently reused.
- Completed the replacement 120-episode audit with 12 city-balanced PPO
  rollouts. It remained non-converged: the adjacent tail-window shift was
  0.756 SD (threshold 0.5), critic explained variance ended at 0.060, and PPO
  KL stayed between roughly 1e-9 and 2e-6. A five-scenario development probe
  found two city wins, three losses, and mean RL improvement -0.00343; this is
  explicitly diagnostic and not confirmatory evidence.
- Tested a tenfold learning-rate development setting (`0.003`) on the same
  balanced 120-episode corpus. It produced healthy nonzero PPO movement (KL
  reaching 0.022 and critic explained variance peaking near 0.60), but entropy
  declined, reward remained non-stationary, and the fixed development probe
  worsened to mean improvement -0.01514. The setting was rejected rather than
  promoted on optimizer activity alone.
- The intermediate learning-rate audit (`0.001`) was extended equally across
  all cities to 160 episodes. It passed the training-only convergence gate on
  32 equal-city blocks (tail trend 0.080 SD, adjacent-window shift 0.499 SD),
  had no KL violations, and ended with critic explained variance 0.607. Its
  deterministic development probe was unchanged from episode 120 and averaged
  +0.00629 over five non-inferential scenarios. This rate is now frozen as the
  pooled-runner default before confirmatory evaluation.
- Sealed the converged reduced `0.001` launch with 25 untouched matched
  city-scenarios (50 RL/heuristic episodes). Interface parity passed all 25
  comparisons. The equal-city descriptive return improvement was +0.03391,
  with 60% RL wins, +0.60 safe completions, 0.64 fewer unfinished evacuees,
  and +0.00991 improvement in normalized risk-time. This is not inferential
  because it has one policy seed. Seattle return was worse, the effect declined
  with scale rank, and Chicago had one additional RL casualty; the generated
  learning assessment therefore correctly says the agent does not yet qualify
  as learning well cross-city.
- Final verification passes 50 focused tests, Python compilation for the RL,
  optimization, multi-city, mapping, and visualization modules, strict JSON
  loading of the principal artifacts, and `git diff --check`.

## 2026-09-06 — Publication figure suite and policy-objective audit

- Added `generate_paper_figures.py`, an idempotent figure pipeline that reads
  the sealed multi-city launch without retraining or overwriting it. It creates
  vector and raster training/evaluation figures, matched policy tables,
  checksum manifests, and OSM shelter/evacuation map panels for every city.
- Evaluated RL, the active-population heuristic, a random feasible-region
  control, and full-budget static predeployment on the identical 25 held-out
  city-scenarios. The derived table contains 100 episodes (25 per policy), and
  all 50 rerun RL/heuristic action returns exactly reproduce the sealed rows.
- Found and repaired an action-count coupling in evaluation: policies with no
  online actions previously reported zero return and zero risk-time by
  construction. `RLBridge` now accumulates the paper objective independently
  across every elapsed simulator interval while retaining action-owned rewards
  for PPO. A regression test proves that a zero-action static policy receives a
  nonzero, correctly decomposed policy-level objective.
- The corrected equal-city means are: policy-objective return -0.4862 (RL),
  -0.5201 (heuristic), -0.6047 (random), and -0.5903 (static); safe completion
  7.48, 6.88, 5.84, and 5.60 of 20, respectively. Relative to the heuristic,
  RL improves return by +0.03391 (scenario-bootstrap interval
  [0.00537, 0.06114]) and safe completion by 3.0 percentage points, but has one
  additional casualty across the 25 scenarios. These remain descriptive
  one-policy-seed results, not confirmatory evidence.
- Reran one matched scenario per city with visualization enabled for all four
  policies. The ten dynamic RL/heuristic map reruns reproduce the sealed
  outcomes, observation digests, hazard digests, and returns exactly. Twenty
  map manifests provide OSM road, hazard, active-pedestrian, shelter, and
  priority-region layers with plotted-source tables and hashes.
- Final validation passes 51 offline tests, Python compilation, the complete
  four-policy scenario matrix audit, map non-intervention checks, and artifact
  checksum generation. The figure bundle contains five analytical figure sets
  in PNG/SVG and ten cross-policy city-map composites.

## 2026-09-08 — Full E0--E6 reporting contract

- Added a versioned full-experiment specification with eight policy seeds, 120
  training episodes per city, an 18-cell capacity/hazard/demand factorial, five
  replications per cell, explicit E0--E6 outputs, and 12 paper figure families.
- Added a simulator-independent schema loader that rejects unknown fields,
  unsafe table paths, incomplete experiment declarations, duplicate factor
  levels, and statistically inadequate confirmatory settings.
- Added a read-only full-suite reporter. It preserves global policy-seed
  dependence across cities, resamples scenarios within fixed cities, uses equal
  city weights, requires the action-count-invariant policy objective for static
  comparisons, and exports statistical summaries plus checksum manifests.
- Added plots for multi-seed training/checkpoint convergence, PPO diagnostics,
  absolute policy performance, city forest effects, capacity-by-hazard effects,
  the safety/casualty/timeliness frontier, action scalability, ablations,
  leave-one-city-out transfer, robustness, and OSM placement/progress maps.
- Backtested the reporter on the sealed one-seed five-city launch. It generated
  six supported analytical/map families and correctly marked E0/E3--E6 panels
  as unavailable; the bundle remains explicitly labeled partial smoke evidence.
- Added independent synthetic tests for suite validation, equal-city training
  blocks, matched two-way bootstrap inference, every optional figure renderer,
  and fail-transparent partial reporting.

## 2026-09-08 — Population/candidate scale-stress implementation

- Added a fixed 5×5 stress design spanning pedestrian populations 2,500,
  5,000, 7,500, 10,000, and 12,500 and OSM candidate counts 25, 50, 75, 100,
  and 125. A 120-transition maximum horizon is encoded as `stopTime=121`.
- Added an atomic, resumable matched backtest runner. The full five-city,
  five-scenario, eight-policy-seed design contains 5,625 episodes; its manifest
  records 42,187,500 requested pedestrian trajectories, checkpoint hashes,
  source-training horizons, and exact RL/heuristic interface-parity checks.
- Split checkpoint inference compatibility from exact PPO-resume
  compatibility. Evaluation now checks observation/action semantics,
  architecture, parameter names, and tensor shapes without treating learning
  rate or rollout size as model inputs; training continuation remains strict.
- Completed a one-cell computational pilot for State College with 2,500
  pedestrians, 25 candidates, and a 120-transition maximum horizon. Both
  policies exhausted the population after transition 29 on an identical hazard
  path. The old 14-transition RL checkpoint produced 21 casualties versus 25
  for the heuristic and objective returns 0.88771 versus 0.87740, but the run is
  explicitly labeled non-inferential and horizon-mismatched.
- Added F13 heatmaps for return, safe completion, casualty reduction, and RL
  episode runtime. Missing factor cells remain visibly absent rather than being
  imputed or fabricated.
- Exported every graph supported by completed evidence to the top-level
  `publication_graphs/` directory in publication PNG/SVG formats with checksum
  manifests and a figure inventory. F01 now exposes raw stochastic block
  returns beneath its moving average, and the incomplete F13 panel is visibly
  labeled as a one-cell pilot.

## 2026-09-08 — Rollout-average diagnostics and factorial map evidence

- Replaced the compact PPO panel with a four-part update diagnostic modeled on
  the supplied reference: trust-region behavior, exploration/concentration,
  critic fit, and rollout-averaged reward outcomes. Optimization statistics
  average all completed epoch/minibatch steps and outcome statistics average
  all ten contributing episodes. The exact 16-update table is exported as
  `publication_graphs/current_evidence/02_ppo_update_averages.csv`.
- Extended the regional action receipt and visualization audit trail through
  the exact lower-level shelter candidate: shelter ID, OSM node ID, local
  coordinates, grid cell, capacity, decision time, and order. RL and the
  active-population heuristic remain on one observation/action/mask/executor
  contract. Static additions are now separately recorded as anticipative
  predeployments at `t=0`, rather than being mislabeled as ordinary initial
  shelters or online decisions.
- Added the fixed five-city map-factorial design: five populations, candidate
  counts 5/10/15/20, hazards 1--5, and a 120-transition horizon. The dry run
  verifies 500 condition cells, 1,500 matched episodes, 500 numbered placement
  comparisons, and 500 evacuation-progress comparisons.
- Added scientifically explicit early-completion handling. Later requested map
  milestones repeat only the frozen absorbing MDP state, carry an
  `absorbing_after_terminal` audit flag, state the true terminal time on the
  figure, and contain no synthesized pedestrian positions.
- Completed a real State College pipeline pilot at population 2,500, five
  candidates, and one hazard. RL and the heuristic made the same three choices
  and both finished at t=37 with 2,490 safe and 10 casualties; static
  predeployment finished at t=36 with 2,492 safe and 8 casualties. These values
  are development diagnostics only because the checkpoint was trained at the
  old 14-transition horizon. Confirmatory mode correctly refuses that source.
- Verified the new artifacts, per-policy manifests, candidate tables, OSM
  checksums, absorbing-state tables, raster-only large-matrix mode, focused
  regression tests, and Python compilation.

## 2026-09-08 — Correction of degenerate shelter-choice visualization

- Audited the first five-candidate map after observing that its highlighted
  priority cell and shelter sequence appeared constant across policies. The
  decision tables proved that priority changed from cell 3 to 4 to 5, but all
  decisions at t=1/6/11 occurred before the first t=20 progress milestone. The
  plot therefore showed only the last decision and was not a valid action
  trajectory visualization.
- Identified a deeper experimental confound: the 120-step scheduling logic
  allowed every pool of 5/10/15/20 candidates to be installed. Candidate-pool
  size and resource budget were therefore coupled, so final site sets could be
  identical by construction. Added a fixed maximum of five additions shared by
  RL, heuristic, and static policies. Candidate level 5 remains a scarcity
  boundary; levels 10--20 now require genuine subset selection.
- Added decision-epoch maps that capture regional active population, hazard
  state, changing priority cell, exact implemented OSM candidate, and decision
  time for every online action. Static placement is kept out of this dynamic
  panel and remains explicitly represented as simultaneous t=0 deployment.
- Completed a replacement nondegenerate State College pilot with population
  2,500, ten candidates, three hazards, and five additions. RL chose cells
  3/9/6/11/4; the heuristic chose 3/11/9/4/5; both left three candidates
  unused. RL and heuristic each recorded 2,463 safe and 37 casualties, with
  objective returns 0.82080 and 0.82180; static recorded 2,458 safe and 42
  casualties. This remains non-confirmatory because the checkpoint was trained
  at the legacy 14-transition horizon and before the fixed-budget contract.
- Added explicit source-policy budget compatibility checks. Confirmatory map
  execution now requires both a 120-transition checkpoint and the same
  five-addition training budget.

## 2026-09-08 — Pedestrian speed units and physical-link congestion

- Audited the movement equation and made its units explicit. The configured
  free-flow value 64 is 64 m/min (1.067 m/s or 3.84 km/h), and each transition
  is one minute. The apparent large jumps arose because map milestones are 20
  minutes apart, not because a pedestrian moves 20 times too far per step.
- Replaced the unused constant edge-capacity placeholder with physical storage
  based on edge length, a declared 3 m effective walking width, and 5.4
  pedestrians/m² jam density.
- Added a synchronized Weidmann pedestrian speed-density model. Directed OSM
  counterflow is combined on one physical segment, pedestrians queued for
  their next link are included in the load, and physical-link occupancy is
  refreshed in six synchronized 10-second substeps per one-minute simulator
  transition without iteration-order allocation.
- Added timestep and episode diagnostics for effective speed, density,
  congestion multipliers, occupied links, congested population, and congested
  person-time. Visualization schema v4 labels both simulator step and elapsed
  minutes while retaining one red dot per active pedestrian and the fixed-scale
  cell-danger heatmap.
- Extended the full-map evidence gate: a source policy must now match the
  120-transition horizon, five-addition budget, and exact congestion contract.
  The existing checkpoint fails all three checks and remains mechanics-only.
- Added toy tests for formula bounds and monotonicity, physical counterflow,
  permutation invariance, movement slowdown, invalid parameters, logging, and
  source-manifest compatibility.
- Completed a congestion-enabled State College mechanics pilot with 2,500
  individual pedestrians, 10 candidates, three hazards, five additions, and a
  120-minute maximum horizon. The most crowded physical link reached 2.34
  pedestrians/m² for RL and the heuristic; its local speed multiplier fell to
  0.370, while the person-time mean remained 0.999 because congestion was
  spatially concentrated. RL selected cells 3/9/6/11/4 and the heuristic
  selected 3/11/9/4/5, confirming distinct decision trajectories. Both had
  2,463 safe completions and 37 casualties; static had 2,458 and 42. These are
  mechanics diagnostics only because the inherited checkpoint predates the
  horizon, resource-budget, and congestion contracts.
- Added one congestion diagnostic to every planned map-factorial condition.
  The full five-city design now specifies 500 conditions, 1,500 episodes,
  11,250,000 pedestrian trajectories, at most 1.35 billion person-transitions,
  and 2,000 comparison/diagnostic PNGs. The old checkpoint is rejected before
  this confirmatory matrix can start.
- Repeated the State College mechanics pilot after introducing six 10-second
  density updates per minute. The policy outcomes and distinct regional
  sequences remained stable: RL selected 3/9/6/11/4, the heuristic selected
  3/11/9/4/5, and their exact OSM candidate sequences differ. The new
  fail-closed completion audit passed 3/3 episode rows, 1/1 condition, 4/4
  figures, dynamic initial-observation parity, the common hazard path, the
  installation budget, and population outcome accounting. The pilot remains
  mechanics-only because its RL checkpoint was trained under the older
  horizon, budget, and no-congestion contracts.

## 2026-09-08 — Individual pedestrian and continuous-danger map encoding

- Replaced the decision-epoch regional-population fill with one red point per
  active pedestrian agent. A fail-transparent compressed table now records the
  exact decision index, time, selected region, candidate OSM node, agent ID,
  coordinates, cell, speed, and affected status for every plotted point.
- Replaced coarse `impactedLevel / 5` rendering with the continuous normalized
  `dangerLevelByCell` value used by the regional MDP. Every cell is rendered on
  the same fixed 0--1 `cividis` scale across policies and times, with an
  explicit colorbar; milestone and decision maps now share the same encoding.
- Corrected nested comparison layouts so policy titles no longer overlap
  decision annotations. The replacement State College pilot is stored under
  `runs/factorial_maps_seed_20260908_v7_pilot` and its paper figures under
  `publication_graphs/factorial_maps/factorial_maps_seed_20260908_v7_pilot`.
- Audited all decision panels: RL red-point counts were 2,488/2,138/1,619/
  1,136/542 and heuristic counts were 2,488/2,138/1,612/1,025/607; every count
  exactly matched both the outcome active population and the sum of cell
  populations. Mean cell danger evolved from 0.0094 to 0.0375. The complete
  69-test regression suite passes.

## 2026-09-08 — 60-minute, 50,000-pedestrian large-network contract

- Replaced the primary scenario with 60 one-minute transitions
  (`stopTime=61`), 50,000 pedestrians, and one regional shelter decision every
  two transitions. The five-level demand sensitivity is retained and rescaled
  to 10,000/20,000/30,000/40,000/50,000 pedestrians.
- Doubled the fixed downtown OSM radii to 2/3/4/5/6 km for State College,
  Reading, Spokane, Seattle, and Chicago. These remain fixed radial samples,
  not whole-city boundaries.
- Versioned the action cadence into the policy inference signature and added
  fail-closed horizon, action-cadence, congestion, resource-budget, and map-
  footprint checks to the map evidence plan. Earlier checkpoints and map
  pilots are not confirmatory evidence under the new transition law.
- Corrected shelter admission at capacity: a pedestrian rejected by a full
  destination now waits or reroutes instead of falling through to the generic
  route-completion arrival outcome. Admission also accounts for represented
  group size atomically and never overfills shelter capacity.
- Updated the full map plan to 500 conditions, 1,500 episodes, 45,000,000
  requested pedestrian trajectories, and at most 2.7 billion one-minute
  person-transitions. Progress maps are prespecified at minutes
  0/10/20/30/40/50/60.
- Replaced per-pedestrian Dijkstra routing with one exact reverse shortest-path
  tree per shelter destination. Pedestrian initialization now samples birth
  nodes directly and avoids constructing random routes that Core immediately
  discarded when assigning the nearest active shelter.
- Live OSM preflight passed for every enlarged footprint. State College,
  Reading, Spokane, Seattle, and Chicago contain respectively
  11,628/9,844/23,609/44,614/63,733 walking-network nodes and
  5,407/1,796/3,264/7,990/16,285 shelter-eligible stamped nodes. The
  checksum-bearing audit is in
  `runs/preflight_60min_large_network_all_20260908/map_preflight.json`.
- A real State College 50,000-person, 60-minute heuristic episode then
  completed with balanced accounting: 12,200 shelter evacuations, 2,679
  casualties, 35,121 unfinished, five deployments at minutes 1/3/5/7/9,
  mean congestion speed ratio 0.5605, and maximum link density 56.83 ped/m².
  This is a mechanics/runtime observation, not policy-performance evidence;
  the extreme density and 24.4% shelter-capacity coverage must be addressed in
  the prespecified E0 calibration before confirmatory training.
- The complete 82-test regression suite passes in the pinned ARM environment.

## 2026-09-19 — Signal-learning diagnosis, repository cleanup, and capacity-control plan

### Work completed

- Audited the recurrent PPO, exact-candidate mask, reward accounting, GNN
  candidate scorer, test suite, and retained research workflows after the
  State College 3,000-pedestrian credit-assignment validation.
- Removed generated root/test bytecode and pytest caches from the working tree.
  No regression source or reproducibility workflow was deleted because all 19
  test modules participate in discovery and the older runners remain referenced
  by their experiment protocols.
- Added `docs/RL_SIGNAL_CURRICULUM_PROPOSAL_20260919.md`, an unverified staged
  plan covering `WAIT`, intervention cost, action-difference local rewards,
  local-to-global annealing, scheduled exploration, conservative masks,
  candidate-to-region attention, variance control, and paired ablations.
- Established the standing maintenance rule that every future model-code change
  must update both this log and `TODO.md` in the same task.

### Capacity-fairness finding

- The current dynamic policies share the same candidate table, executor,
  timing, and maximum number of deployments, but installed capacity is inherited
  from the chosen site's heterogeneous `nodeCap`. Equal action counts therefore
  do not guarantee equal total shelter capacity.
- The current interface audit checks the deployment-count budget but does not
  compare initial capacity, dynamic capacity added, the capacity sequence, or
  final total capacity. The episode summary also does not yet export the final
  capacity fields needed for a fail-closed paired audit.
- The confound is present in completed evidence, not merely theoretical. In
  held-out State College scenario 1 of the 3,000-pedestrian validation, the
  heuristic and RL policies 2/3 ended with 8,500 places, while RL policy 1
  ended with 6,700 places despite sharing the deployment-count contract.
- The planned correction is an ex ante capacity-token schedule shared by RL and
  the heuristic. A policy chooses the location of token `q_k`; raw site capacity
  becomes an eligibility ceiling. The primary matched comparison must end with
  identical initial capacity and identical dynamic capacity `B` for both
  policies. A separate resource-efficiency experiment may allow unused budget,
  but must not be labeled equal implemented capacity.

### Verification

- The cleanup and documentation state pass `git diff --check`.
- The complete repository suite passes 183/183 tests with bytecode generation
  disabled. No model implementation or numerical training result was changed
  by this entry.

### Next steps

- Implement the capacity-token contract and fail-closed capacity parity audit
  before changing the action space or launching another comparative backtest.
- Then implement the staged signal curriculum in the order recorded in
  `TODO.md`, adding unit, integration, checkpoint-resume, and paired-evaluation
  tests at each step.

## 2026-09-20 — Natural-Momentum Counterfactual Control research design

### Diagnosis

- Reframed the learning failure as causal identification under interacting
  stochastic dynamics. Recurrent memory and long GAE can propagate outcomes,
  but they cannot reveal what would have happened under `WAIT`; consequently,
  hazard evolution, crowd momentum, congestion, and decision effects remain
  mixed in one high-variance return.
- Identified selection confounding as a second problem: the controller acts
  when conditions are deteriorating, so a monolithic world model may associate
  intervention with poor outcomes even when intervention is beneficial.
- Concluded that passive system pretraining is necessary but not sufficient.
  It must be paired with randomized or simulator-generated intervention
  contrasts to identify action effects.

### Proposed framework

- Added `docs/NATURAL_MOMENTUM_COUNTERFACTUAL_CONTROL_20260920.md`, an
  unverified design for Natural-Momentum Counterfactual Control (NMCC).
- NMCC separates an action-free hazard/natural-momentum model from an
  intervention-residual model trained on candidate-versus-`WAIT` simulator
  twins under the same structural noise.
- Proposed a dueling causal critic `Q = V_wait + D_action`, exact or anchored
  counterfactual-advantage PPO, a robust receding-horizon optimization teacher,
  uncertainty gating, and staged N0--N8 ablations.
- Specified a keyed structural noise tape so action branches cannot change the
  order of later random draws. Snapshot/restore, factual replay, branch-order
  invariance, equal-capacity tokens, and complete reward accounting are Stage 0
  prerequisites.
- Recommended a minimal 200-snapshot paired experiment before implementing the
  full world model. The primary gate is at least a 50% reduction in advantage
  variance under common-noise pairing without a material mean shift.

### Scientific positioning

- The design combines exogenous-state decomposition, common-random-number
  simulation optimization, counterfactual credit assignment, learned world
  models, and residual action-effect prediction. The proposed contribution is
  their graph-structured integration for crowd-interactive intervention
  control, not the individual ingredients.
- Exact counterfactual causal advantages target the original population
  objective and need not fade. Hand-designed proxy rewards and planner
  imitation remain curricula and must fade before confirmatory evaluation.

### Verification and next steps

- No model source or simulator behavior changed in this design step.
- Documentation passes `git diff --check`.
- Next implementation work is the Stage 0 structural-noise and snapshot
  validity layer, followed by the minimal paired-effect experiment. The full
  NMCC architecture is contingent on that experiment passing its variance,
  effect-ranking, and uncertainty gates.

## 2026-09-20 — NMCC companion note: retrospective evidence and a Stage-0.5 baseline

- Added `docs/CRN_COUNTERFACTUAL_CREDIT_ASSIGNMENT_PROPOSAL_20260920.md`, an
  external (Claude/Cowork) companion analysis to NMCC, not a competing
  framework. It was drafted before its author found NMCC already registered
  in this repository and was revised on discovery to defer to it.
- Computed, from numbers already published in
  `docs/CREDIT_ASSIGNMENT_VALIDATION_RESULTS_20260919.md`, that the implied
  variance ratio between CRN-paired (RL-vs-heuristic, matched hazard
  trajectory) and unpaired per-episode return variance is roughly 2%-5%, i.e.
  a 95%+ reduction. This is a different estimand from NMCC's own
  action-vs-`WAIT` variance-ratio gate and a small sample, so it is offered as
  supportive prior evidence rather than a substitute for NMCC's registered
  minimal 200-snapshot experiment.
- Cross-referenced NMCC's dueling causal critic (`Q = V_wait + D`, `D(b,
  WAIT)=0`) to the classical potential-based-shaping invariance theorem (Ng,
  Harada & Russell, ICML 1999), which proves policy invariance across a full
  multi-step trajectory under any potential function, not only at a single
  decision's argmax. This strengthens, but does not change, NMCC's existing
  invariance argument.
- Proposed an optional `N0.5` interim step: pretrain `V_wait` from independent
  (unpaired) WAIT-continuation Monte Carlo rollouts as soon as the `WAIT`
  action lands, ahead of the harder noise-tape-keying and snapshot/restore
  items later in NMCC's Stage 0, and use it as classical potential-based
  shaping inside the current recurrent PPO's GAE baseline while Stage 0/1 are
  built and verified. Explicitly scoped as a stopgap, not a substitute for
  NMCC's eventual distributional, multi-horizon `M0`.
- Noted that the three registered P0 items (equal-capacity tokens, NMCC Stage
  0, the staged RL signal curriculum) share overlapping `WAIT`-action and
  capacity-token engineering and recommended implementing that shared surface
  once rather than three times.
- Answered the standing question of how the common initial (t=0) shelters are
  selected: `ShelterDatabase.initShelter()` is a row-major round-robin grid
  sweep taking each cell's first available OSM candidate, with no capacity,
  demand, or hazard criterion; this differs from the shared dynamic
  max-capacity site-selection rule (`_candidate_index`/`newShelter`) used by
  every online policy once a region is chosen, and from the `initial_only`
  benchmark's full-budget time-zero predeployment. Common across every policy,
  so not a source of RL-vs-heuristic bias, but flagged as a possible shared
  variance floor.

### Verification and next steps

- No model source, simulator behavior, or run artifact changed in this step.
- Documentation passes `git diff --check`.
- No new implementation is required by this note. NMCC's own Stage 0 remains
  the next implementation work; the optional `N0.5` baseline above may be
  attempted opportunistically once `WAIT` lands, without blocking Stage 0.


## 2026-09-20 (later) — Collapsed action space from exact-candidate to cell-priority

### Motivation

The user requested, after reviewing the exact-candidate action space and a
proposed hierarchical two-agent alternative, a simpler design: the RL policy
(and every heuristic benchmark) chooses only which regional cell to
prioritize next; a shared, deterministic lower layer resolves the specific
building. This is an action-space decision-complexity reduction, distinct
from and complementary to the NMCC structural credit-assignment item above.
Full rationale, a section-by-section diff summary, and the verification
record are in `docs/CELL_PRIORITY_ACTION_SPACE_20260920.md`; this entry is
the required change-discipline summary.

### Implementation

- `DecisionInterface.py`: `RegionalObservationBuilder` no longer freezes one
  action slot per raw candidate at construction. It now resolves exactly one
  slot per regional cell every `build()` call, using
  `ShelterDatabase.previewShelterCandidate` (already computed inside
  `_regional_capacity`) as the per-cell node. `number_of_actions` returns
  `number_of_cells`. `_candidate_action_mask` and
  `_candidate_operational_features` take the resolved per-cell records
  (and nodes) as parameters instead of reading a frozen instance attribute.
  An empty cell contributes a permanently-masked placeholder slot
  (`"empty-cell-{index}"`, zero capacity) so the action table stays a fixed
  size for the whole episode. `RegionalShelterExecutor.execute` now calls
  `ShelterDatabase.newShelter({"cell": cell}, cellTracker)` instead of
  `newShelterCandidate`, and raises `RuntimeError` if the installed OSM
  identifier ever diverges from the observation's prediction for that cell
  (a new fail-closed invariant with no analogue under the old design, where
  prediction and installation were the same lookup by construction).
- `ShelterDatabase.py`: unchanged. `newShelter` / `_candidate_index` /
  `previewShelterCandidate` already implemented the exact shared
  deterministic rule this design needs (max remaining capacity, OSM
  identifier tie break); they were already used by `initShelter` and
  `predeployStaticDemandGreedy`, just not yet by the live dynamic executor.
- `GNN.py` and the four heuristic benchmark policy classes
  (`ActivePopulationHeuristic`, `HazardWeightedDemandHeuristic`,
  `AccessibilityDeficitHeuristic`, `UniformRegionalPolicy`): unchanged. Both
  already operate generically through whatever `candidate_cell_indices`
  mapping `RegionalObservation` supplies (identity under the new design,
  many-to-one under the old one).
- `RLBridge.py`: `MODEL_VERSION` bumped 17 to 18;
  `_model_signature()["action_space"]` changed from
  `"exact_feasible_shelter_candidate"` to
  `"regional_cell_priority_shared_deterministic_site_rule"` so a pre-18
  checkpoint fails closed on load. `candidate_action_count` and the
  PPO/shape-check call sites needed no change: they already read
  `self.num_candidate_actions`, which now equals the cell count because
  `observation_builder.number_of_actions` does.
- `docs/MDP_AND_OPTIMIZATION_DESIGN.md`: rewrote the research-objective
  paragraph, the operational-observation-graph intro, the model-version
  partition-contract note, the Action section, the benchmark-policy
  section, and the PPO-optimization paragraph to describe the cell-indexed
  action space and the shared deterministic site rule.
  `docs/BENCHMARK_MODEL_PROTOCOL.md`: updated the status/scope paragraph;
  the registered policies' formulas were already written generically in
  terms of a region index `c(j)` and needed no formula changes now that
  `c(j) = j`.
- `tests/test_rl_framework.py`: replaced
  `test_same_region_same_capacity_sites_remain_distinguishable` (asserted
  the old design's "distinct slots per candidate" guarantee) with
  `test_cell_with_multiple_equal_capacity_sites_still_yields_one_slot`
  (asserts the new one-slot-per-cell, OSM-id-tie-break behavior). Replaced
  `test_executor_installs_the_exact_selected_candidate_without_substitution`
  (asserted the old design's "no substitution between candidate slots"
  guarantee) with `test_executor_always_installs_the_shared_deterministic_rule_winner`
  and a new `test_executor_rejects_a_site_that_diverges_from_the_prediction`
  proving the new fail-closed invariant. Left
  `test_active_population_tie_uses_stable_high_capacity_candidate_order` and
  `test_forecast_unsafe_candidates_are_removed_from_every_policy_mask`
  unchanged after confirming by direct execution that every assertion in
  both still holds. Updated the `_candidate_action_mask` monkey-patch in
  `test_all_unsafe_training_episode_is_recorded_without_fake_transition` for
  the function's new third parameter. Added
  `test_empty_cell_yields_an_infeasible_placeholder_slot` and
  `test_every_benchmark_policy_and_rl_resolve_the_identical_site_per_cell`.

### Compatibility impact

Breaking change to the action-space contract. Any checkpoint trained under
`MODEL_VERSION <= 17` fails closed on load (both the version number and the
`action_space` signature string changed) rather than being silently
reinterpreted under the new action semantics.

### Verification performed

- AST-parsed `DecisionInterface.py`, `RLBridge.py`, and
  `tests/test_rl_framework.py`: all parsed without error.
- Wrote and ran a standalone, dependency-light harness (imports only
  `DecisionInterface.py`, `ShelterDatabase.py`, `Shelter.py`, and `numpy` --
  no `torch`, no GNN) that reconstructs the exact fakes/fixtures used by
  `tests/test_rl_framework.py` and executes the scenario in every test named
  above, both new and pre-existing. All checks passed, including the
  divergence-detection `RuntimeError`, the OSM-identifier tie break, the
  empty-cell placeholder contract, and cross-policy agreement with
  `previewShelterCandidate`. The harness was scratch and was not committed.
- **Not performed:** the full pinned-environment
  `micromamba activate rlevacuation && python -m unittest discover -s tests
  -v`. The tool used to make this change reaches a sandboxed shell on the
  user's device with no `torch`, no `networkx`, no `rlevacuation`
  environment, and no outbound network access to install them, so every
  test in `tests/test_rl_framework.py` that imports `GNN`, `RLBridge`, or
  `EvacuationVisualizer` at module level (the whole file, since it shares
  one import block) was not executed end to end, including every
  reward/PPO/checkpoint test that exercises `RLBridge`. Their two-line
  `RLBridge.py` edits (a version bump and a signature string, both read
  generically wherever consumed) were inspected by hand rather than tested.
  **This full run must happen in the project's actual environment before
  this change is treated as verified or used for another training run.**
- No training run, backtest, or figure was regenerated.

### Next steps

- Run the full pinned-environment test suite and resolve any failures.
- Confirm whether `backtest.py` / `TrainingLogger.py` / the visualization
  manifest already key per-episode decision logs on `requested_cell` /
  `executed_cell` (needed for the requested RL-vs-heuristic cell-choice-
  divergence analysis) or still assume per-episode-stable candidate
  identities inherited from the old action space; switch the keying if not.
- Proceed with NMCC (the item immediately above in `TODO.md`) independently;
  this change narrows the action space NMCC's causal-effect identification
  will operate over but does not address the statistical credit-assignment
  problem NMCC targets.


## 2026-09-20 (later still) - NMCC Stage 0 and Variant A for the cell-priority action space

### Motivation

The user proposed differencing a with-decision and a without-decision rollout
over the same n timesteps, and correctly identified two obstacles: the system
is stochastic, and the brute-force version is expensive. Both have precise
answers. Stochasticity is handled by sharing the disturbance realization rather
than removing it, so the shared natural trajectory cancels in the difference.
Cost is handled by bounding the branch horizon and letting the critic close the
tail, by using the action-independent WAIT branch as one control variate for
all cells at once, and by setting the branch horizon to the deployment interval
so the acted branch is the trajectory the episode was going to simulate anyway.
Full design and measurements in
`docs/NMCC_CELL_PRIORITY_ADAPTATION_20260920.md`.

### Implementation

- New `CounterfactualBranch.py`: `capture`/`restore` (single-memo deep copy of
  branch-varying state; shared road graph, node/edge objects and routing
  caches; flat-array capture of the mutable flow fields that live on those
  shared objects), `advance_one_timestep` mirroring `Core.simulationEnumerator`
  minus the learner, `CounterfactualBrancher.paired_effect`,
  `counterfactual_advantage`, and the two Stage-0 audits
  `assert_replay_is_bitwise_identical` and
  `assert_hazard_is_action_independent`.
- New `nmcc_testbed.py`: real MapDS/CellTracker/HazardDS/PedDS/SocialForce/
  ShelterDS dynamics on a synthetic grid city, no torch and no osmnx.
- New `nmcc_paired_experiment.py`: NMCC's minimal first experiment, reporting
  variance reduction, mean-shift bias, signal-to-noise and single-sample
  cell-ranking recovery against five promotion gates.
- `RLBridge.py`: Variant-A integration behind `counterfactual_credit`
  (default `False`). Adds `counterfactual_horizon`, `counterfactual_weight`,
  `counterfactual_intervention_cost`; collects the WAIT baseline before the
  executor installs; attaches `A_CF = (R_a - R_0) + gamma^L (V_a - V_0) - c(a)`
  to the transition; blends standardized counterfactual advantages into the PPO
  surrogate where branches exist; emits `nmcc_*` diagnostics.
  `MODEL_VERSION` deliberately not bumped - the architecture is unchanged and
  existing checkpoints stay loadable.
- New tests: `tests/test_counterfactual_branch.py` (torch-free, 10 tests) and
  `tests/test_nmcc_integration.py` (requires torch).

### Compatibility impact

None when `counterfactual_credit` is false, which is the default: every new
code path is guarded. No architecture, checkpoint or observation-schema change.

### Verification performed

- `tests/test_counterfactual_branch.py`: 10 tests, all passing in 4.8 s against
  the real stochastic dynamics.
- `nmcc_paired_experiment.py` executed; report committed to
  `runs/nmcc_paired_report.json`. 99.36% variance reduction, mean-shift z=0.00,
  single-sample cell-ranking recovery +0.984 paired vs +0.457 unpaired, all
  five gates pass. Reproduced at a second configuration (92.13% reduction,
  +0.990 vs +0.721).
- Branch cost measured at three population scales: capture/restore is linear in
  population (~40 us per pedestrian), branch overhead 1.16-1.29x the cost of
  simulating the same interval.
- **Not performed:** anything requiring torch. `tests/test_nmcc_integration.py`,
  the full `python -m unittest discover -s tests`, and the backtest were not
  run, because torch is not installable from the environment this change was
  made in (PyPI and download.pytorch.org are both outside the account's egress
  allowlist, in both the cloud container and the device VM). The `RLBridge`
  integration parses and is guarded but is otherwise unexecuted.

### Next steps

- Run the full suite and `tests/test_nmcc_integration.py` in `rlevacuation`.
- Re-run the paired experiment against a real `Core` on State College and
  replace the synthetic-map numbers before quoting them in the paper.
- Matched training runs with the flag on and off; compare
  `nmcc_counterfactual_advantage_sd` to `nmcc_gae_advantage_sd` before looking
  at reward.
- Then the paired backtest against the heuristic benchmarks.
- Variant B/C (learned residual, dueling causal critic) only after the above.

## 2026-09-20 — NMCC implementation audit and 3,000-person signal backtest

Reviewed the live, uncommitted NMCC implementation and executed the previously
missing Torch integration checks plus a fresh 3,000-person paired-effect run.
Full evidence and commands are in
`docs/NMCC_IMPLEMENTATION_AUDIT_20260920.md`.

### What passed

- All 10 `tests/test_counterfactual_branch.py` tests passed against the real
  stochastic dynamics on the synthetic-map harness.
- The fresh 3,000-person, six-epoch, six-tape paired experiment passed all five
  estimator gates: 94.30% variance reduction, paired signal/noise 46.79,
  paired cell-ranking recovery 0.959 versus 0.712 unpaired, and no estimated
  mean-effect shift (`z=0.00`). Artifact:
  `runs/nmcc_audit_3000_seed_20260920.json`, SHA-256
  `cfbfbd484a8090a30c03e9d358b7263590f432f86c4e163b04a1150febdbac45`.
- A diagnostic-only shim dropping the stale `first_decision_time` test keyword
  allowed the live one-episode counterfactual PPO collection/update test to
  pass. No source file was changed by that shim.

### Blocking findings

- NMCC remains unreachable from production runs: `counterfactual_credit`
  defaults off, `Core` never forwards it, and no configuration declares it.
- The full test suite is 200/204, not green. Two NMCC end-to-end tests use a
  removed constructor argument. Two established recurrent-PPO tests reveal
  that `_serialize_transition` calls `.detach()` on a `None`
  `counterfactual_advantage`, crashing partial-rollout checkpointing even when
  NMCC is off.
- The documented `--real-core` option does not exist; the signal experiment
  always uses the synthetic grid and therefore is not a State College run.
- `nmcc_*` diagnostics are computed but absent from the training-diagnostics
  CSV schema.
- Only Variant A is present; the learned natural-momentum/residual models and
  dueling causal critic remain future work.

### Interpretation

The agent now has access to a demonstrably cleaner local causal signal in the
testbed, but there is no valid evidence yet that it learns to increase return.
A policy-improvement claim would be false at this stage: no production NMCC
training run can be configured, standard multi-episode training currently
fails at checkpoint serialization, and no matched held-out NMCC-on/off
comparison has been run.

## 2026-09-20 — Production Hybrid NMCC, robust recurrent learning, and State College smoke

Replaced the opt-in exact-advantage prototype documented immediately above
with model version 21's complete Hybrid Natural-Momentum Counterfactual Control
learning path. Full design, evidence boundaries, and artifact hashes are in
`docs/HYBRID_NMCC_IMPLEMENTATION_AND_BACKTEST_20260920.md`.

### Design and implementation

- `GNN.py` now exposes one action-independent natural-outcome model and a
  three-member candidate-local causal-residual ensemble on top of the flexible
  regional GNN and episode LSTM. The natural head hard-enforces population
  conservation and bounds final risk and person-time predictions. Ensemble
  means produce causal reward components; dispersion supplies an explicit
  uncertainty estimate and uncertainty-penalized robust action score.
- `RLBridge.py` now closes factual and `WAIT` outcomes at the same exact
  counterfactual horizon while retaining the factual action's complete SMDP
  tail through the next decision or terminal. Recurrent PPO substitutes or
  blends exact paired advantages only where branches exist, trains the
  factorized critic, and jointly fits natural, causal, and dueling-consistency
  Smooth-L1 objectives.
- Robust planner scores guide the actor only after world-model warmup. The
  scores are detached before guidance and teacher imitation, preventing actor
  gradients from changing the learned simulator. The teacher fades with
  optimizer updates. Entropy and sampling temperature also decay by update,
  providing broad early exploration and steadier late decisions.
- Optional NMCC tensors now serialize and restore safely across partial
  rollouts. The checkpoint signature includes the complete NMCC learner,
  exploration, physical-constraint, decision-interface, and capacity-token
  contracts. Model architecture is
  `resolution_flexible_relational_route_gnn_lstm_nmcc_v6`; model version is 21.
- Training CSV output now includes exact branch coverage, exact and raw-GAE
  advantage dispersion, all auxiliary losses, ensemble uncertainty, planner
  guidance, teacher coefficient/loss, entropy, and temperature. Raw GAE SD is
  recorded before PPO normalization.
- `Core.py`, `TrainingCurriculum.py`, and `CityProfiles.py` now validate,
  propagate, journal, and freeze all NMCC and exploration fields. Learner
  settings cannot change between curriculum stages or variants.
- `ShelterDatabase.py`, `DecisionInterface.py`, `Core.py`, and city profiles
  now share `shelterCapacityToken`. Observation previews, initial shelters,
  dynamic installations, and static deployments all use the same configured
  token; raw building capacity remains the deterministic site-selection and
  physical-eligibility signal. Sites rated below one token are filtered before
  candidate sampling and rejected by preview/direct execution. The registered
  State College token is 500.
- Added the complete 64-episode configuration
  `config/state_college_training_curriculum_3000_nmcc_hybrid.json` and an
  explicitly non-confirmatory eight-episode system-smoke configuration.
- Added `nmcc_learning_backtest.py`, which trains recurrent PPO and Hybrid NMCC
  from identical weights and scenario tapes, performs deterministic held-out
  evaluation against PPO and the heuristic, and fails closed unless shelter
  count and added capacity match for every policy.
- Updated stale NMCC integration tests and added tests for physical output
  bounds, population conservation, action independence/locality, detached
  planner guidance, equal visible/executed capacity tokens, and immutable
  curriculum learner settings. Superseded audit documents are retained with
  explicit banners rather than silently rewritten as if they were current.
- Made `nmcc_testbed.py` seed legacy NumPy origin/destination initialization
  from its declared scenario seed while restoring the caller's RNG state. This
  removes test-order dependence from the paired-variance gate. The testbed also
  accepts the capacity token before shelter sampling so its fairness semantics
  match production `Core`.

### Compatibility impact

- General legacy runs still default to `nmccEnabled=false`; the State College
  NMCC curricula enable the complete path explicitly.
- Checkpoints from versions before 21 intentionally fail the architecture
  signature check rather than loading into different causal heads or capacity
  semantics.
- `shelterCapacityToken=500` is now the common city-profile default. This
  removes site-specific implemented-capacity confounding but changes capacity
  semantics from older results; old and v21 evaluation artifacts must not be
  pooled.
- Equal token size does not by itself guarantee equal final capacity if one
  policy has fewer feasible decision epochs. General benchmark-wide deployment
  deadlines and final capacity gates remain open in `TODO.md`; the new matched
  NMCC backtest already fails closed on both count and capacity parity.

### Verification performed

- Full test suite: **210/210 passed** after the final physical-capacity change;
  the latest clean run completed in 11.851 seconds. This includes exact
  counterfactual replay and common-noise invariants, end-to-end branch/PPO
  execution, partial-rollout checkpoint recovery, recurrent observation
  caching, post-action reward accounting, hazard/casualty coupling, learned
  NMCC heads, detached guidance, curricula, and capacity-token execution.
- Matched engineering learning backtest: 16 training episodes, four updates,
  eight held-out scenarios. Every update had exact targets. Counterfactual
  advantage SD was 0.090298 versus raw GAE SD 0.213234 (ratio 0.4235; about
  82.1% lower variance). Hybrid NMCC training-return slope was +0.020739 versus
  +0.020467 for PPO. Held-out mean NMCC-minus-PPO return was -0.014123
  (SD 0.021723); NMCC-minus-heuristic was -0.026568. All policies installed
  exactly two shelters and 600 places in every held-out scenario. Artifact:
  `runs/nmcc_learning_backtest_20260920.json`, SHA-256
  `ae7caeca90bb5fa5b60a312e9d86dc742a71e3879d0dfe8763b369aa9321c80e`.
- Real State College smoke: eight 60-minute episodes with 3,000 individual
  pedestrians on the cached 6-km OSM network. The run completed one four-epoch
  recurrent PPO update over eight whole episodes, 34 decisions, and 268 cached
  frames. Exact-target coverage was 100%; counterfactual SD 0.031394 versus GAE
  SD 0.132154 (ratio 0.2376; about 94.4% lower variance). Natural, causal,
  dueling, and teacher losses were finite; gradient norm was 3.7138, KL
  0.000012, and the update-episode accounting gap was zero. Casualties ranged
  0--42 with mean 13.875/3,000. Artifact directory:
  `runs/state_college_3000_nmcc_v21_physical_smoke_20260920`.
- `git diff --check`, Python AST parsing, and all three NMCC/city-profile JSON
  validations passed.

### Evidence boundary and next work

The small matched backtest and one-update State College smoke demonstrate
working causal supervision and lower target variance; they do not establish
convergence or policy superiority. The State College runner correctly labels
the smoke `training_not_converged`. Confirmatory work requires the registered
64-episode configuration, at least five independent policy seeds, matched
held-out PPO/heuristic evaluation, full final-capacity parity, and a supported
runtime that does not require `KMP_DUPLICATE_LIB_OK=TRUE`.

## 2026-09-20 — Model v22 staged Hybrid NMCC and 2,500-person launch gate

- Added explicit rollout-level phases to `RLBridge`: natural-model pretraining,
  causal-residual pretraining, controller warm-up, and joint optimization.
  During the two model-only phases, PPO, critic, entropy, residual-logit, and
  teacher losses do not update the actor. Causal and dueling losses begin only
  in the causal phase; the complete controller loss begins at warm-up.
- Added separate persisted counters for completed rollout updates and actor
  optimizer steps. Entropy, temperature, guidance, and teacher schedules use
  actor steps, so model pretraining no longer consumes the exploration budget.
- Added a separate final-stage causal weight. The registered 2,500-person pilot
  uses exact causal credit at weight 1.0 during controller warm-up and 0.35 in
  joint optimization, allowing 65% full-tail GAE contribution without
  discarding the lower-variance paired signal.
- Extended `Core`, effective-configuration output, checkpoint signatures, and
  curriculum validation with the fixed staged-learning contract. Checkpoint
  model version is now 22; earlier checkpoints fail closed.
- Added
  `config/state_college_training_curriculum_2500_nmcc_staged_v22.json`: 64
  State College episodes divided 8/8/16/32 across the four stages, with 2,500
  individually represented pedestrians, three calibrated stochastic hazards,
  and five equal 500-person shelter tokens.
- Added integration tests for phase order, actor/guidance exclusion during
  model pretraining, final causal-weight fading, end-to-end natural-model-only
  optimization, and the rollout-aligned 2,500-person curriculum. The complete
  suite passes **214/214 tests**.
- The first physical launch stopped after episode 1 because a textual phase
  label entered a legacy all-numeric diagnostics dictionary. The run did not
  retry. The interface now records a registered numeric phase index, and the
  focused and full test suites pass after the correction. The partial artifact
  is retained at
  `runs/state_college_2500_nmcc_staged_v22_pilot_seed20260920` for auditability.

## 2026-09-21 — Completed 2,500-person staged NMCC pilot and validation

### Execution and artifact recovery

- Restarted the State College pilot under the fresh launch ID
  `state_college_2500_nmcc_staged_v22_retry_seed20260920` with launch seed
  20260920, one policy seed, 64 episodes, 2,500 individually represented
  pedestrians, and the registered 8/8/16/32 N0--N3 curriculum.
- All 64 simulator episodes, eight rollout optimizer gates, the PPO diagnostic
  ledger, and the 3.6 MB model-v22 checkpoint completed. The numerical training
  artifacts were fully written before the campaign process reached plotting.
- The legacy plotting finalizer then exceeded the declared 60-minute total run
  timeout. Inspection isolated a headless-backend defect: `_plot_outputs`
  imported `pyplot` after the learning stack without forcing a non-interactive
  backend. `backtest.py` now forces `Agg` before `pyplot`; a regression test
  verifies that the finalizer writes two valid PNGs.
- The timed-out process entered an OS-level uninterruptible wait and ignored
  SIGINT, SIGTERM, and the pending SIGKILL. The supported `--resume` path reused
  the complete 64-row ledger and checkpoint, generated both campaign figures
  in under two seconds, and finalized the manifest without rerunning simulation.
- Resume previously overwrote `started_utc` and the original command. The runner
  now preserves those origin fields and records `resumed_utc` and
  `resume_command` separately. The recovered manifest was corrected to retain
  its original 2026-09-21T05:52:43.432583+00:00 start.

### Validation results

- Added `analyze_staged_nmcc_training.py`, a deterministic headless audit for
  the registered 64-episode schedule. It validates episode and stage order,
  optimizer gates, finite diagnostics, exact-target coverage, and reward
  accounting; it writes two summary CSVs, five stage-specific PNGs, a JSON
  result, and a Markdown report with a Material Passport and 11/11 statistical
  fallacy scan.
- Credit assignment worked as designed: exact-target coverage was 100%, the
  median counterfactual/GAE SD ratio was 0.258640, the implied median target-
  variance reduction was 93.28%, and the maximum absolute reward-accounting
  gap was 1.39e-16.
- Auxiliary learning was finite and directional: natural loss declined from
  0.056347 to 0.031278 and dueling loss from 0.098856 to 0.020655. Final causal
  loss was 0.002987 and ensemble uncertainty was 0.004513.
- Reward improvement was **not** supported. In the 32-episode joint stage, the
  objective-return slope was -0.004476 per episode with residual-bootstrap 95%
  interval [-0.011351, 0.002593]. The last-eight minus first-eight mean was
  -0.116732 with interval [-0.333637, 0.070123].
- Policy convergence was **not** achieved. The registered gate requires at
  least 100 episodes, the observed tail trend span was 2.214 standard
  deviations versus a 0.5 limit, and the tail target-KL violation rate was 1.0
  versus a 0.1 limit. Final/max KL was 0.024829 against a 0.015 target.
- This separates two claims that must not be conflated: NMCC successfully
  reduced credit-target variance, but recurrent PPO did not translate that
  signal into a stable improving policy. The next experiment must stabilize
  actor trust-region updates and slow rollout-level schedules before simply
  increasing training length.

### Verification and provenance

- Full repository test suite: **218/218 passed** after the plotting and resume-
  provenance fixes.
- All seven campaign/validation PNGs were rendered and visually inspected.
- Training CSV SHA-256:
  `d555306307fa20fc24628466ff20a17ca6c53eca9c901cf0963b00e3fe37bb53`.
- PPO diagnostics SHA-256:
  `4bbab625da431c14bea70c4ce26915fd701f24d0e04bfab27fb50cfa9ad1bb18`.
- Checkpoint SHA-256:
  `543249524b2b21a43ef6b7f1a0ec60936c22900aec9375c0fa179edc90d70123`.
- The launch used `KMP_DUPLICATE_LIB_OK=TRUE`; it remains engineering evidence,
  not confirmatory or publication-grade numerical evidence. No held-out policy
  evaluation and no independent policy-seed replication were performed.

## 2026-09-21 — Model v23 temporal-credit and actor–critic correction

### Design and implementation

- Replaced the coupled λ-GAE actor/critic target with two auditable
  semi-Markov targets. The actor now receives complete episode Monte Carlo
  return-to-go using each finalized action interval's actual duration; this
  label is provably independent of critic predictions. The factorized critic
  receives a separate one-step TD(0) label using the frozen next-decision
  value. Terminal and episode boundaries remain exact.
- Removed current-rollout advantage centering. Actor returns are normalized
  only with statistics from prior rollout batches, stratified by city,
  initial population, hazard count, and within-episode decision position.
  Lagged EMA mean/variance state is checkpointed and updated only after the
  policy attempt completes. Exact NMCC effects use an independent lagged
  baseline if they are ever assigned nonzero actor weight.
- Split parameter and optimizer ownership. Actor/shared-policy parameters and
  critic/NMCC-world-head parameters form a checked, disjoint, exhaustive
  partition. Critic/world training cannot alter actor parameters; actor PPO,
  entropy, residual, and optional teacher losses cannot alter critic or world
  heads. Natural pretraining therefore leaves actor parameters bitwise fixed.
- Split optimization into four default raw-Huber critic/world passes and one
  actor pass. The critic no longer shares a combined loss, PPO value clipping,
  or rollout-dependent target scaling with the actor.
- Made KL enforcement transactional. Before every actor epoch, v23 snapshots
  the actor weights and optimizer state. A full recurrent-rollout KL above the
  registered target restores both snapshots, halves the actor learning rate,
  records the rejection, and does not advance actor update counters or
  schedules. Entropy, temperature, guidance, and teacher schedules now advance
  by accepted actor rollout updates instead of minibatch optimizer steps.
- Bumped the checkpoint contract to model version 23 / GNN-LSTM-NMCC
  architecture v7. Checkpoints now persist separate actor and critic optimizer
  states, accepted actor-rollout count, critic step count, and both lagged
  baseline stores. Model-v22 checkpoints fail closed for training resume.
- Added v23 diagnostics for actor MC-return and normalized-advantage
  dispersion, baseline coverage, TD-error dispersion, separate gradient norms
  and learning rates, attempted versus retained KL, rollback/acceptance state,
  actor rollout updates, and critic optimizer updates. The old GAE-dispersion
  CSV column remains only as an explicitly documented compatibility alias.
- Added the fixed 64-episode State College contract
  `config/state_college_training_curriculum_2500_temporal_credit_v23.json`.
  It preserves the v22 physical scenario and equal five-by-500 shelter budget,
  uses actor/critic learning rates `1e-4`/`3e-4` and one/four epochs, and keeps
  exact NMCC branches as auxiliary supervision with zero direct actor credit,
  planner guidance, or imitation weight.
- Updated the mechanistic credit audit and bounded learning backtest to test
  complete MC/TD(0), lagged baselines, optimizer isolation, accepted-update
  schedules, and retained KL rather than treating GAE as the active contract.

### Verification and current evidence

- The complete repository suite passes **224/224 tests**. New coverage proves
  actor returns are critic-invariant, TD(0) targets use the correct
  duration-aware bootstrap, current batches cannot enter their own baseline,
  optimizer ownership is disjoint/exhaustive, natural pretraining cannot move
  actor parameters, KL rejection restores actor parameters bitwise, and the
  complete v23 checkpoint state round-trips.
- The randomized mechanistic audit passed 2,000/2,000 trajectories: complete
  MC credit reached all 8,946 eligible actor decisions, critic-invariance and
  closed-form error were exactly zero, and no reward component leaked into
  another factorized branch.
- A fresh eight-training/four-held-out synthetic backtest passed its functional
  gate and equal shelter-count/capacity checks. Both actor rollout updates were
  accepted; none were rejected; mean attempted and retained KL were both
  `7.0035e-7`; lagged-baseline coverage increased from `0.0` to `1.0`; and
  mean critic TD-error SD was `0.19869`.
- The NMCC-auxiliary and no-NMCC actor trajectories and held-out returns were
  identical. This is the intended isolation result: auxiliary losses cannot
  perturb the actor when their actor weights are zero. The short training
  return slope was negative and mean held-out difference from the heuristic
  was `-0.01048`; therefore this smoke run validates mechanics and causal
  isolation only. It does **not** establish return improvement or convergence.


## 2026-09-21 — Convergence review and headroom measurement

### Scope

Reviewed the latest complete model (v22 retry, 64 episodes), the v23 smoke run,
the v22/v23 curricula and `RLBridge.py`/`GNN.py` at model version 23, and
measured how much any cell-priority policy can gain over the heuristic. Full
evidence in `docs/CONVERGENCE_REVIEW_20260921.md`. No learner code changed.

### Findings

- The v22 behavior policy is near-uniform for all 64 episodes (normalized
  entropy 0.95-0.99, heuristic agreement 0.18); residual RMS reached 0.12
  logits over six actor updates (~210 actor samples). The prior's logit spread
  is at most 1.0 and is divided by temperature 1.5->1.0. Evaluation takes the
  argmax, which remains the heuristic's cell, so RL-minus-heuristic near zero is
  built in.
- A 0.015 KL target caps each accepted update at ~0.17 logits; concentrating one
  15-way decision needs >=16 consistent updates (>=128 episodes) before noise.
- Critic explained variance is ~0 throughout v22 and v23.
- The v22 counterfactual advantage mean roughly equals its SD; measured on the
  testbed, 62% of the Variant-A target's variance is between states.
- v23 sets every NMCC actor weight to 0.0 (smoke: identical trajectories with
  and without NMCC); its lagged-baseline key collapses to decision position in a
  single-configuration run; its partition leaves the encoder and LSTM
  actor-owned; its actor step budget is ~1/12 of v22's.

### Headroom experiment

- Added `nmcc_headroom_experiment.py` and `headroom_lib.py`; added
  `spacing_m` and `candidate_count` to `nmcc_testbed.py` (defaults unchanged;
  `tests/test_counterfactual_branch.py` 10/10).
- Calibrated synthetic map: 75% safe / 24% unfinished / 0.2% casualties / 4.5
  of 5 tokens under the heuristic (State College v22: 73 / 27 / 0.3 / 4.4).
- Paired against the heuristic: greedy route saving +0.125 [0.061, 0.188]
  (10/12); implementable 2-tape rollout +0.214 [0.109, 0.319] (4/4); perfect-
  information rollout +0.257 [0.197, 0.317] (6/6); uniform random +0.044
  [-0.018, 0.106]. Greedy route saving captures 58% of the perfect-rollout gap;
  the remaining gap is +0.109 [0.046, 0.172].
- Regret is concentrated at t=1 (~0.21 vs <=0.03 later). The heuristic was not
  best at t=1 in any of six scenarios; active population correlates -0.13/-0.18
  with true cell value there; the 10-minute NMCC effect ranks t=1 cells at
  rho ~ 0.5.
- Reports: `runs/headroom_perfect_information_20260921.json`,
  `runs/headroom_mc2_implementable_20260921.json`,
  `runs/headroom_greedy_20260921.json`.

### Evidence boundary

Synthetic uniform-density map only; State College's concentrated population
may change the gap. Rollout results rest on 6 (perfect) and 4 (implementable)
scenarios. torch was unavailable, so no learner was trained or evaluated.

## 2026-09-21 — NMCC-PI: exact within-state policy improvement (model version 24)

Design and evidence: `docs/NMCC_POLICY_IMPROVEMENT_20260921.md`.

### Design

- The actor target is now the MPO E-step `q ∝ π_old·exp(A/η)` built from exact
  CRN-paired full-horizon branches of every feasible cell (first two decisions)
  or the six most probable plus the executed cell (later decisions).
  - Branches are continued by greedy route saving under an independent future
    tape.
  - `A` is the policy-centered within-state advantage, so `V_wait` cancels.
  - `η` is solved so that `KL(q‖π_old) = ε`, and floored at `η_min`.
- The actor fits `q` by cross-entropy (no entropy, teacher or clipping) with
  its own epoch budget and a convergence stop. An exact `KL(π_behavior‖π_θ)`
  trust region with transactional rollback replaces the sampled KL in PI mode.
- The intervention-value ensemble (`GNN.improvement_value_heads`) is trained on
  behavior-centered within-state contrasts with Poisson bootstrap weights.
  - Its held-out within-state Spearman is measured before each rollout's
    training and gates optional model fill-in (off by default).
- New actor prior option `route_time_saving`.

### Implementation

- New files: `NMCCPolicyImprovement.py`, `NMCCPIConfig.py`,
  `nmcc_pi_reference.py`,
  `config/state_college_training_curriculum_2500_nmcc_pi_v24.json`,
  `tests/test_nmcc_policy_improvement.py`, `tests/test_nmcc_pi_torch.py`.
- `RLBridge.py`:
  - MODEL_VERSION 24.
  - Collection before execution, five new `Transition` fields, and 18 new
    constructor settings.
  - Gate, ensemble loss, M-step, exact-KL rollback and 17 `nmcc_pi_*`
    diagnostics.
  - Gate history and RNG state in checkpoints.
  - CSV schema guard.
  - Signature `..._nmcc_v8` with `actor_prior` (inference) and
    `nmcc_policy_improvement` (training).
- `GNN.py`: actor prior switch and intervention-value heads.
- `Core.py`: the 18 keys are wired through `NMCC_PI_CORE_FIELDS` (defaults,
  casting, kwargs, effective configuration).
  - `actor_credit_target` records the PI target.
  - PI rejects staged NMCC phases and non-zero guidance.
- `TrainingCurriculum.py`: the PI keys are learner stage overrides.
- `tests/test_training_curriculum.py`: a v24 test was added.

### Compatibility

- With `nmccPolicyImprovement=false` and `actorPrior="active_population"`, the
  actor loss, sampling and PPO path are unchanged.
- The model does gain new heads, so v23 checkpoints do not load (architecture
  v8).
- A diagnostics CSV with an older header is never appended to. New rows go to
  a `_schema_v24` sibling.

### Verification

- Torch-free: `tests.test_nmcc_policy_improvement` 13/13,
  `tests.test_counterfactual_branch` 10/10, `tests.test_training_curriculum`
  10/10.
- Static checks: `flake8 --select=F` is clean on all changed files; pyright has
  no new errors against the baseline (126 vs 127).
- Reference learner (`runs/nmcc_pi_reference_run1_20260921.json`: 6 iterations
  of 4 episodes, 8 held-out seeds):
  - Held-out return went from +0.096 (current actor) to +0.247 after one update
    and +0.26–0.30 afterwards, versus heuristic +0.100 and route saving +0.280.
  - Versus the heuristic: +0.146 to +0.196, with every CI above zero from
    iteration 2.
  - The policy plateaus at route-saving level. The linear M-step realized only
    0.02–0.06 of the 0.5-nat target, ending at its 200-epoch limit every time.
- **Not run:** `tests/test_nmcc_pi_torch.py` and any torch execution of the
  port, because torch is unavailable in the review environment.

## 2026-09-21 — Learner audit: step size, initialization, signal flow, critic isolation (v24 amendment)

Evidence: `docs/NMCC_POLICY_IMPROVEMENT_20260921.md` §9,
`learner_flow_experiment.py`, `runs/learner_flow_experiment_{8d,32d}_20260921.json`.
v24 had not yet been run, so this amends model version 24 in place. No
checkpoint or artifact of the earlier v24 code exists.

### Findings

- **Step size was the binding constraint, not the step direction.**
  - v22 made 36 accepted actor steps in 64 episodes. The residual RMS grew at
    about 0.003 logits per step, reaching 0.12.
  - v23 moved the residual by 0.0023 logits in two updates.
  - In the numpy replica (hand-written gradients, finite-difference error
    4e-9), the v23 budget closes under 1% of a 0.5-nat target per rollout.
    The v24 M-step closes 20–28% at lr 3e-4 and about 74% at 1e-3.
- **Initialization costs one step.** The zero readout sends exactly zero
  gradient to the encoder and LSTM on the first actor step: representation
  share 0.000 in the replica, residual exactly 0 after v23's first update.
  - With Adam the delay is one step. A 0.05-logit random readout gave the same
    actor regret and a worse early critic ranking, so it was not adopted.
- **The critic was over-isolated.**
  - In `actor_owned` mode the encoder and LSTM receive only policy-loss
    gradient.
  - The TD, NMCC-outcome and within-state intervention-value losses fit heads
    on features they cannot shape. v22's explained variance stayed between
    −0.006 and 0.031.
  - Replica (32-dimensional input): a frozen-representation head plateaus at a
    held-out within-state Spearman of 0.15; a shared representation reaches
    0.28 and is still rising.

### Implementation

- `GNN.EvacPolicy.parameter_roles()`: a validated three-way partition into
  representation, actor head and critic head.
- `RLBridge`:
  - New `representation_mode` setting: `actor_owned` (default, legacy) or
    `shared_phasic`.
  - In `shared_phasic` the critic optimizer also owns the representation, with
    separate Adam moments.
  - The critic loss adds `representation_clone_coef · KL(π_ref‖π)` (exact).
  - After each critic epoch, the full-batch drift is checked; an epoch that
    exceeds `representation_kl_cap` is rolled back transactionally together
    with its optimizer moments.
  - `_set_optimizer_partition_trainable` handles shared tensors.
- PI M-step: an epoch that violates the trust region is rolled back and
  retried at half the rate, up to 3 consecutive times. PPO still stops at the
  first violation.
- 12 learner-flow diagnostics, measured before clipping: per-role gradient
  norms, representation gradient share, drift, rollbacks and readout norms.
- Signature `learner_contract`.
- Config keys `representationMode`, `representationCloneCoefficient` and
  `representationKlCap` (through `NMCC_PI_CORE_FIELDS`). Core records them
  under `policy_improvement_learner`.
- The v24 curriculum sets `shared_phasic` and actorLearningRate 1e-3.
- The GNN zero-readout initialization is unchanged. The measured reasoning is
  recorded in a code comment.

### Compatibility

- `actor_owned` reproduces the previous partition and parameter order
  exactly. `tests.test_nmcc_integration`'s ownership test is unchanged.

### Verification

- Torch-free: `tests.test_nmcc_policy_improvement` 13/13 and
  `tests.test_training_curriculum` 10/10 pass.
- Static checks: `flake8 --select=F` is clean; pyright has no new errors
  against the baseline.
- New torch tests in `tests/test_nmcc_pi_torch.py`: role partition, trainable
  sets per pass, representation gradient from both passes, representation
  movement, and drift within the cap.
- **Not run:** the torch tests. torch cannot be installed in this environment
  (PyPI and download.pytorch.org are both refused by the egress proxy).

## 2026-09-21 — Audit of "weak action-differential signal, tiny policy change" (v24 amendment 2)

Evidence: `docs/NMCC_POLICY_IMPROVEMENT_20260921.md` §10,
`nmcc_pi_signal_audit.py`, `runs/nmcc_pi_signal_audit_20260921.json`,
`runs/nmcc_pi_signal_audit_states_20260921.pkl`.

The audit used 43 real decision states from 12 on-policy episodes of the
calibrated testbed. The behavior policy was the v24 zero-residual
route-saving actor. Every feasible cell was branched to the horizon under 4
independent CRN tapes, and each target was scored on tapes it did not use.

### Findings

- **Signal is not weak where the value is.** Within-state signal SD vs one-tape
  noise SD: 0.089 vs 0.053 at t=1 (SNR 5), 0.068 vs 0.027 at t=11 (SNR 15).
  - Late decisions carry little value and little noise.
- **The one-tape E-step target is valid.** It captures 51% (t=1) and 60%
  (t=11) of the best available within-state gain, and the gain is positive
  in 92–100% of states.
  - A second tape adds about 1 point; the share per update is set by ε, not by
    noise.
- **Conversion.** Replica M-step, leave-episodes-out:
  - v23 budget: 0% of the target fitted, realized KL 0.000, valid gain
    +0.0001 of 0.082. The reported failure is confirmed for v23.
  - v24: 60% fitted, realized KL 0.2, held-out valid gain +0.022 of 0.082
    (78% of the in-sample gain).
  - 96 epochs adds only +0.003 held-out and lowers held-out top-1, so the
    budget stays at 32 epochs.
- **Latent failure in the v24 trust region.** The E-step bounds KL(q‖π_old);
  the M-step capped KL(π_old‖π_new) at 0.5.
  - Fitting the real targets exactly needs 0.68–0.71 in that direction at
    t=1/11 (up to 1.19). The batch mean here was 0.43, so it did not bind.
  - When it did bind, the rollback halved the learning rate, and growth is
    only 5% per update. Three halvings would leave the actor at 1/8 of its
    rate for about 40 updates, longer than the pilot.

### Implementation

- In PI mode the trust region bounds KL(π_new‖π_behavior), the E-step's own
  direction.
  - `nmccPiKlCap` defaults to 0.6; a cap below ε is rejected.
- An overshooting epoch is scaled to the boundary by an 8-step bisection on
  the parameter step (`_line_search_to_trust_region`), accepted, and the
  M-step stops.
- The learning rate is reset to `actor_lr` at the start of every PI update,
  and the lr growth rule applies only in PPO mode. PPO keeps its old rollback.
- New diagnostics: `actor_trust_region_reached`, `actor_line_search_fraction`
  and `actor_reverse_kl`. The signature `learner_contract` records the
  trust-region rule.
- The v24 curriculum sets `nmccPiKlCap` to 0.6.

### Verification

- Torch-free tests pass: `tests.test_nmcc_policy_improvement` and
  `tests.test_training_curriculum`.
- flake8 F is clean; pyright has no new errors (126 vs 127).
- New torch tests: a cap below ε is rejected; the line search lands inside
  the region and within 2^-8 of the boundary. Not run, because torch is
  unavailable here.

### Remaining limit

About a quarter of the available within-state gain is captured per update on
held-out states. With 8 updates in 64 episodes, compounding is the binding
limit. The number of episodes per update is set by `multicity_backtest.py`
(`balanced_rollout_episodes`), not by the curriculum.

## 2026-09-27 — Identifiability of the siting differential, and the branch-budget allocation

Mode: `experiment` + `analysis`. No model code was changed.

Question, as raised: every installation adds the same equal-capacity token, so
every action improves evacuation progress; can the reduction speed of
hazard-weighted exposure time still separate an optimal from a sub-optimal
siting?

### Design

Four measurements, all on the calibrated testbed, all composing with the
existing v25 signal audit:

- `exposure_signal_decomposition.py` — two-way ANOVA per decision state
  separating the state level, the cell effect, the tape effect and the pairing
  residual, at +10/+20/+30 and the horizon.
- `exposure_label_estimators.py` — which one-tape label ranks cells best at
  equal simulation cost, with a leakage-free held-out tape as truth.
- `exposure_budget_allocation.py` — `corr(a_window, a_full)` estimated from
  tape-disjoint cross-products (so residuals cannot inflate it), and the
  resulting cells x tapes x horizon allocation frontier.
- `exposure_component_signal.py` — new collection: the same 16 episodes and
  tape keys at the first decision, recording the four signed reward components
  separately. Its summed components reproduce the audit's scalar values to
  9.99e-16, which the script asserts as a cross-check.

### Findings

- The capacity effect is real and large — installing anywhere rather than
  deferring one interval is worth +0.2316 against a within-state cell spread of
  0.0677, a 3.4x ratio — but a CRN-paired within-state contrast removes it
  identically, because both arms install one identical token from the same
  snapshot under the same tape. The binding constraint is the pairing residual:
  0.0659 against a cell effect of 0.0620 at the first decision, SNR 0.94.
- The differential is not concentrated in hazard-weighted exposure. Shares of
  the total cell-effect variance: safe_completion 0.18, evacuation_time 0.13,
  hazard_exposure 0.03, casualty 0.02, and 0.64 in the covariance between
  components. Only evacuation_time has SNR above one (1.14). Ranking on
  hazard_exposure gives Spearman 0.398 against the scalar return's 0.432; a
  least-squares re-weighting of all four reaches 0.453 with a lower top-1 rate.
  The component-selection lever is measured and negative.
- Early between-cell differences are transient: the safe-completion
  differential peaks at +20 at 1.74x its terminal magnitude and erodes to 1.0.
  So a noiseless +20 contrast correlates only 0.598 with the full-horizon cell
  effect at the first decision (0.752 pooled over all decisions); +30 reaches
  0.813; the full-horizon self-check is 0.997/0.999. Truncation reorders cells
  rather than merely shrinking the signal, and
  `DEFAULT_NMCC_PI_BRANCH_HORIZON = 20` with a bare truncation in
  `BranchValuer._continue` caps the label at Spearman ~0.55 at any tape count.
- Budget should buy horizon first, then tapes: at equal cost +30 x 2 (0.632)
  beats +20 x 3 (0.531), and the full-horizon row is the frontier above 59
  steps per branch (full x 2 0.751, x 3 0.805, x 5 0.863).
- The v25 prior correlates 0.338 with the terminal cell effect; the true best
  cell is in its top 6 only 44% of the time and its top 8 56%, so
  `nmccPiMaxBranches = 6` caps top-1 at 0.44 whatever the label.
- Exhaustive single-tape branching is dominated. v26's first decision (17 cells
  x 1 tape x full, 1003 steps) gives top-1 0.359 and regret 0.551 cell-effect
  SD; 9 cells x 2 tapes x full (1062 steps) gives 0.429 and 0.416 — a 25%
  regret reduction at constant cost, with the interior optimum at 9-12 cells.

### Consequence for convergence

With one tape at the full horizon the label's rank agreement with the quantity
it represents is 0.731 pooled and 0.636 at the first decision. The information
is lost in the label, upstream of the network and of every credit-assignment
mechanism that consumes it; a deployment gate scored against that label inherits
its 0.64 discrimination regardless of `z` or the number of consecutive updates.

### Verification

- The tape-disjoint covariance estimator is validated by its own
  full-versus-full self-check (0.997 first decision, 0.999 all decisions).
- The component collection is validated against the v25 audit's stored scalar
  values (16 states, largest absolute difference 9.99e-16).
- The correlation-to-Spearman and correlation-to-top-1 conversions use a
  Gaussian surrogate over exchangeable cells and are indicative; the induced
  orderings are robust because the underlying correlations differ by far more
  than the surrogate's error.
- 16 episodes and 17 feasible cells on average at the first decision, so
  intervals are wide; no claim here is about State College.

Artifacts: `docs/SITING_SIGNAL_IDENTIFIABILITY_20260927.md`,
`runs/exposure_signal_decomposition_20260927.json`,
`runs/exposure_label_estimators_20260927.json`,
`runs/exposure_budget_allocation_first_20260927.json`,
`runs/exposure_budget_allocation_all_20260927.json`,
`runs/exposure_component_signal_20260927.json`,
`runs/exposure_component_states_20260927.pkl`.

## 2026-09-30 — Seattle population-adaptive three-stage training launch

Mode: `implementation` + `validation` + `experiment`.

- Added `population_density_adaptive` partitioning. A dedicated scenario RNG
  stream now draws the exact pedestrian origins before the grid is cut; each
  road node is weighted by the represented people who will actually begin
  there. Weighted marginal quantiles make dense-origin regions narrower while
  the outer boundaries cover the complete network and the 8x8 tensor/action
  contract remains unchanged. The same sampled origins are then consumed by
  `PedDS`, so partition mass and initialized population cannot disagree.
- Registered
  `config/seattle_training_curriculum_population_foresight_v37.json`: 160
  Seattle episodes across 2,500/5,000/7,500/10,000 people. Stage E is 24
  episodes (six environment-only rollout updates), stage A is 48 episodes
  (twelve decision-interval action updates with the zero-initialized
  continuation head frozen), and stage F is 88 episodes (twenty-two joint
  updates with backward continuation-value learning across eight sequential
  installations). The NMCC M-step ceiling is 256 epochs with exact top-1 and
  absolute-KL stopping; no action prior, gate, fallback, or reward change was
  added.
- Seattle map preflight passed on the cached 5 km walk network: 22,278
  consolidated nodes, 79,174 edges, and 3,834 recognized shelter sites. A real
  simulator smoke conserved 100/100 people and installed 10/10 shelters.
  In the first full 2,500-person episode, population mass was exactly 2,500,
  cell area ranged from 0.542 to 6.005 km², and log area versus realized
  population-density correlation was -0.541.
- Verification: all 266 tests pass. The long run is live at
  `runs/seattle_population_foresight_long_20260930`; episode summaries and PPO/
  NMCC diagnostics are written incrementally.

## 2026-10-01 — NMCC-only causal actor and exact branch acceleration

Mode: `implementation` + `validation`.

- Stopped the unfinished training run and preserved its partial artifacts. No
  `multicity_backtest` or training process remains active.
- Exact NMCC now evaluates all candidates across all future tapes as one
  physical batch. Action-independent hazard evolution is computed once per
  tape and replayed exactly; four isolated simulator-only workers execute the
  candidate pedestrian, routing, congestion, reservation, and exposure
  dynamics without cloning or importing the neural learner.
- The continuation network evaluates every returned branch endpoint in one
  neural batch. Serial and parallel candidate tables are bitwise identical,
  including endpoint observations and live-state restoration.
- Exact NMCC is now the actor's sole objective when policy improvement is
  enabled. PPO still trains the critic, but its sampled-return actor gradient
  no longer competes with the common-noise full-action target. Missing NMCC
  targets raise an error; PPO is not a fallback.
- Continuation values are learned by backward realized returns after the
  environment-only phase, including action warmup. M-step convergence is now
  defined by the optimized `KL(q || pi)` residual; top-1 remains descriptive.
- On the 2,500-person branch benchmark (18 candidates, two tapes, 15 steps),
  four-worker exact evaluation took 35.326 s versus 120.469 s serial (3.41x).
  The full counterfactual and v37 curriculum suites pass (32 tests total), as
  does the focused NMCC convergence contract in the configured PyTorch runtime.

## 2026-10-02 — Clean three-stage ownership and v40 validation

Mode: `implementation` + `validation` + `experiment`.

- Corrected the staged-transfer ownership error. The shared pedestrian,
  hazard, infrastructure, relational-message, pooling, and LSTM representation
  is now learned in Stage E from six observable one-interval physical rates.
  The candidate encoder, candidate-conditioned graph attention, global and
  temporal actor adapters, and actor readout belong to the actor optimizer, so
  Stage A/F can learn a nonlinear siting policy after the environment
  representation is frozen.
- The Stage E dynamics head no longer depends on the actor embedding. It reads
  the candidate's regional embedding, six transparent candidate descriptors,
  the full graph/global context, and recurrent history. Stage A receives the
  current pedestrian distribution, assignment flow, mobility/congestion,
  routes, danger and one-interval forecast, capacity/infrastructure, spatial
  and route edges, wind, and deployment state. Stage F adds the learned
  post-interval continuation differential to the same common-noise physical
  contrast; the actor then distills that forward-looking target.
- Preserved one coherent actor objective: in policy-improvement runs NMCC's
  signed sampled installed-site distribution is the actor target, while PPO's
  complete semi-Markov returns train the value heads and the dedicated
  backward continuation value. PPO return gradients do not alternate with the
  NMCC actor fit. Stage A trains but does not consume continuation; Stage F
  consumes it. No gate, fallback, heuristic target, or model-filled action is
  present.
- Corrected single-update stage schedules so a short stage uses its declared
  high starting exploration and learning rates. Full stages independently
  decay categorical uniform mixing, temperature, entropy, actor LR, and critic
  LR from their own first to last update.
- Extended only the exogenous hazard audit after a physically early terminal,
  without moving pedestrians or adding reward, so policies with different
  completion times can still be verified against a byte-identical full hazard
  trajectory. A controlled prefix comparison confirmed the prior digest
  mismatch was solely unequal trace length.
- Removed unused NMCC ensemble, teacher, guidance, uncertainty, correction-
  gate, alternative actor-objective, heuristic-base, model-fill, and obsolete
  pretraining controls. Removed the historical v28-v34 curriculum tests that
  enforced those retired interfaces and migrated the maintained v36/v37
  configurations.
- A fresh 12-episode/500-person smoke completed all E/A/F stages and matched
  evaluation. Dynamics explained variance was 0.390 after E and 0.724 after F;
  Stage A target top-1 moved 0.219→0.438; Stage F moved 0.406→0.688;
  continuation supplied 16.2% of the branch-value spread. Interface parity
  passed. The smoke policy remained inferior to objective-greedy, as expected
  from one update per stage, so it is mechanics evidence only.
- Verification: 194 current learning, branch, capacity, routing, partition,
  population, curriculum, benchmark, and campaign tests pass. The full
  5,000-person State College v40 run was launched at
  `runs/state_college_5000_staged_nmcc_v40_20261002` with 24 E, 48 A, and 88 F
  episodes, followed by 20 matched deterministic evaluation scenarios.

## 2026-10-03 — Coherent counterfactual PPO and forward-risk potential (v41)

Mode: `research` + `implementation` + `validation`.

- Stopped the incomplete v40 campaign. Legacy Anaconda branch/test processes
  were signaled for termination; macOS reports them as uninterruptible/exiting
  with zero CPU and memory, while no `multicity_backtest` training process is
  active. Validation uses the repository `.venv` runtime.
- Replaced the separate NMCC actor fit with one actor update: matched
  common-random-number branch contrasts estimate the action advantage, and
  PPO alone applies that advantage through its clipped surrogate. The ordinary
  actor-critic value head is also the NMCC branch-end continuation value; the
  dedicated continuation head was removed.
- Changed the learning reward to the semi-Markov potential residual
  `D_ctrl(s_k) - gamma^dt D_ctrl(s_{k+1}) - C_ctrl,k`, normalized by population
  and horizon. `D_ctrl` is capacity-valid forecast plain-plus-danger-weighted
  travel debt and `C_ctrl,k` is realized controllable person-time. Panic and
  casualty still evolve and are reported, but are excluded from actor credit.
- Aligned dynamic NMCC branches with the same controllable objective. Every
  compared action uses the same initial state, equal capacity, and common
  exogenous future, so hazard/panic/self-evolution variation cancels in the
  within-state action contrast rather than becoming a policy label.
- Reduced the policy observation to five pedestrian, five hazard, seven
  infrastructure, eleven global, four momentum, and four candidate features.
  Removed assigned-flow and mobility duplicates; wind is one signed east/north
  velocity pair. Hazard foresight is represented by current danger, one-step
  spread momentum, next-decision expected danger, escalation probability, and
  forecast uncertainty without consuming simulator random draws.
- Added the 4x4, 2,500-person v41 E/A/F curriculum. Fixed experiment and run
  metadata so decision interval, reward, actor optimizer, continuation value,
  actor prior, and KL role describe the effective configuration.
- Verification: 155 focused learning, branch, curriculum, metadata, feature,
  and forecast-contract tests pass. A full 2,500-person training/evaluation
  run has not yet been used to claim policy superiority.

## 2026-10-03 — v41 2,500-person staged pilot and matched evaluation

Mode: `implementation` + `validation`.

- Completed a 20-episode State College pilot on the 4x4 adaptive partition:
  4 Stage E, 8 Stage A, and 8 Stage F episodes, followed by four matched
  deterministic scenarios for RL, objective-greedy, active-population,
  risk-reduction, random, and initial-only policies. Every sequential policy
  installed 2 initial plus 8 dynamic shelters at equal 15-minute intervals;
  total installed capacity was 2,700 for 2,500 pedestrians.
- Verified the live observation rather than only its schema: all 17 cell, 11
  global, 4 momentum, and 4 candidate channels were finite; all four momentum
  channels were nonzero; route-flow edges and feasibility masks were present;
  and every pending actor transition had matched NMCC branch support.
- NMCC/PPO action credit was active and signed. Across the four actor updates,
  NMCC advantage SD ranged from 0.133 to 0.352, actor gradients were nonzero,
  branch coverage was 100%, and PPO retained nonzero policy movement without
  clipping or rollback. The environment stage kept actor gradients exactly
  zero while fitting the shared dynamics representation.
- The short pilot did not learn a useful continuation value. Stage F
  continuation validation explained variance ended at 0.0020, continuation
  supplied only 0.36% of branch-score dispersion, and it changed no top-ranked
  branch. The actor and immediate matched credit work; long-horizon foresight
  is not yet established and requires substantially more continuation data.
- On four matched evaluation scenarios, RL's mean realized hazard-weighted
  time to safety was 223,615 person-minutes versus 220,733 for
  objective-greedy (1.3% worse; two wins), 259,145 for active-population
  (13.7% better; three wins), 286,363 for risk-reduction (21.9% better; three
  wins), and 290,040 for random (22.9% better; four wins). RL reduced the
  hazard-exposure component 6.3% below objective-greedy while using 2.6% more
  plain travel time, indicating a distinct hazard/traffic tradeoff rather than
  heuristic imitation. Initial-only remained an intentionally privileged
  lower bound because it installs all ten shelters at time zero.
- Corrected two audit-only discrepancies exposed by the pilot: a loaded
  optimizer now immediately adopts the new stage's learning-rate schedule,
  and nested checkpoint/figure labels now describe matched NMCC actor credit,
  the shared `Q = V + A` critic, and forward evacuation-debt reward.
- Verification: 156 focused learning, branch, curriculum, metadata, feature,
  and campaign tests pass. Artifacts are under
  `runs/state_college_2500_counterfactual_ppo_v41_pilot20_20261003`.
# 2026-10-03 — Fair static precommitted `initial_only` benchmark

- Removed the artificial early-capacity advantage from `initial_only`. It now
  begins with the same two shelters as RL and installs eight equal tokens at
  the common decision epochs.
- At time zero, the benchmark greedily commits the exact eight-site order on a
  frozen planning copy using the immediate plain-plus-exposure evacuation-debt
  objective. Planned capacity and assignments update between choices, but
  physical time, traffic, pedestrians, and hazards do not advance.
- Restored the live simulator and random streams exactly after planning. At
  runtime the benchmark executes the committed candidate identities without
  re-ranking, adaptive substitution, a safety gate, or a fallback.
- Added explicit schedule and realized deployment-time metadata plus backtest
  assertions for two-initial/eight-sequential capacity and timing parity.
- Re-evaluated the corrected schedule against the frozen v41 RL checkpoint on
  the original four held-out State College scenarios at 2,500 pedestrians.
  Mean realized HWTTS was 223,615 person-minutes for RL versus 280,589 for the
  corrected static schedule: RL reduced it by 56,974 (20.3%) and won 3/4
  matched pairs. Restricted mean time to safety improved in all four pairs.
  Both arms had 10 shelters, 2,700 places, identical hazard digests, and the
  exact `[15,30,45,60,75,90,105,120]` installation schedule.

## 2026-10-04 — v43 critic-target, reporting, and efficient exact-NMCC repair

Mode: `implementation` + `validation`.

- Corrected the actor-critic identity. The NMCC actor advantage is centered on
  a uniform feasible shelter and therefore is not `Q_pi - V_pi`; it is no
  longer subtracted from the factual Monte Carlo return to construct a critic
  target. The critic now learns the positive complete-episode on-policy
  return-to-go, and Stage F evaluates each branch as its realized decision-
  interval effect plus that same critic's endpoint continuation value.
- Removed the fixed `0.05` advantage scale floor from active PPO
  normalization. Every nonconstant batch is now centered and normalized by
  its actual standard deviation; a truly constant batch maps to zero.
- Separated positive learning and evaluation quantities. Training logs retain
  the positive potential-progress return. The physical evaluation reports
  positive normalized cost (lower is better) and
  `1 / (1 + normalized_cost)` (higher is better). Intermediate simulator rows
  no longer mislabel partial learning reward as the complete episode
  objective. Stale metadata describing a negative objective was corrected.
- Replaced per-decision branch-process startup with a persistent isolated
  worker pool. Workers are reused through all eight decisions, keep exact
  snapshot/common-random-number simulator dynamics, and exchange large results
  through temporary files. Serial/parallel branch outputs remain identical;
  a local repeated-batch probe fell from about 0.96 seconds including startup
  to about 0.02 seconds on the reused pool.
- Changed the 4x4 verification curriculum to one rollout episode per update
  and 24 actor PPO epochs. The original 8 epochs left retained KL near zero;
  a rejected 64-epoch probe overshot to KL 0.03--0.09. The final fixed budget
  reuses existing branch evidence and adds no simulator calls.
- Added direct policy-improvement diagnostics: expected physically branched
  NMCC advantage before/after the actor update and its gain. This distinguishes
  correct movement from entropy, heuristic agreement, or an optional MPO
  target-fit view.
- Completed the ten-episode v43 State College smoke (`2 E + 4 A + 4 F`) at 250
  pedestrians. All learning and evaluation scores were positive; all eight
  actor updates executed; retained KL ranged from 0.00045 to 0.01297; seven of
  eight updates increased expected NMCC advantage; entropy moved from 1.000 to
  0.985; and Stage-F continuation validation explained variance rose from
  0.406 to 0.659. Exact physical branching averaged 2.92 seconds per decision
  batch and completed without deadlock or missing branches.
- Two-scenario matched evaluation is deliberately not a performance claim.
  The smoke policy scored 0.661 versus 0.737 for objective-greedy and therefore
  remains undertrained. It verifies coherent learning and policy motion, not
  convergence or superiority.
- Verification: all 274 repository tests pass under the pinned repository
  `.venv`. Final artifacts are under
  `runs/state_college_250_coherent_nmcc_v43_smoke10_r5_20261004`.

## 2026-10-04 — v44 complete cost ledger and dense NMCC-PPO repair

Mode: `implementation` + `validation`.

- Audited the six screenshot findings against v43 rather than applying them
  blindly. Four defects still reproduced: panic/casualty deletion from the
  learning cost, the Stage-F endpoint-debt sign omission, discarding three of
  four candidate labels in the actor loss, and a sampled KL diagnostic that
  was not the same categorical distribution used by NMCC. The old report that
  continuation did not learn and the proposal to replace decision-interval
  branching with full-episode rollout iteration did not reproduce and were not
  implemented.
- Unified factual reward, branch cost, and evaluation around a complete
  time-to-safety ledger. Active pedestrians remain counted after panic;
  casualties accrue plain unresolved time; common-random-number differencing
  removes shared exogenous variation instead of deleting adverse outcomes.
  The unassigned/panicked debt now uses remaining episode time rather than a
  new 180-minute cliff at every decision.
- Corrected the Stage-F paired target to
  `C_wait-C_action + gamma^dt(D_wait'-D_action') + gamma^dt(V_action-V_wait)`.
  This is the candidate difference of the same potential-shaped reward used
  by factual trajectories, so the endpoint debt cancels correctly against the
  learned continuation value.
- Replaced the executed-action-only actor surrogate with one dense clipped-PPO
  surrogate over all exactly branched actions. Advantages are centered by the
  NMCC E-step under the stored behavior distribution, RMS-scaled without
  changing within-state baselines, and zero for unbranched actions. The critic
  remains trained on complete on-policy returns.
- Replaced NMCC's sampled-action KL report with exact
  `KL(pi_new || pi_behavior)` over the stored full categorical distribution.
  This is now directly comparable with entropy, clipping, and the E-step trust
  region.
- All 277 repository tests pass. The ten-episode 250-person 4x4 smoke completed
  all E/A/F stages and two matched evaluation scenarios. Every one of eight
  actor updates increased expected NMCC advantage (mean +0.00668); exact KL
  averaged 0.00903; entropy ended at 0.926; continuation validation explained
  variance ended at 0.778; and continuation changed the branch-best action in
  25% of the final update's states.
- The shortcut diagnostic changed materially: training-return Pearson
  correlation with realized cost fell from +0.755 in the matched v43 smoke to
  -0.107, and with panic onsets from +0.787 to -0.086. On the two held-out
  scenarios, RL mean HWTTS was 15,827 person-minutes versus 15,243 for
  objective-greedy (3.8% worse), but substantially better than the v43 smoke's
  23,250. This is mechanics evidence, not a convergence or superiority claim.
  Artifacts are under
  `runs/state_college_250_complete_ledger_dense_nmcc_v44_smoke10_20261004`.

## 2026-10-04 — Complete-ledger objective-greedy primary benchmark

Mode: `implementation` + `validation`.

- Replaced maximum active population as the primary paired comparator with
  `objective_greedy`. Maximum active population remains the practice rule,
  random remains the floor, and the scheduled `initial_only` policy isolates
  the value of feedback. One-line risk/accessibility scores remain available
  as appendix diagnostics rather than separate primary claims.
- Made objective-greedy expose the same positive score used by evaluation:
  `1 / (1 + post-installation HWTTS cost / (population * horizon))`. Its
  zero-horizon assignment still enforces exact shelter capacity, current
  routes/speeds, strict-better rerouting, and the public one-interval danger
  forecast. The monotone transformation does not change its site ordering.
- Made the complete person ledger an executable invariant. Rational active,
  panicked active, casualty, and safe populations must sum to the initialized
  population. Panicked people and casualties retain unresolved time-to-safety
  debt; safe people contribute zero. A mismatch fails candidate evaluation.
- Updated single-city and multicity pairing, parity references, plots,
  manifests, campaign commands, experiment designs, result generation, and
  current protocols to name objective-greedy as primary. Default evaluation
  now runs the concise core set: RL, objective-greedy, max-population practice
  rule, random floor, and scheduled initial-only feedback control. Other
  registered strategies remain explicitly selectable.
- This changes benchmark evaluation/reporting, not PPO/NMCC learning. Existing
  v44 weights remain meaningful, but a fresh evaluation artifact is required
  for the new primary-inference schema. Python compilation and all 278
  repository tests pass.

## 2026-10-04 — v45 uniform-reference reward and full State College validation

Mode: `implementation` + `smoke validation` + `full training/evaluation`.

- Replaced the no-further-installations debt potential with the fixed uniform
  feasible-site policy value. The factual decision reward is now
  `V_mu(s_k) - C_k - V_mu(s_{k+1})`, with terminal value zero. Its episode sum
  is savings relative to uniform-random siting, so scheduled capacity arrival
  no longer gives random or bad siting an automatic positive return.
- Added a dedicated random-reference value head and replay target. Stage E uses
  exact uniform feasible actions and learns this stationary reference value;
  Stage A learns action quality over one physical decision interval with the
  reference frozen; Stage F adds a dedicated policy-savings continuation value.
  NMCC's paired label uses the same decomposition as the factual reward and
  PPO critic, so actor, critic, and continuation now optimize one Bellman
  system.
- Kept the complete person ledger. Panicked and casualty states remain in
  unresolved time-to-safety cost, while casualties stay a reported outcome
  rather than a standalone reward term. The reward is signed; below-random
  actions produce negative labels rather than being clipped or suppressed.
- Added direct tests for signed good/bad action credit, zero uniform-site mean,
  telescoping reward, backward random-policy cost-to-go, and the NMCC
  `V_mu + W_pi` branch decomposition. The complete supported suite passes:
  282 tests.
- The corrected 250-person E/A/F smoke completed all stages. After correcting
  numerical conditioning so `V_mu` learns the unamplified cost scale, Stage-E
  explained variance reached 0.83--0.92, accounting error was numerical zero,
  and all actor stages produced mixed positive/negative labels and nonzero
  gradients. Its eight actor episodes were intentionally too small for a
  performance claim.
- Completed the 96-episode, 2,500-person 4x4 curriculum and deterministic
  20-scenario evaluation at
  `runs/state_college_2500_random_policy_value_v45_full96_20261004`.
  Stage E's last-eight uniform-policy mean return was `0.016 +/- 0.042 SE`;
  Stage A mean return was `+0.065`; Stage F mean return was `+0.083`. Maximum
  reward accounting error was `1.18e-16`. In the final eight Stage-F episodes,
  continuation validation explained variance averaged `0.443`, continuation
  supplied `0.739` of branch-score variation, and it changed the top branch in
  `29.7%` of decisions.
- Held-out evaluation confirms the reward-floor claim but not heuristic
  superiority. RL normalized cost was `0.5760`, significantly below random's
  `0.6471` (paired saving `0.0710`, 95% bootstrap CI `[0.0166, 0.1305]`, 16/20
  wins). RL was statistically tied with complete-ledger objective-greedy
  (`0.5733`; paired difference `-0.0027`, CI `[-0.0510, 0.0580]`) and slightly,
  nonsignificantly worse than the best one-line heuristics (`0.5592--0.5627`).
  The corrected scheduled `initial_only` control scored `0.6767`; the much
  stronger `static_greedy` and `rl_precommit` references deploy at time zero
  and are not fair online comparators.
- The run is not a convergence claim. It has only 96 episodes versus the
  registered 100-episode minimum, and 85% of tail actor updates exceeded the
  descriptive `0.015` KL target. The return curve is positive and stationary
  in its tail, but optimization remains noisy. Casualties remain stochastic
  reporting: RL had 28.15 mean versus objective-greedy's 26.7, with no
  significant paired difference.
- Corrected the diagnostic plot's stale evacuation-debt label. It now names
  the uniform-random-relative return, and the run includes
  `random_reference_reward_validation.png` for Stage-E calibration,
  continuation quality/influence, stagewise exploration, and the exact
  telescoping-accounting check.

## 2026-10-05 — v48 restored exploration and calibrated NMCC actor M-step

Mode: `implementation` + `full test` + `smoke validation`.

- Restored independent Stage-P exponential exploration decay: uniform action
  mixing now moves from `0.35` to `0.02`, categorical temperature from `1.25`
  to `0.8`, and the actor learning rate from `0.001` to `0.0001`. Stage E
  remains exactly uniform by its reference-policy contract and does not train
  the actor.
- Replaced raw, small-magnitude branch-advantage regression with the coherent
  NMCC M-step. Each retained decision stores the complete dual-constrained
  E-step distribution `q`; the actor minimizes `KL(q || pi_theta)` over all
  feasible cells for 24 replay passes. Exact branched-pair ordering remains a
  diagnostic. No KL gate, rollback, gradient clipping, fallback policy, or
  action correction was added.
- A first diagnostic using pair-only target log-odds produced large movement
  but worsened full-distribution target KL because unbranched logits were
  unconstrained. That run was stopped and the incomplete design was replaced,
  not retained as a fallback. A direct integration test now requires every
  actor update to reduce KL to its NMCC target.
- All 290 tests pass. The completed eight-episode, 250-person smoke and two
  matched held-out scenarios are at
  `runs/state_college_250_target_v48_actor_scale_smoke8_r2_20261005`.
- Across six Stage-P updates, mean policy KL was `0.00829` versus `0.0000271`
  in v47; final entropy was `0.9464` versus `0.9998`; mean target KL fell from
  `0.1717` before each M-step to `0.1484` after it; pairwise branch-order
  accuracy averaged `0.823`; and expected NMCC advantage gain averaged
  `0.00794` versus `0.000235` in v47.
- On the same two held-out scenarios, v48 RL normalized cost was `0.3370`,
  improving on v47 RL's `0.3415`, random's `0.3559`, max-active's `0.3430`,
  and scheduled initial-only's `0.3686`. Objective-greedy remained stronger at
  `0.3133`. This smoke validates update direction and scale, not convergence
  or benchmark superiority.

## 2026-10-06 — v49 coherent NMCC policy iteration

Mode: `implementation` + `full test` + `staged smoke validation`.

- Corrected the NMCC E/M distribution mismatch. The E-step now constructs its
  target from the raw masked actor distribution that the M-step and
  deterministic evaluation actually use. Temperature and uniform mixing remain
  data-collection exploration only.
- Removed persistent actor-target replay. Every NMCC target is now fitted for
  24 passes inside the policy iteration that generated it and then discarded;
  targets tied to older actor priors or older continuation values are never
  optimized by a later policy.
- Removed the separate continuation replay learner. Stage P uses the same
  critic trained on the current on-policy return-to-go for both actor branch
  continuation and value diagnostics. The fixed uniform-random reference value
  remains the only persistent replay dataset because its policy never changes.
- Added regression tests that require the stored NMCC prior to equal the raw
  actor while the sampled action still follows the exploration distribution,
  prohibit actor/continuation replay in checkpoints, require every direct
  actor fit to reduce target KL, and require continuation diagnostics to refer
  to the shared on-policy critic. All 290 repository tests pass.
- Completed the 2E + 6P, 250-person staged smoke and two matched evaluation
  scenarios at
  `runs/state_college_250_target_v49_smoke8_seed4902_20261006`. Every one of
  the six Stage-P M-steps reduced target KL; the mean reduction was `0.0320`,
  every update increased expected NMCC advantage, and normalized entropy moved
  from `1.0000` to `0.9388` as exploration decayed from `0.35` to `0.02`.
- This short checkpoint is not yet a good policy. Its held-out normalized cost
  was `0.3384`, versus `0.3280` for random and `0.2925` for objective-greedy.
  Six actor iterations validate data flow and update direction, not convergence
  or benchmark superiority. The continuation critic also remains immature:
  on-policy explained variance ended at `-0.022` and its branch-score share was
  only about `0.010`.
- No new worker was left by the successful v49 run. Twenty-five workers from
  older interrupted runs remain in macOS `UE` (uninterruptible/exiting) state,
  parented to PID 1 with 4 KB RSS. Both TERM and KILL were delivered to their
  exact PIDs; the kernel will not reap them before reboot.

## 2026-10-06 — v50 simulator-graded residual rollout policy iteration

Mode: `implementation` + `full-horizon branch integration` + `smoke training`.

- Replaced the active four-cell, one-interval MPO target and learned endpoint
  continuation with complete simulator rollouts. Each tested cell starts from
  the same exact snapshot and keyed future; later installations are selected
  deterministically by a frozen copy of the current GNN/LSTM actor. No critic,
  value bootstrap, gate, rollback, or model-filled action grades the actor.
- Made the actor explicitly residual: the fixed base logit is the normalized
  exact objective-greedy assignment gain already present in the observation,
  and the zero-initialized neural actor learns a bounded correction. At
  initialization deterministic RL therefore selects exactly the myopic base.
- Replaced fixed-size MPO updates with inverse-variance-weighted regression on
  every pair of tested cells. Targets are complete cost-to-go differences;
  weights use the variance of paired common-random-number differences. Actor
  target replay remains iteration-local, so older exploratory panels do not
  outvote a new frozen-policy iteration.
- Replaced four uniformly sampled branches with a bounded coverage panel: the
  current actor's strongest cells plus two least-evaluated feasible cells.
  Per-cell counts persist in the checkpoint and rotate neglected feasible
  cells into later panels. `nmccPiMaxBranches=0` requests exhaustive grading.
- Preserved the v49 random-policy reward definition and complete ledger. In
  the single policy-improvement stage, `V_mu` is evaluated directly by a
  uniform-siting Monte Carlo continuation at each decision rather than by an
  untrained Stage-E value approximation. Observations, simulator dynamics and
  the held-out evaluation objective are unchanged.
- Added unit tests for coverage rotation, exhaustive mode, required executed
  action inclusion, paired-variance weighting, correct lower-cost ranking,
  exact zero-residual objective-greedy behavior, isolated worker execution and
  no critic update in the rollout actor path.
- Completed a four-episode, 250-person smoke at
  `runs/state_college_250_rollout_pi_v50_smoke4_seed5001_20261006`. Both actor
  fits were nonzero; pairwise accuracy was `0.851` then `0.806`, policy KL was
  `0.0805` then `0.0280`, and measured expected rollout-score gain was
  `+0.1159` then `+0.0119`. The critic made zero updates by construction.
  Fifteen of sixteen cells were graded; the remaining cell had no feasible
  sampled shelter candidate. Average recorded branch time was `5.54 s` then
  `4.59 s` per decision. Tqdm displayed one misleading long elapsed interval,
  but logged episode wall times were `70–92 s` for the updated episodes.

## 2026-10-06 — v51 residual and diagnostic coherence review

Mode: `targeted implementation` + `full regression` + `fresh smoke training`.

- Removed the residual actor's `4*tanh(.)` ceiling. The zero-initialized neural
  correction is now a linear readout added to the fixed, normalized
  objective-greedy base score, so large simulator-measured corrections retain
  their scale and gradient instead of saturating.
- Removed the obsolete categorical improvement target from active transitions
  and checkpoints. Each frozen-policy iteration fits all of its physically
  branched pair contrasts for the configured number of passes and then clears
  them. Only fixed-policy random-reference data can use persistent replay, and
  the one-stage v51 curriculum does not populate that replay.
- Split entropy reporting into two explicitly different quantities: episode
  `entropy` is the normalized distribution actually sampled after one
  temperature transform and one uniform exploration mixture;
  `actor_entropy_before/after_update` is the raw temperature-one actor that
  paired rollout regression fits and deterministic evaluation ranks.
  Evaluation now applies neither training temperature nor exploration mixing.
- Replaced stale target-KL diagnostics with actual pairwise regression loss and
  accuracy before/after fitting, raw residual RMS/max, and exact
  `KL(new actor || frozen iteration prior)`. KL is descriptive: there is no
  acceptance gate, rollback, clipping target, or correction controller.
- Removed the retired held-out controller gate from experiment convergence.
  Rollout policy iteration is audited using complete branch-target coverage,
  finite pairwise fits, complete rollout batches, and tail stationarity; its KL
  cannot block evaluation. Updated training figures distinguish sampled
  behavior entropy from raw actor entropy and no longer draw a PPO KL limit for
  the rollout learner.
- The full repository regression suite passes `299` tests. A fresh four-episode
  State College smoke completed at
  `runs/state_college_250_rollout_pi_v51_smoke4_seed5101_20261006`. Both actor
  fits reduced paired loss (`1.0200 -> 0.8547`, `0.6987 -> 0.5560`) and improved
  pairwise accuracy (`0.653 -> 0.792`, `0.648 -> 0.690`). Raw residual RMS moved
  `0 -> 0.581` on the first update and `0.516 -> 0.386` on the second; exact
  actor KL was `0.0500` and `0.0689`. Behavior entropy was logged separately
  from raw actor entropy, the actor made `32` optimizer steps, the critic made
  `0`, pending actor transitions were `0`, and persistent reference replay was
  `0`. Fifteen of sixteen cells were graded; the ungraded cell was never
  feasible in the smoke scenarios.
- This smoke validates wiring and optimizer response only. It does not establish
  convergence, foresight, or superiority over objective-greedy; those require
  the 96-episode multi-seed training and matched deterministic evaluation.

## 2026-10-07 — v52 diagnostic and decision-aligned rollout fit

Mode: `full diagnostic` + `targeted learner/runtime correction` + `matched smoke`.

- Confirmed that per-state target normalization erased action consequence, the
  minimum sampled variance made inverse-variance fitting ill-conditioned, and
  precision-only regression overvalued immaterial near-ties. The final actor
  fit keeps one iteration-wide target scale and weights paired regression by
  measured regret divided by paired standard error.
- Forced every rollout panel to measure both the executed action and the exact
  objective-greedy base. Production policy improvement now uses four shared
  futures; a six-future audit found only 40% two-tape versus 73.3% four-tape
  top-cell agreement with the six-tape ranking.
- Added bounded worker protocol reads, transactional cleanup after partial
  startup or interruption, and a parent-death watchdog. Three repaired smokes
  added no new orphan workers. Historical macOS `UE` workers remain kernel
  stuck and require a restart.
- The final matched smoke update improved cost-aware pair accuracy
  `0.844 -> 0.858` and expected rollout advantage `0.00206 -> 0.00606`, with
  KL `0.266` and raw actor entropy `0.836 -> 0.820`. The focused regression
  suite passes 166 tests.
- Two deterministic held-out smoke scenarios beat random but trailed
  objective-greedy. A six-state foresight ablation found no useful momentum or
  forecast dependence after one update. These are explicit undertraining
  results; no benchmark-superiority or foresight claim is made.
- Full evidence and the next-run contract are in
  `docs/ROLLOUT_PI_V52_DIAGNOSTIC_20261007.md`.

## 2026-10-07 — v54 coherent terminal clearance-tail objective

Mode: `objective extension` + `pipeline regression` + `four-episode smoke`.

- Added one explicit service-level term to the complete person-minute ledger:
  the slowest 10% of restricted safety-completion times receive unit extra
  weight. T90, the slow-tail mean, and T100 are reported separately. Safe
  completion times are retained exactly, unfinished people retain their
  capacity-valid remaining time, and a casualty can never improve clearance.
- Applied the same term to immediate objective-greedy scores, full-horizon
  simulator-graded actor branches, frozen-policy continuation, uniform-random
  reference continuation, factual reward closure, and deterministic
  evaluation. The reward remains savings relative to uniform-random siting;
  no standalone installation, completion, or realized-death bonus was added.
  Expected casualty loss remains Rao-Blackwellized in the complete ledger, so
  the return already measures expected casualties avoided relative to random
  without adding a second noisy Bernoulli term.
- Added exact weighted completion-time bookkeeping for grouped pedestrians,
  explicit `clearanceTailFraction` and `clearanceTailExtraWeight` curriculum
  parameters, v54 smoke/full curricula, logger fields, and regression tests.
  Older experiment files retain their original objective because the new
  extra weight defaults to zero.
- The supported runtime passed 115 RL-framework tests plus 62 focused rollout,
  benchmark, branch, curriculum, and population tests. The four-episode
  State College smoke completed at
  `runs/state_college_250_rollout_pi_v54_tail_smoke4_seed5401_20261007`.
  Reward accounting error was at most `5.6e-17`; the one actor update reduced
  paired loss `1.234 -> 0.558` and improved pairwise ranking accuracy
  `0.840 -> 0.899`. This validates wiring, not policy performance.
- In the two harsh smoke scenarios at least 10% of the population remained
  unresolved, so the slow-tail mean reached its declared 360-minute ceiling.
  This is a useful sensitivity warning: the complete base ledger continues to
  distinguish policies, while the tail fraction should be reported in a
  prespecified sensitivity analysis rather than tuned on held-out results.

## 2026-10-08 — v55 constrained categorical rollout policy iteration

Mode: `learner replacement` + `full regression` + `background comparison campaign`.

- Replaced pairwise score regression as the actor's optimizing loss with a
  listwise categorical M-step, `KL(q || pi_theta)`, over the complete feasible
  action table. The NMCC E-step computes `q` from full-horizon, common-random-
  number simulator costs and constrains every iteration to
  `KL(q || pi_old) <= 0.05`; ungraded actions retain their frozen-policy mass.
- Preserved the zero-initialized objective-greedy residual architecture. The
  fixed greedy logit remains present after every update, while the constrained
  categorical target prevents a single noisy iteration from replacing it.
  Paired rank accuracy is descriptive only and has no gradient authority.
- Replaced stale cross-iteration coverage selection with an iteration-local
  panel containing the executed action, objective-greedy base, strongest
  remaining base-score contenders, and two uniformly sampled feasible extras.
  The first two high-impact installations are exhaustive. Four shared futures
  grade each selected cell to the physical episode horizon, and the paired
  contrast standard error supplies the categorical temperature floor.
- Actor targets are discarded after each four-episode policy iteration. No
  actor replay, branch-label replay, cell-coverage state, critic grade, learned
  continuation grade, correction gate, rollback, or fallback controller is
  used. The complete ledger, random-reference reward, simulator, observation,
  and deterministic evaluation objective were not changed.
- Updated diagnostics to report categorical target KL before/after fitting,
  target-to-prior step KL, target temperature, paired contrast uncertainty,
  eligible/panel coverage, retained-policy KL, entropy, and physical rollout
  advantage. Updated checkpoint/model signature to v55.
- The full repository regression suite passes 305 tests. A 12-episode,
  250-person smoke plus all-benchmark matched evaluation is running first;
  after it succeeds, the same background job starts the 128-episode,
  2,500-person training and 20-scenario matched evaluation.
