# TODO

This list is ordered by correctness and research-validity risk. Check an item only after its acceptance criteria are met.

## Change discipline

- Every model-code change must append a dated `LOG.md` entry describing the
  design, implementation, compatibility impact, and verification performed.
- The same change must update `TODO.md`: close only items whose acceptance
  criteria were verified and state the next unresolved implementation or
  experiment steps.
- A model-code task is not complete until its tests, `LOG.md`, and `TODO.md`
  agree with the implemented behavior.

## Current priority — complete the clean v40 staged run (2026-10-02)

- [x] Make the three stages own distinct jobs. Stage E trains the shared
  graph/LSTM environment representation and six candidate-conditioned
  traffic/hazard rates while the actor is frozen. Stage A freezes that
  representation, trains the complete nonlinear actor adapter on
  one-decision-interval installed-site contrasts, and warms the backward
  continuation value without using it. Stage F retains the same observable
  action features and activates the continuation value in NMCC targets.
- [x] Remove inactive ensemble, teacher, guidance, uncertainty, alternative
  M-step, gate, model-fill, and heuristic-base-policy controls from the active
  code/configuration surface. Delete the historical curriculum tests that
  asserted those retired contracts; retain tests of current E/A/F ownership,
  exact branches, backward returns, schedules, and deterministic evaluation.
- [x] Verify a fresh 500-person E/A/F smoke run. Stage E joint dynamics
  explained variance rose from 0.390 to 0.724; Stage A branch-target top-1
  agreement moved 0.219 to 0.438; Stage F moved 0.406 to 0.688 and used a
  continuation spread equal to 16.2% of the total within-state value spread.
  The eight-dynamic-policy initial-state, capacity, partition, random-stream,
  and full hazard-trajectory parity audit passed. This 12-episode smoke was
  inferior to objective-greedy and is not performance evidence.
- [ ] Complete the active 5,000-person State College run at
  `runs/state_college_5000_staged_nmcc_v40_20261002`: 24 E episodes, 48 A
  episodes, 88 F episodes, followed by 20 deterministic matched scenarios for
  all eleven policies. At each stage boundary audit actor ownership, dynamics
  explained variance, target-fit KL/top-1, continuation train/validation
  explained variance, continuation ranking share, and the independent
  exploration/LR reset and decay.
- [ ] After evaluation, report realised hazard-weighted time to safety,
  congestion, exposure, completion, unfinished, panic, and casualty outcomes
  against every benchmark. Treat heuristic agreement as descriptive only and
  do not infer convergence or superiority unless the recorded evidence
  supports it.

## Current priority — validate scalable sampled NMCC v38 (2026-10-02)

- [x] Remove exhaustive per-state physical branching from the active learner.
  Every NMCC state now includes the executed action and a coverage sample;
  production uses two installed actions under two common-noise futures. The
  MPO target changes only exactly observed probability mass and leaves every
  unobserved action at its behavior-policy probability.
- [x] Remove the redundant WAIT simulation from policy improvement. Equal
  capacity and natural system evolution cancel directly between the two
  installed-site branches. Preserve the complete 15-timestep congestion,
  movement, hazard, and dedicated-continuation calculation.
- [x] Verify the direct installed-site contrast against the corresponding
  difference in the former full action table, and measure the real map path.
  The 5,000-person first decision improved from 157.3 to 22.0 seconds; a fresh
  2,500-person eight-decision episode completed in 157.6 seconds including map
  initialization.
- [ ] Start a fresh v38 E/A/F run. The v37 checkpoint must not resume because
  the NMCC sampling and actor-target contract changed. Confirm that sampled
  actions cover every candidate and all eight decision positions, target KL is
  nonzero, the M-step converges, the continuation explained variance becomes
  positive, and episode wall time remains near the measured scale.
- [ ] Deterministically evaluate the completed v38 policy on matched scenarios
  against every registered heuristic. Compare realised hazard-weighted time to
  safety, congestion, exposure, safe completion, unfinished share, panic, and
  casualties; do not treat heuristic action agreement as a success criterion.

## Current priority — validate random-centred Seattle training (2026-10-01)

- [ ] Complete the active 2,500-person State College precursor run at
  `runs/state_college_n2500_random_centered_foresight_v37_20261001/`: 24
  environment episodes, 48 action episodes, 88 foresight episodes, followed by
  eight matched deterministic scenarios for all eleven registered strategies.
  Do not interpret or compare policies until the launcher writes a complete
  manifest and the capacity/interface parity audits pass.

- [x] Give every ten-shelter run `population + 200` total capacity. The
  registered Seattle levels now use capacities 270, 520, 770, and 1,020 per
  shelter. Population-matched runs use the absolute `15,30,...,120` schedule,
  reject an undersized detected candidate pool, and fail immediately rather
  than postponing a scheduled action. After the eighth sequential installation
  at `t = 120`, assert that every active non-panicked pedestrian has a valid
  capacity reservation and a route to that shelter. Panicked pedestrians
  remain deliberately outside the shelter-assignment controller.
- [x] Verify that danger speed effects are level-based rather than recursively
  compounded, and that moving to a lower danger level recomputes speed from
  free-flow `maxSpeed`, allowing full recovery.
- [x] Replace the learning reward's WAIT baseline with the signed uniform mean
  over all feasible equal-capacity installation sites. Scalable v38 NMCC uses
  exact sampled equal-capacity site contrasts; natural evolution and capacity
  cancel directly, and WAIT is no longer simulated by policy improvement.
- [x] Separate the signed learning reward from evaluation. Deterministic
  held-out comparison now uses negative realised hazard-weighted time to
  safety, including observed active/exposure person-time and a terminal
  remaining-time residual. Casualty has no standalone coefficient, while a
  person who never reaches safety retains time-to-safety debt.
- [x] Reset exploration, entropy, temperature, and actor/critic learning-rate
  decay at the start of each E/A/F curriculum stage and hit both configured
  endpoints exactly. Keep heuristic agreement descriptive only. Ensure the
  NMCC M-step uses the scheduled actor rate instead of restoring the base rate.
- [x] Make final-stage evaluation deterministic and matched across RL and all
  dynamic/static heuristics. Cycle the final Seattle stage's 2,500/5,000/
  7,500/10,000 population variants so each policy sees the same scenario seed,
  capacity, hazard realization, and ten-shelter budget.
- [ ] Start a fresh checkpoint; the reward/baseline and evaluation contracts
  intentionally invalidate the interrupted run. Confirm each stage begins at
  its high exploration/LR endpoint and finishes at its low endpoint, and that
  all scheduled episodes install at `15,30,...,120` with zero unassigned
  rational population after the last installation.
- [ ] Run deterministic matched evaluation against
  `objective_greedy`, `risk_reduction`, `route_saving`, active-demand,
  hazard-weighted, accessibility, random, static-greedy, learned-precommit,
  and initial-only policies. Report the realised HWTTS objective by population,
  together with congestion, exposure, completion, panic, and casualty trends.

## Current priority — validate v37 decision-interval NMCC (2026-09-30)

- [x] Evolve every clean-NMCC candidate and its shared common-noise WAIT from
  the same snapshot through exactly one shelter decision interval.
- [x] Add a dedicated NMCC continuation head and train it from backward
  semi-Markov return-to-go across each episode's eight sequential
  installations. Keep the environment reward on its unchanged zero-horizon
  installation branch.
- [x] Make the NMCC M-step fit the masked actor itself at the base actor
  learning rate, without exploration-mixture distortion or gradient clipping;
  stop on absolute `KL(q || pi) <= 0.01` plus exact top-1 agreement.
- [x] Replace the redundant momentum vector with six traffic/hazard velocities
  and expose physical congestion delay separately from general mobility delay.
- [x] Enforce the common ten-shelter budget in evaluation. Dynamic policies
  install two initially plus eight sequentially even after early clearance;
  `initial_only` now commits eight sites from the time-zero state and installs
  them at the common `t = 15,30,...,120` schedule. The frozen-policy matched
  comparison is complete: RL reduced mean HWTTS by 20.3% and won 3/4 pairs on
  the original four held-out State College scenarios. Expand the sample before
  making an inferential superiority claim.
- [ ] Run the v37 staged curriculum from a fresh checkpoint. Confirm that the
  continuation explained variance becomes positive after the first fitted
  update, NMCC residual KL and top-1 meet their fit criteria, branch endpoint
  values acquire nonzero within-state dispersion, and all episodes contain
  eight decisions unless the horizon or candidate contract is violated.
- [ ] Freeze the trained v37 checkpoint and compare it with
  `objective_greedy`, the remaining dynamic heuristics, random, static greedy,
  learned precommitment, and initial-only at 2,500/5,000/7,500/10,000 people.
  Report shelter-count parity, action-sequence divergence, congestion/exposure
  outcomes, and traffic/hazard feature ablations before attributing an
  advantage to foresight.

## Current priority — v36 congestion and policy-use audit (2026-09-28)

This section is the active continuation of the September 22--28 working
session. It supersedes conflicting historical model-v22--v26 next-step items
below for the current v36 line. The older items remain in this file as
provenance and must be revalidated against the current MDP before reuse.

### Current contract carried forward from the session

- The policy objective is to reduce total expected future hazard-weighted
  time-to-safety for active pedestrians. The implemented decision reward is
  the positive, population/horizon-normalized reduction in remaining ordinary
  travel time plus hazard exposure relative to the same-state `WAIT` branch.
- Casualties and panic remain reported outcomes, not standalone reward terms.
  There is no terminal-residual reward.
- Pedestrians change shelter only for a strictly better valid reservation;
  otherwise they keep following the shortest path to their assigned shelter.
  Unassigned pedestrians retain the configured horizon-valued remaining-time
  cost. Danger-level speed modifiers do not compound at every timestep.
- Every dynamic run has two initial shelters and eight sequential
  installations at minutes 15, 30, ..., 120 of a 180-minute episode. Each
  shelter has capacity `population / 10`, so total installed capacity equals
  the simulated population and 60 minutes remain after the final action.
- Training is staged: environment dynamics first, then action learning. The
  active policy has no engineered actor prior, fallback selector, correction
  gate, or post-selection action replacement. Feasibility/safety masks remain
  part of the MDP. Exploration and learning rates decay on their registered
  schedules; frozen evaluation uses deterministic masked ranking.
- Current multiscale evidence is
  `runs/state_college_v36_complete_multiscale_nmcc_20260927/`: 120 balanced
  training episodes and 112 frozen evaluation episodes at populations 2,500,
  5,000, 7,500, and 10,000. RL reward relative to the best dynamic benchmark
  changes from `-0.0574`, to `-0.0068`, to `+0.0362`, to `+0.0810` across
  those four levels. This is diagnostic evidence from one city and four
  evaluation seeds per population, not a general superiority claim.

### P0 — Complete before another long training run

- [x] **Add the objective-matched myopic benchmark and run the first paired audit.**
  - `objective_greedy` scores every feasible cell with the exact signed
    same-state reduction in remaining plain plus forecast-danger-weighted
    travel time used by the current RL objective. It has no physical rollout,
    temporal memory, or continuation value.
  - The frozen v36 policy and benchmark were compared on the existing 16
    held-out State College scenario keys. RL lost episode return in 14/16 and
    had 20.3% mean action agreement; pooled restricted mean time-to-safety was
    3.40 minutes worse. At 10,000 pedestrians RL had lower RMTTS in 3/4
    scenarios despite lower immediate reward, a possible high-congestion
    continuation effect that is not yet statistically or causally established.
  - Artifacts: `runs/state_college_v36_complete_multiscale_nmcc_20260927/objective_greedy_analysis_20260930/`.

- [ ] **Separate genuine self-evolution understanding from immediate-target underfit.**
  - At retained decision states compare the objective-greedy ordering, the
    frozen RL ordering, a common-noise 15-minute physical-branch ordering, and
    a continuation-valued ordering. Report top-1 agreement, Spearman rank,
    regret, and subsequent realized outcome by decision position and population.
  - Evaluate at least 12 new matched seeds per population. Then retrain matched
    no-momentum/no-recurrent-history and no-hazard-forecast ablations; frozen
    feature masking is diagnostic but cannot establish causality.
  - Acceptance: show that RL preferentially departs from the myopic benchmark
    where future physical evolution reverses the immediate ranking, and loses
    that advantage when temporal/forecast information is removed.

- [ ] **Measure which observation groups the frozen v36 policy actually uses.**
  - Add a read-only matched-evaluation audit for the existing checkpoint with
    grouped masking/permutation and local sensitivity for: current load and
    mobility delay; route assignments and route-flow edges; hazard and hazard
    momentum; shelter capacity/utilization; spatial/geometry features; and
    recurrent history.
  - Do not infer importance from feature availability or neural weights alone.
    Report action-distribution change, selected-cell change, reward,
    completion, unfinished share, and congestion person-minutes by population.
  - Follow the frozen audit with the minimum retrained ablations needed for a
    causal architecture claim: no route-flow information, no load/congestion
    information, and no temporal/candidate-attention context. Use the same
    population blocks, action schedule, capacities, seeds, and evaluation
    contract as the full model.
  - Acceptance: identify which groups explain the 7,500/10,000 advantage and
    whether the low-population deficit is a scale-conditioning or action-
    coverage problem. Preserve all negative results.

- [ ] **Make current congestion observable without adding an oracle.**
  - Keep the existing administrator-observable inputs, but separate physical
    congestion from generic low speed. Add regional current congestion-speed
    ratio, congested-population share, and occupied-link/storage share computed
    from the same pre-action physical-link state used by the simulator.
  - Replace or rename the static `network_load_share`. It currently measures
    initial population relative to link storage, so
    `network_load_reduction_velocity` is identically zero. Retain the static
    quantity as an explicit scenario-load descriptor and create a genuinely
    time-varying load/clearance quantity if a momentum feature is needed.
  - Do not expose future action outcomes, candidate rollouts, or perfect
    candidate rankings as policy inputs. The observation must remain available
    from the current emergency-operations state.
  - Bump the model/signature version, reject incompatible checkpoints, document
    units and normalization, and add deterministic tests for zero load,
    increasing density, jam-floor behavior, regional aggregation, and temporal
    change.

- [ ] **Test whether the immediate causal label misses action-induced congestion.**
  - The current zero-horizon candidate evaluator uses each pedestrian's
    current speed for every hypothetical destination. It rewards immediate
    reservation/route improvement but does not recompute the congestion that
    the candidate's new routes create.
  - On a retained set of pre-action states, compare the current immediate
    all-candidate ordering with a common-noise physical branch that advances
    every equal-capacity candidate through one complete 15-minute decision
    interval. Also record the later realized ordering from the live episode.
  - Compare rank agreement, top-1 recall, regret, and compute cost by
    population. Use candidate-versus-candidate contrasts so the common benefit
    of installing one capacity token cannot dominate the siting differential.
  - Change the training target only if the congestion-aware label improves
    held-out ordering at matched states. Do not introduce a fallback policy,
    correction gate, teacher ranking, or post-action override.

### P1 — Training after the P0 measurements

- [ ] **Run one fresh, balanced engineering pilot for the accepted v37 contract.**
  - Start from a new checkpoint after any observation or target-schema change.
    Use 32 environment-dynamics episodes followed by 128 action-learning
    episodes, balanced in repeating 2,500/5,000/7,500/10,000 blocks so every
    four-episode optimizer batch contains all four population levels.
  - Preserve the v36 shelter schedule, equal capacities, positive exposure-
    removal objective, no actor/base prior, smooth exploration decay, learning-
    rate decay, and full feasible-action NMCC coverage unless the P0 label
    experiment justifies a preregistered alternative.
  - Evaluate on at least 12 new matched seeds per population before extending
    training. Report policy KL, selection-TV movement, NMCC fit/ranking,
    reward and component trajectories, per-installation action value,
    congestion, completion, unfinished share, and exact shelter sequences.
  - Acceptance: the learned policy must move, beat random at every population,
    retain the high-population advantage, and not obtain that advantage merely
    by rerouting more pedestrians or choosing larger capacity (capacities are
    equal by contract).

- [ ] **Run a multi-seed confirmatory evaluation only after the engineering pilot passes.**
  - Train at least three independent policy seeds under the frozen accepted
    contract and evaluate at least 20 untouched matched scenario seeds per
    population.
  - Report seed- and scenario-level uncertainty for reward, completion,
    unfinished share, congestion person-minutes, and benefit per rerouted
    pedestrian. Casualty and panic remain documented outcomes only.
  - Separate the scale claim (advantage grows with demand) from the policy
    superiority claim. Test the population-by-policy interaction rather than
    relying on four separate means.

### Explicit non-priorities for the next cycle

- Do not run a longer single-seed copy of v36 before the P0 audits; it cannot
  distinguish missing congestion information from inadequate optimization.
- Do not add a complete engineered candidate ranking, perfect-information
  benchmark, correction gate, conservative fallback, casualty reward, or
  terminal-residual reward.
- Do not expand to the full multi-city campaign until the State College
  congestion mechanism and feature contribution are identified with matched
  ablations.

## P0 — Blockers before further training

- [ ] **Re-divide the NMCC branch budget: fewer cells, more tapes, full horizon.**
  - Measured in `docs/SITING_SIGNAL_IDENTIFIABILITY_20260927.md`: the label,
    not the network, is where the siting information is lost. With one tape at
    the full horizon its rank agreement with the quantity it represents is
    0.731 pooled over decisions and 0.636 at the first decision, where the
    regret is.
  - Branch the candidate prior's top 9-12 cells instead of every feasible cell
    at the exhaustive decisions (executed and base actions still always
    included), and set `nmccPiTapes = 2`. Cost-neutral against the present
    configuration: 17 cells x 1 tape x full is 1003 simulator timesteps per
    decision for expected top-1 0.359 and regret 0.551 cell-effect SD; 9 cells
    x 2 tapes x full is 1062 steps for 0.429 and 0.416.
  - Change `DEFAULT_NMCC_PI_BRANCH_HORIZON` from 20 to 0. A noiseless +20
    contrast correlates only 0.598 with the full-horizon cell effect at the
    first decision, so a curriculum that omits the key is silently capped at
    Spearman ~0.55 at any tape count.
  - If truncation is ever needed for cost, add the bootstrap term
    `gamma^k * V(s_{+k})` from the value head to `BranchValuer._continue`,
    which currently returns accrued reward with no tail. The missing 40% of the
    ordering at +20 is exactly what the critic exists to supply.
  - Raise `nmccPiMaxBranches` from 6 to at least 9 at later decisions, or
    accept that top-1 there cannot exceed 0.44: the v25 prior places the true
    best cell inside its top 6 only 44% of the time and its top 8 56%.
  - Do not pursue a single-component or re-weighted label. Measured and
    negative: hazard_exposure carries 3% of the cell-effect variance with a
    worse SNR than the scalar (0.85 vs 0.94), and a least-squares re-weighting
    of all four components gains 0.021 Spearman with a lower top-1 rate.
  - Acceptance criteria:
    - the allocation is configuration-driven, with the per-decision step cost
      recorded in the run diagnostics;
    - a pilot at the new allocation reports the label's own Spearman against a
      held-out tape at a sample of decision states per epoch, so a failure to
      fit is distinguishable from an unfittable target;
    - held-out within-state top-1 and mean regret improve against the v26
      pilot at matched simulator cost;
    - `LOG.md` and `TODO.md` updated, and the earlier model versions remain
      reproducible.

- [ ] **Improve the candidate prior before improving the label.**
  - The v25 prior correlates 0.338 with the terminal cell effect; every subset
    branching scheme inherits that as a ceiling. A prior at 0.6 would let a
    6-cell subset reach the recall a 12-cell subset reaches today, halving the
    branch cost.
  - Acceptance criteria: `corr(prior, a_full)` measured on held-out states by
    `exposure_budget_allocation.py`, improved over 0.338, with the recall curve
    reported.

- [x] **Implement and backtest model v25 staged NMCC candidate scoring.**
  - Exact CRN candidate branches now retain paired natural and action-caused
    targets for all six physical outcome coordinates.
  - Natural, causal, controller-warm-up, and joint phases are compatible with
    NMCC policy improvement; actor updates are disabled in the first two.
  - The actor learns listwise candidate scores plus a best-versus-rival margin,
    explores with a decaying epsilon-greedy schedule, and deploys deterministic
    masked argmax at evaluation.
  - Actor/critic warm-up and cosine learning-rate schedules, forecast/safety/
    route-benefit masks, and a capacity-token deadline safeguard are active.
  - The matched synthetic backtest passed 8/8 gates; exact top-1 improved
    0.333 -> 0.667 and mean held-out return improved 0.2154 -> 0.2442 with
    exact 120-person dynamic-capacity parity. Artifacts:
    `runs/nmcc_score_backtest_v25.json` and `.png`.

- [x] **Run and audit the registered v25 State College train-only pilot.**
  - The fresh 64-episode, 2,500-individual launch completed all 16 optimizer
    updates and the 0/1/2/3 phase order. The actor was exactly frozen in phases
    0/1 and received nonzero accepted updates in phases 2/3.
  - Natural loss decreased 0.04220 -> 0.02090; dueling loss decreased
    0.10576 -> 0.01476; exact-target fit KL improved on 11/11 actor updates;
    mean exact top-1 increased 0.4083 -> 0.4878 (7 improving and 4 worsening
    updates). Credit signal is validated, but decision convergence is not.
  - Return did not converge: actor-period slope −0.000207/episode and
    last-eight minus first-eight +0.02744 with 95% bootstrap interval
    [−0.11754, 0.17614]. The 100-episode stationarity minimum was not met.
  - Artifacts: `runs/state_college_2500_nmcc_score_v25/` and the reproducible
    `analyze_nmcc_score_training.py` audit.

- [x] **Replace the v25 one-rollout fit with model v26 persistent fitted NMCC.**
  - Checkpointed episode-level replay now retains every exact branch label and
    recurrent prefix, uses episode-disjoint train/validation splits, and refits
    dedicated intervention heads on the growing dataset instead of discarding
    labels after one optimizer update.
  - The first two decisions branch every feasible action through physical
    terminal; later decisions use bounded cellular-automata branches. The
    relational GNN/LSTM system encoder is learned by the world-model and critic
    objectives, then frozen while wide-and-deep control heads are fitted.
  - Deployment uses the route-saving base until three consecutive held-out
    lower confidence bounds on exact paired gain are positive. Candidate
    ordering then uses candidate-minus-fixed-base ensemble advantages, which
    are invariant to critic-head offsets and do not count the route prior
    twice.
  - A continued 48-episode matched engineering run opened the gate at episode
    33 and ended with held-out paired-gain LCB `+0.009815`. On 12 untouched
    common-random-number tapes, v26 mean return was `0.29101`, versus `0.25032`
    for route saving and `0.24156` for v25, with exact shelter count/capacity
    parity. All 10 engineering gates and all 266 repository tests passed.
    This validates the learning path, not State College convergence or policy
    superiority. Artifacts: `runs/nmcc_value_backtest_v26.json` and `.png`.

- [ ] **Close the map-backed realized-capacity gate before v26 evaluation.**
  - The train-only pilot installed all five 500-person tokens in only 53/64
    episodes; the hard safety/feasibility mask produced 1-4 installations in
    11 episodes. Mean realized dynamic capacity was 2,312.5 rather than the
    2,500-place budget.
  - Add a hazard-forecast capacity-viability layer that proves at episode start
    that a common safe K-token installation schedule exists and advances the
    common deployment deadline when future safe capacity would disappear.
    Never relax the hard hazard-safety threshold merely to spend a token.
  - Fail closed or resample the training/evaluation scenario before policy
    execution when no safe full-budget schedule exists. The feasibility test
    must depend only on the common hazard tape and candidate inventory, not on
    the realized RL or heuristic trajectory.
  - Add deterministic tests where safe candidate availability shrinks over
    time; every dynamic policy must install the same total token capacity or
    the paired scenario must be rejected before outcomes are compared.

- [ ] **Run deterministic held-out v26 evaluation only after the capacity gate passes.**
  - Run the registered 96-episode, 2,500-pedestrian v26 State College curriculum
    from a fresh checkpoint, then freeze it. Compare it with v25, route-saving,
    and the registered MC/rollout benchmark on identical held-out scenario
    tapes that passed the common ex-ante capacity-viability check.
  - Require exact realized capacity parity, positive paired return, casualty
    nonworsening, and intervals for safe completions, active person-time, and
    exposure person-time. Training-return movement is not evidence for this
    gate.
  - Then train at least three independent policy seeds and at least 100
    episodes per seed before making a convergence or superiority claim.

- [x] **Superseded: validate and pilot NMCC-PI model v24.** v24 is retained as
  historical evidence; the current learning backtest compares v25 with v26
  and the shared route-saving controller. v25 addressed v24's missing staged
  physical-outcome learning, and v26 replaces v25's disposable small-sample
  actor fit.
  Historical design: `docs/NMCC_POLICY_IMPROVEMENT_20260921.md`.

- [x] **Act on the 2026-09-21 convergence review before the next training run.**
  Model v26 resolves the review's learning-path items: deterministic held-out
  CRN evaluation replaces training-return/KL convergence; route saving is the
  explicit fallback and benchmark; the system GNN/LSTM belongs to world/critic
  learning and is frozen during control refits; persistent exact-branch fitted
  policy iteration replaces the weak PPO actor; early decisions use terminal
  branches; ensemble bootstrap uncertainty is paired to the fixed base action;
  and infeasible deployment epochs retry without relaxing safety. Evidence and
  historical ranking remain in `docs/CONVERGENCE_REVIEW_20260921.md`.
  The separate map-backed realized-capacity gate above remains unresolved and
  is deliberately not treated as part of this closed learning-path item.


- [ ] **Enforce equal implemented shelter capacity in RL-versus-heuristic evaluation.**
  - Freeze a scenario-level dynamic capacity budget `B` and an ex ante
    capacity-token schedule `q_1, ..., q_K`, preferably equal-sized tokens.
  - Make the `k`th installation add exactly `q_k` under every dynamic policy;
    use raw candidate capacity only as a physical eligibility limit.
  - Keep the initial shelter identities and capacities identical across the
    matched pair and include a dedicated initial-capacity digest.
  - If `WAIT` is introduced, permit it only while deployment slack remains;
    require all tokens to be placed by the common deployment deadline in the
    primary capacity-controlled comparison.
  - Log initial capacity, dynamic budget, token-schedule digest, installed
    dynamic capacity, final total capacity, and capacity-time area.
  - Fail the paired-evaluation gate before statistics unless RL and heuristic
    both satisfy `dynamic_capacity_added == B` and have identical final total
    capacity. Never repair a mismatch by conditioning the heuristic on the
    realized RL trajectory.
  - Add deterministic tests with heterogeneous candidate `nodeCap` values to
    prove that different selected locations still produce identical capacity
    totals and that physically undersized candidates are masked.
  - **Implemented 2026-09-20:** one validated `shelterCapacityToken` now drives
    observation, initial/dynamic/static shelter execution, checkpoint
    provenance, city profiles, and curricula. Sites whose raw capacity is less
    than one token are filtered before sampling and masked in legacy pools; a
    heterogeneous-capacity test proves both rejection and equal capacity at
    different selected locations. The matched NMCC learning
    backtest fails closed on count or capacity mismatch and passed all eight
    scenarios at two additions/600 places per policy. Remaining before this
    broader item closes: add the initial-capacity digest, capacity-time area,
    common deployment deadline, and fail-closed parity gate to every general
    benchmark runner, not only the NMCC learning backtest.
  - **Production audit 2026-09-22:** equal token *size* did not guarantee equal
    realized total capacity. In the v25 State College pilot, hard masks left no
    action at some epochs and only 53/64 episodes spent all five tokens. Add the
    common hazard-only capacity-viability schedule described above before this
    item can close.

- [x] **Collapse the action space from exact-candidate to cell-priority, with a shared deterministic site rule.** (2026-09-20)
  - Replace `RegionalObservationBuilder`'s frozen one-row-per-raw-candidate
    action table with exactly one action slot per regional cell, resolved
    every decision epoch from `ShelterDatabase.previewShelterCandidate`
    (maximum remaining capacity, OSM identifier as the tie break).
  - Route `RegionalShelterExecutor` through `ShelterDatabase.newShelter`
    (the same deterministic rule already used by `initShelter` and
    `predeployStaticDemandGreedy`) instead of `newShelterCandidate`, and
    fail closed if the installed building ever diverges from the
    observation's prediction for that cell.
  - Bump `RLBridge.MODEL_VERSION` to 18 and change the
    `_model_signature()["action_space"]` string so a pre-18 checkpoint
    (fit to the larger per-candidate action space) cannot be loaded and
    silently misinterpreted.
  - Update `docs/MDP_AND_OPTIMIZATION_DESIGN.md` (Action, benchmark policy,
    and PPO sections) and `docs/BENCHMARK_MODEL_PROTOCOL.md` to describe the
    cell-indexed action space; full design and diff rationale in
    `docs/CELL_PRIORITY_ACTION_SPACE_20260920.md`.
  - Rewrite `tests/test_rl_framework.py`'s two tests that encoded the old
    design's per-candidate-slot guarantees into tests of the new shared-
    site-rule guarantee, and add coverage for the empty-cell placeholder
    slot and for every benchmark policy plus RL resolving the identical
    site per cell.
  - **Verified:** AST-parsed every edited file; a standalone
    dependency-light harness (no `torch`/GNN) exercised every test scenario
    above directly against the edited `DecisionInterface.py` and
    `ShelterDatabase.py` and all passed. **Not yet verified:** the full
    pinned-environment `python -m unittest discover -s tests -v` — the tool
    used to make this change had no access to the `rlevacuation` conda
    environment or outbound network to install `torch`/`networkx`. Run the
    full suite in the actual environment before training against this
    action space, and confirm whether `backtest.py` / `TrainingLogger.py`'s
    per-episode decision logs already key on `requested_cell`/
    `executed_cell` (needed for the RL-vs-heuristic cell-choice-divergence
    analysis) or still assume per-episode-stable candidate identities.
  - This is an action-space complexity reduction, not a structural
    credit-assignment fix; it is complementary to, not a substitute for,
    the NMCC item immediately below.

- [x] **NMCC Stage 0 and Variant A implemented for the cell-priority action space.** (2026-09-20)
  - `CounterfactualBranch.py` adds exact simulator snapshot/restore (shared road
    graph and routing caches, deep-copied branch state, flat-array capture of
    the mutable flow fields on shared node/edge objects) and a paired
    act-versus-`WAIT` branch runner under common random numbers.
  - Stage-0 audits pass on real dynamics in
    `tests/test_counterfactual_branch.py`: bitwise factual replay from a
    restored snapshot, snapshot reusability, branch-order invariance, hazard
    action-independence, non-disturbance of the live episode, and agreement of
    the fast outcome read with `RegionalObservationBuilder`.
  - Audit finding: the keyed structural noise tape was already ~90% present.
    `_hazard_uniform`, `_panic_uniform` and `_panic_susceptibility_uniform` are
    already counter-based and order-invariant, so `U^cas`, `U^panic`, `U^move`
    needed no work. Hazard is exogenous (verified, not assumed), so restoring
    the generator state reproduces `U^H`.
  - `nmcc_testbed.py` drives the real dynamics on a synthetic grid city with no
    torch and no OSM, so the branch machinery is testable in seconds and in any
    environment.
  - **Measured (runs/nmcc_paired_report.json):** 99.36% variance reduction on
    the per-decision effect; mean effect unchanged (z=0.00); single-sample
    cell-ranking recovery +0.984 paired versus +0.457 unpaired; signal-to-noise
    54.4 versus 2.19. Branch overhead 1.16-1.29x the cost of simulating the same
    interval, i.e. about 2.2x per training episode.
  - **Still to verify:** the `RLBridge` Variant-A integration
    (`counterfactual_credit`, default off) has not been executed anywhere -
    torch is not installable in the environment this change was made from.
    Run `tests/test_nmcc_integration.py` and the backtest in `rlevacuation`
    before relying on it. Watch `nmcc_counterfactual_advantage_sd` against
    `nmcc_gae_advantage_sd` first; the reward curve is the wrong first
    diagnostic.
  - Learned residual world model, dueling causal critic and `WAIT`-as-an-action
    are deliberately deferred: NMCC says Variant C follows only after Variant A
    shows a material advantage-variance reduction in a real environment.

- [ ] **Validate Natural-Momentum Counterfactual Control before another large training run.**
  - Implement action-independent structural noise keys for hazard, casualty,
    panic, movement, and policy randomness; verify branch-order invariance.
  - Add complete simulator snapshot/restore and prove that an unmodified
    factual replay is bitwise identical for state, outcomes, and reward.
  - Collect a small State College candidate-versus-`WAIT` paired dataset under
    common noise and an independent-noise control, following
    `docs/NATURAL_MOMENTUM_COUNTERFACTUAL_CONTROL_20260920.md`.
  - Verify that paired causal return differences reduce variance by at least
    50% without changing mean effects beyond Monte Carlo uncertainty.
  - Retrospective evidence in
    `docs/CRN_COUNTERFACTUAL_CREDIT_ASSIGNMENT_PROPOSAL_20260920.md`, computed
    from the already-completed 09-19 validation run rather than a new
    experiment, is directionally consistent with this gate; it does not
    replace the registered minimal experiment.
  - Train and validate a small intervention-residual head before committing to
    the full world model; require held-out effect sign, ranking, uncertainty,
    and top-action-regret gates.
  - If the causal premise passes, implement the action-free hazard/natural-
    momentum model, intervention-residual ensemble, dueling causal critic,
    robust receding-horizon teacher, and counterfactual-advantage PPO in staged
    ablations N0--N8.
  - Do not use learned counterfactual predictions in production policy updates
    unless uncertainty is calibrated and periodically anchored to exact paired
    simulator branches.

- [ ] **Implement and ablate the staged RL signal curriculum.**
  - Add a recurrently recorded `WAIT` action without allowing idle-state reward
    farming, plus fixed intervention cost and nonpositive-benefit waste cost.
  - Compute deterministic action-difference features and bounded local rewards
    against the same-state `WAIT` baseline; fade the local coefficient to zero.
  - Replace fixed entropy with a checkpointed, update-indexed normalized target
    schedule and ensure behavior/replay log-probabilities use the same policy.
  - Add hard physical/safety masks, per-head critic target normalization,
    risk-stratified complete-episode batches, and candidate-to-region attention
    in the documented implementation order.
  - Run E0--E5 paired ablations from
    `docs/RL_SIGNAL_CURRICULUM_PROPOSAL_20260919.md` before another full-scale
    training claim. Coordinate these ablations with NMCC: proxy rewards and
    planner imitation must fade, while an exact causal control-variate
    advantage may remain because it preserves the original global objective.

- [x] **Add a shared pedestrian link-congestion transition.**
  - Interpret `maxSpeed=64` as 64 m/min with an explicit one-minute timestep.
  - Freeze physical-link density before each 10-second internal movement
    substep, combine OSM counterflow, and apply the Weidmann speed-density
    relation to every policy.
  - Refresh downstream-link loads within each one-minute MDP transition,
    export person-time-weighted diagnostics, and verify boundedness,
    monotonicity, counterflow, substep use, and iteration-order invariance on
    toy networks.
  - Reject confirmatory use of checkpoints trained without the same congestion
    transition contract.

- [ ] **Calibrate congestion assumptions before confirmatory training.**
  - Prespecify width sensitivity at 2/3/4 m and jam-regularization sensitivity
    at 0.025/0.05/0.10 before comparing policy outcomes.
  - Verify numerical integration stability at 5/10/15-second substeps on a
    representative scenario before freezing the 10-second primary contract.
  - Verify that each population level spans informative densities without
    universal free flow or universal jam-floor behavior.
  - Freeze the accepted common parameters without selecting them on RL versus
    heuristic performance.
  - Resolve the State College 50,000-person mechanics result before freezing
    the design: maximum density reached 56.83 ped/m² (well above the 5.4
    ped/m² jam point) and the mean speed ratio was 0.5605. Determine from E0
    whether this represents an intended extreme-demand regime or excessive
    spatial concentration.

- [ ] **Calibrate shelter supply for the 50,000-person demand contract.**
  - The first large-network mechanics run provided 12,200 total implemented
    shelter places for 50,000 pedestrians, leaving a structural upper bound of
    24.4% on shelter evacuation before casualties.
  - Prespecify whether the scientific problem intentionally studies severe
    shelter scarcity. If not, change the capacity/resource regime before any
    RL training, not after comparing policy performance.

- [x] **Define one paper-defensible regional MDP objective.**
  - Reward is safe completion minus casualty cost minus hazard-weighted person-time.
  - All terms use fixed population/horizon scaling and causal post-action intervals.
  - The casualty coefficient is justified by a finite-horizon dominance bound.
  - Policy-level evaluation accumulates the same objective over every elapsed
    interval independently of action count, so zero-action static policies do
    not receive an artificial zero return.

- [x] **Unify the RL and benchmark decision interfaces.**
  - All dynamic policies receive the same immutable regional observation and exact feasible-cell mask.
  - The benchmark selects the feasible cell with maximum active population.
  - All strategies use the same deployment timing, resource budget, cell action encoding, and lower-level execution path.

- [x] **Replace the hidden lower-level scoring heuristic with an explicit optimizer.**
  - The upper level selects only a region.
  - The lower level selects the maximum-capacity candidate in that region with deterministic tie breaking.
  - Previewed capacity and executed candidate are guaranteed to use the same rule.

- [x] **Fix full-reward argument mapping and terminal signaling.**
  - The reward interface is keyword-only and rejects unknown or positional inputs.
  - Dead reward inputs were removed so the public interface matches the implemented formula.
  - `Core` now signals the true final transition explicitly; population exhaustion is also terminal.
  - Sentinel and terminal-bonus regression tests cover the dispatch contract.

- [x] **Correct PPO minibatch construction.**
  - Batches use `self.minibatch_size`, including a correctly sized final partial batch.
  - Transition indices are reshuffled at the start of every PPO epoch.
  - Tests verify complete, nonduplicated coverage and expected batch sizes for a long episode.
  - A synthetic episode exercises the complete optimizer update.

- [x] **Keep PPO probability tensors strictly one-dimensional.**
  - Old and new action log-probabilities, actions, values, returns, and advantages are normalized to `(T,)`.
  - Ratio construction rejects shape mismatches instead of allowing accidental `(B, B)` broadcasting.
  - Regression tests cover vector ratios and deliberate broadcast-shaped input rejection.

- [x] **Make the behavior policy genuinely on-policy.**
  - The final action mask is built before sampling and includes gating, budget, deployment, and candidate constraints.
  - Training samples directly from the masked categorical policy without epsilon-random mixing.
  - Selected actions are never replaced after their log-probabilities are recorded.
  - Non-RL strategies cannot populate PPO memory, and mask/sampling tests cover all fallback paths.

- [x] **Make shelter action scores depend on each candidate cell.**
  - Each cell uses a shared residual scorer over local features, four-neighbor messages, and pooled city context.
  - The fixed prior exactly ranks active population; PPO learns when hazard, speed, or capacity information justifies departing from the benchmark.
  - Mean-plus-maximum pooling preserves both system load and sparse urgent regions without memorizing cell identities.
  - Tests cover batched output shape, exact prior ranking, direct local gradients, and neighbor-message gradients.

- [x] **Replace episode-local PPO with complete multi-episode on-policy rollouts.**
  - Hold the behavior policy fixed for complete multi-episode rollouts and normalize advantages across the resulting decision batch.
  - Pooled rollouts contain whole city blocks (ten episodes for five cities), so every optimizer update is city-balanced.
  - Persist partial rollout tensors, terminal boundaries, policy RNG, optimizer RNG, and every PPO hyperparameter for exact resume.
  - Record per-episode behavior entropy separately from update-only KL and gradient diagnostics.

- [x] **Make initial shelters usable and dynamic strategy comparisons semantically equivalent.**
  - Define when and why pedestrians become shelter-seeking.
  - Route eligible pedestrians to available initial shelters before any dynamic placement action.
  - Do not count arrival at an arbitrary non-shelter destination as shelter evacuation unless that definition is explicitly justified.
  - Verify that a reachable, capacitated initial shelter receives flow in a deterministic toy network.
  - Verify that all dynamic strategies receive the same online shelter budget and evacuation semantics; label initial-only as an anticipative bound.

- [x] **Reconnect hazard parameters to simulated effects.**
  - Define CSV units (probability versus percent; variance versus standard deviation) and validate ranges at load time.
  - Use each hazard's configured spread, casualty, and speed-reduction values in the active simulation path.
  - Apply speed effects before movement and ensure they persist for the intended timestep.
  - Call hazard lifecycle updates every timestep and define how heat/smoke combine when hazards overlap or terminate.
  - Calibrate casualty thresholds against reachable danger values and add deterministic boundary tests.

## P1 — Required for trustworthy experiments

- [x] **Create reproducible, paired scenario seeds.**
  - Derive and persist a seed per replication.
  - Reuse the same scenario seed for RL, initial-only, random, and heuristic evaluation.
  - Separate scenario randomness from policy/action randomness.
  - Isolate exogenous hazard evolution and use person-time-keyed casualty
    shocks; verify the full hazard-trajectory digest across matched policies.
  - Save the launch seed, per-replication seeds, effective configuration, commit hash, and dependency versions with artifacts.

- [x] **Isolate checkpoints by experiment launch.**
  - Store checkpoints inside the launch directory.
  - Require an explicit resume/checkpoint option instead of silently loading by address.
  - Save optimizer/scheduler state and model/config metadata when training continuation is intended.
  - Fail clearly on incompatible grid or model dimensions.

- [x] **Represent unfinished pedestrians honestly at episode end.**
  - Add a `stranded`/`unfinished` outcome instead of forcing active agents to `Arrival`.
  - Write a final CSV summary after finalization, or remove post-log mutation.
  - Ensure console totals, CSV totals, plots, and exported summaries use the same final state.
  - Add the terminal transition flag and terminal outcome to reward tests.

- [x] **Enforce route and population invariants.**
  - Reject a route containing a missing edge and retry without mutating the requested origin/destination unexpectedly.
  - Bound destination selection for tiny graphs and disconnected components.
  - Either initialize exactly `pedVol` agents or report initialization failures as a separate metric and adjust denominators.
  - Assert that `arrival + casualty + evacuated + stranded == initialized_population`.

- [x] **Build a fast deterministic test suite for the optimization/RL boundary.**
  - Cover regional observations, shelter capacity/rerouting, reward terms, action masks, PPO tensor shapes, logger schema, final outcome accounting, and statistical parity checks.
  - Use offline toy-domain integration tests without OSM/network access.
  - Add a one-episode training smoke test that checks finite rewards, losses, gradients, and checkpoint reload.
  - Run the suite in CI with fixed seeds.

- [x] **Implement valid evaluation statistics.**
  - Report confidence intervals or paired differences across matched replications, not only means.
  - Separate training convergence metrics from evaluation outcomes.
  - Confirm that reward normalization is reset or persisted according to the documented experimental design.
  - Do not use results produced before the P0 fixes as evidence of policy performance.
  - Gate held-out evaluation on a prespecified, machine-readable convergence
    audit and classify superiority only from the paired return interval.

- [ ] **Execute the preregistered backtest and audit the produced evidence.**
  - Run five independent 320-episode training replications and 50 held-out matched scenarios.
  - Inspect PPO stability, population conservation, interface parity, confidence-interval width, and RL-versus-heuristic paired effects.
  - Extend the preregistered sample only if precision is inadequate; preserve the initial analysis.
  - A reduced 2-policy × 160-episode, 20-scenario pilot passed the convergence and interface gates and produced a positive but inconclusive return difference; it is not a substitute for this confirmatory run.

- [x] **Add reproducible city-map evacuation milestone figures.**
  - Use the same OSM-derived road graph and map extent for matched policies.
  - Show active pedestrians, stochastic hazard evolution, shelters, and the
    selected priority region without changing simulator or policy state.
  - Export the plotted source tables and a provenance/hash manifest alongside
    publication-quality PNG and SVG outputs.
  - Limit default rendering to representative held-out pairs while supporting
    an explicit milestone schedule.

- [x] **Implement exact shelter-choice and full-factorial map evidence.**
  - Export the exact OSM candidate installed at every dynamic decision, not
    only the selected priority cell.
  - Distinguish common initial shelters, sequential installations, and static
    time-zero predeployments in both tables and map symbols.
  - Decouple the five-addition resource budget from candidate-pool size so the
    10/15/20 levels require genuine subset selection; retain level 5 as the
    explicit candidate-scarcity boundary.
  - Capture every active pedestrian as a red point and continuous normalized
    cell danger as a fixed-scale heatmap at each dynamic decision epoch,
    together with priority region and exact candidate; do not imply that
    coarse milestone snapshots are action snapshots.
  - Fix the illustrative matrix at five cities, populations 10,000--50,000,
    candidate counts 5/10/15/20, hazard counts 1--5, and seven milestones
    through minute 60.
  - Represent early completion as an explicitly labeled absorbing state with
    no invented pedestrian positions and preserve actual terminal time.
  - Validate the 500-cell/1,500-episode plan and complete a real three-policy
    OSM pilot with checksum-bearing comparison figures.

- [ ] **Execute the full 500-cell shelter-decision map matrix.**
  - First train and freeze a converged pooled policy at the 60-transition,
    one-hour horizon with `pedVol=50000`, `shelterActionInterval=2`,
    `maxAdditionalShelters=5`, the enlarged 2--6 km OSM footprints, and the
    exact 10-second congestion integration contract; all earlier checkpoints
    are mechanics-only.
  - Run `map_factorial_backtest.py` for all five cities, five populations,
    four candidate levels, five hazard counts, and three policies.
  - Resume atomically if interrupted and audit all 1,500 episode rows, 500
    observation-parity checks, 2,000 comparison/diagnostic PNGs, common hazard
    paths, outcome accounting, candidate availability, terminal flags, and
    artifact hashes before using the maps in the paper.

- [x] **Generate an auditable paper-figure and policy-control bundle.**
  - Plot balanced training return, reward decomposition, PPO update size,
    critic fit, entropy, and heuristic action agreement.
  - Compare RL, heuristic, random feasible-region, and static predeployment on
    one matched 25-scenario matrix using equal-city estimates and within-city
    scenario bootstrap intervals.
  - Export absolute performance, paired improvements, city outcomes, final
    shelter configurations, and evacuation milestone composites in all five
    cities.
  - Verify map rendering is non-interventional and record every derived table
    and figure in a checksum manifest.

- [x] **Generalize the experiment to five increasing-scale cities.**
  - Version State College, Reading, Spokane, Seattle, and Chicago profiles using
    fixed OSM study points/radii and externally defined 2020 Census scale order.
  - Keep one 8x8 observation/action interface and one lower-level optimizer in
    every city; expose transferable road-density and physical-scale features to
    all policies without exposing the city identifier.
  - Train one pooled policy on an exactly balanced, block-randomized city
    schedule and evaluate matched held-out scenarios within every city.
  - Macro-weight cities equally and bootstrap policy seeds plus scenarios within
    city; do not imply that five deliberately selected sites represent all cities.
  - Validate profile schemas, map cache identity, resume schedules, parity, and
    stratified inference with offline tests.

- [ ] **Run the full five-city confirmatory backtest.**
  - Prewarmed all five enlarged OSM graph/building footprints; retain the
    hash-bearing `runs/preflight_60min_large_network_all_20260908/map_preflight.json`
    artifact with the confirmatory launch.
  - Train the eight pooled policy seeds for 120 episodes per city specified by
    `config/full_experiment_suite.json`; extend all seeds and cities equally if
    the preregistered convergence gate fails.
  - Evaluate the 90 untouched factorial scenarios per city and inspect the
    macro, city-specific, casualty-safety, regime, and scale-trend results
    before paper claims.
  - The sealed one-seed reduced audit is descriptively promising (+0.03391
    equal-city return over 25 matched scenarios) but fails the cross-city
    learning-well claim because Seattle worsened and Chicago had one additional
    casualty; do not substitute it for the five-seed confirmatory run.

- [x] **Implement the complete E0--E6 figure and evidence contract.**
  - Freeze eight policy seeds, 120 training episodes per city, and a 3x3x2
    capacity/hazard/demand evaluation with five replications per cell.
  - Validate every experiment table against a versioned schema and make the
    paper build fail when the policy-seed/scenario/factor matrix is incomplete.
  - Generate multi-seed learning, PPO, policy comparison, city heterogeneity,
    regime, frontier, scalability, ablation, transfer, robustness, shelter-map,
    and evacuation-progress figures without rerunning or changing simulation.
  - Preserve a partial mode for smoke backtesting, with conspicuous readiness
    diagnostics that prevent partial evidence from being reported as complete.

- [ ] **Execute and populate the full E0--E6 experiment tables.**
  - Run E0 calibration before policy selection and freeze accepted operating
    ranges without using confirmatory policy outcomes.
  - Complete all eight pooled training seeds and fixed checkpoint evaluations.
  - Populate E2--E6 result tables using matched scenarios and the schemas in
    `config/full_experiment_suite.json`.
  - Run `generate_full_experiment_figures.py --require-complete-suite` and audit
    every confidence interval, map provenance record, and artifact checksum.

- [ ] **Execute the 5×5 population/candidate scale-stress matrix.**
  - Train all eight pooled policy seeds using the 60-transition design horizon
    and two-transition shelter-action interval; earlier checkpoints are
    permitted only for computational pilots.
  - Evaluate populations 10,000--50,000 and 25--125 sampled candidates in all
    five cities with five matched stochastic scenario replications per cell.
  - Preserve the 5,625-episode execution manifest, interface-parity audit,
    runtime measurements, and F13 population-by-candidate figure.
  - The completed one-cell State College pilot validates execution and common
    random numbers but is not performance evidence.

- [ ] **Repair the supported runtime without unsafe OpenMP workarounds.**
  - Build or select one environment with compatible PyTorch, NumPy, Pandas, OSMnx, GeoPandas, Shapely, and Matplotlib.
  - Reject `KMP_DUPLICATE_LIB_OK=TRUE` because it can produce scientifically unreliable numerical behavior.
  - Record the resolved dependency versions in the experiment manifest.
  - A pinned all-Conda-forge `environment.yml` is present, but the active
    machine currently uses a legacy PyPI-PyTorch/Conda-LLVM environment that
    loads duplicate OpenMP runtimes. Create and verify the declared environment
  before producing numerical paper evidence.

- [x] **Make Hybrid NMCC production-runnable and verify the learning path.** (2026-09-20)
  - Add a validated and checkpointed `Core`/curriculum contract for exact
    common-noise branch credit, natural and causal auxiliary models, dueling
    consistency, ensemble uncertainty, robust planner guidance, teacher decay,
    and entropy/temperature exploration schedules.
  - Repair optional counterfactual transition checkpointing and close exact
    branch outcomes at horizon `L` while preserving uncensored full-tail SMDP
    credit for the factual action.
  - Implement an action-independent natural-outcome head with hard population
    conservation and bounded person-time/risk outputs, plus three candidate-
    local causal residual heads and a factorized recurrent critic.
  - Detach robust-planner teacher scores from PPO/imitation gradients and delay
    guidance until after world-model warmup.
  - Persist NMCC target coverage, exact and GAE dispersion, all auxiliary
    losses, uncertainty, guidance, teacher, entropy, and temperature fields.
  - Verify exact branch replay, factual invariance, completed optimizer update,
    checkpoint reload, bounded heads, causal locality, detached guidance, and
    fixed learner configuration. The complete suite passed 210/210 tests.
  - Run a matched 16-episode engineering learning backtest with identical
    seeds, weights, and capacity budgets. Exact causal-target SD was 0.090298
    versus raw GAE SD 0.213234 (ratio 0.4235); all four updates had exact
    targets. Held-out NMCC-minus-PPO mean was -0.014123 over eight scenarios,
    so this is functional/signal evidence and explicitly not a policy-
    superiority result.

- [ ] **Complete confirmatory Hybrid NMCC evidence on State College.**
  - The eight-episode, 3,000-person physical smoke is complete: exact-target
    coverage was 100%, counterfactual/GAE SD was 0.2376 (about 94.4% lower
    variance), all auxiliary losses and gradients were finite, and casualty
    outcomes ranged from 0 to 42. It was one optimizer update and is explicitly
    not convergence or policy-superiority evidence.
  - Finish the registered 64-episode, 3,000-individual-pedestrian training run
    for at least five independent policy seeds in a supported runtime.
  - Run matched held-out NMCC-on/recurrent-PPO/heuristic scenarios with
    identical hazard tapes, initial weights where applicable, action counts,
    and capacity tokens. Report seed-level intervals for return, casualties,
    safe completions, evacuation person-time, and exposure person-time.
  - Calibrate ensemble uncertainty and test causal-cell ranking on held-out
    exact branches; do not select hyperparameters using confirmatory returns.
  - Implement the real-`Core` path in `nmcc_paired_experiment.py`, or retire
    that obsolete CLI in favor of the production State College runner.
  - Require the full capacity ledger and supported OpenMP environment before
    using results as application or publication evidence.

- [x] **Complete the model-v22 2,500-person staged NMCC pilot.** (2026-09-21)
  - The fresh launch completed the registered 64-episode 8/8/16/32 schedule,
    all eight optimizer gates, a 3.6 MB checkpoint, and exact population and
    reward accounting. The complete suite now passes 218/218 tests.
  - Exact NMCC target coverage was 100%; the median counterfactual/GAE SD ratio
    was 0.2586 (93.3% implied target-variance reduction). Natural loss fell
    0.05635 to 0.03128 and dueling loss fell 0.09886 to 0.02066.
  - The policy result is negative: N3 return slope -0.004476/episode (bootstrap
    95% interval [-0.011351, 0.002593]), last-eight minus first-eight mean
    -0.116732 ([-0.333637, 0.070123]), and policy convergence false. Tail KL
    violation rate was 1.0 against the 0.015 target.
  - The validation report, seven figures, two summary tables, JSON audit, and
    checkpoint are under
    `runs/state_college_2500_nmcc_staged_v22_retry_seed20260920`. Treat this as
    single-seed engineering evidence only.
  - The original process completed simulation but stalled in its plotting
    finalizer. `backtest._plot_outputs` now forces `Agg`, a PNG regression test
    covers the path, and supported resume finalized the existing launch without
    rerunning an episode. Resume now preserves the original start and command
    while recording separate resume provenance.

- [x] **Implement the model-v23 temporal-credit and actor-stability contract.** (2026-09-21)
  - The actor now uses complete duration-aware Monte Carlo return-to-go and a
    prior-rollout-only regime/decision-position baseline; critic prediction
    error and current-batch centering cannot censor delayed actor credit.
  - The factorized critic uses independent one-step semi-Markov TD(0) targets
    and raw Huber loss. Actor and critic/world heads have disjoint optimizer
    ownership and separate one/four-pass update schedules.
  - KL enforcement is transactional: violating actor epochs restore both
    parameters and optimizer state, back off only actor LR, and do not advance
    accepted-rollout schedules. Entropy, temperature, guidance, and teacher
    schedules are indexed by accepted actor rollouts.
  - The v23 State College curriculum keeps NMCC exact branches as auxiliary
    supervision with zero direct actor-credit, planner-guidance, and imitation
    weights. This removes the failed fixed `0.35` local/global blend from the
    registered actor while preserving the paired model-learning experiment.
  - Verified by 224/224 tests and a fresh eight-training/four-held-out bounded
    backtest. Two actor updates were accepted, none rejected, retained KL was
    far below target, baseline coverage reached 100% on the second batch, and
    the auxiliary/no-auxiliary actors remained exactly matched.

- [ ] **Run the registered v23 staged validation before scaling training.**
  - First run the 64-episode, one-seed State College engineering pilot from
    `config/state_college_training_curriculum_2500_temporal_credit_v23.json`.
    Do not resume the model-v22 checkpoint; v23 is intentionally incompatible.
  - Audit accepted/rejected actor rollouts, attempted versus retained KL,
    baseline coverage by decision position, actor-gradient and MC-return
    dispersion, critic TD error/explained variance, casualty nonworsening, and
    exact reward/capacity accounting at every rollout gate.
  - The bounded smoke backtest did not show reward improvement: its training
    slope was negative and four held-out scenarios averaged `-0.01048` versus
    the heuristic. Treat the smoke only as a mechanics/isolation test.
  - If retained KL remains orders of magnitude below `0.015` during the full
    pilot, preregister an actor-step-size ablation before adding episodes; do
    not tune on held-out return. If rollbacks are frequent, retain the lower
    LR selected by transactional backoff.
  - Only after the one-seed engineering gate passes, run at least five policy
    seeds and at least 100 episodes per seed with matched held-out hazard tapes,
    shelter-count/capacity parity, casualty nonworsening, and seed-level return
    intervals. Do not claim improvement or convergence before those criteria.

## P2 — Reliability and maintainability

- [ ] **Complete and evaluate the Seattle population-adaptive foresight run.**
  - The 2026-09-30 launch was stopped on request and its partial artifacts were
    preserved under `runs/seattle_population_foresight_long_20260930`; it is
    not an active process. Start a fresh run only after the revised exact-NMCC
    smoke check passes in the supported training environment.
  - Stage boundaries are fixed: environment-only episodes 1--24,
    decision-interval action learning 25--72, and continuation/foresight
    learning 73--160. Do not relabel a stage from curriculum names alone;
    verify `nmcc_training_phase_index`, continuation loss, branch horizon,
    M-step fit KL, exact top-1, and optimizer counters at each boundary.
  - After training, run matched held-out Seattle evaluation against at least
    the objective-greedy heuristic, the simpler heuristic, and random policy,
    with the same ten shelters, capacity, population origins, hazards, and
    installation times. Compare the resulting scale/congestion behavior with
    the registered State College evaluation before claiming that a larger
    network benefits RL.
  - Produce population-stratified reward, M-step convergence, continuation
    explained-variance, congestion, action-location, evacuation, and benchmark
    comparison plots. Treat a single Seattle seed as engineering evidence, not
    a general city-size conclusion.

- [ ] **Make all runtime paths project-relative or explicitly configurable.**
  - Resolve the CSV, run root, cache root, and checkpoint root from one project/config object.
  - Add a smoke test that launches from a working directory outside the repository.

- [x] **Complete dependency and environment specification.**
  - Add Matplotlib and any supported optional extras.
  - Pin or constrain versions known to work together, especially OSMnx, GeoPandas, Shapely, PyTorch, Pandas, and NumPy.
  - Add a supported Python version and setup/run commands to `README.md`.

- [ ] **Decide whether guidance optimization is supported.**
  - If supported, instantiate and integrate `GuidanceDS`, update pedestrian routing, expose actions, and log meaningful guidance metrics.
  - If not supported, remove the dormant configuration, imports, and metrics from the active interface.

- [ ] **Reduce dead code and ambiguous interfaces.**
  - Remove unused imports, variables, duplicated assignments, obsolete commented blocks, and unconsumed reward fields.
  - Replace broad exception swallowing in routing/model paths with narrow exceptions and actionable messages.
  - Add type hints to cross-module contracts and use consistent naming for status fields.

- [ ] **Improve repository hygiene.**
  - Add a `.gitignore` for `.DS_Store`, `__pycache__/`, generated run artifacts, local checkpoints, and regenerable caches.
  - Decide which small fixtures must remain tracked; remove generated artifacts from version control in a separate, reviewed cleanup.
  - Keep experiment metadata and intentionally published result artifacts in a clearly named directory.

- [ ] **Expand project documentation.**
  - Document architecture, configuration fields and units, outcome definitions, action timing, reward equations, seeding, checkpoint behavior, and expected artifacts.
  - Add a short quick-start and a small offline example that does not require a full 68-replication launch.

## Current priority — validate coherent counterfactual PPO v41 (2026-10-03)

- [x] Use one coherent Bellman system: NMCC estimates matched action
  advantages, PPO is the sole actor optimizer, and the shared critic supplies
  both the PPO baseline and NMCC endpoint continuation value.
- [x] Replace the random-site/WAIT learning reward with the potential-based
  reduction in controllable forecast hazard-weighted evacuation debt; keep
  random-site comparison as a diagnostic only.
- [x] Exclude panic and casualty from action credit while retaining them as
  separately reported outcomes. Align physical NMCC branches with that same
  action-responsive cohort.
- [x] Reduce the policy input contract and express wind as signed east/north
  velocity. Add expected danger, spread momentum, escalation probability, and
  uncertainty as the concise hazard-distribution forecast.
- [x] Correct the effective experiment metadata and checkpoint contract for
  the 4x4, 2,500-person v41 curriculum.
- [x] Complete the 20-episode v41 engineering pilot and four-scenario matched
  evaluation. Immediate matched NMCC credit and PPO policy updates work, and
  RL is competitive with objective-greedy, but this is not convergence or a
  multi-seed superiority result.
- [ ] Improve continuation-value data efficiency before treating Stage F as
  successful. The pilot ended with validation explained variance 0.0020,
  0.36% continuation share, and zero branch-best changes. Diagnose target
  scale/normalization and increase backward continuation observations without
  adding a heuristic target, gate, or fallback controller.
- [ ] Run the registered 96-episode v41 State College curriculum, followed by
  deterministic matched-scenario evaluation against objective-greedy,
  risk-reduction, route-saving, random, static, and initial-only comparators.
  Treat heuristic agreement as descriptive only.
- [ ] Before claiming superiority, require: nonzero NMCC advantage dispersion;
  retained PPO policy movement; improving held-out controllable exposure cost;
  positive continuation-value rank contribution in Stage F; identical
  exogenous tapes/capacity/schedule across policies; and confidence intervals
  over multiple independent policy seeds.

## Current priority — v43 performance validation after wiring repair (2026-10-04)

- [x] Correct the critic target so a uniform-site-centered NMCC actor contrast
  cannot redefine the on-policy value baseline.
- [x] Remove attenuation of small Stage-F advantages, make reported objective
  score/cost positive and unambiguous, and add direct expected-NMCC-advantage
  movement diagnostics.
- [x] Reuse persistent isolated exact-branch workers across all eight decisions
  and verify serial/parallel physical parity and clean worker shutdown.
- [x] Tune the cheap actor fitting budget on training diagnostics only. Keep 24
  PPO epochs: 8 was effectively stationary and 64 overshot; the final smoke
  stayed below the 0.015 descriptive KL boundary without a gate or rollback.
- [x] Complete the 250-person 4x4 E/A/F smoke and matched evaluation. Mechanics,
  continuation learning, schedules, reward sign, branch coverage, and actor
  motion pass; the ten-episode policy is still inferior to objective-greedy.
- [ ] Run a preregistered longer v43 curriculum with multiple training seeds.
  Do not tune further on the two smoke evaluation scenarios. Require held-out
  objective improvement across seeds, not merely entropy reduction or NMCC
  training-table gain, before claiming that policy quality is restored.

## Current priority — v44 complete-ledger NMCC validation (2026-10-04)

- [x] Replace the controllable-only learning ledger with one complete
  time-to-safety ledger. Panicked pedestrians continue to accrue active and
  exposure time; casualties carry unresolved plain-time debt; neither outcome
  can improve a shelter action by disappearing from its label.
- [x] Replace the repeated fixed 180-minute unassigned charge with remaining
  episode time, and use the same debt definition in factual rewards, physical
  NMCC branches, Stage-F continuation composition, and evaluation.
- [x] Correct Stage F to compare
  `C_wait-C_action + gamma^dt(D_wait'-D_action')` before adding the shared
  critic continuation difference.
- [x] Use every physically branched candidate in one dense clipped-PPO actor
  update; leave unbranched candidates at zero NMCC advantage rather than
  inventing labels.
- [x] Compute exact full-categorical NMCC policy KL against the stored behavior
  distribution. Retain sampled-action approximate KL only for ordinary PPO
  transitions without a stored categorical behavior policy.
- [x] Pass all 277 repository tests and complete the ten-episode 250-person
  E/A/F smoke plus two matched evaluation scenarios. All eight actor updates
  increased expected matched NMCC advantage; final continuation validation EV
  was 0.778 and final normalized entropy was 0.926.
- [x] Complete the 96-episode, 2,500-person single-seed E/A/F run and 20-scenario
  matched evaluation. RL reduced mean HWTTS by 19,585 person-minutes versus
  max-active-population but was statistically tied with objective-greedy; the
  final checkpoint was still moving and is not a convergence claim.
- [x] Make complete-ledger objective-greedy the primary comparator and score it
  with the same positive realized-objective transformation used by evaluation.
  Register random, max-population, and scheduled initial-only as floor,
  practice-rule, and feedback controls; retain weaker one-line policies only as
  optional appendix diagnostics. All 278 tests pass.
- [ ] Run a multi-seed, longer v44 curriculum before making a superiority or
  convergence claim. Use objective-greedy—not max active population—as the
  primary paired comparator, and report weaker rules only in secondary tables.
- [ ] Add a separately registered simulation-optimization reference and a
  rolling-horizon capacitated location-allocation benchmark before claiming
  proximity to an online optimizer. Do not relabel time-zero predeployment or
  any one-line heuristic as that OR benchmark.

## Current priority — v45 uniform-reference policy improvement (2026-10-04)

- [x] Replace the no-more-shelters potential with the fixed uniform feasible-site
  policy value `V_mu`; use `V_mu(s)-C-V_mu(s')` as the signed factual reward.
- [x] Make Stage E collect exact uniform-policy data and learn `V_mu`; freeze
  that reference in Stages A/F so the reward zero does not drift with the actor.
- [x] Align NMCC, PPO critic, and actor labels with the same random-relative
  Bellman decomposition, including the dedicated Stage-F policy-savings value
  `W_pi`.
- [x] Verify signed bad-action penalties, zero-in-expectation random credit,
  reward telescoping, backward value targets, branch decomposition, exact
  accounting, and end-to-end checkpoint/replay transfer. All 282 tests pass.
- [x] Complete the corrected 250-person smoke and the 96-episode 2,500-person
  State College curriculum plus 20 matched deterministic evaluation scenarios.
- [x] Confirm the core reward claim: RL significantly beats uniform random;
  random siting no longer earns scheduled-capacity credit. Confirm that
  continuation has actual ranking influence late in Stage F.
- [ ] Do not claim superiority yet. RL is statistically tied with
  objective-greedy and slightly, nonsignificantly worse than the strongest
  simple online heuristics. Repeat with multiple independently trained policy
  seeds before changing reward or features again.
- [ ] Diagnose actor update scale using the existing dense NMCC records. Tail
  KL exceeded the descriptive target in 85% of updates even though late KL
  occasionally settled. Prefer a justified optimizer/schedule adjustment over
  a KL gate, rollback, or fallback controller.
- [ ] Profile the intermittent branch-worker memory-compression stalls seen in
  the long run. Prove whether retained simulator object graphs survive between
  persistent batches before changing worker lifecycle; preserve exact branch
  parity and do not add partial-result fallback behavior.
- [ ] Treat `static_greedy` and `rl_precommit` as explicit time-zero/offline
  references only. The fair feedback comparator is scheduled `initial_only`;
  all online policies must retain the common 15-minute installation schedule.

## Current priority — v48 NMCC actor-scale validation (2026-10-05)

- [x] Restore stage-local decay for categorical uniform exploration and action
  temperature, and retain independent actor/critic learning-rate decay.
- [x] Replace raw advantage-difference regression with the complete NMCC
  E-step target and train the actor head by a replayed full-distribution
  `KL(q || pi_theta)` M-step over all feasible cells.
- [x] Increase actual actor optimization capacity to a `0.001` initial learning
  rate and 24 replay passes without adding a KL gate, rollback, fallback,
  gradient clipping, or action correction.
- [x] Require a direct integration test to show that an actor update reduces
  KL to the NMCC policy target; pass all 290 tests and the 250-person staged
  smoke/evaluation.
- [ ] Run the 96-episode, 2,500-person v48 curriculum with multiple independent
  policy seeds before drawing a performance conclusion. The smoke policy now
  moves and beats random/max-active, but still trails objective-greedy.
- [ ] On the long run, report target-KL reduction, NMCC expected-advantage
  gain, entropy, deterministic action diversity, and held-out objective cost
  together. None is sufficient alone to claim convergence or foresight.

## Current priority — v49 coherent NMCC policy iteration (2026-10-06)

- [x] Align the NMCC E-step prior with the raw masked actor optimized by the
  M-step; keep temperature and uniform mixing solely in behavior exploration.
- [x] Delete persistent actor-target replay so targets generated by old actor
  priors and continuation critics cannot train a new policy iteration.
- [x] Delete the separate continuation replay learner and use the shared
  current-policy critic consistently in Stage P.
- [x] Pass all 290 tests and complete a fresh 2E + 6P staged smoke plus matched
  evaluation. All six actor fits reduced target KL and increased expected
  NMCC advantage; exploration decayed from `0.35` to `0.02`.
- [ ] Run the 2,500-person 16E + 80P v49 curriculum with independent training
  seeds. The smoke checkpoint remains below random and objective-greedy, so do
  not claim policy improvement from optimizer diagnostics alone.
- [ ] In the long run, require the current-policy continuation critic to gain
  positive held-out/on-policy explained variance and material branch-ranking
  influence. If it does not, diagnose the value target and sample efficiency;
  do not restore stale continuation replay.
- [ ] Reboot macOS before the next expensive campaign to reap the 25 historical
  `UE` workers that PID 1 did not clear after exact TERM and KILL signals.

## Current priority — v50 simulator-graded rollout policy iteration (2026-10-06)

- [x] Replace the four-cell one-interval MPO/critic label with full-horizon,
  same-snapshot, common-future simulator grading under a frozen continuation
  actor.
- [x] Initialize from the exact objective-greedy policy and learn only a neural
  correction by inverse-variance-weighted paired regression.
- [x] Give neglected feasible cells bounded, persistent coverage through two
  least-evaluated panel slots; retain `nmccPiMaxBranches=0` as explicit
  exhaustive evaluation rather than a hidden fallback.
- [x] Preserve the random-reference reward, complete ledger, observations,
  simulator and evaluation objective; evaluate the fixed random-policy
  potential directly in the one-stage learner.
- [x] Complete a 250-person smoke and verify nonzero actor/representation
  gradients, `0.81–0.85` pairwise fit accuracy, positive rollout-score gains,
  decayed exploration, zero critic updates, and 15/16 cell coverage (the last
  cell had no feasible shelter candidate).
- [ ] Run the 96-episode, 2,500-person v50 curriculum with at least three
  independent training seeds, then perform matched deterministic evaluation
  against objective-greedy, random, max-active and the online rollout teacher.
- [ ] Report achieved improvement as `(J_random - J_RL) / (J_random -
  J_rollout)` together with decision time, branch-panel coverage, pairwise
  ranking accuracy and uncertainty. Do not infer policy quality from reward,
  KL or fit accuracy alone.

## Current priority — v51 residual correction and honest diagnostics (2026-10-06)

- [x] Replace the saturating bounded residual with a zero-initialized,
  unbounded linear correction to the fixed objective-greedy base score.
- [x] Remove the obsolete categorical actor target and ensure every actor fit
  uses all pair contrasts from only the current frozen-policy iteration.
- [x] Report sampled behavior entropy separately from the raw temperature-one
  actor entropy before and after fitting; apply no exploration transform during
  deterministic evaluation.
- [x] Report actual pairwise regression loss/accuracy before and after fitting,
  raw residual magnitude, and exact actor-to-iteration-prior KL. Keep all of
  these descriptive: no gate, rollback, fallback action, or safe controller.
- [x] Remove the retired held-out controller gate and PPO KL threshold from
  rollout-policy-iteration convergence and plots. Use complete branch coverage,
  finite actor fits, complete rollout batches, and tail stationarity instead.
- [x] Pass the full test suite and a fresh four-episode, 250-person integration
  smoke with two loss-reducing actor updates, zero critic updates, no pending
  actor transitions, and no persistent replay in the one-stage learner.
- [ ] Run the 96-episode, 2,500-person v51 curriculum for at least three
  independent policy seeds, then perform matched deterministic evaluation
  against objective-greedy, uniform random, max-active, and the online rollout
  teacher. The smoke is not evidence of benchmark superiority.
- [ ] In the long run, plot behavior entropy, raw actor entropy, paired loss and
  accuracy, residual RMS, actor KL, per-cell coverage, complete objective cost,
  and deterministic action diversity together. Confirm that the residual
  improves held-out cost beyond its objective-greedy base rather than merely
  fitting noisy branch panels.

## Current priority — v53 decision-aligned rollout policy iteration (2026-10-07)

- [x] Preserve raw complete-cost magnitude across states with one
  iteration-wide scale.
- [x] Include the executed and objective-greedy base actions in every branch
  panel and use four shared futures in the production curriculum.
- [x] Replace unstable minimum-variance precision weights with regret divided
  by median-stabilized paired standard error, so clear consequential wins drive
  the actor.
- [x] Bound worker protocol reads and close all partially constructed or
  interrupted worker pools; verify repaired runs create no new orphans.
- [x] Pass 166 focused tests and a matched smoke whose expected physical
  rollout advantage moves in the correct direction.
- [ ] Restart macOS to clear the historical uninterruptible workers before the
  expensive campaign.
- [ ] Run the 128-episode, 2,500-person curriculum in
  `config/state_college_training_curriculum_2500_rollout_pi_v53.json` with at
  least three independent policy seeds.
- [ ] Evaluate deterministically on at least 30 matched scenarios against
  objective-greedy, random, max-active, and the online rollout teacher. Report
  forward-looking ablations at later decisions; do not infer foresight from
  feature presence or one-update optimizer diagnostics.

## Current priority — v55 categorical rollout policy iteration (2026-10-08)

- [x] Replace pairwise regression as the primary actor loss with the complete
  categorical `KL(q || pi_theta)` M-step.
- [x] Keep objective-greedy as the zero-residual initial policy and constrain
  each simulator-graded improvement target to `KL(q || pi_old) <= 0.05`.
- [x] Use four-episode policy iterations and discard actor targets immediately
  after each iteration; do not serialize or restore stale actor replay.
- [x] Grade the objective-greedy base, executed action, strongest base-score
  contenders, and uniformly sampled feasible extras on four shared, complete-
  horizon futures; exhaustively grade the first two installations.
- [x] Make paired contrast uncertainty set the target temperature floor and
  keep pairwise ranking accuracy diagnostic-only.
- [x] Pass all 305 repository tests and validate both v55 curricula.
- [ ] Complete the background 12-episode smoke and six matched scenarios
  against every available benchmark.
- [ ] If the smoke succeeds, complete the chained 128-episode, 2,500-person
  training and 20-scenario deterministic matched evaluation. Assess held-out
  cost, improvement over objective-greedy, forward-looking branch changes,
  action diversity, target-fit KL, and retained-policy KL together.
