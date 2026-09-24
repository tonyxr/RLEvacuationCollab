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

## P0 — Blockers before further training

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
