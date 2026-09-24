The codes are personal properties of Xiaoru Shi, Department of Industrial and Systems Engineering, University of Washington, Seattle. Any unauthorized use will be subject to legal consequences.

## Reproducible environment

Use the all-Conda-forge CPU environment in `environment.yml`; mixing pip
PyTorch with Conda scientific packages can load incompatible OpenMP runtimes.
On Apple Silicon, create a native environment rather than an x86/Rosetta one:

```bash
micromamba create --platform osx-arm64 -f environment.yml
micromamba activate rlevacuation
python -m unittest discover -s tests -v
```

The convergence-first five-city runner uses a conservative PPO learning rate
of `0.0003`; use `--learning-rate` only for a separately identified sensitivity
run.

## Current production campaign

The current campaign first learns with exactly 5,000 individual pedestrians in
Malibu/Santa Monica Bay, State College, Spokane, Seattle, and Chicago. Training
uses one stationary nominal distribution: exactly three hazard instances and
50% first-exposure panic susceptibility. It produces one pooled, city-agnostic
policy. Only after convergence and a held-out 5,000-person RL-versus-heuristic
behavior gate does the separate factorial evaluation vary city, population
(5,000–25,000), hazard count (1–5), and panic (10%–90%). Guidance points are
deprecated and are not populated. The paper's social-force recurrence,
synchronized link congestion, and 5 m topology-rebuilding intersection
consolidation are active in every episode.

The Southern California profile is a 35 km point-centered bounding footprint
covering the Malibu coast, Santa Monica Bay, and substantial north and west Los
Angeles. It uses the full OSM walking network, not the presentation-only
major-road skeleton.

Validate the complete plan without starting the long run:

```bash
python full_experiment_campaign.py --dry-run
```

The registered plan trains one fixed policy for 120 episodes per city (600
episodes), caches it only after convergence under an exact checksummed
contract, runs 50 held-out nominal RL/heuristic behavior-gate episodes, and
only then executes 37,500 matched full-factorial episodes. Uncertainty for the
primary campaign is conditional on this frozen policy and resamples held-out
scenario seeds; it does not claim variation over independently trained policy
seeds. Start or resume it with:

```bash
python full_experiment_campaign.py
python full_experiment_campaign.py --resume
```

The campaign first validates all OSM graphs, consolidated topology, and the
shared 20-site candidate budget. Evaluation is journaled after every episode
and exports a dedicated RL-versus-density-heuristic end-to-end deployment
latency table. See `docs/CODE_PATH_DIAGNOSTIC_20260914.md` for the active-code
inventory and model contracts.

## Supplementary OR journal experiment package

The revised INFORMS/IISE-oriented design is frozen in
`config/or_journal_experiment_suite.json` and explained in
`docs/OR_JOURNAL_EXPERIMENT_PROTOCOL.md`. It adds a demand-aware static policy,
an `rl_precommit` timing-isolation control, whole-episode timing, a controlled
empty-center/populated-ring test, and policy/action-space latency benchmarks.

```bash
python or_journal_experiments.py --mode dry-run
python or_journal_experiments.py --mode tractable
python or_journal_experiments.py --mode audit \
  --output-dir runs/or_journal_full_20260909
```

`--mode tractable` executes the computational and controlled-mechanism tiers;
it does not claim to have completed the five-city evacuation trials. The run
manifest, raw tables, audit, and publication-ready PNG/SVG figures are written
to the selected output directory. City-level efficacy continues through the
existing backtest runners after the State College gate.

The confirmatory extreme-case sensitivity and timing experiment is frozen in
`config/extreme_sensitivity_experiment.json` and documented in
`docs/EXTREME_SENSITIVITY_EXPERIMENT_PROTOCOL.md`. It compares RL with the
realized-population heuristic and a non-anticipative static expected-demand
plan over five population levels, five city scales, five grid resolutions, and
four empty-center extreme variants. It also times both registered cell
partition modes:

```bash
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/rlevac_matplotlib_cache \
python extreme_sensitivity_experiments.py \
  --output-dir runs/extreme_sensitivity_full_20260909
```

Regional discretization has two explicit options. `equal_area` uses equally
spaced projected-metre boundaries; `node_density_adaptive` uses OSM road-node
quantiles so dense coordinate ranges receive smaller cells. Both retain exactly
`cellX * cellY` contextual graph nodes. The model uses the union of
four-neighbour and road-crossing spatial edges plus current pedestrian-to-
shelter route edges. Shared pedestrian, hazard, and infrastructure encoders can
process any runtime node count, while the confirmatory design still trains a
separate checkpoint for each resolution. Actions rank forecast-safe regional
cells. Every RL and heuristic policy then uses the same deterministic lower-
level rule to select the physical shelter site inside the chosen cell, so the
action remains administrator-interpretable and site execution is not a hidden
policy-specific choice. The full administrator-facing feature and reward
contract is documented in `docs/MDP_AND_OPTIMIZATION_DESIGN.md`. The
established behavior remains the default adaptive mode. Select a mode in any
backtest with, for example:

```bash
python backtest.py --override cellPartitionMode=equal_area
python multicity_backtest.py \
  --override cellPartitionMode=node_density_adaptive
python cell_partition_experiment.py
```

Model version 17 adds an episode-level LSTM over the minute-by-minute GNN
observations, explicit improvement/deterioration momentum features, and
separate critic heads for safe completion, casualties, evacuation time, and
hazard exposure. Recurrent PPO minibatches preserve whole episodes. The final
executed shelter action owns outcomes through the true simulation terminal;
deployment-budget exhaustion and temporary candidate infeasibility no longer
censor later casualties or exposure.

Model version 21 adds the production Hybrid Natural-Momentum Counterfactual
Control (NMCC) learner. At each shelter decision, exact common-noise
action-versus-`WAIT` branches provide a fixed-horizon causal target. An
action-independent natural-evolution head and a candidate-local three-model
causal residual ensemble learn that target, while recurrent PPO uses the exact
paired advantage, a factorized outcome critic, uncertainty-penalized planner
guidance, and decaying entropy/temperature exploration. The planner scores are
detached from the actor loss, so policy imitation cannot rewrite the learned
environment model. Post-action outcomes are closed exactly at the NMCC horizon
and the ordinary SMDP transition continues through the complete episode tail.

Model version 22 adds explicit rollout-gated NMCC training. Natural-dynamics
pretraining updates only the action-independent world model; causal
pretraining then adds the candidate residual ensemble; controller warm-up
enables PPO, the factorized critic, and scheduled planner guidance; final joint
optimization blends the exact local causal advantage with full-tail GAE. Actor
entropy, temperature, teacher, and guidance schedules advance only while the
controller is trainable, so world-model pretraining cannot consume the
controller's exploration budget.

Model version 23 replaces the coupled GAE actor/critic label with two explicit
temporal contracts. The actor receives the complete duration-discounted return
to go from each installed shelter through the true episode terminal, with no
critic bootstrap. It is centered and scaled only by a lagged baseline keyed by
city, population, hazard count, and within-episode decision position; the
current rollout cannot subtract away its own improvement. The factorized
critic receives an independent one-step semi-Markov TD(0) target and raw Huber
loss. Actor and critic/world heads have disjoint optimizers; one actor pass and
four critic passes are the registered defaults. A full-rollout KL violation
restores both actor parameters and actor-optimizer state, backs off only the
actor learning rate, and does not advance actor schedules. NMCC branches remain
available as auxiliary world-model supervision, but the registered v23 pilot
sets their direct actor credit, planner guidance, and imitation weights to
zero.

Model version 25 replaces the v24 behavior-anchored actor target with direct
candidate-score learning. Exact cellular-automata branches now retain a
six-coordinate physical outcome vector for every candidate and its paired
`WAIT` trajectory. Training is ordered as natural-system identification,
candidate-caused effect identification, score-controller warm-up, and joint
optimization. The actor receives listwise and margin supervision only among
candidates evaluated from the same simulator snapshot; deployment takes the
highest-scoring feasible candidate. Training exploration is a scheduled
epsilon-greedy mixture over the masked scores, while evaluation is always
deterministic argmax. Actor and critic learning rates use independent warm-up
and cosine decay schedules. Forecast danger, physical site availability,
remaining capacity tokens, hazard safety margin, and optional positive
risk-time benefit are hard constraints rather than reward penalties.

Model version 28 uses one transparent base rule: install the safe candidate
with the largest capacity-capped reduction in future active plus exposure
time. Shelter capacity is reserved when a route is assigned, conserved while
people are en route, and released on admission, rerouting, casualty, panic,
termination, or shelter loss. A persistent fitted-policy controller may
improve on that rule. The
first two, highest-leverage shelter decisions are branched
exhaustively through the physical episode terminal. Every exact labelled
episode is retained in a bounded, checkpointed replay dataset; complete
episodes are assigned to fitting, early stopping, or a disjoint deployment
gate. A dedicated
intervention-value ensemble is refit over the growing dataset through the full
relational GNN and episode LSTM, with held-out early stopping and direct
pairwise candidate-difference supervision. The system encoder is frozen during
these refits and feeds a wide-and-deep intervention model: a linear path over
the normalized physical features stabilizes small-data fitting while a deep
path preserves nonlinear GNN/LSTM capacity. Deployment stays on the simple
risk-time-reduction prior until the lower confidence bound of gate-set paired
gain is positive. Only decisions that branched every feasible candidate may
certify the gate. After
that gate opens, the prior contributes only its common base-score level and the
conservative physical-advantage estimate supplies candidate ordering, so the
heuristic is not double-counted. Ensemble uncertainty is computed on each
candidate's paired value
difference from the base action, so arbitrary head-specific value offsets
cannot corrupt the conservative ranking. The
scientific reward remains the four-term safe-completion, casualty,
evacuation-time, and exposure objective.

The production State College v28 curriculum is:

```bash
python multicity_backtest.py \
  --cities state_college_pa \
  --launch-id state_college_2500_risk_value_v28 \
  --policy-replicates 1 \
  --train-episodes-per-city 104 \
  --training-curriculum config/state_college_training_curriculum_2500_risk_value_v28.json \
  --train-only --no-require-convergence --no-policy-cache
```

For pooled training across cities and the registered population × hazard-count
× panic scenario distribution, use
`config/scenario_general_training_curriculum_risk_v28.json`. Its schema-v2
`learner_overrides` declares one immutable learner contract while scenario
fields vary by stage and variant. It schedules 120 episodes per city and covers
the complete 3 × 3 × 3 factorial of 1,000/2,500/5,000 pedestrians, 1/3/5
hazard instances, and 0.1/0.5/0.9 panic susceptibility before nominal
consolidation. Every factorial cell appears twice per city and every stage ends
on a complete pooled PPO rollout boundary.

The earlier 2,500-person State College v25 curriculum is retained as an
ablation:

```bash
python multicity_backtest.py \
  --cities state_college_pa \
  --launch-id state_college_2500_nmcc_score_v25 \
  --policy-replicates 1 \
  --train-episodes-per-city 64 \
  --training-curriculum config/state_college_training_curriculum_2500_nmcc_score_v25.json \
  --train-only --no-require-convergence --no-policy-cache
```

Audit a completed v25 score-training launch with the model-specific diagnostics
rather than the legacy v22 8/8/16/32 analyzer:

```bash
python analyze_nmcc_score_training.py \
  runs/state_college_2500_nmcc_score_v25
```

The completed one-seed pilot produced 16 optimizer updates. Natural loss fell
from 0.04220 to 0.02090, dueling loss from 0.10576 to 0.01476, exact-target fit
KL improved in all 11 actor updates, and mean exact top-1 increased from 0.4083
to 0.4878. Training return did not converge, and this train-only run did not
test superiority over a heuristic. Its dedicated JSON and six-panel figure are
`runs/state_college_2500_nmcc_score_v25/nmcc_score_training_audit.json` and
`nmcc_score_training_diagnostics.png`.

Before a map-backed run, reproduce the bounded matched v25-v26 engineering
backtest:

```bash
python nmcc_learning_backtest.py \
  --train-episodes 24 --eval-episodes 12 --overwrite
python nmcc_learning_backtest.py \
  --train-episodes 24 --continuation-episodes 24 --eval-episodes 12 \
  --resume-v26-only
```

Its JSON and four-panel plot are written to
`runs/nmcc_value_backtest_v26.{json,png}`. The harness reports persistent-data
growth, training and episode-heldout rank/top-1, the paired-gain confidence
bound and controller gate, exploration decay, reward components, and exact
capacity parity. This is an engineering signal/optimization check, not State
College convergence evidence. The recorded 48-episode v26 run passed all ten
gates: final held-out paired-gain lower bound `+0.00982`, twelve-seed return
gain `+0.04069` over the fixed route-saving base, and exact capacity parity.

Each dynamic installation uses an equal-capacity shelter token. The default and
the registered State College NMCC study use 500-person tokens, so five successful
installations add exactly 2,500 places regardless of which feasible physical
sites a policy chooses. Raw site capacity remains an eligibility constraint.
This does not by itself guarantee equal *realized* total capacity: the v25
State College pilot installed all five tokens in only 53/64 episodes because
hard safety/feasibility masks sometimes left no action. Confirmatory comparison
therefore requires a common hazard-only capacity-viability schedule and must
fail closed before statistics unless RL and heuristic install the same total
capacity.

The complete 3,000-person State College training contract is in
`config/state_college_training_curriculum_3000_nmcc_hybrid.json`:

```bash
python multicity_backtest.py \
  --cities state_college_pa \
  --launch-id state_college_3000_nmcc_v21 \
  --train-episodes-per-city 64 \
  --training-curriculum config/state_college_training_curriculum_3000_nmcc_hybrid.json \
  --train-only --no-require-convergence --no-policy-cache
```

The first full staged pilot uses 2,500 individual pedestrians and four
rollout-aligned stages (8/8/16/32 episodes):

```bash
python multicity_backtest.py \
  --cities state_college_pa \
  --launch-id state_college_2500_nmcc_staged_v22 \
  --policy-replicates 1 \
  --train-episodes-per-city 64 \
  --training-curriculum config/state_college_training_curriculum_2500_nmcc_staged_v22.json \
  --train-only --no-require-convergence --no-policy-cache
```

The checkpoint-incompatible v23 rerun uses the same population, hazard,
capacity, and 8/8/16/32 episode budget while changing only the temporal-credit
and optimizer contract:

```bash
python multicity_backtest.py \
  --cities state_college_pa \
  --launch-id state_college_2500_temporal_credit_v23 \
  --policy-replicates 1 \
  --train-episodes-per-city 64 \
  --training-curriculum config/state_college_training_curriculum_2500_temporal_credit_v23.json \
  --train-only --no-require-convergence --no-policy-cache
```

The bounded synthetic v23 backtest is recorded at
`runs/temporal_credit_learning_backtest_v23_smoke.json`. Both actor updates
were accepted, none were rolled back, mean attempted and retained KL were
`7.00e-7`, and lagged-baseline coverage rose from zero at the first update to
one at the second. The NMCC-auxiliary and no-NMCC policies remained exactly
matched, as required by disjoint optimizer ownership. Eight training episodes
and four held-out scenarios are an execution/identifiability check, not
evidence of reward improvement or convergence.

The completed single-seed engineering pilot is in
`runs/state_college_2500_nmcc_staged_v22_retry_seed20260920`. It completed all
64 episodes and eight optimizer gates. Exact NMCC target coverage was 100% and
the median counterfactual/GAE SD ratio was 0.2586, but the joint-stage return
slope was -0.004476 per episode and the registered policy-convergence gate
failed. This is evidence that the counterfactual credit path works, not that
the policy improves or converges. Regenerate the stage-specific audit with:

```bash
python analyze_staged_nmcc_training.py \
  runs/state_college_2500_nmcc_staged_v22_retry_seed20260920 \
  --bootstrap-draws 10000
```

The command writes five stage-specific diagnostic figures, two summary CSVs,
a machine-readable validation JSON, and `STAGED_TRAINING_VALIDATION.md`.

For a short execution check, use the eight-episode smoke curriculum. It tests
integration only and must not be reported as convergence or policy evidence:

```bash
python multicity_backtest.py \
  --cities state_college_pa \
  --launch-id state_college_3000_nmcc_v21_smoke \
  --train-episodes-per-city 8 \
  --training-curriculum config/state_college_training_curriculum_3000_nmcc_hybrid_smoke.json \
  --train-only --no-require-convergence --no-policy-cache
```

The partition comparison is registered in
`config/cell_partition_experiment.json`. Its deterministic State College
characterization writes auditable cell tables and the F8a geometry/balance
figure; evacuation accuracy remains a separately trained paired experiment.

The cross-city research design and commands are documented in
`docs/MULTICITY_EXPERIMENT_PROTOCOL.md`.

The secondary benchmark definitions and their operations-research literature
basis are registered in `docs/BENCHMARK_MODEL_PROTOCOL.md`. The dynamic
hazard-weighted and accessibility-deficit policies use the same regional
observation, exact-candidate mask, cadence, budget, and candidate executor as
RL. Run a complete strategy evaluation with:

```bash
python multicity_backtest.py \
  --strategies rl,heuristic,hazard_weighted,accessibility_deficit,random,static_greedy,rl_precommit,initial_only
```

The completed integration backtest, parity audit, five-city map preflight, and
boundary between engineering validation and confirmatory evidence are recorded
in `docs/BENCHMARK_BACKTEST_RESULTS_20260913.md`.

## Earlier staged-training studies

The operational and continuation curricula are retained to reproduce earlier
engineering studies. They are not used by the current full campaign; its
registered curriculum is
`config/staged_training_curriculum_5000_convergence.json`.

```bash
python multicity_backtest.py \
  --launch-id staged_training \
  --policy-replicates 1 \
  --train-episodes-per-city 120 \
  --training-curriculum config/staged_training_curriculum_5000_convergence.json \
  --train-only --require-convergence --convergence-min-episodes 100
```

The executed 200-episode training record, state-sensitivity audit, matched
five-city benchmark results, conservative residual calibration, artifact hashes,
and the explicit boundary between directional evidence and a robustness claim are in
`docs/STAGED_TRAINING_RESULTS_20260913.md`.

The earlier versioned E0--E6 design, required result schemas, and
publication-figure workflow are documented in
`docs/FULL_EXPERIMENT_FIGURES.md`. Validate such a legacy launch
without plotting using:

```bash
python generate_full_experiment_figures.py \
  --launch-dir runs/FULL_EXPERIMENT_LAUNCH \
  --validate-only
```

Use `--require-complete-suite` for the paper build. The strict build requires
all eight policy seeds, the full 90-scenario-per-city factorial, and every
E0--E6 table; it will not silently turn smoke results into confirmatory figures.

The five-city population/candidate stress test is separately executable and
uses a true 60-transition, 60-minute maximum horizon:

```bash
python population_candidate_backtest.py --dry-run
python population_candidate_backtest.py --pilot
```

Its fixed 5×5 levels, full-run requirements, provenance checks, and F13 paper
figure are documented in `docs/FULL_EXPERIMENT_FIGURES.md`.

The companion map-factorial runner produces exact, numbered shelter-choice
maps and seven aligned evacuation-progress panels for RL, the shared-interface
active-population heuristic, and static time-zero predeployment:

```bash
python map_factorial_backtest.py --dry-run
python map_factorial_backtest.py --pilot
python map_factorial_backtest.py \
  --source-launch-dir runs/FULL_60_MINUTE_TRAINING_LAUNCH \
  --launch-id factorial_maps_CONFIRMATORY
```

Its fixed design is in `config/map_visualization_experiment.json`. A full run
contains 500 visual condition cells and 1,500 matched episodes. Confirmatory
mode refuses a non-converged policy or a checkpoint trained under a different
60-minute horizon, two-minute decision cadence, enlarged OSM footprint,
five-addition budget, or pedestrian-congestion contract. The design fixes a
maximum of five additional shelters
independently of the candidate-pool level, so policies select a genuine subset
when 10, 15, or 20 candidates are available.

Both milestone and decision-epoch maps use one red point per active pedestrian
agent and a shared, continuous 0--1 heatmap of `dangerLevelByCell`. The fixed
scale permits direct comparison across policies and times. Decision-point
coordinates are exported as `decision_epoch_pedestrians.csv.gz`, while cell
danger values remain in `decision_epoch_cells.csv`.

Pedestrian speed units, the synchronized physical-link Weidmann congestion
model with six 10-second load updates per one-minute transition, calibration
assumptions, diagnostics, and sensitivity requirements are documented in
`docs/PEDESTRIAN_SPEED_AND_CONGESTION.md`. Every map condition also produces a
congestion diagnostic alongside the shelter sequence, decision-epoch priority
map, and evacuation-progress comparison.
