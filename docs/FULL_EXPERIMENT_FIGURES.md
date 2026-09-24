# Full E0--E6 experiment and figure pipeline

## Purpose

The full experiment is specified in
`config/full_experiment_suite.json`. The specification is executable metadata,
not a narrative checklist: it fixes the sample sizes, factorial conditions,
required result columns, experiments E0--E6, and the figure families before
confirmatory results are inspected. `ExperimentSuite.py` validates this
contract without importing the simulator or PyTorch.

The figure generator is read-only with respect to training and simulation:

```bash
python generate_full_experiment_figures.py \
  --launch-dir runs/FULL_EXPERIMENT_LAUNCH \
  --require-complete-suite
```

It fails closed when a required table, policy seed, matched scenario, or
factorial cell is absent. During development, omit `--require-complete-suite`
to render only figures supported by currently available evidence. The output
`full_suite_readiness.json` labels that bundle `partial_available_data` and
lists every missing input; a skipped panel is never replaced by simulated or
placeholder values.

To audit table and sample readiness without importing Matplotlib:

```bash
python generate_full_experiment_figures.py \
  --launch-dir runs/FULL_EXPERIMENT_LAUNCH \
  --validate-only
```

## Fixed design

The confirmatory design uses eight independently initialized pooled policies
and 120 training episodes per city for each policy. Evaluation crosses three
initial shelter-capacity regimes, three hazard regimes, and two pedestrian
demand patterns. Five common-random-number replications in every factorial
cell give 90 scenarios per city. State College, Reading, Spokane, Seattle, and
Chicago are fixed study sites and receive equal weight in the macro estimate;
they are not resampled as though they were a probability sample of cities.

The primary metric is `objective_episode_return`. This is the same simple
safe-completion/casualty/risk-weighted-person-time equation accumulated over
the full episode, independent of how many online actions a policy takes. The
field `episode_return` remains available as the sum of action-owned PPO
rewards, but it must not be used to compare a dynamic policy with static
predeployment. The reporting code automatically uses
`objective_episode_return` and the full-suite schemas require it for every
policy comparison.

For paired RL--heuristic intervals, the reporter resamples a global policy-seed
index and independently resamples scenarios within each fixed city. It then
macro-averages the five city estimates. The same trained seed index is used
across cities in each bootstrap draw, preserving the actual dependence induced
by evaluating one policy in all cities.

## Required source tables

The exact columns are defined once in `config/full_experiment_suite.json`.
Core E1 files remain at the launch root:

- `training_episode_summary.csv`
- `evaluation_episode_summary.csv`

Experiment-specific tables are written under `full_suite_tables/`:

- `scenario_calibration.csv` (E0)
- `checkpoint_evaluation.csv` (E1 fixed development-set learning curve)
- `regime_evaluation.csv` (E2 full policy/factorial comparison)
- `scalability_evaluation.csv` (E3 hierarchy versus flat action)
- `ablation_evaluation.csv` (E4)
- `transfer_evaluation.csv` (E5 leave-one-city-out)
- `robustness_evaluation.csv` (E6)
- `scale_stress_evaluation.csv` (E6 population/candidate stress test)

## Population and candidate scale stress

The scale-stress design crosses five pedestrian populations
`[10000, 20000, 30000, 40000, 50000]` with five sampled OSM shelter-candidate
counts `[25, 50, 75, 100, 125]` in every city. Candidate count therefore tests
the breadth and spatial quality of the feasible implementation set while the
resource budget remains fixed at five additional shelters.

The configured horizon is 60 one-minute simulator transitions. Because the simulator
iterates over `range(1, stopTime)`, the runner sets `stopTime=61`; an episode
may still terminate earlier when every pedestrian has reached a terminal
outcome. One shelter may be implemented every two transitions, beginning at
the first decision epoch, subject to the five-addition budget. Run the
auditable plan, a single computational pilot, or the full
matrix with:

```bash
python population_candidate_backtest.py --dry-run
python population_candidate_backtest.py --pilot
python population_candidate_backtest.py \
  --source-launch-dir runs/FULL_60_MINUTE_TRAINING_LAUNCH \
  --launch-id population_candidate_scale_CONFIRMATORY
```

The full matrix contains 5 cities × 25 factor cells × 5 stochastic scenario
replications. Each scenario is evaluated by eight independently trained RL
policies and one matched heuristic, for 5,625 episodes. The runner rejects an
incomplete checkpoint set and rejects confirmatory checkpoints trained at a
different horizon or decision cadence. Pilot mode permits a mismatched source
only to validate execution and labels that limitation in its manifest.

A horizon-compatible pooled checkpoint launch can be trained with the existing
five-city runner (the remaining training distribution and convergence rules are
those recorded in that launch manifest):

```bash
python multicity_backtest.py \
  --launch-id pooled_five_city_60_minute \
  --policy-replicates 8 \
  --train-episodes-per-city 120 \
  --train-only \
  --override stopTime=61 \
  --override pedVol=50000 \
  --override shelterActionInterval=2 \
  --override maxAdditionalShelters=5
```

The OSM products use `paper_figures/map_figure_index.csv`. Every indexed map
must carry `non_interventional=true`. Existing four-policy map composites are
copied into the full bundle when present; otherwise the indexed policy panels
are composed without rerunning the episode.

## Full shelter-decision and progress map matrix

`config/map_visualization_experiment.json` fixes the larger illustrative map
matrix requested for the paper. It crosses the five cities with populations
`[10000, 20000, 30000, 40000, 50000]`, sampled OSM candidate counts
`[5, 10, 15, 20]`, and hazard counts `[1, 2, 3, 4, 5]`. Two shelters are
common at time zero and at most five additional shelters may be installed. The
resource budget is independent of candidate-pool size: level 5 is an explicit
scarcity boundary with three available additions, whereas levels 10, 15, and
20 require policies to choose a strict subset. This separation is essential;
allowing the installation budget to equal the candidate count would make all
policies eventually install the same set. One matched visual replication per
cell yields 500 condition cells and 1,500 episodes across RL, the
active-population heuristic, and static predeployment. Each condition also
exports a three-panel congestion diagnostic showing mean realized walking
speed ratio, the minimum occupied-link speed ratio, and maximum physical-link
density over elapsed minutes for all three policies.

RL and the heuristic receive the same regional observation, exact-candidate
mask, action timing, maximum budget, and direct candidate executor. Their exact
chosen shelter is exported at every decision as the OSM
node identifier, local coordinates, candidate cell, capacity, decision time,
and decision order. Static placement is not presented as an online policy: its
additional candidates are explicitly marked as anticipative choices made at
`t=0` under the same maximum shelter budget.

A separate decision-epoch comparison captures every online action rather than
sampling those actions at coarse evacuation milestones. Each panel displays
one red point for every active pedestrian agent, the continuous normalized
danger of every cell on a common 0--1 heatmap, the exact selected OSM candidate
and its host region, active population, and decision time.
Static predeployment is omitted from this dynamic panel because it has no
online decision epochs; its simultaneous choices remain in the shelter
implementation comparison.

The visual encoding is identical in the milestone maps. Individual decision
coordinates are retained in a compressed audit table rather than inferred
from regional counts, and the untransformed `dangerLevelByCell` values that
drive the heatmap are retained in the cell table. A fixed scale is used across
all policies, cities, factor levels, and timesteps; no panel-specific color
normalization is permitted.

Each progress comparison contains aligned panels at
`t=[0,10,20,30,40,50,60]` minutes. If evacuation completes early, the system records
the true terminal boundary and displays later requested panels as explicitly
labeled absorbing MDP states. Hazard and shelter layers are frozen and no
post-terminal pedestrian positions are synthesized. The audit tables carry an
`absorbing_after_terminal` indicator and the actual terminal time.

```bash
python map_factorial_backtest.py --dry-run
python map_factorial_backtest.py --pilot
python map_factorial_backtest.py \
  --source-launch-dir runs/FULL_60_MINUTE_TRAINING_LAUNCH \
  --launch-id factorial_maps_CONFIRMATORY
```

The runner is atomic and resumable, validates RL--heuristic initial-observation
parity, checks every source-image checksum before composition, and records
conditions whose public map contains fewer eligible candidates. Full mode is
blocked unless the source launch is converged and was trained with the same
60-minute horizon, two-minute decision cadence, enlarged city footprint,
five-addition resource budget, and congestion transition law. Pilot mode may
use an older checkpoint only to test the pipeline and records every mismatch
as a non-confirmatory limitation.

## Figure families

| ID | Figure | Scientific role |
|---|---|---|
| F01 | Training reward and fixed checkpoint performance | Separates reward stationarity from genuine held-out improvement |
| F02 | Rollout-averaged PPO optimization diagnostics | Shows epoch/minibatch-mean KL, critic fit, entropy, benchmark agreement, and the mean outcome over every episode contributing to each update |
| F03 | Absolute policy performance | Compares RL, active-population, hazard-weighted, accessibility-deficit, random, static, and precommitment controls available in the result table |
| F04 | City-specific paired forest plot | Shows heterogeneity hidden by the macro estimate |
| F05 | Capacity-by-hazard heatmap | Identifies operating regimes where learning helps or harms |
| F06 | Safety--casualty--timeliness frontier | Displays all three operational objectives without changing the reward |
| F07 | Hierarchical action scalability | Relates candidates to action count, latency, and decision quality |
| F08 | Reward and architecture ablation | Tests which parsimonious components matter |
| F09 | Leave-one-city-out transfer | Tests performance in cities excluded from training |
| F10 | Operating robustness | Tests demand, hazard, capacity, and availability shifts |
| F11 | Final shelter installations on OSM | Shows where capacity was implemented |
| F12 | Evacuation progress on OSM | Shows pedestrians, hazards, priority regions, and shelters over time |
| F13 | Population-by-candidate scale stress | Shows RL--heuristic safety, casualty, and return effects together with RL runtime |

Each analytical figure is saved as 320-dpi PNG and vector SVG. The reporter
also creates `figure_statistical_summary.csv`, `full_suite_readiness.json`, and
`full_experiment_figure_manifest.json`. The manifest records the suite hash,
source-table hashes, estimation method, artifact paths, and output checksums.

## Smoke-result compatibility

The earlier sealed one-seed launch can be rendered with:

```bash
python generate_full_experiment_figures.py \
  --launch-dir runs/multicity_five_city_lr1e3_learning_audit_arm_20260906
```

That command intentionally produces a partial bundle. It is useful for visual
and schema backtesting, but its figures remain descriptive smoke evidence and
cannot be substituted for the eight-seed E0--E6 study.
