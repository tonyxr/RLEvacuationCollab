# OR Journal Experiment Implementation and Results — 2026-09-09

> Historical-result warning: these results use the former cell-priority,
> two-stage action model. They do not validate model version 13's exact-
> candidate actions or revised four-component reward and must be rerun before
> being cited as evidence for the current model.

## Executive status

The revised experiment has been implemented as two explicitly separated
evidence tiers. The controlled mechanism and computational scaling tier is
complete and audited. The five-city evacuation tier is designed and enabled by
the simulator changes, but is not complete; its 4,800 training episodes and up
to 10,000 all-strategy evaluation episodes require a dedicated compute run.

No controlled or legacy result is substituted for the missing city-level
evidence. The figure-readiness audit marks those panels as pending.

## Implemented model controls

1. `static_greedy` performs non-anticipative expected-demand placement at time
   zero. It uses the same maximum-capacity lower-level site rule as the dynamic
   policies and is matched on final shelter count.
2. `rl_precommit` loads the same frozen learned policy as sequential RL and
   repeatedly selects its full shelter set at time zero. Candidate availability
   and installed capacity update after each choice, while pedestrian locations
   and the hazard state remain fixed. This isolates the value of sequential
   feedback and deployment timing from neural ranking ability.
3. The single- and multi-city evaluators now load every trained policy seed for
   both `rl` and `rl_precommit`.
4. Every simulator episode records wall time, CPU time, and wall time per
   simulated minute.
5. The visualization layer recognizes both new static/precommit strategies and
   records their sites as time-zero deployments.
6. Regional cell construction now exposes `equal_area` and
   `node_density_adaptive` modes while retaining exactly n-squared actions.
   Run metadata records the full edge arrays and their digest, and model
   version 9 checkpoints bind inference to the selected partition contract.

## Completed State College partition characterization

The registered deterministic characterization used the cached 2 km State
College walking network (11,628 OSM road nodes and 33,204 directed edges) at
grid sides 4 through 32. At the production 8 by 8 resolution, the adaptive
quantile partition reduced the coefficient of variation of road nodes per cell
from 1.039 to 0.461, a 55.6% reduction. No adaptive cell was empty, compared
with no empty equal-area cells at this resolution; at 32 by 32 the respective
empty-cell fractions were 7.9% and 22.7%. The adaptive 8 by 8 cell-area
max/min ratio was 21.09, verifying materially finer resolution in dense road
coordinate ranges.

These are map-preprocessing diagnostics, not evacuation outcomes. They verify
that the requested adaptive mechanism operates as intended. The registered
paired accuracy comparison requires independently trained policies for each
partition mode and grid resolution and remains pending compute.

## Completed controlled experiment

The full tractable run used eight independent policy seeds, 500 contextual
bandit updates per seed, and 200 held-out cases per variant and seed. The raw
evaluation contains 19,200 strategy-case rows; the training table contains
4,000 rows. All diagnostics are finite.

The extreme case is a 9 by 9 grid in which all eight cells surrounding the
center contain pedestrians and the center contains none. In the safe-center
variant, the center is the accessibility medoid and the heuristic cannot choose
it because it ranks feasible cells by local active population.

| Variant | RL utility improvement over heuristic | 95% cluster CI | Exact sign p | Interpretation |
|---|---:|---:|---:|---|
| Safe center | 0.09027 | [0.08858, 0.09197] | 0.0078125 | RL selected the empty center in 100% of held-out cases; heuristic selected it in 0%. |
| Dangerous center | 0.01229 | [0.01135, 0.01322] | 0.0078125 | RL avoided the center and attained the oracle action in 81.4% of cases versus 38.6% for the heuristic. |
| Center unavailable | 0.01532 | [0.01426, 0.01637] | 0.0078125 | The feasibility mask was obeyed; RL attained the oracle action in 83.3% of cases versus 36.7% for the heuristic. |
| Asymmetric ring | 0.00000 | [0.00000, 0.00000] | 1.0 | Both policies selected the dominant ring cell; RL did not manufacture a difference when the heuristic was already optimal. |

The first 50 training updates had mean deterministic utility 0.28835. The last
50 had mean 0.34196. These curves are training diagnostics, not independent
hypothesis tests.

This controlled result demonstrates that the architecture can learn a spatially
contextual decision different from the local-population heuristic. It does not
establish reduced casualty or evacuation time in a city simulation.

## Completed computational scaling experiment

All measurements used one CPU thread, 50 warm-up calls, and 500 measured calls
per design point. The reported latency covers policy scoring and feasible-action
selection; it excludes state construction and the shared site executor.

The regional actor contains 22,978 parameters. Its median latency increased
from 0.255 ms at 16 cells to 2.246 ms at 1,024 cells. A log-log fit gave slope
0.526 (R-squared 0.926) with cell count. The active-population heuristic was
much faster: 0.00238 ms at 64 cells and 0.00417 ms at 1,024 cells. At the
production 8 by 8 grid, RL median/p95 latency was 0.339/0.581 ms versus
0.00238/0.00246 ms for the heuristic. Both are well below the two-minute
deployment cadence; total simulator latency remains a separate outcome.

At a fixed 8 by 8 regional representation, hierarchical RL latency was nearly
constant as raw candidate count increased: its fitted candidate-count slope was
0.0013. A vectorized flat candidate actor had slope 0.8687. The flat actor was
faster for small candidate sets, crossed the hierarchical actor between 400 and
800 sites, and at 6,400 candidates required 4.440 ms versus 0.343 ms for the
hierarchical actor (12.95 times slower). Its input representation was 33.19
times larger. This supports the computational action-space claim only. The
grid-resolution decision-quality noninferiority experiment remains pending.

## Completed confirmatory extreme sensitivity and timing experiment

The separately frozen `extreme_policy_sensitivity_v2_confirmatory` run completed
16 independent policy seeds at each of five grid resolutions, 500 updates per
seed, and 100 held-out scenarios for every condition/variant. The audit matched
40,000 training updates, 80 checkpoints, 332,800 policy evaluations, 120 paired
contrasts, 7,500 state-aggregation timings, and 50 partition-construction
timings. Artifact hashes were independently rechecked with no mismatch, and all
checkpoint tensors were finite.

At the 30,000-person Spokane 9 by 9 base condition, mean RL utility improvement
was 0.0803 over the active-population heuristic and 0.0935 over static in the
safe-center case. The corresponding policy-seed 95% intervals were
[0.0701, 0.0906] and [0.0833, 0.1037]. In the dangerous-center case, RL improved
utility by 0.0091 and 0.0209, respectively, while selecting the dangerous center
in zero held-out cases. When the center was unavailable, improvements were
0.0174 and 0.0203. Under asymmetric demand, RL exactly matched the already
optimal heuristic and improved 0.3164 over static.

Across all sensitivity levels, RL beat the heuristic in all 45 registered
safe-center, dangerous-center, and unavailable-center contrasts and tied it in
all 15 asymmetric-ring contrasts. RL beat static in 56 of 60 contrasts. Static
had small, nonsignificant advantages in the State College and Reading
safe-center cases, and tied RL in the 11 by 11 and 13 by 13 safe-center cases.
Overall, 101 of 120 contrasts were significant after one Holm correction over
the registered family.

Median controlled 500-update training time per policy seed rose from 13.46 s at
25 cells to 44.92 s at 169 cells. The descriptive log-log exponent was 0.622
(95% CI 0.441--0.802), although a linear-in-cell-count fit was better over the
five measured levels. At the base condition, median online evaluation was
1.006 ms for RL, 0.007125 ms for the recomputed population heuristic, and
0.000292 ms for static lookup; the RL median was 1.402 ms at 169 cells, with a
1.976 ms 95th percentile. RL evaluation was essentially flat over population
and city at a fixed 9 by 9 representation.

Population affected state construction rather than neural scoring. On the
primary adaptive Spokane 9 by 9 partition, state aggregation increased from
0.188 ms at 10,000 pedestrians to 0.775 ms at 50,000, with descriptive exponent
0.871. At 30,000 pedestrians, state aggregation was approximately flat over
25--169 cells after node-to-cell membership was constructed. Equal-area and
node-density-adaptive preprocessing timings are both reported.

The complete result is in
`runs/extreme_sensitivity_full_20260909/EXTREME_SENSITIVITY_RESULTS.md`. Its
decision utility remains a controlled mechanism outcome, not casualty or
evacuation-time evidence. The 21 reconstructed five-city simulator episode
times are retained only as incomplete legacy-v8 observational evidence.

## Figure inventory

Completed in both 320-dpi PNG and SVG:

- F4: controlled training diagnostics.
- F6/F7: grid latency, candidate latency, and input-storage scaling.
- F8a: State College equal-area/adaptive geometry, road-node imbalance, and
  empty-cell prevalence.
- F9a: extreme-case population and danger designs with oracle and heuristic
  selections.
- F9b: extreme-case center-selection and regret comparison.
- F12: publication-figure readiness and missing-evidence audit.
- F13: RL-minus-heuristic/static sensitivity heatmap over population, city
  scale, grid resolution, and four extreme variants.
- F14: controlled RL training, update decomposition, state aggregation, and
  clearly labeled incomplete legacy simulator timing.
- F15: online policy-evaluation time for RL, heuristic, static, and oracle.
- F16: preprocessing timing under equal-area and node-density-adaptive cells.

Pending until the confirmatory city trials complete:

- F1 city-population outcome curves.
- F2 paired casualty/RMTS forest plot.
- F3 sequential RL versus `rl_precommit` and `static_greedy`.
- F5 dynamic State College map sequence.
- F8b grid-resolution and partition-mode accuracy/noninferiority curve.
- F10 full runtime decomposition.
- F11 robustness heatmap.

## Compute gate and next execution

The frozen city design includes 4,800 training city-episodes, 4,500 primary
RL/heuristic evaluation episodes, and 10,000 episodes if every registered
strategy is run across both capacity regimes. A prior State College computational pilot required 95.8–102.9
seconds for each 120-minute, 2,500-pedestrian episode. Even a linear half-horizon
projection places the complete design above 205 core-hours before accounting
for larger maps, extra training cost, or contention. The existing 60-minute,
50,000-pedestrian pooled launch is incomplete and therefore cannot supply
confirmatory checkpoints.

The required order is:

1. Complete the State College gate at 10,000, 30,000, and 50,000 pedestrians.
2. Verify population accounting, feasible first actions, finite PPO diagnostics,
   matched random streams, and observed runtime.
3. Complete all eight pooled policy seeds at the frozen 60-minute horizon and
   two-minute decision cadence.
4. Open held-out evaluation only after the convergence gate passes.
5. Run the five-city/five-population comparisons under both fixed-resource and
   proportional-capacity regimes.
6. Train grid-specific policies before evaluating grid-resolution accuracy.
7. Generate the pending figures only from complete design cells.

## Reproducibility

The completed run is rooted at `runs/or_journal_full_20260909`. Its manifest
records design hash
`f5400ba2633f6fc6b019630f86d5c551903a964a58d287dac47926f31195b052`,
environment information, every artifact hash, and an audit matching all
expected row/checkpoint counts. The extreme-sensitivity design hash is
`704ce5efb4c56b39ce09984c09de432113dd4016aa0e8d7c0bc949109db0175e`.
Forty tests in the modules compatible with the default runtime passed,
including five new extreme-sensitivity tests. Two legacy modules could not be
collected in that runtime because its pandas/geospatial binaries require a
newer NumPy; this environment limitation is separate from the experiment audit.
