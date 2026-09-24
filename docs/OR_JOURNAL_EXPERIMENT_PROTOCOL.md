# OR Journal Experiment Protocol

## Scope and evidentiary standard

This protocol targets a methods paper for an INFORMS or IISE Transactions
audience. It separates efficacy, mechanism, and computational claims. A graph
is generated only when its corresponding design cells are complete; an absent
or failed run is never silently imputed.

The model deploys shelter capacity through one administrator-facing decision:
the policy selects an exact candidate site from a stable masked table, and the
executor installs that site without substitution. Dynamic decisions are
available at a fixed cadence and share the same budget, observations,
candidate set, hazard realization, and executor.

The frozen machine-readable design is
`config/or_journal_experiment_suite.json`. Its SHA-256 digest is copied to each
run manifest before any result is generated.

The supplementary spatial-discretization design is frozen separately in
`config/cell_partition_experiment.json`. This preserves the original
confirmatory protocol hash while registering `equal_area` and
`node_density_adaptive` as an explicit two-level preprocessing factor.

## Hypotheses and estimands

H1 (planner efficacy): the learned sequential policy reduces casualty fraction
and restricted mean time to safety relative to the active-population heuristic.
The estimand is the equal-city macro mean paired difference. The two primary
outcomes use Holm correction at familywise alpha 0.05.

H2 (value of sequential information): the sequential learned policy improves
outcomes relative to `rl_precommit`, which uses the identical frozen neural
selector and shelter budget but commits every site at time zero. This
comparison isolates feedback and timing from learned ranking ability.

H3 (static planning): the sequential learned policy improves outcomes relative
to `static_greedy`, a non-anticipative expected-demand capacity placement, and
the legacy round-robin `initial_only` rule. The demand-aware comparator is the
substantive static benchmark; round robin remains only as a diagnostic.

H4 (candidate-action scalability): exact-site scoring grows acceptably with
the candidate-site count when one shared contextual scorer is applied to every
site. The computational comparison is against a context-free candidate MLP;
it does not stand in for a decision-quality comparison.

H5 (reward alignment): training is numerically stable and safe-completion,
casualty, evacuation-time, hazard-exposure, and residual-prior components make
identifiable contributions under preregistered ablations. A smooth learning
curve alone is not sufficient.

## Factorial design

The confirmatory design crosses five fixed city sites with populations of
10,000, 20,000, 30,000, 40,000, and 50,000. Population is evaluated under two
capacity regimes:

1. Fixed resource: shelter number and realized capacity remain fixed as demand
   increases. This is an operational stress estimand.
2. Proportional capacity: capacity per initial pedestrian remains approximately
   constant. This separates computational scaling from worsening scarcity.

Eight independently trained policy seeds are crossed with ten held-out scenario
seeds per city-population-capacity cell. Scenario random streams are common across all
strategies. Policy seeds, not individual episodes, are the independent
replicates for learned-policy inference. Cities are fixed design sites and are
macro-averaged rather than resampled as if they were a random city sample.

State College is the execution gate. Three policy seeds and three held-out
scenarios at 10,000, 30,000, and 50,000 pedestrians must complete with valid
accounting, finite diagnostics, matched-interface provenance, and a recorded
runtime projection before the remaining cities are allocated.

The partition-mode study first characterizes geometry on the fixed State
College network at grid sides 4, 6, 8, 10, 12, 16, 20, 24, and 32. It then
compares held-out decision quality at grid sides 4, 6, 8, 10, 12, and 16.
Policies are trained independently for every grid/mode combination. Paired
mode contrasts use common map, candidate, pedestrian, hazard, policy-seed, and
scenario-seed factors. The same State College gate applies before extension to
the other four cities.

## Strategies

- `rl`: adaptive learned exact-candidate policy.
- `heuristic`: candidate whose host region has maximum active population.
- `rl_precommit`: the learned policy repeatedly selects from the initial state
  at time zero while candidate availability is updated between selections.
- `static_greedy`: expected-demand placement at time zero with no realized
  hazard information.
- `initial_only`: legacy round-robin static placement, retained as a diagnostic.
- `random`: dynamic uniform feasible-candidate policy, retained as a negative
  control.

Every policy receives the same final shelter-count budget. Static policies may
receive an inherent early-capacity advantage; this makes them conservative
comparators for demonstrating the value of later adaptive information.

## Outcomes

Primary operational outcomes are casualty fraction and restricted mean time to
safety (RMTS). RMTS assigns the horizon to casualties and unfinished evacuees,
avoiding the survivorship bias of a conditional mean among successful evacuees.
Safe-completion fraction, unfinished fraction, normalized risk-weighted
person-time, and the preregistered scalar objective are secondary outcomes.

Runtime reporting has three layers:

1. Training wall and CPU time per policy seed.
2. Whole-episode simulator wall time and wall time per simulated minute.
3. Online decision latency for observation scoring and action selection,
   reported as median and 95th percentile after warm-up on a pinned CPU thread.

## Extreme-case mechanism test

The requested extreme case uses a 9 by 9 grid with positive population in all
eight cells around an empty center. The production neural architecture receives
the production seven pedestrian, three hazard, eight infrastructure, eight
global, and eight candidate features, together with the spatial and route
relations. It is trained as a
one-decision contextual bandit on four variants: safe center, dangerous center,
center unavailable, and an asymmetric ring.

The benchmark heuristic must choose a populated ring cell. In the symmetric
safe-center case, the center can be the accessibility medoid and therefore can
be optimal despite having no local population. This is a mechanism test of
spatial context, masking, and danger response. It is not evidence of improved
city evacuation outcomes.

## Statistical analysis

Primary city comparisons use paired differences under common random numbers.
Uncertainty is computed with a two-way bootstrap over policy seeds and scenario
seeds within each city; cities are equally weighted and not resampled. The
extreme-case summaries cluster uncertainty by policy seed. Exact sign
randomization is preferred when the number of policy seeds permits complete
enumeration.

Scaling is evaluated with raw latency plots on log axes and a log-log model

`log(median latency) = intercept + slope * log(problem size)`.

The slope is descriptive unless repeated on additional machines. Grid accuracy
is reported separately from grid latency; the paper must not infer preserved
decision quality from computational measurements alone.

## Figure set

The full paper package reserves the following figures:

1. City-by-population effect curves under fixed and proportional capacity.
2. Paired forest plot for casualty fraction and RMTS.
3. Sequential RL versus learned precommit and demand-aware static placement.
4. PPO reward decomposition, entropy, KL, gradient norm, and seed variability.
5. Four-timepoint city maps showing red pedestrian dots, danger heatmap,
   candidate sites, active shelters, and newly selected shelters.
6. Grid dimensionality versus RL and heuristic online latency.
7. Candidate-site count versus contextual exact-site and context-free MLP latency/storage.
8. Spatial discretization: (a) equal-area versus node-density-adaptive geometry
   and road-node balance; (b) grid resolution and partition mode versus paired
   decision quality, including a noninferiority margin.
9. Extreme ring design and center-selection/regret results.
10. Training, simulation, state-construction, policy, and shared-execution time
    decomposition.
11. Robustness heatmap by hazard, demand, capacity error, and candidate loss.
12. Design-cell completion matrix and reasons for missing runs.

For maps, each red dot denotes one represented pedestrian only when agents are
ungrouped. If population aggregation is active, the legend must report the
represented group size rather than implying one dot per person.

## Reproducible commands

Use the pinned project environment:

```bash
/private/tmp/rlevac_arm_env/bin/python or_journal_experiments.py --mode dry-run
/private/tmp/rlevac_arm_env/bin/python or_journal_experiments.py --mode tractable
```

The `--quick` flag is a pipeline pilot and its manifest is labeled
`pilot_complete`. It must not be cited as confirmatory evidence. City-level
execution continues through `backtest.py`, `multicity_backtest.py`, and
`population_candidate_backtest.py` after the State College gate.
