# Cell-partition experiment protocol

## Material Passport

- Material type: Experiment Plan + partial Experiment Result
- Material ID: `cell_partition_comparison_v1`
- Status: `PARTIALLY_EXECUTED`
- Registered design: `config/cell_partition_experiment.json`
- Completed evidence: deterministic State College map characterization
- Pending evidence: separately trained evacuation-policy accuracy comparison

## Research question and hypotheses

The study asks whether placing more regional cells in road-dense urban areas
improves the representation/decision tradeoff relative to equal-area cells
without changing the regional context-grid size; the exact action count is the
number of shelter candidates and is recorded separately.

- H6a (representation): at fixed n, `node_density_adaptive` lowers the
  across-cell coefficient of variation and Gini coefficient of OSM road-node
  counts and reduces empty-cell prevalence.
- H6b (decision quality): at fixed n, adaptive-grid RL is noninferior on
  casualty fraction and restricted mean time to safety (RMTS), with superiority
  analyzed only if noninferiority is established.
- H6c (resolution): increasing n changes held-out casualty fraction and RMTS;
  this accuracy curve is estimated separately from the latency curve.
- H6d (runtime): partition construction is offline. Online policy inference is
  expected to depend on n-squared, not physical cell area; whole-episode state
  construction may nevertheless differ because node/candidate assignment is
  redistributed.

## Partition interventions

Let x-min/x-max and y-min/y-max denote the occupied road-network extent in the
same local metre coordinates used by the simulator.

The axis counts X and Y are registered experiment inputs. A condition contains
M = X times Y region nodes. X and Y are fixed within a trajectory and policy
checkpoint but vary across resolution conditions. Every node uses the same
feature schema and shared GNN parameters, so model parameter count does not
grow with M; the observation row count, regional edges, and candidate-to-region
mapping are rebuilt for the selected partition.

`equal_area` uses linearly spaced boundaries on each axis. Every cell therefore
has area

\[
A = \frac{x_{max}-x_{min}}{X}\frac{y_{max}-y_{min}}{Y}.
\]

`node_density_adaptive` uses empirical road-node quantiles

\[
b^x_i = Q_x(i/X),\qquad b^y_j = Q_y(j/Y).
\]

This produces smaller rectangles where node-coordinate mass is concentrated.
A minimum width of 0.0001 of the corresponding axis span prevents degenerate
boundaries. The construction is separable and axis-aligned, so it preserves a
rectangular X by Y tensor but does not promise equal two-dimensional node mass.

Both modes use the same row-major regional context identifiers, four-neighbour
GNN edges, exact-candidate action table and mask, shelter budget, executor, and
reward.
The road graph is fixed before scenarios are sampled. Consequently, adaptive
boundaries cannot use realized pedestrian movement, danger, casualties, or
policy outcomes.

## Execution phases

### Phase A — deterministic map characterization (complete)

On the registered 2 km State College walking network, construct both modes at
n = 4, 6, 8, 10, 12, 16, 20, 24, and 32. Record every boundary, cell area,
road-node count, node density, empty-cell indicator, CV, Gini coefficient, and
edge digest. Node conservation and exactly n-squared cells are hard checks.

### Phase B — State College computational gate (pending)

At n = 8, train three independent policies for each mode for 120 episodes per
seed. Evaluate the learned policy and active-population heuristic on three
common held-out scenario seeds at populations 10,000, 30,000, and 50,000. This
requires 720 training episodes and 72 evaluation episodes. Expansion is
allowed only if:

1. all population identities and hazard-trajectory matches pass;
2. every learned checkpoint is converged and finite;
3. both policies see identical partition edges within each matched comparison;
4. no checkpoint is reused across modes;
5. training, episode, state-construction, inference, and site-execution times
   are separately recorded.

### Phase C — five-city accuracy study (pending)

Cross five cities, five population levels, grid sides 4, 6, 8, 10, 12, and 16,
and both partition modes. Use eight policy seeds and ten held-out scenarios.
The frozen matrix contains 300 city-population-grid-mode cells, 24,000 learned
evaluation episodes, and 3,000 heuristic episodes. Pooled policies receive
complete city blocks per PPO update. Every grid/mode condition is trained
separately, totaling 57,600 city-training episodes under the registered plan.

Separate training is an experimental-control decision, not an architectural
limitation. The shared node encoder and candidate scorer can be instantiated at
any registered X and Y, but a checkpoint is not transferred across resolutions
in the confirmatory analysis because spatial aggregation and neighborhood
semantics change with the intervention. Mixed-resolution training, if later
studied, is a distinct transfer experiment and is not pooled with these results.

The adaptive mode is not compared with an equal-area policy checkpoint applied
post hoc: the partition changes state aggregation and action semantics, so such
a comparison would be invalid. RL-versus-heuristic contrasts are computed
within each partition condition; adaptive-versus-equal contrasts are paired by
all common scenario factors.

## Outcomes and analysis

Primary outcomes are casualty fraction and RMTS. Secondary outcomes are safe
completion, unfinished fraction, normalized risk-weighted person-time, online
decision latency, whole-episode time, and preprocessing/state-construction
time. Road-node balance is a mechanism diagnostic, not a safety outcome.

Within city, estimate paired adaptive-minus-equal differences using a two-way
bootstrap over policy and scenario seeds. Macro-average city estimates with
equal weights; do not treat episodes as independent replicates. Apply Holm
correction across the two primary outcomes. Report effect estimates and 95%
confidence intervals. The numerical noninferiority margins for casualty and
RMTS must be set before Phase B outcomes are opened; absent those margins, only
two-sided difference intervals may be reported.

Grid accuracy and latency are plotted separately. A log-log slope may describe
latency scaling but cannot establish an accuracy law. Failures and timeouts
remain missing with reasons and are never replaced by favorable seeds.

## Required figures

1. F8a (complete): equal-area and adaptive State College maps at 8 by 8, plus
   road-node CV and empty-cell fraction versus n-squared.
2. F8b (pending): casualty-fraction difference and RMTS difference versus
   n-squared, with 95% intervals, mode-specific curves, and preregistered
   noninferiority bands.
3. F8c (pending): online decision latency and state-construction time versus
   n-squared, with raw seed points and median/p95 summaries.
4. F8d (pending): city-stratified forest plot of paired adaptive-minus-equal
   casualty and RMTS effects, with the equal-city macro estimate.

A city-by-mode node-balance and empty-cell heatmap is retained as a
supplementary preprocessing diagnostic.

F8a is descriptive of road-network discretization. F8b is the figure that can
support or reject a decision-quality claim.

## Reproducibility

Characterization command:

```bash
/private/tmp/rlevac_arm_env/bin/python cell_partition_experiment.py
```

The simulator modes are selected without source edits:

```bash
python multicity_backtest.py --override cellPartitionMode=equal_area
python multicity_backtest.py --override cellPartitionMode=node_density_adaptive
```

Each simulator run records the mode, minimum-width constraint, full edge
arrays, edge SHA-256 digest, graph hash, code state, and seeds. Model version 17
checkpoints reject a mismatched partition contract in the confirmatory study,
even though the shared architecture can process a different runtime node count
in a separately registered transfer experiment.

## Claim boundary

The adaptive input is OSM road-node density. It must not be described as census
population density, pedestrian density, or a demographic equity measure. The
completed Phase A result supports only the statement that the adaptive method
allocates finer geometric resolution to road-node-dense coordinate ranges and
improves node balance on the registered State College network.
