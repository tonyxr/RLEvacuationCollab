# Benchmark Model Protocol

## Status and scope

This document registers the secondary benchmark extension added on 2026-09-12.
It does not change the frozen primary estimand: the confirmatory comparison
remains sequential RL versus the dynamic maximum-active-population heuristic.
The additional policies provide negative, risk-priority, accessibility, and
static controls on the same held-out scenarios.

All dynamic strategies receive the same `RegionalObservation`, cell action
mask, decision cadence, shelter budget, and hazard realization. Every formula
below is already written as a region/cell-level rule, `c(j)` mapping a
candidate index to its host region; as of `docs/CELL_PRIORITY_ACTION_SPACE_20260920.md`
(2026-09-20), the action space itself is cell-indexed, so `c(j) = j` and one
row exists per cell rather than per raw candidate building. No formula below
changed. An action names the cell to prioritize; the specific building
installed is resolved by the shared deterministic rule
(`ShelterDatabase._candidate_index`: maximum remaining capacity, OSM
identifier as the tie break) common to every policy here and to RL. Exact
score ties among cells use stable cell-table order. No benchmark contains a
fitted coefficient.

## Registered policies

Let \(J_t\) be the feasible action slots at decision epoch \(t\) (one per
regional cell), let \(c(j)\) be slot \(j\)'s host region (\(c(j)=j\) under the
cell-priority action space), and let \(N_{it}\),
\(D_{it}\in[0,1]\), and \(Q_{it}\) be active demand, danger, and remaining
installed-shelter capacity in region \(i\).

### Maximum active population (`heuristic`)

\[
a_t=\arg\max_{j\in J_t}N_{c(j)t}.
\]

This is a demand-priority rule related to weighted-demand location and maximal
covering models. It is the primary prespecified comparator because the RL actor
contains the same rule as a fixed residual-policy prior.

### Hazard-exposure-weighted demand (`hazard_weighted`)

\[
a_t=\arg\max_{j\in J_t}N_{c(j)t}(1+D_{c(j)t}).
\]

The score is the current cell contribution to the simulator's
hazard-weighted person-time term. The fixed (1+D) scaling avoids a tunable
benchmark weight and gives otherwise identical exposed demand between one and
two times the priority of unexposed demand. This is a risk-priority benchmark;
it is not asserted to solve joint shelter-location and evacuation routing.

### Maximum accessibility deficit (`accessibility_deficit`)

Define local capacity shortfall

\[
U_{it}=\max(N_{it}-Q_{it},0).
\]

Let (S_t^+=\{j:Q_{jt}>0\}), let (d(i,S_t^+)) be projected-metre Euclidean
distance between cell centroids and the nearest usable-shelter cell, and let
(D_{\max}) be the study-area cell-centroid diagonal. The registered score is

\[
a_t=\arg\max_{j\in J_t}
U_{c(j)t}\left(1+\frac{d(c(j),S_t^+)}{D_{\max}}\right).
\]

If (S_t^+) is empty, normalized distance is one for every cell. The policy
therefore combines unmet local demand and spatial isolation while remaining
linear-time in cells for the small set of currently usable shelter cells. It
is a transparent p-median-style diagnostic, not an optimal capacitated
location-allocation solution. Network-time and capacity-constrained assignment
remain evaluation outcomes and potential stronger optimization benchmarks.

### Uniform random feasible candidate (`random`)

\[
a_t\sim\operatorname{Uniform}(J_t).
\]

The random policy has an isolated policy RNG and is a negative control.

### Static initial-only (`initial_only`)

The final shelter-count budget is deployed at time zero in deterministic
row-major candidate order. Because it receives all shelter capacity early, it
is an anticipative timing diagnostic rather than an equal online competitor.
The demand-aware `static_greedy` strategy remains the substantive static OR
comparator, while `rl_precommit` isolates sequential feedback from learned
ranking.

## Literature basis

The literature supports the constructs used in these benchmarks; it does not
establish that the exact registered score is optimal.

1. Hakimi, S. L. (1964). Optimum locations of switching centers and the
   absolute centers and medians of a graph. *Operations Research, 12*(3),
   450–459. https://doi.org/10.1287/opre.12.3.450
   Establishes weighted network median/center foundations for distance-based
   accessibility objectives.
2. Church, R., & ReVelle, C. (1974). The maximal covering location problem.
   *Papers of the Regional Science Association, 32*, 101–118.
   https://doi.org/10.1007/BF01942293
   Supports population-weighted coverage as a facility-location objective.
3. Kılcı, F., Kara, B. Y., & Bozkaya, B. (2015). Locating temporary shelter
   areas after an earthquake: A case for Turkey. *European Journal of
   Operational Research, 243*(1), 323–332.
   https://doi.org/10.1016/j.ejor.2014.11.035
   Jointly considers shelter selection, assigned population, capacity and
   utilization.
4. Coutinho-Rodrigues, J., Tralhão, L., & Alçada-Almeida, L. (2012). Solving a
   location-routing problem with a multiobjective approach: The design of urban
   evacuation plans. *Journal of Transport Geography, 22*, 206–218.
   https://doi.org/10.1016/j.jtrangeo.2012.01.006
   Incorporates risks of routes and shelter sites together with path length in
   urban evacuation location-routing.
5. Zhao, X., Xu, W., Ma, Y., Qin, L., Zhang, J., & Wang, Y. (2017).
   Relationships between evacuation population size, earthquake emergency
   shelter capacity, and evacuation time. *International Journal of Disaster
   Risk Science, 8*, 457–470.
   https://doi.org/10.1007/s13753-017-0157-2
   Uses population-weighted evacuation time, shortest paths, shelter capacity,
   and maximum evacuation distance in shelter location-allocation.
6. Bayram, V., & Yaman, H. (2018). Shelter location and evacuation route
   assignment under uncertainty: A Benders decomposition approach.
   *Transportation Science, 52*(2), 416–436.
   https://doi.org/10.1287/trsc.2017.0762
   Provides a strong stochastic-optimization precedent for jointly handling
   uncertain demand, disrupted networks/shelters, assignment, fairness, and
   expected evacuation time.
7. Drezner, Z. (1995). Dynamic facility location: The progressive p-median
   problem. *Location Science, 3*(1), 1–7.
   https://doi.org/10.1016/0966-8349(95)00003-Z
   Provides a direct precedent for changing demand and facilities installed
   sequentially at specified times, with users assigned to their closest open
   facility.

## Evaluation and interpretation

The primary inference remains paired RL minus `heuristic` under common random
numbers. The two new heuristic comparisons are secondary and must be reported
with effect estimates and confidence intervals without changing the primary
multiplicity family post hoc. Absolute performance plots may include every
registered strategy.

The single-city evaluator additionally writes `benchmark_comparison.csv`,
`benchmark_comparison.md`, and `all_benchmark_comparisons.png`. Its
`episode_return` analysis resolves to the action-count-invariant
`objective_episode_return`; normalized risk time is reconstructed from the
full-horizon objective accumulator. This prevents static policies from
receiving artificial zero risk-time or return values merely because they make
no online actions.

The accessibility heuristic uses projected centroid distance because this
quantity is available at negligible online cost through the common regional
state. It must be described as an accessibility proxy. Claims about actual
network accessibility must use realized route time/distance outcomes or a
separate network-assignment optimization benchmark.

Recommended complete evaluation strategy list:

```text
rl,heuristic,hazard_weighted,accessibility_deficit,random,static_greedy,rl_precommit,initial_only
```

The completed engineering-backtest record is in
`docs/BENCHMARK_BACKTEST_RESULTS_20260913.md`.
