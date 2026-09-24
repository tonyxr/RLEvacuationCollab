# Extreme-Case Policy and Computational Sensitivity Protocol

## Material Passport

- Material type: Experiment Plan
- Material ID: `extreme_policy_sensitivity_v2_confirmatory`
- Verification status: frozen for execution
- Machine-readable design: `config/extreme_sensitivity_experiment.json`
- Claim boundary: controlled one-decision mechanism and computational timing;
  not city-level evacuation efficacy

## Research questions

This experiment asks whether the learned regional policy uses spatial context
that the maximum-local-population heuristic and a non-anticipative static plan
cannot use, and how training, state construction, and online decision time scale
with population, city-network complexity, and grid dimensionality.

The extreme construction places positive population in every cell immediately
surrounding an empty center. Four variants are fixed before evaluation:

1. `safe_center`: the empty center is safe and available;
2. `dangerous_center`: the empty center is hazardous;
3. `center_unavailable`: the empty center is removed from the feasible mask;
4. `asymmetric_ring`: one surrounding cell contains most demand.

The primary response is decision utility, defined as accessibility-weighted
served demand minus 0.45 times selected-cell danger. Oracle regret, center
selection, and oracle selection are diagnostics. This utility is not a casualty
or evacuation-time outcome.

## Policies and information sets

- `rl` is the production graph actor architecture trained as a contextual
  one-decision actor-critic. A separate actor is trained for each grid size; it
  is shared across populations, cities, and extreme variants.
- `population_heuristic` recomputes the feasible cell having the largest
  realized active population. It cannot select the empty center.
- `static_expected_demand` is computed from uniform expected ring demand and
  expected danger. It knows candidate feasibility but not realized demand or
  hazard. Its online operation is a stored-plan lookup; offline planning time
  is reported separately.
- `oracle` selects the feasible action having maximum realized decision utility
  and is used only as a regret reference.

All policies face the same candidate mask and scenario realization. Scenarios
use common random numbers across policies.

## Frozen sensitivity design

The base condition is 30,000 pedestrians, Spokane, and a 9 by 9 grid. Three
one-factor-at-a-time axes are evaluated:

- population: 10,000, 20,000, 30,000, 40,000, and 50,000;
- fixed city sites in increasing registered scale: State College, Reading,
  Spokane, Seattle, and Chicago;
- grid side: 5, 7, 9, 11, and 13, corresponding to 25--169 actions.

The shared base condition is deduplicated, producing 13 unique condition cells
and 15 reported axis levels. Each condition is crossed with four extreme
variants, 100 held-out scenario seeds, and 16 independently initialized and
trained policy seeds. The raw evaluation contains 332,800 policy observations.

Training uses 500 updates per policy seed and grid, batches of 64, AdamW at
0.0008, weight decay 0.0001, gradient clipping at 0.5, value-loss weight 0.5,
and entropy coefficient 0.01. This produces 40,000 recorded updates. Torch is
pinned to one CPU thread for timing.

## Confirmatory inference

For each axis level, extreme variant, and comparator, the estimand is the paired
mean RL-minus-comparator decision utility. Scenario-level differences are first
averaged within policy seed. The 16 policy-seed means define the standard error,
normal 95% interval, and an exactly enumerated two-sided sign-randomization
test. Holm adjustment controls familywise error across all 120 registered
comparisons. The minimum attainable two-sided p-value with 16 seeds is
0.0000305, so the registered family can resolve an adjusted 0.05 threshold.

Center- and oracle-selection rates and oracle regret are reported for every
strategy. Results are not pooled as though the repeated scenario seeds were
independent trained policies.

## Timing definitions

Timing is decomposed because the components have different operational scopes:

1. RL training wall time includes controlled case/tensor construction and 500
   actor/value updates for one policy seed. The heuristic and static policies
   require zero parameter training. One RL actor is reused across all
   populations and cities at a fixed grid, so those axes require no retraining.
2. State aggregation measures road-node-sampled pedestrian assignment and
   cell `bincount`. It is crossed with every city, population, grid, and both
   `equal_area` and `node_density_adaptive` partition modes. It excludes
   routing, congestion, hazard propagation, and movement.
3. Online policy evaluation includes graph representation construction and a
   deterministic forward pass for RL, a realized argmax computation for the
   heuristic, and stored-plan lookup for static. Case generation is also
   retained as a separate total-pipeline field.
4. Cached GraphML load and partition construction are measured on the five
   registered OSM graphs. Static offline planning is timed separately.
5. Full-simulator episode timing is an explicitly labeled observational table
   reconstructed from 21 completed 50,000-person episodes in an incomplete
   legacy model-v8 launch. It cannot support a current-version efficacy or
   convergence claim.

Median and 95th-percentile times are reported. Log-log regressions estimate a
descriptive scaling exponent for training versus cell count, RL evaluation
versus cell count, state aggregation versus population and cell count under
each partition, partition construction versus road-node count, and cached graph
load versus road-node count. With five factor levels, the exponents are
descriptive machine-specific summaries, not asymptotic complexity proofs.

## Pilot amendment

The first 20-update, two-seed run was a pipeline test only and was not used as
confirmatory evidence. It correctly exposed that 20 updates did not learn a
policy distinct from the heuristic. A separate 500-update 9 by 9 calibration
recovered the intended safe-center mechanism. Before inspecting any
confirmatory held-out results, the frozen design was amended from 160 to 500
updates and from 8 to 16 policy seeds. Paths and the amendment rationale are
stored in the machine-readable configuration.

## Outputs and acceptance checks

The run is accepted only if it contains exactly 40,000 training-update rows,
80 trained policies, 332,800 evaluation rows, 120 paired comparisons, 7,500
state-aggregation timings, 50 partition timings, 30 joined scenario-timing
rows, and nine scaling-model rows; all required timing values must be finite
and nonnegative. The manifest records the configuration hash, environment, Git
state, expected/observed row counts, artifact paths, and SHA-256 hashes.

Publication figures are:

- F13: six-panel RL-minus-heuristic/static extreme-case sensitivity heatmap;
- F14: RL training, update decomposition, population-state construction, and
  clearly labeled incomplete full-simulator timing;
- F15: policy evaluation latency over all three sensitivity axes;
- F16: population, city-graph, and grid preprocessing sensitivity under both
  registered cell-partition options.

## Reproduction

```bash
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/rlevac_matplotlib_cache \
python extreme_sensitivity_experiments.py \
  --output-dir runs/extreme_sensitivity_full_20260909
```

Use `--quick` only for a noninferential pipeline validation.
