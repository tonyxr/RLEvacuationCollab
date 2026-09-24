# Cell-priority action space: design, implementation, and verification

## Material Passport

- Origin: Claude (Cowork) implementation change, made directly in this
  repository at the user's explicit instruction
- Origin Mode: `code` — `DecisionInterface.py` and `RLBridge.py` were edited,
  `tests/test_rl_framework.py` was updated and extended, and this document,
  `TODO.md`, and `LOG.md` were updated to match
- Origin Date: 2026-09-20
- Verification Status: `PARTIALLY VERIFIED` — see "Verification performed"
  below. The repository's pinned conda environment (`rlevacuation`) was not
  reachable from the tool used to make this change, so the full
  `python -m unittest discover -s tests` run has not yet been executed and
  must be run before this change is treated as merge-ready.
- Version Label: `cell_priority_action_space_v1` (`RLBridge.MODEL_VERSION = 18`)

## 1. Motivation

This document supersedes the exact-candidate action space described in the
original `docs/MDP_AND_OPTIMIZATION_DESIGN.md` ("Action" section) with a
cell-priority action space. The motivation is decision complexity, not
credit assignment directly: selecting among individual candidate buildings
(often several visually and operationally near-identical sites inside the
same regional cell) gives the policy a much larger and less informative
branching factor than selecting which cell to prioritize. This is a
different lever from `docs/NATURAL_MOMENTUM_COUNTERFACTUAL_CONTROL_20260920.md`
(NMCC), which addresses the *statistical* credit-assignment problem (global
exogenous variance swamping the effect of an action). The two are
complementary: NMCC will still need to identify the causal effect of
choosing cell A over cell B, and a smaller, more meaningful action space
makes that identification problem strictly easier, not a substitute for it.

Concretely: previously, `RegionalObservationBuilder` froze one action slot
per raw candidate building at episode start (`_initial_candidate_records`).
A cell with two similarly-capacitated candidates contributed two nearly
redundant actions to the policy's decision. The new design collapses every
cell to exactly one action slot. The slot's specific building is resolved,
every decision epoch, by the shared deterministic rule already implemented
in `ShelterDatabase._candidate_index` (maximum remaining capacity, OSM
identifier as the tie break) and already used by `initShelter` and
`predeployStaticDemandGreedy`. That rule is now used for the RL policy and
every heuristic benchmark alike, through `ShelterDatabase.newShelter`.

## 2. What changed

### `DecisionInterface.py`

- `RegionalObservationBuilder._initial_candidate_records` (froze one row per
  raw candidate at construction) was replaced by
  `_initial_position_bounds` (computes only the position-normalization
  bounds and validates OSM-identifier uniqueness once, at construction).
- A new method, `_cell_action_records`, resolves exactly one action record
  per regional cell every `build()` call, using the preview node
  (`ShelterDatabase.previewShelterCandidate`) `_regional_capacity` already
  computed. A cell with no remaining candidate gets an infeasible
  placeholder slot (`"empty-cell-{index}"`, zero capacity) so the action
  table stays a fixed size for the whole episode.
- `number_of_actions` now returns `number_of_cells` directly.
- `_candidate_action_mask` and `_candidate_operational_features` are now
  parameterized by the per-cell records (and nodes) resolved for the current
  decision epoch, rather than reading a frozen instance attribute. Node-
  dependent route/safety features are skipped (left at their neutral
  default) for an empty cell's placeholder slot.
- `RegionalShelterExecutor.execute` now calls
  `ShelterDatabase.newShelter({"cell": cell}, cellTracker)` instead of
  `newShelterCandidate(candidate_osm_id, cell, cellTracker)`, and asserts,
  fail-closed, that the OSM identifier actually installed matches the one
  the observation predicted for that cell (`RuntimeError` naming both
  identifiers on divergence). This is a new invariant that did not exist
  under the exact-candidate design, where "prediction" and "installation"
  were the same lookup by construction.
- The module docstring, the `RegionalObservation` docstring, and the
  `RegionalShelterExecutor` docstring were updated to describe the new
  contract.

### `ShelterDatabase.py`

No changes. `newShelter`, `_candidate_index`, and `previewShelterCandidate`
already implemented exactly the shared deterministic rule this design
needs; they were already used by `initShelter` and
`predeployStaticDemandGreedy`, just not yet by the live dynamic executor.

### `GNN.py`

No changes. `EvacPolicy` already falls back to an identity
`candidate_cell_index` mapping and gathers per-candidate features generically
by whatever mapping `RegionalObservation` supplies; it does not encode any
assumption about how many raw candidates share a cell.

### Heuristic benchmark policies (`ActivePopulationHeuristic`,
`HazardWeightedDemandHeuristic`, `AccessibilityDeficitHeuristic`,
`UniformRegionalPolicy`)

No changes. Each already selects through
`observation.candidate_cell_indices[feasible]`, which is now an identity
mapping (slot `a` is cell `a`) rather than a many-to-one mapping. Their
formulas in `docs/BENCHMARK_MODEL_PROTOCOL.md` are already written in terms
of a region/cell index `c(j)`; under the new design `c(j) = j`, so no
formula changes, only the collapse of what counts as a distinct action.

### `RLBridge.py`

- `MODEL_VERSION` bumped `17 → 18`.
- `_model_signature()["action_space"]` changed from
  `"exact_feasible_shelter_candidate"` to
  `"regional_cell_priority_shared_deterministic_site_rule"`, so an old
  checkpoint fails closed on load rather than silently reinterpreting cell
  indices as candidate indices (or vice versa).
- `_model_signature()["training_environment"]["candidate_action_count"]`
  needs no code change: it already reads
  `int(self.num_candidate_actions)`, which now equals the cell count because
  `self.observation_builder.number_of_actions` does.
- The class docstring was updated to describe the shared deterministic site
  rule.

### `tests/test_rl_framework.py`

- `test_same_region_same_capacity_sites_remain_distinguishable` (asserted
  that two same-capacity candidates in one cell got separate, distinguishable
  action slots — the old design's core guarantee) was replaced by
  `test_cell_with_multiple_equal_capacity_sites_still_yields_one_slot`,
  which asserts the opposite: exactly one slot, resolved by the OSM-id tie
  break.
- `test_executor_installs_the_exact_selected_candidate_without_substitution`
  (asserted that two distinct candidates in the same cell could be targeted
  by two distinct actions with no substitution — the old design's other core
  guarantee) was replaced by
  `test_executor_always_installs_the_shared_deterministic_rule_winner`,
  which asserts that choosing a cell always installs whichever building the
  shared rule resolves there, and a new
  `test_executor_rejects_a_site_that_diverges_from_the_prediction`, which
  proves the new fail-closed invariant added to `RegionalShelterExecutor`.
- `test_active_population_tie_uses_stable_high_capacity_candidate_order` was
  left unchanged: every one of its assertions (`action_index == 0`,
  `capacity_added == 50.0`, `executed_candidate == executed_cell == 0`)
  continues to hold under the new design, because the deterministic rule
  independently resolves the same higher-capacity site the old frozen-table
  sort put first. This was confirmed by direct execution (see "Verification
  performed"), not by inspection alone.
- `test_forecast_unsafe_candidates_are_removed_from_every_policy_mask` was
  left unchanged: the base test fixture (`make_core`) already has exactly
  one raw candidate per cell, so it was already a cell-indexed action space
  in practice.
- The `_candidate_action_mask` monkey-patch in
  `test_all_unsafe_training_episode_is_recorded_without_fake_transition` was
  updated to accept the mask function's new third parameter (`records`).
- New tests added: `test_empty_cell_yields_an_infeasible_placeholder_slot`
  (an empty cell keeps its slot, masked infeasible, with a unique
  placeholder id and zero capacity) and
  `test_every_benchmark_policy_and_rl_resolve_the_identical_site_per_cell`
  (every registered benchmark policy's predicted site for whichever cell it
  picks matches an independent call into
  `ShelterDatabase.previewShelterCandidate` for that cell).

## 3. What this does for the experiment

The user's stated experimental goal is to start every policy (RL and every
heuristic benchmark) from an identical initial shelter configuration
(already the case: `ShelterDatabase.initShelter()` is a common,
policy-independent round-robin baseline invoked once before any dynamic
policy acts) and observe how the RL agent's cell choices diverge from the
heuristic benchmarks' cell choices as the evacuation unfolds. This is now
directly supported without further schema changes:

- `RegionalActionReceipt.requested_cell` / `executed_cell` already record,
  per decision epoch, which cell each policy chose (these fields existed
  before this change but were, under the exact-candidate design, one of
  several proxies for "which decision mattered"; they are now the
  single, unambiguous quantity that differs across policies).
- `RegionalActionReceipt.candidate_osm_node_id` / `capacity_added` /
  `candidate_cell_i` / `candidate_cell_j` record the resolved building, for
  the (secondary) confirmation that the shared site rule behaved
  identically regardless of which policy chose the cell.
- Because `j*(·)` is shared, any observed divergence in evacuation outcomes
  between RL and a heuristic is now attributable, by construction, to which
  cells were prioritized and in what order — not to which building was
  built once a cell was chosen. This is the analytical payoff of the
  change: it removes a nuisance nested inside the old exact-candidate action
  space that the confirmatory comparison did not intend to measure.

Whichever downstream harness assembles per-episode decision logs
(`backtest.py`, `TrainingLogger.py`, or the visualization manifest) should
be checked to confirm it is already surfacing `requested_cell` /
`executed_cell` per decision epoch per policy in its output tables; if it is
currently keyed by `requested_candidate` under an implicit assumption that
each candidate identity is stable and unique across the whole episode (true
under the old design, no longer true under this one, since the specific
building behind a cell's slot can change across epochs), that keying should
be switched to cell index for any report or figure whose intent is to
compare cell-prioritization behavior across policies. This was out of scope
for this change (no such per-episode logging code path was located and
edited) and should be confirmed before the next training/backtest run whose
output will be used for this comparison.

## 4. Verification performed

- `python3 -c "import ast; ast.parse(...)"` on every edited file
  (`DecisionInterface.py`, `RLBridge.py`, `tests/test_rl_framework.py`):
  all parsed without error.
- A standalone, dependency-light harness (no `torch`, no GNN) was written
  that imports `DecisionInterface.py`, `ShelterDatabase.py`, and
  `Shelter.py` directly, reconstructs the exact fakes and fixtures used by
  `tests/test_rl_framework.py` (`FakeCellTracker`, `FakePedestrianStore`,
  `make_core`), and executes the scenario in every test named in section 2
  above, both new and pre-existing. All checks passed, including the
  divergence-detection RuntimeError, the OSM-identifier tie break, the
  empty-cell placeholder contract, and cross-policy agreement with
  `previewShelterCandidate`. This harness was scratch and was not committed.
- **Not yet performed:** the full pinned-environment test suite
  (`micromamba activate rlevacuation && python -m unittest discover -s
  tests -v`, per `README.md`) could not be run because the tool used to
  make this change reaches a sandboxed shell on the user's device that does
  not have `torch`, `networkx`, or the `rlevacuation` conda environment
  installed, and has no outbound network access to install them. Every
  test in `tests/test_rl_framework.py` that imports `GNN`, `RLBridge`, or
  `EvacuationVisualizer` at module level (i.e., the whole file, since it is
  one module-level import block) was therefore not executed end to end.
  **This must be run in the project's actual environment before this
  change is treated as verified**, per this repository's own change-
  discipline convention. In particular, the reward/PPO/checkpoint tests in
  `tests/test_rl_framework.py` that exercise `RLBridge` end to end were not
  run at all under this change (only their two-line `RLBridge.py` edits
  were inspected by hand and confirmed to be otherwise inert: a version
  bump and a signature string, both read generically wherever they are
  consumed).
- No training run, backtest, or figure was regenerated under this change.

## 5. Compatibility impact

This is a breaking change to the action-space contract. Any checkpoint
trained under `MODEL_VERSION <= 17` fails closed on load under this version
(both the version number and the `action_space` signature string changed).
This is intentional: an old checkpoint's action distribution was fit to a
different, larger, per-candidate action space and cannot be reinterpreted
as a cell-priority policy.
