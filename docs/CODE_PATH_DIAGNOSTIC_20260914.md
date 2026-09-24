# Production code-path diagnostic — 2026-09-14

## Scope and conclusion

The active production workflow is the staged-training and full-factorial
campaign launched by `full_experiment_campaign.py`. Its simulator path is
coherent and contains no active guidance-point subsystem. Social-force,
network congestion, panic, OSM topology, shelter capacity, and planner actions
all participate in the same transition loop.

The repository also contains earlier, separately executable sensitivity and
publication workflows. They are not imported by the production campaign, but
they remain used by their own protocols and regression tests. They were not
deleted merely because they are outside the new confirmatory campaign.

## Active production path

```text
full_experiment_campaign.py
├── multicity_backtest.py                 staged PPO training + policy cache
│   └── backtest._run_episode
│       └── Core.py                       environment orchestration
│           ├── OSMProcessor.py           graph/facility extraction and topology rebuild
│           ├── MapDatabase.py            routable node/edge model
│           ├── HazardDatabase.py         CA spread and Gaussian heat/smoke fields
│           ├── PedestrianDatabase.py     panic, social force, congestion, movement, outcomes
│           ├── SocialForce.py             equations 19–22
│           ├── NetworkCongestion.py       synchronized physical-link congestion
│           ├── ShelterDatabase.py         candidate selection, activation, capacity
│           ├── CAProcessor.py             cell observations
│           └── RLBridge.py                common decision interface and PPO
└── factorial_backtest.py                 cached-policy evaluation and latency comparison
    └── backtest._run_episode              same simulator path for every policy
```

`DecisionInterface.py`, `GNN.py`, `RewardProcessor.py`, `TrainingLogger.py`,
`TrainingCurriculum.py`, `CityProfiles.py`, `CellPartitioning.py`,
`NetworkOptimization.py`, and the entity classes are transitively active.

## Explicitly excluded or deprecated

- `Guidance.py` and `GuidanceDatabase.py` are compatibility-only deprecated
  modules. Both expose module- and class-level deprecation markers and emit a
  `DeprecationWarning` on construction.
- `Core` forces both guidance configuration counts to zero.
- `MapDS.nodeInit` never adds an intersection to `guidanceCanList`.
- No production module imports `GuidanceDatabase`; it has no route into the
  active campaign.
- Empty guidance fields retained in `Cell`/`CAProcessor` are schema-compatible
  zero-valued containers, not an active intervention or candidate source.

## Social-force and panic contracts

- Social force is mandatory in an active `Core`; a default processor also
  exists for isolated pedestrian simulations.
- Safe cells (levels 0–1) supply the paper's binary self-driven force.
- Gaussian heat and smoke fields are normalized and combined into the
  emergency impact force. Velocity is recurrently updated before network
  movement; congestion is applied afterward on frozen physical-link loads.
- The OSM graph is the environmental collision constraint.
- Each active, non-panicked individual in a level 3–5 cell receives exactly
  one panic-onset trial per simulator timestep. Panic is permanent and removes
  the intended shelter route.
- At every subsequently reached node, a panicked individual chooses the most
  occupied incident physical edge with probability 0.5, or a uniformly random
  incident physical edge with probability 0.5. Frozen loads and counter-based
  random draws eliminate iteration-order effects.
- Positive panic rates reject cohort approximation; the full experiment uses
  one modeled agent per pedestrian.

## Intersection consolidation decision

The production graph is replaced—not supplemented—by OSMnx's topology-rebuilt
intersection graph. Reconnecting edges are retained, invalid zero-length
connector edges and isolates are removed, and street counts are recomputed on
the graph that is actually routed.

A 5 m OSMnx tolerance is used. OSMnx buffers each node before merging, so this
captures the nearby node clusters common at divided or large intersections
without the aggressive over-collapse observed at 10–15 m. The preflight
records raw and consolidated node/edge counts, removed connector counts, query
identity, graph-cache path, and cache checksum for every city.

The current checksummed preflight passed the 20-candidate minimum in every
city (`runs/diagnostic_map_preflight_current_20260914/map_preflight.json`):

| City | Raw nodes | Consolidated nodes | Active edges | Eligible sites |
|---|---:|---:|---:|---:|
| Malibu | 742 | 465 | 1,252 | 21 |
| State College | 16,490 | 6,973 | 25,804 | 2,375 |
| Spokane | 23,609 | 13,129 | 49,838 | 1,692 |
| Seattle | 44,614 | 22,278 | 79,174 | 3,834 |
| Chicago | 63,733 | 33,396 | 123,692 | 8,091 |

Malibu uses a 4 km study radius. Its original 2 km radius produced only 12
eligible facility nodes, below the common 20-candidate contract. The 4 km
footprint has 21 eligible sites while retaining the population-based city
scale rank. Study radius is therefore correctly treated as a morphology/design
choice, not as the city-scale outcome variable.

## Full experiment contract

- Cities: Malibu, State College, Spokane, Seattle, and Chicago.
- Population: 5,000; 10,000; 15,000; 20,000; 25,000.
- Hazard sources: 1–5.
- Panic: 10%, 30%, 50%, 70%, 90%.
- Benchmarks: static initial-only, active-population density, random,
  hazard-weighted demand, and maximum accessibility deficit.
- Training: 120 balanced episodes per city for each of eight policy seeds
  (4,800 episodes total). Stage sizes are 4/4/8/20/84 per city so every PPO
  update remains inside one stage.
- Evaluation: 625 factor cells × 10 scenario replications × (8 RL seeds + 5
  benchmarks) = 81,250 episodes.
- Every completed evaluation episode is appended and `fsync`-committed to a
  resumable JSONL journal.
- Complete policies and their exact training ledgers are stored in a
  content-addressed, checksum-verified cache. An exact contract match restores
  both, allowing training to be skipped without fabricating convergence data.
- Full-factorial evaluation fails closed unless the source launch is complete
  and its recorded convergence audit passed. A nonconverged source requires an
  explicit diagnostic-only command-line override.
- RL-versus-heuristic deployment latency pairs cover density, random,
  hazard-weighted, and accessibility-deficit policies under the same city,
  population, hazard count, panic level, and scenario replication. The timed
  boundary covers observation construction, policy/heuristic selection, and
  the shared facility-level execution and rerouting.

## Retained secondary workflows

`or_journal_experiments.py`, `extreme_sensitivity_experiments.py`,
`cell_partition_experiment.py`, `population_candidate_backtest.py`,
`map_factorial_backtest.py`, the figure generators, checkpoint audit tools, and
training-shard assembly are independent research/audit entry points. Each has
an associated configuration, protocol, consumer, or test. They are not dead
code and remain intentionally retained.

`script.py` is a seven-line compatibility entry point to `backtest.main`; it is
retained to avoid breaking historical invocations. `GuidanceDatabase.py` is
similarly retained only for import compatibility.

The obsolete pilot curriculum was removed. No regression tests were deleted:
the older tests still exercise retained research workflows, while the new
social-force/panic/guidance/consolidation/cache tests cover the active model.

## Environment diagnostic

The machine's default Anaconda environment is internally inconsistent
(Pandas 2.2.3 imports against NumPy 1.21.5 despite different package metadata).
That is not a repository logic failure. `requirements.txt` and
`environment.yml` now define one bounded Python 3.11 stack matching the tested
NumPy 1.26 / Pandas 2.1 / OSMnx 1.6 / NetworkX 3.1 / PyTorch 2.4 generation.

## Validation commands

```bash
python full_experiment_campaign.py --dry-run
python multicity_backtest.py --preflight-maps-only --launch-id map_preflight
python -m unittest discover -s tests -v
```

The first command validates the curriculum and exact episode plan without
launching it. The second validates and caches all required OSM road and
facility inputs. The third runs the full regression suite.

Final verification completed with 152/152 tests passing, all Python sources
compiling, and `git diff --check` reporting no whitespace errors.

The bounded real-map integration diagnostic at
`runs/diagnostic_social_panic_latency_20260914` completed an eight-episode PPO
rollout, checkpoint reload, and matched Malibu RL/heuristic evaluation. It
verified byte-identical initial observations, masks, random streams, hazard
trajectory, map contract, and cell partition across the comparison. The one
diagnostic decision took 8.69 ms for RL and 2.29 ms for the density heuristic;
these values validate instrumentation and are not inferential results.

A separate deterministic-hazard Malibu transition diagnostic produced 13
panic onsets, 10 herd choices, 10 random choices, one casualty, and 391
social-force person-timesteps. This confirms that danger-triggered persistent
panic and both node-choice branches execute inside the complete map-backed
simulation loop.
