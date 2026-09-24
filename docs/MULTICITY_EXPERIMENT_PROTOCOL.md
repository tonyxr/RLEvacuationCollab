# Five-city transfer and scale experiment

## Research question

Does one pooled regional shelter-priority policy improve the preregistered
evacuation return over the active-population heuristic across heterogeneous
urban road networks, without losing performance as city scale increases?

This is a fixed-site experiment over five deliberately selected cities. It is
not a probability sample of United States cities, so the primary estimand is
the equal-city macro average over these five sites. A broader claim about all
cities requires a larger probability-based site sample.

This document describes the active single-policy pooled five-city experiment.
Historical multi-seed E0--E6 manifests remain separate and must not be used to
describe this campaign.

## Prespecified cities and scale order

Cities are ordered by the stable 2020 Decennial Census municipal population,
not by an RL outcome or an OSM graph statistic observed after selection.

| Rank | City | 2020 population | OSM study radius | Purpose |
|---:|---|---:|---:|---|
| 1 | Malibu–Santa Monica Bay / North & West Los Angeles, California | 10,654 | 35,000 m | regional wildfire stress site |
| 2 | State College, Pennsylvania | 40,501 | 3,000 m | college town |
| 3 | Spokane, Washington | 228,989 | 4,000 m | midsize city |
| 4 | Seattle, Washington | 737,015 | 5,000 m | large city |
| 5 | Chicago, Illinois | 2,746,388 | 6,000 m | major city |

The authoritative population URLs, fixed study points, exact radii, and common
simulation settings are versioned in `config/city_profiles.json`. The increasing
radii form an explicit computational and geographic scale stress test. The
Southern California region is intentionally much larger than the municipal
profiles and includes the Malibu coast, Santa Monica Bay, and substantial
north and west Los Angeles. These footprints do not represent complete
municipal boundaries or the full exposed population.

## Why fixed point-and-radius OSM queries

Whole administrative boundaries would make the five cases radically different
computational problems, especially for Chicago, and boundary/geocoder updates
could alter the study area between runs. Each profile therefore uses a fixed
latitude/longitude and radius with an OSM walking network. Roads and
building/amenity features use the identical geographic query. The graph cache
key hashes the complete query specification, and run artifacts record the OSM
query, GraphML SHA-256 digest, node count, edge count, and stamped-feature count.

OpenStreetMap data are licensed under the Open Database License. Static paper
figures must display `© OpenStreetMap contributors` and identify the ODbL; the
existing evacuation visualizer does this automatically.

## Shared MDP and transfer interface

All cities use an 8 by 8 regional observation graph and a stable table of 20
candidate shelter sites. An action names one exact feasible candidate; there
is no hidden lower-level site optimizer. A candidate slot maps to its OSM node
and regional cell for the full episode, while installed or otherwise
unavailable candidates are disabled by the action mask. The primary
configuration uses `node_density_adaptive` axis-quantile cell boundaries; the
registered partition sensitivity uses `equal_area` boundaries. Both modes
retain the same 64-node four-neighbour context graph and separately trained
checkpoints. RL and every dynamic benchmark receive the same
`RegionalObservation`, exact-candidate mask, decision opportunities, and
deployment budget.
The training run initializes 5,000 individual pedestrians, advances 60
one-minute transitions, and permits one regional shelter deployment every ten minutes,
subject to the fixed five-addition budget.

The pooled policy is not given a city identifier. Its inputs are normalized
decision variables that can transfer across maps:

- normalized regional population, mobility delay, danger, and remaining
  shelter capacity;
- normalized candidate capacity and relative east/north map position for every
  exact candidate slot; and
- time remaining, active-population share, and deployment-budget share.

The heuristic remains prespecified to select the feasible candidate whose
regional cell has the largest active population and deliberately ignores the
additional fields.
The two additional deterministic heuristic definitions are registered in
`docs/BENCHMARK_MODEL_PROTOCOL.md`; they are secondary comparisons and do not
replace the primary RL-minus-active-population estimand.

The action-training reward uses a fixed 10-timestep post-deployment window:

`r_k = (Delta safe_k - 3 Delta casualty_k) / P
       - active_person_time_k / (P H)
       - hazard_exposure_person_time_k / (P H)`.

The full-episode held-out objective uses the same four components. There is no
shelter-service shaping or site-selection bonus, so training cannot improve by
exploiting an auxiliary objective that is absent from evaluation. Changing
cities therefore does not change the scientific objective or comparator.

## Training design

One policy seed trains one checkpoint across all five cities.
Within every block of five episodes, each city appears exactly once in a
seeded random order. The convergence-first default is 120 episodes per city,
or 600 episodes total, all at 5,000 individual pedestrians, exactly three
hazard instances, and 50% first-exposure panic susceptibility. Population,
hazard-count, and panic variation are reserved for evaluation. PPO updates use the smallest
rollout of at least eight episodes
that contains an integer number of complete city blocks. For this five-city
suite, that is ten episodes (two observations of every city) per rollout. The
default checkpoint therefore ends on a city-balanced rollout boundary.

The pooled PPO learning rate is `0.0003`, with four optimization epochs,
0.10 policy/value clipping, a 0.015 target-KL stop, and 0.005 entropy weight.
These conservative settings accompany bounded, regularized residual logits and
a clipped smooth-L1 critic loss.

Training and held-out evaluation seeds come from disjoint deterministic
streams. Full factorial evaluation cannot begin until the policy passes the
training-only convergence audit and a 5,000-person held-out RL-versus-heuristic
behavior gate. A failed audit requires an equal extension
for every city; cities must not be selectively continued. Return stationarity is assessed on complete equal-city block means,
not the raw mixed-city episode sequence, so city baseline difficulty cannot
create a false trend. Once held-out evaluation is opened, the launch is sealed
against further training and evaluation overwrite.

## Evaluation and analysis

The behavior gate contains five new nominal stochastic scenarios per city, and
the full factorial then varies all registered population, hazard-count, and
panic levels. Every benchmark and the frozen RL policy are evaluated on the
same city-scenario. Required parity checks include the city and map query,
first observation and feasibility mask, deployment budget, random component
streams, and complete exogenous hazard trajectory.

The primary estimate is the equal-city macro mean of RL-minus-heuristic episode
return, conditional on the one frozen trained checkpoint. The bootstrap
resamples held-out scenarios independently within each city, then averages the
five city estimates. It does not resample policy seeds or the five deliberately
selected cities. Report the macro estimate and
interval first, followed by every city-specific estimate. Casualty, unfinished
population, restricted time to safety, safe completion, and hazard-weighted
person-time remain confirmatory components of the same objective.

The scale-robustness diagnostic regresses the five city-level paired effects on
prespecified scale rank only as a descriptive trend. With five fixed sites it
must not be presented as a powered inferential test. A policy is not called
cross-city robust if the macro interval is favorable but one or more cities
show material casualty degradation.

## Commands

Create the pinned all-Conda-forge environment first. On Apple Silicon, keep
the environment native so PyTorch is not loaded through Rosetta:

```bash
micromamba create --platform osx-arm64 -f environment.yml
micromamba activate rlevacuation
```

Do not combine a pip PyTorch wheel with Conda NumPy/scikit-learn on macOS and
do not set `KMP_DUPLICATE_LIB_OK`; both compromise the numerical runtime used
to support scientific claims.

Validate the versioned design without network access:

```bash
python multicity_backtest.py \
  --validate-profiles-only \
  --launch-id multicity_profile_validation
```

Download/cache and audit the five public map footprints before training:

```bash
python multicity_backtest.py \
  --preflight-maps-only \
  --launch-id multicity_map_preflight
```

Run the current one-policy convergence-first campaign:

```bash
python full_experiment_campaign.py --dry-run
python full_experiment_campaign.py
```

The map preflight requires internet access only for uncached OSM queries. The
full experiment must use the compatible environment recorded in its manifest.
Existing checkpoints from earlier observation/action schemas are intentionally
incompatible with model version 17 and must not be reused. Version 17 adds
minute-level recurrent graph histories, complete final-action accounting, and
factorized critic heads.

## Material passport

- Direct evidence: versioned city catalog, profile snapshot and hash, OSM graph
  hashes, scenario manifests, observation digests, and raw episode summaries.
- Statistical inference: paired city-specific estimates and the fixed-site,
  city-stratified macro bootstrap.
- Limitation: these five sites support heterogeneity and scale-stress evidence,
  not population-level inference to every city.
