# RLEvacuationCollab redesign: hazard regimes, timing decisions, belief-based control

Date: 2026-10-12. Base: the current model (complete-ledger reward, cell-priority action, NMCC rollout policy improvement, v52 diagnostics). All changes sit behind configuration keys that default to the current behaviour; with the keys off, a 250-person State College smoke must reproduce today's diagnostics bit for bit.

Companion docs in the project: `regime_aware_framework_2026-10-12`, `corner_case_evaluation_design_2026-10-12`, `RIDESHARE_REDESIGN_2026-10-12` (same section structure).

---

## Part A. Theory

### A.1 Why the learned policy cannot beat objective-greedy today
Three structural facts, all measured on the calibrated testbed:
1. **The decision is forced.** With `requirePopulationMatchedShelterCapacity`, a shelter is installed every 15 minutes on a fixed timetable. The policy only chooses *where*. The large value in the problem is *when* (initial-only 0.37 vs sequential 0.98 success), and that lever is not in the action set.
2. **The remaining choice is a near-tie.** Installing anywhere is worth +0.23; the choice among cells has within-state SD 0.068. The siting signal is 3.4× smaller than the common capacity effect, and two simulated futures identify the best cell only 40% of the time.
3. **The environment has no regime.** Wind is fixed for the whole episode, all hazards ignite at t = 0, and the forecast is a deterministic momentum extrapolation. A greedy rule that reads the current danger map therefore has the same information as any policy with memory. Nothing in the environment punishes a snapshot decision.

So the frozen v36 policy scoring 0.108 below objective-greedy (CI crossing zero) is the expected outcome, not a learner bug. The v52 fixes to the rollout target were necessary, but they cannot create a gap the environment does not contain.

### A.2 The regime-switching crowd-interacting POMDP, evacuation instance
State x_t = (e_t, μ_t, z_t):
- **e_t, hazard regime.** A Markov-modulated process on its own random stream: wind mode (direction and speed jump at a hazard rate, with persistence), intensity mode (spread-rate multiplier, e.g. calm/gusty), road-closure events on links, and secondary ignitions (Poisson, latent until the cell is observed burning). Public signals: a wind forecast with lead L minutes and a miss probability; closure warnings. The hazard remains action-independent, so the Exo-MDP structure and the CRN counterfactual machinery hold.
- **μ_t, crowd state.** Pedestrian positions, routes, panic; as today.
- **z_t, persistent endogenous state the action moves.** Installed capacity and occupancy per region, remaining budget, the time left to the deadline, and the crowd's committed routes.

Action a_t = **defer** or **install in region r** (coarse region, with the deterministic in-region rule choosing the building). Decision epochs stay on the 15-minute clock, but a deferred token is kept, and a deadline constraint (remaining tokens ≤ remaining epochs) keeps the schedule feasible. Observation o_t: today's `RegionalObservation` plus the forecast ensemble (mean, dispersion), wind forecast vector, warning flag, and the budget/deadline slack.

### A.3 What breaks the snapshot policy, channel by channel
| Regime event | Why the greedy snapshot fails | What a belief policy does | Persistent state |
|---|---|---|---|
| Wind shift (forecast with lead, sometimes missed) | Forecast-greedy commits to the current front direction; after the shift the capacity is behind the fire or unreachable | Hedge: hold a token, or place on the side the ensemble says is likely next | Installed capacity, routes |
| Intensity jump (latent) | Greedy under-places ahead of a fast front | Infer from observed spread rate; move early | Capacity |
| Road closure (warned) | Demand-weighted scores ignore route risk; a high-demand cell about to be cut off looks best | Place on the safe side before the closure | Routes, occupancy |
| Secondary ignition (latent) | All rules chase the first front; nothing reserves | Keep a reserve token when the ignition hazard is high | Budget |
| Decoy demand (designed) | A large population that self-evacuates before the shelter opens outranks a small population that will be cut off | Lookahead over crowd dynamics | Routes |

### A.4 Value decomposition (identical to the rideshare paper)
On identical hazard tapes:
- **adaptation value** = regime-oracle greedy (told the regime) − objective-greedy on schedule;
- **foresight value** = belief-rollout over public signals − objective-greedy;
- **ceiling** = hindsight rollout (replays the realized hazard tape; clairvoyant);
- **RL share** = (RL − best tuned rule) / (ceiling − best tuned rule), plus an information ablation with the forecast features zeroed.

RL is justified in a scenario only if adaptation + foresight value ≥ 3 paired SE. The timing value (0.37 → 0.98) says the gap exists at the whole-schedule level; the gate tests whether it survives when greedy is also sequential.

### A.5 Identifiability
The label noise problem (two futures → 40% top-1) is handled by moving the decision to the region + timing level (fewer, more distinct alternatives), by allocating futures adaptively to contenders (OCBA), and by training on hindsight targets per tape. The rollout-on-greedy construction guarantees the learned policy is never worse than greedy in expectation, given accurate evaluation.

---

## Part B. Code changes

Version tag: `v53r`. Configuration keys live in `learner_overrides` (curricula) and stage `overrides`; all default to current behaviour.

### B.0 Files
| File | Change |
|---|---|
| `HazardDatabase.py` | **new class** `HazardRegimeProcess`; `HazardDS.add_hazard_at`, `set_regime`, ensemble forecast. |
| `Core.py` | regime keys, `random_stream_seeds["hazard_regime"]`, regime advance in `simulationEnumerator`, logging. |
| `CounterfactualBranch.py` | regime state in `HazardFrame`; belief vs realized futures; OCBA allocation in `candidate_effects_many_futures`. |
| `DecisionInterface.py` | regime/public-signal features; region-level actions plus `defer`; deadline mask; executor handles defer; new benchmark `ForecastGreedyHeuristic`, `BeliefRolloutPolicy`. |
| `GNN.py` | defer logit from the pooled graph; region-level scorer. |
| `RLBridge.py` | decision bookkeeping with deferral; hindsight targets; exogenous critic channels; benchmark registry. |
| `NMCCPolicyImprovement.py` | branch panels at region + defer level; OCBA. |
| `config/*.json` | new curriculum `state_college_regime_v53.json`; evaluation suite `regime_value_decomposition.json`. |
| `scripts/value_decomposition.py` | **new**: runs all policies on common tapes and writes the table. |
| `tests/test_v53r_*.py` | **new** (B.9). |

### B.1 `HazardDatabase.py`: `HazardRegimeProcess`
```python
@dataclass
class HazardRegimeConfig:
    enabled: bool = False
    # wind regime
    wind_shift_rate_per_minute: float = 0.0        # hazard of a direction/speed jump
    wind_shift_degrees: tuple[float, ...] = (-90.0, -45.0, 45.0, 90.0)
    wind_speed_multipliers: tuple[float, ...] = (0.6, 1.0, 1.5)
    wind_min_dwell_minutes: int = 20
    forecast_lead_minutes: int = 15                # public warning lead
    forecast_miss_probability: float = 0.2
    forecast_false_alarm_rate_per_minute: float = 0.0
    # intensity regime (latent)
    intensity_switch_rate_per_minute: float = 0.0
    intensity_multipliers: tuple[float, ...] = (0.7, 1.0, 1.6)
    # road closures
    closure_rate_per_minute: float = 0.0
    closure_duration_minutes: tuple[int, int] = (20, 60)
    closure_warning_lead_minutes: int = 10
    # secondary ignitions (latent)
    secondary_ignition_rate_per_minute: float = 0.0
    secondary_ignition_window: tuple[int, int] = (10, 90)

class HazardRegimeProcess:
    def __init__(self, config: HazardRegimeConfig, rng: np.random.Generator, cell_tracker, map_ds): ...
    # state
    wind_direction_degrees: float; wind_speed_multiplier: float; intensity_multiplier: float
    active_closures: dict[edge_id, end_time]; scheduled: list[(time, kind, payload)]
    public: dict  # {"wind_forecast_direction","wind_forecast_speed","wind_warning","closure_warning_edges","time_to_change"}
    def reset(self, start_time: int = 0) -> None            # draws the whole episode's event list from self.rng
    def advance(self, sim_time: int, hazard_ds: "HazardDS", map_ds) -> None
        # applies due events: hazard_ds.configure_wind(...), hazard_ds.set_intensity(...),
        # map_ds.close_edge(e, until) / reopen, hazard_ds.add_hazard_at(cell, sim_time)
        # updates self.public (forecast issued lead minutes before a scheduled change,
        # missed with forecast_miss_probability; false alarms as configured)
    def snapshot(self) -> dict ; def restore(self, state: Mapping) -> None
    def belief_particles(self, k: int, rng) -> list["HazardRegimeProcess"]
        # k copies: each keeps the current state and the *public* schedule (warned
        # changes), and resamples the latent part of the future from the hazard rates
```
Event list is drawn at `reset` from the regime stream so the path is a CRN tape. `advance` is called once per simulation minute.

`HazardDS` additions:
```python
def set_intensity(self, multiplier: float) -> None      # scales spreadRate in _spreadUpdateStochastic
def add_hazard_at(self, cell, sim_time) -> None         # new Hazard with age 0, impactedCells=[cell]
def forecast_distribution_by_cell(self, horizon_steps, particles: Optional[Sequence[HazardRegimeProcess]] = None)
```
With `particles`, the forecast is the ensemble: for each particle, run the *deterministic* forecast under that particle's wind and intensity (`_deterministic_forecast_states` already takes the wind from `self`; temporarily `configure_wind` per particle and restore). `expected` = mean, `uncertainty` = across-particle variance normalised to [0,1], `escalation_probability` = share of particles in which the cell becomes affected. Without particles, the current behaviour is unchanged. No simulator random stream is consumed.

`MapDatabase.MapDS` (attribute `core.mapDS`): new `close_edge(edge_id, until)` / `reopen_edge(edge_id)` maintaining `self.closed_edges: dict`; `shortestPath` and `_routing_tree` skip closed edges (invalidate the routing-tree cache on every change); `PedestrianDatabase.pedestrianNetworkInteraction` reroutes a pedestrian whose next link is closed via `MapDS.assignEvacuationRoute`. Pedestrians already on the link finish it. Closures are stored in `HazardFrame`. Neighbour lookups stay in `CAProcessor.CellTracker.getNeighborCells`.

### B.2 `Core.py`
- New configuration keys (defaults off): `hazardRegimeEnabled`, `hazardWindShiftRatePerMinute`, `hazardWindShiftDegrees`, `hazardWindSpeedMultipliers`, `hazardForecastLeadMinutes`, `hazardForecastMissProbability`, `hazardIntensitySwitchRatePerMinute`, `hazardIntensityMultipliers`, `hazardClosureRatePerMinute`, `hazardClosureDurationMinutes`, `hazardSecondaryIgnitionRatePerMinute`, `hazardSecondaryIgnitionWindow`. Validated in `_validate_effective_configuration`, recorded in `_effective_configuration`.
- `random_stream_seeds["hazard_regime"]` derived like the others; `self.hazard_regime = HazardRegimeProcess(cfg, rng=default_rng(seed), ...)` after `hazardDS` is built; `hazardDS.regime = self.hazard_regime`.
- `simulationEnumerator`: at the top of each minute, `if self.hazard_regime.enabled: self.hazard_regime.advance(sim_time, self.hazardDS, self.mapDS)` before `spreadUpdate()`.
- `_record_hazard_state` logs wind direction, speed multiplier, intensity, closures, and the public forecast.

### B.3 `CounterfactualBranch.py`
- `HazardFrame` gains `regime_state: dict` and `closures: tuple`. `_capture_hazard_frame`/`_apply_hazard_frame` snapshot and restore them (`HazardRegimeProcess.snapshot/restore`, `MapDatabase` closures).
- `_hazard_timeline(snapshot, future_seed, mode)`: `mode="realized"` (current behaviour; `future_seed=None` replays the tape) or `mode="belief"`: build one particle with `regime.belief_particles(1, rng(future_seed))`, install it, reseed the exogenous streams, evolve. `BranchResult` records the mode so a clairvoyant value can never be mislabelled.
- `candidate_effects_many_futures(cells, futures, mode, allocation="uniform"|"ocba")`: with `ocba`, run `futures_min` (2) futures for every candidate, then allocate the remaining budget in rounds of 2 to the candidates whose lower confidence bound overlaps the current best (OCBA-style, using the paired variance estimate already computed). Return per-candidate mean, paired SE, and the number of futures spent.

### B.4 `DecisionInterface.py`
**Observation.** `RegionalObservation` gets per-cell `forecast_ensemble_mean`, `forecast_ensemble_dispersion`, `forecast_escalation`, and global `wind_forecast_cos`, `wind_forecast_sin`, `wind_forecast_speed`, `wind_warning`, `closure_warning_share`, `remaining_tokens`, `remaining_epochs`, `deadline_slack = remaining_epochs − remaining_tokens`. `RegionalObservationBuilder._hazard_layer` calls `forecast_distribution_by_cell(horizon, particles=regime.belief_particles(K))` with `K = forecastEnsembleSize` (default 8) when the regime is enabled; otherwise unchanged. `policy_features()` appends the new columns; the frame schema version increments so checkpoints are not mixed.

**Action set.** `number_of_actions = number_of_regions + 1`; index 0 = defer. Regions are the existing `cellX × cellY` partition (4 × 4 in the State College curricula). The in-region building is resolved by `ShelterDatabase._candidate_index` as today. `_candidate_action_mask`:
```python
mask[0] = remaining_tokens < remaining_epochs          # defer allowed only with slack
mask[1:] = (remaining_tokens > 0) & region_has_capacity
```
When `deadline_slack == 0`, defer is masked, so every token is placed by the last epoch. Episodes still end with exactly the budgeted shelters, so capacity parity with the benchmarks holds.

**Executor.** `RegionalShelterExecutor.execute(observation, action)`: `action == 0` returns a receipt with `installed=False`; `RLBridge` keeps `deployments_made` unchanged and `tokens_remaining` as is.

**Benchmarks.**
- `ObjectiveMatchedGreedyHeuristic`: unchanged scoring, never defers (installs on schedule) — the primary comparator.
- `ForecastGreedyHeuristic` (new): greedy on `forecast_ensemble_mean`-weighted demand; never defers.
- `DeadlineGreedyHeuristic` (new, tuned): defers while `max score < θ` and slack > 0; θ tuned by CEM on validation tapes. This is the strongest same-information rule and must be reported.
- `BeliefRolloutPolicy` (new): at each epoch, for every feasible action, K belief futures (mode `belief`) with greedy continuation; picks the best mean; this is the public-information lookahead planner (tier 2).
- `HindsightRolloutPolicy` (new): same with mode `realized`; the clairvoyant ceiling, never reported as a policy.

### B.5 `GNN.py`
The region scorer stays (shared weights, permutation-invariant). Add a defer head: `defer_logit = MLP(pooled_graph ⊕ global_features)`. Logits = concat(defer_logit, region_logits); masking as in `RLBridge._safe_masked_logits`. Parameter count still independent of M.

### B.6 `RLBridge.py`
- `PendingDecision` gains `deferred: bool`; a deferred decision's interval reward is credited to the defer action exactly as an install's is to the install (the complete-ledger reward is unchanged).
- `_decision_due`: unchanged timetable; tokens are no longer implied by `deployments_made` but tracked in `tokens_remaining`.
- **Hindsight targets.** `_collect_rollout_policy_improvement` already branches every panel action on common futures. Add the `mode` switch: training targets use `realized` (hindsight per tape: legitimate for learning), evaluation planners use `belief`. Store both values in the diagnostics so the clairvoyance gap is visible.
- **Exogenous critic channels.** The value head (training only) receives the realized regime tape summary for the branch horizon: future wind direction/speed, intensity, closures, ignitions. The actor never does. This is the input-dependent baseline; it reduces target variance without bias.
- **OCBA** in `_collect_rollout_policy_improvement`: call `candidate_effects_many_futures(..., allocation="ocba")`; the pair regression weights (v52) use the returned paired SE.
- Benchmark registry: add `forecast_greedy`, `deadline_greedy`, `belief_rollout`, `hindsight_rollout`; the evaluator refuses to label `hindsight_rollout` as a policy result.

### B.7 Configuration and scenarios
`config/state_college_regime_v53.json`: copy of `..._complete_ledger_v44.json` with
```json
"hazardRegimeEnabled": true, "hazardWindShiftRatePerMinute": 0.015, "hazardWindShiftDegrees": [-90,-45,45,90],
"hazardForecastLeadMinutes": 15, "hazardForecastMissProbability": 0.2,
"hazardIntensitySwitchRatePerMinute": 0.01, "hazardIntensityMultipliers": [0.7,1.0,1.6],
"hazardClosureRatePerMinute": 0.005, "hazardSecondaryIgnitionRatePerMinute": 0.004,
"forecastEnsembleSize": 8, "actionSpace": "region_defer", "nmccPiTapes": 4, "nmccPiAllocation": "ocba"
```
`config/regime_value_decomposition.json`: scenarios `calm` (all rates 0), `wind_shift`, `intensity`, `closure`, `secondary_ignition`, `decoy` (a constructed start state: one large self-evacuating population and one small population that a scheduled closure cuts off), `compound`. Each with 20 held-out tapes (seeds disjoint from training) and 20 validation tapes for tuning θ.

### B.8 `scripts/value_decomposition.py`
For each scenario: run `objective_greedy`, `forecast_greedy`, `deadline_greedy` (tuned), `belief_rollout` (K = 8), `hindsight_rollout`, `static_greedy`, `initial_only`, and the RL checkpoints (5 seeds), all on the same tapes. Write per scenario: mean return per policy, paired differences vs the best tuned rule with seed-level bootstrap CI, adaptation value, foresight value, RL share, Holm-adjusted p across scenarios, and event-study plots around the regime change. Output `runs/value_decomposition_<date>/table.md`.

### B.9 Tests (`tests/test_v53r_*.py`)
- `test_v53r_identity.py`: regime disabled, 250-person smoke → identical learner diagnostics and episode ledger digests to the current model.
- `test_v53r_regime_crn.py`: same seed → identical event list and `advance` trace; `snapshot/restore` round-trips; `assert_hazard_is_action_independent` still passes with the regime on.
- `test_v53r_forecast_ensemble.py`: with K particles and zero shift rate the ensemble equals the deterministic forecast; with a scheduled, warned shift the ensemble mean moves toward the new direction before the shift; dispersion > 0 only when the latent part varies.
- `test_v53r_action_space.py`: `number_of_actions == regions + 1`; defer masked when slack is 0; every episode ends with exactly the budgeted shelters for every policy.
- `test_v53r_benchmarks.py`: `hindsight_rollout ≥ belief_rollout ≥ objective_greedy` in expectation on 8 toy tapes (tolerance); `deadline_greedy` with θ = −∞ equals `objective_greedy`.
- `test_v53r_ocba.py`: budget accounting exact; the selected best under OCBA matches the exhaustive 8-future best at least as often as uniform allocation on the stored audit states (`runs/nmcc_v25_signal_audit_states_20260921.pkl`).
- `test_v53r_learner.py`: deferred decisions receive interval credit; the critic input includes the regime channels and the actor input does not (shape assertions).

### B.10 Run plan and gates
```
pytest tests/test_v53r_*.py
python scripts/value_decomposition.py --config config/regime_value_decomposition.json --skip_rl   # bounds and rules; ~2 days on 4 workers
```
**Gate 1 (before training):** adaptation + foresight value ≥ 3 paired SE in a scenario. Expected to pass: wind_shift, closure, decoy, compound. Expected marginal: intensity. Expected null: calm (stationarity holds for siting; timing value is reported separately via `initial_only`).

```
python full_experiment_campaign.py --curriculum config/state_college_regime_v53.json   # 5 policy seeds
python scripts/value_decomposition.py --config config/regime_value_decomposition.json --rl_runs runs/v53r_seed*
```
**Gate 2 (RL result):** RL ≥ best tuned rule (`deadline_greedy`) by ≥ 2 SE in the scenarios that passed Gate 1, and RL share ≥ 0.5 of the ceiling. Because the actor is distilled from rollout-on-greedy with hindsight targets, "RL < objective_greedy" is a bug by construction and stops the run.

Then the second site (Spokane or Seattle from `city_profiles.json`) with the same scripts for the transfer axis.

### B.11 What is deliberately not changed
Pedestrian model, social force and congestion, the complete-ledger reward and its four components, cell partitioning, the GNN encoders, the NMCC branch workers and their lifecycle fixes (v52), the checkpoint contracts.
