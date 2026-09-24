# Publication graph inventory

Generated on 2026-09-08 from completed, checksum-tracked experiment artifacts.
No values were simulated, imputed, or invented solely to fill a figure.

## Graphs available from current evidence

Both PNG and SVG are supplied for analytical figures.

| ID | Graph | Location | Evidence status |
|---|---|---|---|
| F01 | Training reward and convergence | `current_evidence/01_training_reward_and_convergence.*` | Descriptive; one policy seed; fixed-checkpoint evaluation still absent |
| F02 | Rollout-averaged PPO optimization diagnostics | `current_evidence/02_ppo_optimization_diagnostics.*` and `current_evidence/02_ppo_update_averages.csv` | Descriptive; 16 complete rollout updates; one policy seed |
| F03 | Absolute policy performance | `current_evidence/03_primary_policy_performance.*` | 25 matched scenarios; one trained policy seed |
| F04 | City-specific paired effects | `current_evidence/04_city_specific_paired_effects.*` | Five fixed cities; one trained policy seed |
| F06 | Safety, casualty, and timeliness frontier | `current_evidence/06_safety_casualty_timeliness_frontier.*` | 25 matched scenarios; one trained policy seed |
| F11 | Final shelter installations | `current_evidence/maps/f11_*.png` | One visualized matched scenario per city |
| F12 | Evacuation progress | `current_evidence/maps/f12_*.png` | One visualized matched scenario per city |
| F13 | Population/candidate stress | `scale_stress_pilot/13_population_candidate_scale_stress.*` | Computational pilot; 1 of 25 factor cells |

The revised F01 displays raw equal-city block returns behind the eight-block
moving average so stochastic variation is not mistaken for a smooth learning
trend. Every F02 optimization point averages its completed PPO epoch/minibatch
steps, and every F02 outcome point averages all ten episodes contributing to
that rollout update; the last episode is never substituted for the rollout
mean. F13 visibly identifies itself as pilot evidence and leaves all untested
cells blank.

## Exact shelter choices and 120-transition map progression

The executable full design contains 500 condition cells and 1,500 matched
episodes. The completed development pilot is under
`factorial_maps/factorial_maps_seed_20260908_v7_pilot/` and contains:

- `shelter_decision_sequence_comparison.png`: the exact numbered OSM candidate
  selected by RL and the heuristic at each online decision, with static sites
  correctly labeled as time-zero predeployments;
- `priority_decision_epoch_comparison.png`: every changing RL and heuristic
  priority cell at t=1,6,11,16,21, together with one red point per active
  pedestrian, a fixed 0--1 continuous cell-danger heatmap, and the exact
  candidate installed; static is excluded because it has no online decision
  epoch;
- `evacuation_progress_comparison.png`: aligned t=0,20,40,60,80,100,120 map
  panels for all three policies, including explicit absorbing-state labels
  after genuine early completion;
- `map_factorial_figure_index.csv` and `map_factorial_manifest.json`: source
  paths, checksums, factors, seeds, terminal-state rules, and the explicit
  horizon-mismatch limitation.

The revised pilot uses 10 candidates, a strict five-addition budget, and three
hazards. RL selected cells `3,9,6,11,4`; the heuristic selected
`3,11,9,4,5`; both left three candidates unused. This one-cell result validates
rendering and auditability only. At every decision, the number of exported
pedestrian point records equals both the active count and the sum of regional
active populations. Mean cell danger evolves from 0.0094 at the first decision
to 0.0375 at the fifth on the common scale. The available checkpoint was trained at 14
transitions before the fixed-budget contract and cannot support confirmatory
claims. Full execution is intentionally blocked until a converged
120-transition, five-addition source policy exists.

## Graphs awaiting experiment data

| ID | Graph | Required missing result table |
|---|---|---|
| F05 | Capacity-by-hazard performance | `regime_evaluation.csv` |
| F07 | Hierarchical versus flat action scalability | `scalability_evaluation.csv` |
| F08 | Reward and architecture ablation | `ablation_evaluation.csv` |
| F09 | Leave-one-city-out transfer | `transfer_evaluation.csv` |
| F10 | Operating robustness | `robustness_evaluation.csv` |

These figures will be generated automatically when the declared tables are
populated. Empty panels or synthetic values are not used as substitutes.

## Provenance

- `current_evidence/full_experiment_figure_manifest.json` records input and
  output checksums for F01--F04, F06, F11, and F12.
- `scale_stress_pilot/full_experiment_figure_manifest.json` records checksums
  for F13.
- Each subdirectory contains `full_suite_readiness.json`, which states why the
  available evidence is not yet the complete confirmatory suite.

## Regeneration

```bash
python generate_full_experiment_figures.py \
  --launch-dir runs/multicity_five_city_lr1e3_learning_audit_arm_20260906 \
  --output-dir publication_graphs/current_evidence

python generate_full_experiment_figures.py \
  --launch-dir runs/population_candidate_scale_seed_20260908_pilot \
  --output-dir publication_graphs/scale_stress_pilot \
  --skip-maps
```
