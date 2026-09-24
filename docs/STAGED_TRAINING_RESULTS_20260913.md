# Staged PPO training and robustness audit — 2026-09-13

> Historical-result warning: this audit predates model version 13's
> administrator-facing observation, exact-candidate action space, and revised
> reward. Its checkpoint is intentionally interface-incompatible with the
> current model.

## Decision statement

The staged regional GNN policy is operational. A conservative evaluation
calibration is directionally better than all four heuristic benchmarks on the
equal-city mean of a fresh two-scenario backtest and is nonworse than the active
population heuristic in four of five cities. It is **not yet justified to claim
robust or statistically confirmed superiority**. The single trained policy seed
is not inferentially sufficient, the preregistered training-return trend gate
did not pass, and Chicago retains a small negative return point estimate.

## Policy and state design

The actor is a shared four-neighbour spatial scorer with one masked logit per
8x8 regional cell. A fixed active-population prior supplies the transparent
benchmark behavior; PPO learns residual corrections. It has no city identifier
and therefore cannot memorize a city-specific action table.

Each cell exposes active population, mean movement speed, hazard danger,
remaining shelter capacity, shelter utilization, unmet active-evacuee demand,
deployable candidate capacity, candidate count, and road-node density. Global
state exposes time and deployment budget, safe completion, casualties, active
population, network load, and free-flow crossing-time context. After a cell is
selected, every policy uses the same lower-level maximum-capacity shelter-site
selector inside that cell.

## Training executed

The final checkpoint continues one optimizer and RNG state through 200 full
episodes across State College, Reading, Spokane, Seattle, and Chicago. Every PPO
rollout contains each city twice, so an update cannot be dominated by one city.
Every episode uses a 60-minute horizon and a maximum of five dynamic shelter
installations.

1. Operational curriculum: 100 episodes (20 per city). Population increases
   from 10,000 to 50,000; hazards increase from one to multiple; target-scale
   training randomizes hazard count, candidate loss, and initial-shelter
   scarcity before nominal consolidation.
2. Hazard continuation: 100 episodes (20 per city). At 50,000 people it applies
   slow spread, fast spread, high casualty, and severe mobility-reduction
   variants, followed by 80 nominal episodes so the convergence tail contains
   no curriculum transition.

Training uses person-weighted cohorts of at most 20 people. The simulator still
computes congestion, shelter admissions, casualties, rewards, and terminal
accounting in persons. A separate paired fidelity calibration at 5,000 people
and a 14-minute horizon passed its registered group-20 limits; this establishes
an engineering surrogate, not exact equivalence.

All 200 staged episodes preserved

`initialized = safely completed + casualties + unfinished`.

The final update remained numerically stable: approximate KL 0.00773 against a
0.03 limit, clip fraction 0.158, finite gradient norm 0.454, and a complete
10-episode rollout.

## Training stopping audit

The first 100-episode run failed its stationary-tail gate. The 100-episode
continuation improved adjacent-window stability but also failed the declared
trend criterion:

| Criterion | Observed | Limit | Result |
|---|---:|---:|---|
| Complete episodes | 100 | at least 100 | pass |
| Complete city-balanced rollouts | yes | required | pass |
| Finite PPO diagnostics | yes | required | pass |
| Tail KL violation rate | 0.000 | at most 0.100 | pass |
| Adjacent-window shift | 0.067 SD | at most 0.500 SD | pass |
| Tail trend span | 1.676 SD | at most 0.500 SD | **fail** |

The final checkpoint must therefore be described as trained but not converged
under the registered rule. The rule was not relaxed after seeing the result.

## Dynamic-state responsiveness

A counterfactual logit audit passed all four required state-family checks:

- dynamic population movement;
- regional hazard exposure;
- evolving physical/mobility environment; and
- dynamic shelter demand and fulfillment.

The final actor is especially responsive to local movement speed and deployable
capacity. Its canonical danger response is detectable but small. These probes
show numerical use of the inputs; they do not show that each learned direction
is causally correct.

## Fresh robustness backtest of the unscaled checkpoint

The frozen checkpoint was evaluated on two new stochastic scenarios per city
against active population, hazard-weighted demand, accessibility deficit, and
random feasible selection. This produced 50 complete trajectories. Pairing was
verified for scenario seed, initial observation and mask, deployment budget,
random streams, exogenous hazard trajectory, observation/action schemas,
lower-level site optimization, OSM query, and partition digest.

Mean results across the five equally weighted fixed cities are:

| Policy | Episode objective | Safe completed | Casualties |
|---|---:|---:|---:|
| **RL** | **-1.37945** | **8,866.7** | **12,466.5** |
| Active population | -1.40700 | 8,627.5 | 12,933.2 |
| Hazard-weighted demand | -1.41092 | 8,601.9 | 12,990.7 |
| Accessibility deficit | -1.40924 | 8,693.0 | 13,002.7 |
| Random feasible | -1.52233 | 7,308.6 | 14,109.8 |

Against the primary active-population heuristic, RL improves the mean objective
by 0.02755, safe completion by 239.2 people, and casualties by 466.7 people.
The descriptive two-way-bootstrap interval for objective improvement is
[-0.06517, 0.11846]. It crosses zero and cannot establish superiority.

Return improvement by city is +0.1175 (State College), -0.0894 (Reading),
-0.0475 (Spokane), +0.1740 (Seattle), and -0.0169 (Chicago). Thus, the favorable
aggregate does not satisfy the registered all-city robustness criterion.

## Full individual-agent sentinel

One additional State College scenario was run at the full 50,000-person scale
with `pedestrianGroupSize=1`, so all evacuees were represented individually.
The frozen policy loaded and both trajectories completed with verified interface
parity and exact accounting. RL reduced casualties from 3,353 to 2,082, but it
also reduced safe completion from 12,200 to 9,800 and increased unfinished
evacuees from 34,447 to 38,118. Its episode objective was therefore worse by
0.02843. Wall time was 408 seconds for RL and 380 seconds for the heuristic.

This sentinel verifies deployment-scale execution compatibility but supplies
additional evidence against a robustness claim for the unscaled checkpoint.
One city-scenario pair cannot estimate individual-agent average performance.

## Conservative residual calibration

The development results showed frequent and sometimes harmful departures from
the active-population prior. A derived evaluation-only checkpoint therefore
scales the learned final residual logits by 0.25 while leaving the fixed active
prior, GNN representation, critic, and all simulator components unchanged.
Scaling candidates 1.0, 0.5, and 0.25 were compared on the already-opened
development scenarios using worst-city return first and mean return second.
Their worst-city improvements were -0.304, -0.225, and -0.073, respectively;
their equal-city means were +0.145, +0.131, and +0.124. Scale 0.25 was frozen.

The derived checkpoint is marked `training_resume_allowed=false`: its optimizer
state cannot be resumed because the weight transformation invalidates exact AdamW
continuity. Its parent hash and transformation are embedded in the payload. A
new counterfactual audit confirms that all four required dynamic state families
remain responsive after shrinkage.

On two new scenarios per city, the conservative checkpoint produced:

| Policy | Episode objective | Safe completed | Casualties |
|---|---:|---:|---:|
| **Conservative RL** | **-1.25256** | 8,993.1 | **10,144.1** |
| Active population | -1.28326 | **9,403.1** | 10,909.9 |
| Hazard-weighted demand | -1.32756 | 9,333.1 | 11,660.0 |
| Accessibility deficit | -1.29064 | 9,262.7 | 10,984.7 |
| Random feasible | -1.32979 | 7,618.1 | 10,633.1 |

Against active population, conservative RL improves mean objective by 0.03070
with a descriptive interval of [-0.01750, 0.07900]. It reduces casualties by
765.8, but completes 410 fewer evacuations and leaves 1,175.8 more unfinished.
Return improvements are +0.1112 (State College), +0.0625 (Reading), 0.0000
(Spokane), +0.0009 (Seattle), and -0.0210 (Chicago). This is a marked reduction
in cross-city downside, not proof of superiority.

On the already-opened 50,000-individual State College sentinel, conservative RL
improves objective by 0.0466, preserves 12,200 safe completions, and reduces
casualties by 1,025 relative to the heuristic. It leaves 1,025 more unfinished.
Because the scenario was reused for calibration diagnosis, this is not held-out
evidence.

## What can and cannot be concluded

Supported now:

- the end-to-end staged trainer, checkpoint continuation, GNN inference,
  feasibility mask, lower-level shelter selection, and matched benchmark runner
  work as intended;
- the policy demonstrably responds to all requested dynamic state families;
- on its untouched two-scenario screen, conservative RL has the best equal-city
  mean objective and casualty outcome among the five evaluated policies; and
- conservative residual shrinkage materially reduces the observed cross-city
  downside while preserving learned dynamic-state responses.

Not supported yet:

- statistical superiority, because only one independently initialized policy
  was trained;
- convergence under the registered return-stationarity rule;
- nonworsening performance in every city;
- favorable individual-agent average performance at 50,000 people (only one
  reused calibration scenario is favorable for the conservative policy); and
- generalization beyond the five fixed OSM city sites and registered hazard
  family.

## Required next experiment for a defensible robustness claim

Train at least one additional independent policy seed (preferably five), retain
the frozen stopping rule, and evaluate at least two untouched scenarios per city.
The next training revision should add an explicit city-tail/CVaR term and/or an
unfinished-population penalty so casualty reduction does not conceal weaker
completion. The 0.25 calibration must remain fixed if it is tested
confirmatorily; no further tuning may use the completed held-out launches.

## Material passport

| Artifact | Origin and transformation | Status |
|---|---|---|
| `config/staged_training_curriculum_operational.json` | Registered population, hazard, and infrastructure curriculum | source design |
| `config/staged_training_curriculum_continuation.json` | Target-scale hazard perturbations followed by stationary nominal consolidation | source design |
| `runs/staged_operational_full_v2_20260913/` | First 100 staged episodes | complete; not converged |
| `runs/staged_operational_continuation_20260913/` | Exact optimizer/checkpoint continuation for 100 more episodes | complete; not converged |
| `runs/staged_operational_continuation_20260913/policies/policy_001/regional_policy.pt` | Final frozen checkpoint; SHA-256 `bcfb9b17192d840bd44591ed2f7fe276d768a308c5afa22d3761ccf2eb28c2a6` | trained policy |
| `runs/staged_operational_continuation_20260913/policy_sensitivity_audit.json` | Canonical counterfactual state-family probes | all families responsive |
| `runs/staged_development_screen_v3_20260913/` | One development scenario per city, all five policies | directional only |
| `runs/staged_robustness_backtest_v1_20260913/` | Two fresh scenarios per city, all five policies | complete; descriptive |
| `runs/cohort_fidelity_group20_20260913/` | Paired group-1 versus group-20 calibration at population 5,000 | engineering gate passed |
| `runs/staged_individual_sentinel_v1_20260913/` | One 50,000-person individual-agent RL/heuristic pair | complete; RL objective worse |
| `conservative_policy_checkpoint.py` | Scales only learned residual logits; records parent hash; blocks training resume | tested utility |
| `runs/staged_residual_scale_050_dev_20260913/` | Scale-0.5 development calibration | not selected |
| `runs/staged_residual_scale_025_dev_20260913/` | Scale-0.25 development calibration | selected |
| `runs/staged_conservative_robustness_v1_20260913/` | Two untouched scenarios per city, selected checkpoint versus all benchmarks | complete; descriptive |
| `runs/staged_conservative_individual_dev_v1_20260914/` | Selected checkpoint on the reused 50,000-individual sentinel | diagnostic only |

All paths are relative to the repository root. Run manifests record commands,
seeds, dependency versions, city specifications, OSM cache hashes, observation
features, policy contracts, and random-stream provenance.
