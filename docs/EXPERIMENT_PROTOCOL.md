# Preregistered backtest protocol

## Primary question

Does a trained regional RL policy improve the undiscounted evacuation objective relative to the prespecified active-population heuristic on held-out stochastic hazard scenarios?

## Experimental design

This document specifies the single-city matched backtest. The pooled five-city
extension, fixed-site macro estimand, and scale ordering are specified separately
in `docs/MULTICITY_EXPERIMENT_PROTOCOL.md`.

The default paper run trains five policies from independent training seeds, each for 320 episodes. The horizon is a multiple of the fixed eight-episode PPO rollout, so no collected experience is discarded. Every trained policy is evaluated on the same 50 held-out scenario seeds. The active-population heuristic is evaluated once on each held-out scenario. Random regional selection is a secondary diagnostic; `initial_only`, when requested, is an anticipative all-capacity-at-time-zero bound and must not be described as an equal online competitor.

The RL actor is initialized as a transparent residual policy. Its fixed prior
ranks exact feasible candidate sites by active population in their host region,
exactly matching the benchmark for deterministic zero-residual selection. A
shared candidate scorer learns corrections from the administrator-facing
regional dashboard, candidate capacity and map position, and incident-level context. Training
remains stochastic; evaluation uses the deterministic masked argmax.

The first held-out RL/heuristic scenario pair is rendered at five evenly
spaced milestones by default. Only RL policy replicate one is rendered for
that representative pair; all policy replicates remain in the numerical
analysis. This prevents figure generation from dominating storage or runtime.
The number of rendered pairs, strategies, selected policy replicate, and exact
milestones are command-line parameters and are recorded in the manifest.

Training and evaluation seeds come from disjoint deterministic streams derived from the recorded launch seed. Hazard evolution owns a dedicated generator. Pedestrian casualty potential outcomes use a counter-based draw keyed by scenario, timestep, and pedestrian identifier, so their values do not depend on iteration order or the number of agents another policy has already evacuated. Policy sampling uses separate generators, and RL evaluation is deterministic.

## Fairness checks

For every held-out scenario, the runner requires:

1. the same scenario seed across dynamic policies;
2. a byte-identical first `RegionalObservation`, including the feasible-action mask;
3. the same maximum dynamic shelter budget;
4. the same independent component-stream seeds;
5. a byte-identical digest of the complete exogenous hazard trajectory;
6. the same observation schema and stable exact-candidate action table;
7. byte-identical candidate-to-OSM-site and candidate-to-region mappings.

The digest check is intentionally limited to the first decision. Later observations should differ when earlier actions change pedestrian routes and outcomes.

Common random numbers synchronize initialization, hazard evolution, and pedestrian hazard shocks. Policy-dependent pedestrian state determines whether a keyed shock is encountered and at what severity, but it cannot change the hazard path or the potential draw assigned to another person-time pair. This preserves the intended stochastic decision problem while preventing a policy from changing its comparator's exogenous scenario through random-number consumption.

## Outcomes

The primary outcome is episode return under the stated reward equation.
Confirmatory components are casualties, safe completions, active evacuation
person-time, hazard-exposure person-time, unfinished population, and restricted
mean time to safety. The restricted mean assigns the finite horizon to every
casualty or unfinished pedestrian, so a policy cannot appear faster merely
because few pedestrians reached safety. Conditional mean safe-completion time,
shelter utilization, action agreement with the heuristic, entropy, KL
divergence, clipping fraction, gradient norm, value loss, and explained
variance are diagnostics rather than alternative objectives.

## Statistical analysis

For each trained policy and held-out scenario, compute RL minus heuristic for outcomes where larger is better and heuristic minus RL where smaller is better. Positive values therefore always favor RL.

Point estimates average the complete policy-seed by scenario matrix. Confidence intervals use a two-way nonparametric bootstrap that independently resamples trained policies and held-out scenarios. The reported randomization p-value applies sign flips to policy-level mean improvements, preserving the training seed as the independent unit for that test. All sign assignments are enumerated exactly when computationally feasible (including the preregistered five-seed design); larger designs use a recorded Monte Carlo approximation. Report effect estimates and intervals even when a p-value crosses 0.05. With five independent policy seeds, the smallest attainable two-sided exact p-value is 0.0625, so estimation and interval width—not a binary 0.05 threshold—must lead interpretation.

Five training seeds are a pragmatic minimum, not a guarantee of adequate precision. If the confidence interval is too wide to support the intended claim, add training seeds and new held-out scenarios under a documented extension; do not selectively rerun only unfavorable scenarios.

## Convergence criteria

Training convergence is assessed across episodes and policy seeds using:

- undiscounted episode return and a fixed-window moving mean;
- entropy and approximate KL;
- clipping fraction and gradient norm;
- value loss and explained variance.

A visually flat reward curve alone is not proof of optimality. A model is considered numerically stable only when rewards and gradients remain finite, KL early stopping is not persistently triggered, entropy does not collapse prematurely, and held-out outcomes do not degrade.

Before opening held-out evaluation, every policy seed must pass an automated
training-only audit. The default audit requires at least 100 episodes and uses
the last 20% of training episodes: all reward and PPO diagnostics must be
finite, the fitted tail trend may span no more than 0.5 recent standard
deviations, the shift from the preceding equal-length window may be no more
than 0.5 standard deviations, and no more than 10% of tail optimizer updates may exceed
the PPO target KL of 0.015. The final checkpoint must also end at a complete
eight-episode rollout boundary. Failure stops the confirmatory evaluation and
preserves the checkpoint for a documented, all-seed training extension.

Convergence does not imply superiority. After held-out evaluation, the runner
labels the primary result `rl_superior` only when the two-way-bootstrap 95%
interval for paired episode-return improvement lies wholly above zero;
otherwise it reports `inconclusive` or `rl_inferior` without changing the
policy or evaluation set. Runs with fewer than two independent policy seeds or
two held-out scenarios are labeled descriptive-only regardless of interval
location.

## Generated artifacts

`backtest.py` writes:

- `experiment_manifest.json` with command, Git state, dependencies, seeds, design, and status;
- per-policy versioned checkpoints and PPO diagnostics;
- per-timestep run logs and per-episode JSON summaries;
- consolidated training and evaluation CSV files;
- `interface_parity.json`;
- paired estimates, two-way bootstrap intervals, randomization p-values, and paper-ready Markdown table;
- a return convergence plot, a reward-decomposition/PPO diagnostic panel, and
  a paired-comparison plot when Matplotlib is available;
- a machine-readable convergence audit and a separate primary-performance
  assessment that cannot promote an inconclusive interval to superiority;
- representative OSM milestone panels in PNG and SVG, individual milestone
  PNGs, source-layer CSV files, and a hash-bearing visualization manifest.

Artifacts produced by code versions before this interface revision are not valid evidence for this study.

## Material passport

- Direct evidence: seed manifests, observation digests, episode summaries, and diagnostic CSV files generated by the runner.
- Statistical inference: paired point estimates, two-way bootstrap intervals, and policy-level randomization tests.
- Recommendation: base paper claims on held-out paired results across independent policy seeds, not on training reward or a single checkpoint.
