# Recurrent reward and credit-assignment validation protocol

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan + run + validate
- Origin Date: 2026-09-19
- Verification Status: preregistered before the validation campaign
- Version Label: `credit_validation_v1`

## Claims under test

1. Interval rewards exactly reconstruct the complete post-action objective,
   including terminal casualties after the final deployment.
2. Duration-aware component GAE assigns a delayed outcome to every eligible
   earlier action without leaking it into another reward branch.
3. The trained LSTM provides a differentiable causal path from a later decision
   to observations 10, 30, and 60 frames earlier; resetting memory removes that
   path.
4. Across additional training, policy return and component-critic calibration
   improve rather than merely fluctuate with scenario difficulty.
5. Frozen trained policies improve held-out episode return over the registered
   active-population heuristic without worsening casualties.

## Frozen experiment

- City: State College, Pennsylvania.
- Population: 3,000 individual pedestrians per episode.
- Hazard sources: 3.
- Conditional Level-5 60-minute casualty distribution: `[40, 9]`.
- Training: 3 independent policy seeds, 32 episodes per seed, 8-episode PPO
  rollouts (four update blocks per seed).
- Evaluation: 8 held-out scenario seeds shared by all three policies and the
  heuristic.
- Primary outcome: held-out episode-return improvement over the heuristic.
- Safety outcome: held-out casualty difference.
- Learning diagnostics: equal-seed update-block return, component value losses,
  component explained variance, KL, entropy, and residual magnitude.

## Success criteria

### Hard mechanism gates

- At least 2,000 randomized reward trajectories with maximum scalar and
  component telescoping errors no larger than `1e-6`.
- At least 2,000 randomized delayed-terminal GAE trajectories with maximum
  closed-form error no larger than `1e-6`, zero cross-component leakage, and
  nonzero credit for every earlier action.
- For eight independent graph fixtures at each lag of 10, 30, and 60 frames,
  both actor and casualty-critic gradients to the earliest observation must be
  nonzero under full memory and exactly zero when memory is reset every frame.
- Every physical training episode must have a terminal reward-accounting gap no
  larger than `1e-6`.

### Outcome evidence

- Reward improvement is supported only if the update-block trend is positive
  across policy seeds and the final held-out mean return improvement is
  positive. Strong evidence additionally requires its two-way bootstrap 95%
  interval to exclude zero.
- Improved learned credit is supported only if delayed-credit mechanism gates
  pass and critic diagnostics improve across update blocks. A mechanism pass by
  itself is not described as performance improvement.
- Safety requires a nonworsening held-out casualty point estimate. Zero
  casualties in all held-out arms is reported as uninformative, not as proof of
  safety superiority.

## Interpretation controls

- Training return is not compared episode-by-episode because scenario seeds
  differ. Analysis uses rollout blocks and policy-seed stratification.
- Training episodes are not reused as held-out evidence.
- Timing metrics are excluded because the current OpenMP environment is
  hardware and suspension sensitive.
- The campaign remains below the registered 100 episodes per policy required
  for a formal convergence claim; it is an extensive diagnostic, not the final
  efficacy experiment.
