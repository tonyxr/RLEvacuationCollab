# Convergence-first 5,000-pedestrian revision

## Legacy-run diagnosis

The superseded production launch had 813 complete training episodes when it
was inspected: all 600 episodes for policy seed 1 and 213 for seed 2. Policy 1
failed the registered stationarity gate because its adjacent tail-window shift
was 0.653 standard deviations against a maximum of 0.500. Its trend and KL
checks passed, so the failure was not simply an exploding optimizer.

The primary defect was unequal action credit. With a two-timestep deployment
cadence, the first four shelter actions usually received two timesteps of
reward, while the fifth remained pending until the 60-minute terminal boundary
and absorbed roughly 51 timesteps plus delayed effects from prior actions. This
made action returns structurally non-comparable. A second defect reapplied the
panic level as a Bernoulli probability every eligible minute: even a 10% level
reached 65% cumulative onset after ten qualifying minutes. Casualty input was
also applied as a per-minute probability, making sustained exposure far more
lethal than the nominal percentage suggested.

Runtime was dominated by individual pedestrian simulation, not PPO. In the
813-row legacy ledger, mean wall time was 36.39 seconds at 5,000 pedestrians,
64.38 seconds at 10,000, 104.15 seconds at 15,000, 137.15 seconds at 20,000,
and 182.67 seconds at 25,000. This approximately linear population scaling is
why convergence is now established at 5,000 before any scale-generalization
claim is attempted.

## Corrected contracts

- Every action receives one non-overlapping 10-timestep reward window.
- Training and held-out evaluation use the same unshaped population-outcome
  objective: safe completions minus three times casualties, active
  person-time, and danger-weighted exposure person-time. There is no
  shelter-service or site-selection bonus.
- Each action selects one exact feasible shelter candidate from the stable
  20-site episode table. The 8 by 8 regional graph supplies transferable
  context; it is not a coarse action followed by a hidden site optimizer.
- The actor starts exactly at the active-population heuristic. Its learned
  residual logits are bounded to [-1,1] and L2-regularized.
- PPO uses learning rate 0.0003, four epochs, 0.10 policy/value clipping,
  target KL 0.015, entropy 0.005, gradient clipping, and clipped smooth-L1
  critic loss.
- Panic level is a one-time susceptibility probability at first exposure to
  danger level 3 or higher. Panic is permanent; each subsequent node choice is
  independently 50% herd and 50% random.
- Casualty input is cumulative over 60 minutes at level 5. Levels 0--3 are
  nonlethal, level 4 is half severity, and overlapping sources combine through
  complementary survival.
- The Southern California site is centered at (33.98, -118.60) with a 35 km
  point radius/bounding footprint and uses the full OSM walking graph. The
  existing major-road regional graph remains presentation-only.
- Topology-rebuilding 5 m intersection consolidation stays active. The
  consolidated graph now has a query-, tolerance-, and source-hash-validated
  disk cache, avoiding duplicate consolidation between preflight and training.

## Gated execution order

1. Preflight and cache every OSM graph, including the expanded regional graph.
2. Train one pooled policy for 120 episodes per city under a single stationary
   baseline: 5,000 people, three hazards, and 50% panic susceptibility.
3. Require that policy to pass the equal-city stationary-tail convergence gate.
4. Publish the checkpoint only after convergence to the checksummed cache.
5. Run 50 held-out nominal RL-versus-heuristic episodes and require conditional
   fixed-policy return superiority with nonworsening casualty point estimates.
6. Only after that behavior gate, open the full 5-city by 5-population by
   5-hazard by 5-panic factorial evaluation.

The final factorial inference is conditional on the one frozen trained policy.
Held-out scenario seeds are resampled within each fixed city; there is no claim
about variability across independently retrained policies.
