# Decision-interval NMCC with learned continuation (model v37)

> Historical v37 contract. Model v38 retains the decision interval and learned
> continuation but replaces exhaustive action tables with exact sampled
> installed-site contrasts; see `SCALABLE_SAMPLED_NMCC_V38.md`.

## Scope

Model v37 changes the NMCC improvement calculation, its dedicated value fit,
and the exact NMCC actor update. The MDP reward, route-improvement rule, action
mask, capacity assignment, candidate executor, casualty reporting, panic
model, and hazard simulator are unchanged. In exact NMCC mode, PPO continues
to fit the return critic but no longer applies a second sampled-return gradient
to the actor. Non-NMCC PPO runs are unchanged.

The registered curriculum is
`config/state_college_training_curriculum_multiscale_env_action_v37.json`.
Every online policy begins with two shelters, installs eight equal-capacity
shelters at `t = 15, 30, ..., 120`, and continues to `t = 180`. Every policy
therefore finishes with ten shelters and population-matched aggregate capacity.
Historical v37 runs installed all ten `initial_only` shelters at `t = 0`.
The active benchmark contract supersedes this with a time-zero committed site
order executed at `t = 15,30,...,120`; see `BENCHMARK_MODEL_PROTOCOL.md`.

## Forward-looking NMCC target

At decision state `s_k`, every feasible shelter cell `a` and one `WAIT`
control are restored from the same simulator snapshot and evolved with common
random numbers for exactly one decision interval, `L = 15` timesteps. This
captures pedestrian movement, congestion, route reassignment, hazard spread,
wind-driven forecast changes, and shelter use before the next installation.

Let `D_L(s_k,a)` be the signed action-versus-WAIT reduction in the existing
plain-plus-hazard-exposure time-to-safety objective over that branch. Let
`s'_(k,a)` and `s'_(k,WAIT)` be the paired endpoints. The NMCC score is

```text
Q_NMCC(s_k,a) = scale * D_L(s_k,a)
                 + gamma^L * [V_NMCC(s'_(k,a)) - V_NMCC(s'_(k,WAIT))].
```

`V_NMCC` is a dedicated state-value head. It is not the PPO critic, a heuristic
ranking, a fallback action, or a correction gate. Subtracting the shared WAIT
continuation keeps the score causal and auditable. The WAIT term is constant
within a decision and therefore cannot manufacture a different candidate
ordering.

The environmental reward remains the approved zero-horizon installation
effect. NMCC's longer branch is a learning target only; changing its horizon
does not redefine episode reward.

## Backward learning across the eight installations

Each rollout stores the eight realized installation rewards. For an episode
with decisions `k = 1,...,8`, the continuation target is computed backward:

```text
Y_8 = r_8
Y_k = r_k + gamma^(Delta t_k) * Y_(k+1),  k = 7,...,1.
```

The dedicated continuation head is fitted to `Y_k` with Huber loss after the
environment-only pretraining phase, including action warmup and joint
optimization. A branch ending after installation `k` reads the learned value
of the next-decision state. The final acted branch has zero continuation
because no installation remains. This is fitted value iteration: the current
rollout provides the backward targets, and subsequent NMCC updates use the
improved continuation estimate.

## NMCC actor convergence

The E-step still builds the complete feasible-action target

```text
q(a|s) proportional to pi_behavior(a|s) * exp(Q_NMCC(s,a) / eta),
```

under the configured KL budget. The M-step fits the trainable masked actor,
not the actor mixed with uniform exploration. Exploration remains part of
on-policy data collection only. In exact NMCC mode this is the actor's only
optimization objective; a missing full-action target is an error, not a switch
back to PPO. During the supervised M-step:

- the actor's scheduled learning rate is used;
- NMCC gradients are not clipped;
- the configured pass budget is an upper bound; and
- fitting stops when mean `KL(q || pi_actor) <= 0.01`, the objective actually
  minimized by the M-step. Exact top-choice agreement remains diagnostic
  because nearly tied actions need not share an argmax at this tolerance.

There is no action gate, policy fallback, heuristic continuation, or PPO actor
fallback.

## Exact branch execution cost

Hazard evolution is action-independent, so each random future is advanced once
and its exact hazard frames are replayed in every candidate branch. Candidate
pedestrian/routing/congestion dynamics remain fully simulated. Independent
branches run in isolated simulator-only interpreters, which neither import nor
clone the PyTorch learner. Four workers evaluate all candidates and both tapes
concurrently; worker failure raises directly and never substitutes a serial or
approximate label.

## Concise traffic and hazard foresight

The observation now exposes physical congestion delay per cell as
`1 - congestion_speed_ratio`, separate from total mobility delay. The recurrent
momentum input was reduced from nine partly redundant terms to six rates,
normalized per initial population and per decision interval:

1. active-population clearance velocity;
2. congestion-delay reduction velocity;
3. remaining-route-time reduction velocity;
4. forecast-exposure reduction velocity;
5. hazard-front growth velocity (positive is explicitly adverse); and
6. hazardous-congestion reduction velocity.

The fifth term distinguishes a physically expanding hazard front from changes
caused merely by pedestrians moving between cells. The sixth supplies the
traffic-hazard interaction that a density-only or current-danger heuristic
cannot calculate. Current danger, one-interval wind-aware forecast danger,
route-assignment flows, and congestion remain spatial cell features processed
by the relational GNN; the six rates summarize how those fields are changing
for the LSTM and for branch endpoint valuation.

## Required diagnostics

Training records the branch horizon, physical-branch time, continuation-batch
time, branch-worker count, continuation difference mean/dispersion,
continuation Huber loss, continuation target dispersion, continuation explained
variance, residual target KL, top-1 agreement, and fit-convergence indicator.
Evaluation fails if any policy finishes with a shelter count different from the
shared target. These are validation checks and reports, not action guardrails.
