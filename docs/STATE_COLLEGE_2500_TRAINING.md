# State College 2,500-person training run

## Material Passport

- Material type: code experiment plan and execution record
- Material ID: `state_college_2500_gnn_v16`
- Model contract: version 16
- City: State College, Pennsylvania
- Population: 2,500 individual pedestrian agents
- Training episodes: 232 completed
- PPO rollout target: 8 action-bearing episodes
- Final curriculum: `config/state_college_training_curriculum_2500_extended_232.json`
- Status: complete; the unchanged convergence audit passed
- Run ID: `state_college_2500_gnn_v16_seed_20260918`

## Initial registered training command

```bash
python multicity_backtest.py \
  --cities state_college_pa \
  --launch-id state_college_2500_gnn_v16_seed_20260918 \
  --launch-seed 20260918 \
  --policy-replicates 1 \
  --train-episodes-per-city 120 \
  --training-curriculum config/state_college_training_curriculum_2500.json \
  --train-only \
  --require-convergence \
  --convergence-min-episodes 100 \
  --no-policy-cache
```

The run is training-only. It does not use its training trajectories as evidence
of superiority over the heuristic. Any comparative claim requires a separate
matched-seed held-out evaluation with the frozen checkpoint.

## Execution result

The first 120 episodes completed but did not pass the registered stationarity
audit. Training was continued from the exact saved optimizer, action RNG,
minibatch RNG, and checkpoint state. Registered monotone curriculum extensions
to 160, 176, 224, and finally 232 episodes preserved every completed schedule
entry and kept the population, hazard, panic, model, reward, PPO, and
convergence contracts unchanged. The final command was:

```bash
python multicity_backtest.py \
  --cities state_college_pa \
  --launch-id state_college_2500_gnn_v16_seed_20260918 \
  --launch-seed 20260918 \
  --policy-replicates 1 \
  --train-episodes-per-city 232 \
  --training-curriculum config/state_college_training_curriculum_2500_extended_232.json \
  --train-only \
  --resume \
  --require-convergence \
  --convergence-min-episodes 100 \
  --no-policy-cache
```

The completed campaign represents 580,000 simulated individual-pedestrian
episodes. Every one of the 232 episodes initialized exactly 2,500 pedestrians,
and all 232 contained a valid administrator decision. The 29 complete PPO
rollouts produced 228 minibatch gradient steps. The final checkpoint contains
no pending episode or transition and all parameters are finite.

The final 47-episode audit window passed every registered condition:

- return trend span: 0.100 standard deviations, at most 0.5;
- adjacent-window return shift: 0.092 standard deviations, at most 0.5;
- KL violation rate: 0.0, at most 0.10;
- mean normalized entropy: 0.830;
- mean episode return: 0.0932;
- complete rollout and finite diagnostics: yes.

The convergence result establishes a stationary, numerically stable training
tail. It does not establish superiority over a heuristic. In the first versus
final 47-episode windows, mean return increased from -0.0188 to 0.0932, safe
completions increased from 1,701.6 to 1,795.7, mean safe-completion time fell
from 30.63 to 27.01 minutes, and hazard-exposure person-time fell from 7,884.8
to 6,956.4. Mean casualties were noisy and slightly higher, 1.38 versus 1.40,
so casualty improvement must be judged only in a matched-seed held-out
comparison, not from these unpaired training episodes.

The final checkpoint is
`runs/state_college_2500_gnn_v16_seed_20260918/policies/policy_001/regional_policy.pt`
with SHA-256
`dd183ce1bdeb55e29d9c2e2457659cceea6bb95f9e3089a2e1e35293ed494b38`.
The run manifest records every adaptive continuation and confirms that the
convergence thresholds were not changed.

## Updated optimization contract

> Historical note: this section documents the model-v16/v17 run named above.
> The active model-v23 contract is defined in
> `docs/MDP_AND_OPTIMIZATION_DESIGN.md` and uses complete actor Monte Carlo
> returns, an independent TD(0) critic, lagged baselines, and transactional KL
> rollback rather than GAE and retained KL-violating epochs.

- Policy collection and PPO replay contain no dropout, so stored and replayed
  likelihoods refer to the same parameterized policy.
- Advantages are duration-aware GAE values normalized across a complete
  multi-episode rollout.
- Entropy is normalized by the logarithm of the feasible candidate count.
- Policy and value objectives use clipping; gradients use global norm clipping.
- KL is measured on the entire rollout after each epoch. It stops further
  epochs and controls the next update's learning rate.
- Exact optimizer, action RNG, minibatch RNG, partial rollout, episode count,
  feature contract, and model weights are atomically checkpointed.
- An all-unsafe episode records outcomes without inventing a policy action. A
  final partial batch of complete action trajectories is flushed at the end of
  the registered campaign.
