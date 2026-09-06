# How PPO-DAP works

PPO-DAP adds action suggestions from a diffusion model to Proximal Policy Optimization (PPO). The **actor** is the policy that chooses actions; the **critic** estimates their value. The diffusion model acts as an **action prior**: a learned distribution of plausible actions for a given state.

The scientific reference is [paper version 6](https://arxiv.org/abs/2409.01427v6). This page explains how the repository implements its training flow.

## Two training stages

**Offline pretraining.** A conditional diffusion model learns from recorded state-action trajectories. This gives it an initial action distribution before online interaction begins.

**Online learning.** The policy collects new environment interactions. PPO learns from this fresh experience, while the prior generates additional action suggestions at the states the policy visited. Critic estimates guide these suggestions toward promising actions.

The suggestions affect the actor through a small imitation loss and, optionally, a distribution-matching penalty. They do not enter PPO's probability-ratio estimator, advantage calculation, or critic update.

## What each dataset is used for

| Data | Paper notation | Use in this implementation |
| --- | --- | --- |
| Recorded trajectories | `D_off` | Train the action prior; support explicitly configured initialization or diagnostics. |
| Fresh policy rollouts | `D_on` | Compute PPO and generalized advantage estimation (GAE); update the critic and the prior's small trainable subset. |
| Generated action suggestions | `D_syn` | Supply auxiliary actor losses at the visited states. |

Only fresh policy rollouts supply PPO's on-policy training data. The code checks these data roles when constructing training inputs.

## One online iteration

1. **Save the starting models.** Keep fixed copies of the policy, critic, and prior needed to evaluate this iteration consistently.
2. **Collect experience.** Run the policy in the environment to obtain a fresh rollout.
3. **Prepare PPO inputs.** Compute advantages, return targets, and the action probabilities recorded during collection.
4. **Generate and guide actions.** Sample candidate actions from the prior at the visited states and apply critic-based guidance.
5. **Update the actor.** Combine the PPO objective with the configured auxiliary losses.
6. **Update the critic.** Learn from the current rollout.
7. **Adapt the prior when scheduled.** Update only the designated small parameter subset; keep the main diffusion network fixed.
8. **Record diagnostics and finish.** Complete the iteration before preparing the next one.

## How value guidance is applied

| Mechanism | Paper reference | Role |
| --- | --- | --- |
| Candidate weighting and resampling | Eq. (7) | Give greater weight to candidate actions with higher critic estimates. |
| Guidance during denoising | Eq. (8) | Use critic gradients within action generation. |
| Optional prior regularization | Eq. (9) | Encourage the actor to stay close to an approximation of the prior. |

For the optional regularizer, the implementation uses a Gaussian approximation held fixed during the actor update. Its Kullback-Leibler (KL) divergence is tractable, but it is not the exact KL divergence to the full diffusion distribution.

## Which parameters change

| Component | Online update |
| --- | --- |
| Actor | PPO and the configured auxiliary losses. |
| Critic | Current environment rollouts. |
| Main diffusion network | Kept fixed. |
| Prior adaptation parameters | Updated from current rollouts when scheduled. |

Updating only a small subset is called **parameter-efficient tuning (PET)** in the paper; low-rank adaptation (LoRA) is one such approach. The implementation checks which parameters each loss may update.

## Reproducibility and diagnostics

The runtime maintains separate random-number streams for policy actions, candidate generation, resampling, prior adaptation, and diagnostic replay. It validates all inputs needed for the next iteration before installing them together. If an iteration fails, the run stops instead of retrying with a changed random state.

Checkpoints are taken between completed iterations. Resuming the same run requires restoring model state, random-number state, and any environment state required by the adapter. If exact restoration is unavailable, the runtime rejects the resume request. Actual environment support must be established by the experiment integration.

Diagnostics report training quantities. In `v0.1.0`, they do not automatically stop training, change hyperparameters, or guarantee a performance bound.

See [validation and limitations](THEORY_CONFORMANCE.md) for the scope of the implementation checks and [the implementation guide](IMPLEMENTATION.md) for source locations.
