# PPO-DAP algorithm guide

This page is a reader-facing guide to the implementation in this repository. The paper remains the scientific reference: [arXiv:2409.01427v6](https://arxiv.org/abs/2409.01427v6).

## 1. Two-stage protocol

PPO-DAP separates the diffusion prior from the online PPO estimator.

### Stage I — offline action prior

A conditional diffusion model is pretrained on logged trajectories `D_off`. This stage learns a state-conditioned action prior over behavior-supported actions. `D_off` does not become an on-policy PPO batch later.

### Stage II — strictly on-policy PPO with guided proposals

Online training repeatedly collects a fresh rollout `D_on`, computes the PPO/GAE quantities from that rollout, and uses the diffusion prior only to generate auxiliary action proposals around the states that the current policy actually visited.

The important consequence is simple:

> **PPO remains on-policy.** Offline logs and purely synthetic actions do not enter the PPO likelihood-ratio estimator.

## 2. Data roles

| Data object | Source | May affect | Must not be used as |
| --- | --- | --- | --- |
| `D_off` | logged/offline trajectories | Stage-I prior training; explicitly allowed initialization or diagnostics | current PPO/GAE rollout |
| `D_on` | fresh environment interaction from the current policy | PPO, GAE, critic update, PET update, monitoring | offline behavior data |
| `D_syn` | prior proposals generated at current on-policy states | low-weight actor auxiliary terms | on-policy PPO trajectories |

This separation is enforced throughout the implementation rather than left as a training-script convention.

## 3. One online iteration

The runtime follows one ordered iteration spine:

1. **Freeze entry state.** Capture the actor, critic and prior authorities used for the current iteration.
2. **Collect a fresh rollout.** The environment produces the current `D_on` under the behavior policy for this iteration.
3. **Prepare PPO inputs.** Compute GAE, returns and the frozen behavior quantities used by PPO.
4. **Generate proposals.** At states from `D_on`, the diffusion prior generates multiple candidate actions.
5. **Apply value guidance.** Candidate actions are concentrated toward high-value regions through the paper's value-guidance mechanisms.
6. **Update the actor.** Optimize the fresh on-policy PPO objective together with the allowed low-weight auxiliary terms.
7. **Update the critic.** Update the value/Q owner using the current online batch.
8. **Update PET when triggered.** Adapt only the designated PET/LoRA subset; the prior backbone remains frozen online.
9. **Monitor and commit.** Diagnostics are read-only, the iteration is committed, and only then may the next iteration authority be constructed.

## 4. Value-guided proposals

The implementation separates the three roles that are easy to conflate in a monolithic training loop:

- **Eq. (7):** critic-based energy weighting / resampling of candidate actions;
- **Eq. (8):** gradient guidance performed inside the denoising process rather than as a post-hoc action correction;
- **Eq. (9):** actor-side regularization through the tractable detached Gaussian proxy used by the implementation.

The Gaussian proxy is a computational device. It is not claimed to equal the full diffusion action distribution or to provide an exact theory-KL identity. See [Theory conformance](THEORY_CONFORMANCE.md) for the audited claim boundary.

## 5. Parameter ownership

| Component | Online update owner | Key restriction |
| --- | --- | --- |
| Actor | actor parameters `θ` | PPO uses fresh `D_on`; synthetic proposals enter only allowed auxiliary terms |
| Critic | shared value/Q parameters `φ` | updated from current online data |
| Diffusion prior backbone | frozen online | no hidden full-prior optimization during Stage II |
| PET / LoRA subset | `ψ_PET` | only the designated small parameter subset is adapted |

The implementation treats these ownership rules as contracts so gradients cannot silently cross into the wrong model component.

## 6. Runtime and randomness

Production randomness is explicit and non-aliased across the behavior policy, raw proposals, optional guided proposals, Eq. (7) resampling, PET updates and diagnostic replay. The runtime does not silently reseed through hidden default generators.

Multi-iteration execution uses a complete successor bundle that is validated before atomic installation. A failed production iteration is terminal for that run rather than silently retried with altered random state.

## 7. Checkpoint and resume

Checkpointing is intentionally strict:

- a checkpoint is legal only after a successful committed iteration and before constructing the next iteration bundle;
- resume restores the same run, lineage, RNG state and checkpointable environment state;
- if exact restoration is unavailable, resume fails closed rather than creating an approximate continuation.

This is an implementation reproducibility contract, not an additional claim from the paper.

## 8. Monitoring

The audit layer reports diagnostics such as gradient- and KL-related quantities. In `v0.1.0` these diagnostics are **report-only**: they do not trigger hidden early stopping, automatic hyperparameter changes or an active safety response.

## 9. What this release does not claim

The theory-core release does not claim that:

- its clean-room diffusion prior is the paper's unique possible prior or a specific named reverse solver;
- finite TD-MAE is a true-Q oracle or strict theoretical proof;
- the Gaussian proxy is the real diffusion distribution;
- Proposition 1 / Eq. (14) is stronger than stated in the paper;
- runtime initial-state handling uniquely identifies the paper's `ρ₀` or provides exact `J/ΔJ` oracles;
- finite monitoring creates a guaranteed threshold response.

For the complete audited wording, see [THEORY_CONFORMANCE.md](THEORY_CONFORMANCE.md).
