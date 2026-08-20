# PPO-DAP implementation map

This document explains how the public `ppo_dap` package is organized. It is meant to answer a practical question: **where does each part of the paper live in the code?**

The stable release is `v0.1.0`. The package is a theory-core library; the paper-scale MuJoCo experiment harness is intentionally separate and is not part of this release.

## Package map

| Paper / runtime concern | Main public paths | Role |
| --- | --- | --- |
| Action representation | `src/ppo_dap/actions/` | environment/action-space contracts and immutable action carriers |
| PPO / GAE / values | `src/ppo_dap/estimators/` | GAE, PPO preparation, value/Q snapshots and targets |
| Actor / critic / PET losses | `src/ppo_dap/objectives/` | optimization objectives with explicit parameter ownership |
| Diffusion action prior | `src/ppo_dap/prior/` | denoiser, training noise, Eq. (6), sampler, trainer and publication boundary |
| Value guidance | `src/ppo_dap/value_guidance/` | Eq. (7), Eq. (8) and Gaussian proxy support |
| Composition boundaries | `src/ppo_dap/interfaces/` | actor, critic and PET composition / authority contracts |
| Iteration model | `src/ppo_dap/algorithm/` | training state, iteration report, ports and the single iteration spine |
| Production runtime | `src/ppo_dap/runtime/` | runner construction, persistent RNG, multi-iteration continuation, bundle replacement and checkpoint/resume |
| Controlled initialization | `src/ppo_dap/warm_start/` | explicitly scoped warm-start dataset, losses and atomic execution |
| Diagnostics | `src/ppo_dap/audit.py` | report-only training diagnostics and audit evidence |

## Execution layers

### 1. Mathematical kernels

The lowest-level modules implement constrained mathematical operations such as Gaussian distributions, GAE, PPO quantities, diffusion-noise handling and value-guidance operations.

Relevant paths include:

- `src/ppo_dap/distributions/`
- `src/ppo_dap/estimators/`
- `src/ppo_dap/prior/`
- `src/ppo_dap/value_guidance/`

### 2. Ownership-aware objectives

`src/ppo_dap/objectives/` and `src/ppo_dap/interfaces/` compose the mathematical pieces while enforcing who is allowed to receive gradients. This is where the implementation prevents offline/synthetic data or the wrong parameter owner from silently entering an update.

### 3. Iteration spine

`src/ppo_dap/algorithm/iteration.py` is the framework-neutral single-iteration spine. State and identity carriers live beside it in `algorithm/state.py` and `algorithm/ports.py`.

### 4. Production runtime

`src/ppo_dap/runtime/` adds the capabilities required to run the frozen iteration semantics repeatedly:

- environment capability boundaries;
- persistent production RNG;
- complete iteration bundles;
- atomic successor installation;
- Stage-II multi-iteration execution;
- exact same-run checkpoint/resume.

The runtime is deliberately framework-neutral. A MuJoCo/Gym-style experiment environment must implement the explicit environment capability rather than being hard-coded into the algorithm package.

## Why some files have `g*` and `v*` names

You will see filenames such as:

- `runtime/g7_environment.py`
- `runtime/g7_stage_ii.py`
- `runtime/v1_bindings.py` through `v4_bindings.py`
- `tests/g3/` through `tests/g7/`

These labels come from the internal implementation and validation milestones used while reconstructing the paper. They are preserved in `v0.1.0` because the released source and tests were independently validated byte-for-byte.

They do **not** mean that the repository contains multiple competing PPO-DAP algorithms. The public algorithm is the single theory-conformant release described in [ALGORITHM.md](ALGORITHM.md).

Renaming these modules would be a source-level API/refactor change and is therefore deferred to a future API-stabilization release rather than mixed into documentation cleanup.

## Tests

The repository ships the full validation suite used for the public theory-core release. Test directories mirror the same historical implementation milestones:

- `tests/g3/` — PPO core, rollout provenance and warm-start contracts;
- `tests/g4/` — diffusion-prior kernels and publication compatibility;
- `tests/g5/` — integrated value-guidance / actor / critic / PET slices;
- `tests/g6/` — audit and diagnostic integration;
- `tests/g7/` — production environment, RNG, multi-iteration, rearm and checkpoint/resume.

Run all tests with:

```bash
uv run pytest
```

The validated `v0.1.0` release completed `350 passed / 0 failed`.

## Public API status

`v0.1.x` should be treated as a **research theory-core package**, not yet as a fully stabilized high-level SDK. Internal module names are public source but are not all promised as long-term semantic-versioning API endpoints.

The next user-facing layer is the independent paper experiment harness. It should depend on the released core and provide:

- environment adapters;
- dataset manifests/loaders;
- Stage-I and Stage-II run configuration;
- evaluation and paper metrics;
- run manifests and resource logging.

That experiment layer may validate PPO-DAP, but it must not redefine the frozen algorithm to match a target result.

## Further reading

- [README](../README.md)
- [Algorithm guide](ALGORITHM.md)
- [Theory conformance](THEORY_CONFORMANCE.md)
- [Machine-readable release provenance](PROVENANCE.json)
- [v0.1.0 release record](releases/v0.1.0.md)
