# PPO-DAP clean-room theory-v6 — v0.1.0

This release packages an independent clean-room implementation based on
[`arXiv:2409.01427v6`](https://arxiv.org/abs/2409.01427v6). The source
authority is `TianciGao/DPPO` annotated tag `theory-v6-local-ready`, peeled to
commit `64337ea47e0939a1f019b6c21ef5d35641a42e4c`.

## Included

- the `ppo_dap` theory-core library;
- the complete clean-room test suite;
- locked Python 3.12 / CPU Torch package metadata;
- curated theory-conformance and byte-level provenance records.

The source authority completed a native-Linux suite with `350 passed / 0
failed`. The public candidate must independently pass fresh-clone, locked
install, wheel/sdist, clean-install, provenance, privacy, README-command, and
full-suite validation before this release is published.

## Claim boundary

Algorithm implementation and theory conformance are complete for the 247
audited requirements. This release is not an experimental reproduction. It
does not claim reproduction of paper rewards, learning curves, runtime,
training duration, GPU behavior, or benchmarks.

The six fail-closed boundaries remain authoritative:

1. The clean-room prior is not claimed to be the paper's unique `p_psi` or a
   named reverse solver.
2. Finite TD-MAE is not true-Q, a visited-set supremum oracle, or a strict-eta
   proof.
3. The Gaussian proxy is not the real diffusion distribution or exact
   theory-KL identity.
4. Proposition 1 and Eq. (14) are not a formal theorem, guarantee, training
   objective, or numerical oracle.
5. The runtime initial-state source is not the paper's unique `rho_0` and is
   not an exact `J` or `Delta J` oracle.
6. Finite monitoring is report-only; it provides neither a threshold
   guarantee nor an active response.

Experiment adapters, server runs, GPU validation, empirical results, PyPI,
and container publication are outside this release.
