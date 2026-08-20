# PPO-DAP clean-room theory-v6

This repository is an independent clean-room implementation derived from
[`arXiv:2409.01427v6`](https://arxiv.org/abs/2409.01427v6). It was developed
without using the original project repository or its implementation as an
algorithm authority.

## Release status

- Clean-room source authority: `TianciGao/DPPO` annotated tag
  `theory-v6-local-ready` at
  `64337ea47e0939a1f019b6c21ef5d35641a42e4c`.
- Public package version: `0.1.0`; the final public release tag is `v0.1.0`.
- Algorithm implementation: complete.
- Theory conformance audit: complete (`247/247` requirements; `181` supported,
  `39` explicit project choices, `11` correctly scoped optional requirements,
  `16` paper-underspecified fail-closed requirements, and `0` blockers).
- Open-question ledger: `24 Resolved / 0 Open`.
- Empirical reproduction: **not completed and not claimed**.

The validated implementation evidence includes a native-Linux full suite of
`350 passed / 0 failed` with exit code `0`. This is implementation-validation
evidence, not evidence that the paper's rewards, learning curves, runtime,
training duration, GPU behavior, or benchmarks have been reproduced.

## Claim boundaries

The theory-conformance result retains the six audited fail-closed boundaries:

- the clean-room prior is not claimed to be the paper's unique `p_psi` or any
  named reverse solver;
- finite TD-MAE is not true-Q, a visited-set supremum oracle, or a strict-eta
  proof;
- the Gaussian proxy is not the real diffusion distribution or an exact
  theory-KL identity;
- Proposition 1 and Eq. (14) are not elevated to a formal theorem, guarantee,
  training objective, or numerical oracle;
- the runtime initial-state source is not claimed to be the paper's unique
  `rho_0` and supplies no exact `J` or `Delta J` oracle;
- finite monitoring remains report-only and supplies neither a threshold
  guarantee nor an active response.

## Install and verify

Python `3.12` and `uv 0.12.0` are required. For the published release:

```bash
git clone https://github.com/TianciGao/DiffPPO.git
cd DiffPPO
git checkout v0.1.0
uv sync --frozen --all-groups
uv run python -c "import ppo_dap; print(ppo_dap.__file__)"
uv run pytest
```

Before `v0.1.0` is published, release-candidate validation uses the same
commands after checking out `release/cleanroom-v0.1.0` at its exact candidate
commit.

## Public scope

This is a library/theory-core release. It contains the clean-room
`ppo_dap` package, its tests, locked package metadata, public release notes,
theory-conformance boundaries, and byte-level provenance. It does not include
an experiment harness, benchmark configuration, trained models, or a claim of
production deployment readiness. See
[the conformance statement](docs/THEORY_CONFORMANCE.md),
[release notes](RELEASE_NOTES.md), and
[provenance manifest](docs/PROVENANCE.json).

The later experimental-server track is separate from this theory release. GPU
smoke tests, complete training, multi-seed studies, ablations, and empirical
result publication cannot redefine the frozen algorithm and are not part of
this release.
