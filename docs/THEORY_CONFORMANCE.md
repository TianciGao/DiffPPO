# Theory conformance statement

PPO-DAP `v0.1.0` is a clean-room implementation based on paper v6
([`arXiv:2409.01427v6`](https://arxiv.org/abs/2409.01427v6)). Its algorithm
implementation and theory-conformance audit are complete. Experimental
reproduction is not complete and is not claimed.

## Audited inventory

The primary and independent secondary audits covered all `247/247`
requirements:

| Classification | Count |
| --- | ---: |
| Conformant and supported | 181 |
| Conformant by explicit project choice | 39 |
| Paper-optional and correctly scoped | 11 |
| Paper-underspecified and fail-closed | 16 |
| Conformance blocker | 0 |

- Primary audit packet SHA-256:
  `70ff3b1e1632e7d40c71c38660b856738cfd2022e758e7dd71c41fca1b4a5349`
- Independent secondary ledger SHA-256:
  `0247efe70611d2eb6a4677fbe645dad5d710204dbe7268fb1b2fa5cac1d28be2`
- Secondary classification, evidence, residual, and cross-cutting mismatches:
  `0/0/0/0`

All 24 tracked open questions were resolved. Six were closed only through the
following fail-closed claim boundaries; their paper residuals were not filled
with new paper semantics:

- the clean-room prior is not the paper's unique `p_psi` or a named reverse
  solver;
- finite TD-MAE is not true-Q, a visited-set supremum oracle, or strict-eta
  proof;
- the Gaussian proxy is not the real diffusion distribution or exact
  theory-KL identity;
- Proposition 1 and Eq. (14) are not a formal theorem, guarantee, training
  objective, or numerical oracle;
- the runtime initial-state source is not the paper's unique `rho_0` and does
  not provide an exact `J` or `Delta J` oracle;
- finite monitoring is report-only and creates no threshold guarantee or
  active response.

These boundaries prohibit using finite diagnostics or empirical performance
as theory proof. They also prohibit using this release to claim reproduction
of paper rewards, curves, runtime, GPU behavior, or benchmarks.

## Implementation boundary

The package preserves the audited Algorithm 1 ordering, online/offline/
synthetic data isolation, actor/critic/prior/PET gradient ownership, explicit
non-aliasing RNG authorities, multi-iteration continuation, and exact
same-run checkpoint/resume semantics. Checkpoint/resume does not change the
algorithm or authorize a fresh-run fork.

The later experiment-server track is independent and requires separate
authorization.
