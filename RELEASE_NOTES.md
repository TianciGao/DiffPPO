# PPO-DAP v0.1.0

Published 20 August 2026 · [Download this release](https://github.com/TianciGao/DiffPPO/releases/tag/v0.1.0)

This release provides the `ppo_dap` research library implementing PPO with a Diffusion Action Prior, based on [paper version 6](https://arxiv.org/abs/2409.01427v6).

## Included

- PPO and advantage estimation, a conditional diffusion action prior, critic-based action guidance, and auxiliary policy losses.
- Parameter-efficient prior adaptation, random-number management, and checkpoint/resume components.
- The algorithm test suite and a locked Python 3.12 / CPU PyTorch environment.
- A wheel, source archive, checksums, and validation records.

## Validation

The release record reports **350 tests passed, 0 failed**. It also records checks of dependency installation, package builds, installation into a clean environment, imports, documented commands, and agreement with the reviewed source files. The published release had the same file tree as the validated candidate.

The download page includes these records:

| File | Purpose |
| --- | --- |
| `R2_VALIDATION.json` | Results of the release checks. |
| `R2_RECONCILIATION.json` | Links the validated candidate to the published release. |
| `SHA256SUMS` | Checksums for the wheel and source archive. |

## Scope

This version supplies algorithm components. It does not include a complete MuJoCo experiment runner, benchmark configurations, or pretrained models. Full reproduction of the paper's rewards, learning curves, runtime, and GPU measurements with this implementation remains incomplete.

The [implementation review and its limitations](https://github.com/TianciGao/DiffPPO/blob/main/docs/THEORY_CONFORMANCE.md) describe the 247 reviewed requirements, the 24 resolved implementation questions, and the six retained limitations. Software checks are not evidence of reproduced experimental results.

The [provenance record](https://github.com/TianciGao/DiffPPO/blob/main/docs/PROVENANCE.json) identifies the original release files and their source history. The fixed release tag is `v0.1.0`; later branch documentation can be revised without changing the release. The earlier implementation remains available under the [historical tag](https://github.com/TianciGao/DiffPPO/tree/legacy-pre-cleanroom-main).

License: [MIT](https://github.com/TianciGao/DiffPPO/blob/v0.1.0/LICENSE).
