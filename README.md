# PPO-DAP v0.1.0 release branch

This branch was used to prepare the first release of **PPO with a Diffusion Action Prior (PPO-DAP)**. For the original released files, use the fixed [`v0.1.0` tag](https://github.com/TianciGao/DiffPPO/releases/tag/v0.1.0). For the main project introduction and current documentation, see [`main`](https://github.com/TianciGao/DiffPPO/tree/main).

## What the library provides

PPO-DAP uses a diffusion model to suggest actions at states visited by the policy. Critic estimates guide those suggestions, which contribute a small auxiliary policy loss. PPO's training estimates and critic updates use fresh environment interactions.

The release contains the algorithm components, tests, and a locked Python 3.12 / CPU PyTorch environment. It implements the method described in [paper version 6](https://arxiv.org/abs/2409.01427v6), independently of the earlier repository implementation.

The release record reports **350 tests passed, 0 failed**. The implementation review covered **247 requirements**. These checks concern software behavior and correspondence with the paper; full experiment reproduction remains unfinished.

## Install and check the release

Install **Python 3.12.3** and **uv 0.12.0** first:

```bash
git clone https://github.com/TianciGao/DiffPPO.git
cd DiffPPO
git checkout v0.1.0
uv sync --frozen --all-groups
uv run python -c "import ppo_dap; print(ppo_dap.__file__)"
uv run pytest
```

A complete MuJoCo experiment runner, benchmark configurations, and pretrained models are outside this release. Follow the [experiment branch](https://github.com/TianciGao/DiffPPO/tree/experiment/paper-v6-e1/experiments/paper-v6) for that work.

## Documentation

- [Release notes and downloads](RELEASE_NOTES.md)
- [Validation and limitations](docs/THEORY_CONFORMANCE.md)
- [Original release checksums and source history](docs/PROVENANCE.json)
- [Paper citation](https://github.com/TianciGao/DiffPPO/blob/main/CITATION.cff)

The checksum record refers to the original tagged release. Documentation edits on this branch do not alter that tag or its downloadable packages.

License: [MIT](LICENSE).
