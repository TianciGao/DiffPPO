# DiffPPO: PPO with a Diffusion Action Prior

**Research code for PPO-DAP, a method for improving exploration in continuous-control reinforcement learning.**

[![Paper](https://img.shields.io/badge/arXiv-2409.01427v6-b31b1b.svg)](https://arxiv.org/abs/2409.01427v6)
[![Release](https://img.shields.io/badge/release-v0.1.0-blue.svg)](https://github.com/TianciGao/DiffPPO/releases/tag/v0.1.0)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

PPO-DAP combines **Proximal Policy Optimization (PPO)** with a diffusion model that suggests actions for a given state. The model first learns from recorded trajectories. During online training, a critic scores its suggestions, which provide a small additional learning signal for the policy. PPO's probability ratios, advantage estimates, and critic updates use only fresh environment interactions.

This repository implements the method described in [our paper](https://arxiv.org/abs/2409.01427v6). The Python package is named `ppo_dap`.

## Research overview

The paper evaluates PPO-DAP on eight MuJoCo tasks with an online budget of one million environment steps per task, following offline pretraining. It reports improved early learning and final returns that match or exceed the strongest on-policy baselines on six of the eight tasks. See the [paper](https://arxiv.org/abs/2409.01427v6) for the full comparisons and protocol.

**These are the paper's reported results. Reproducing them with this implementation remains unfinished.** The released software provides the algorithm components and their tests; experimental tooling is being developed separately.

## How it works

1. **Learn action suggestions.** Train a state-conditioned diffusion model on recorded trajectories.
2. **Collect new experience.** Run the current policy in the environment and compute PPO's training quantities.
3. **Guide the policy.** Generate candidate actions at the visited states, guide them using critic estimates, and use them in a small auxiliary policy loss.
4. **Adapt the prior.** Update a small subset of diffusion-model parameters using the new experience, keeping its main network fixed.

Generated actions are used only in the auxiliary policy terms. The [algorithm guide](docs/ALGORITHM.md) explains the data flow and update order.

## Install and check

Install **Python 3.12.3** and **uv 0.12.0** first. The release uses a locked CPU PyTorch environment.

```bash
git clone https://github.com/TianciGao/DiffPPO.git
cd DiffPPO
git checkout v0.1.0
uv sync --frozen --all-groups
uv run python -c "import ppo_dap; print(ppo_dap.__file__)"
uv run pytest
```

The [release record](docs/releases/v0.1.0.md) reports **350 tests passed, 0 failed**. These checks cover software behavior; they do not run the paper's training experiments. A wheel and source archive are available on the [download page](https://github.com/TianciGao/DiffPPO/releases/tag/v0.1.0).

## Where to start

| Goal | Page |
| --- | --- |
| Understand the method | [Algorithm guide](docs/ALGORITHM.md) |
| Find the relevant code | [Implementation guide](docs/IMPLEMENTATION.md) |
| Understand what was checked and its limits | [Validation and limitations](docs/THEORY_CONFORMANCE.md) |
| Follow experiment development | [Experiment documentation](https://github.com/TianciGao/DiffPPO/tree/experiment/paper-v6-e1/experiments/paper-v6) |

## Versions and branches

| Version or branch | Purpose |
| --- | --- |
| [`v0.1.0`](https://github.com/TianciGao/DiffPPO/releases/tag/v0.1.0) | Fixed software release for installation and version-specific comparisons. |
| [`main`](https://github.com/TianciGao/DiffPPO/tree/main) | Main documentation, algorithm library, and initial experiment configuration tools. |
| [`experiment/paper-v6-e1`](https://github.com/TianciGao/DiffPPO/tree/experiment/paper-v6-e1) | Experiment interfaces, evaluation, and reporting tools under development. |
| [`release/cleanroom-v0.1.0`](https://github.com/TianciGao/DiffPPO/tree/release/cleanroom-v0.1.0) | Branch used to prepare the first release; use the release tag for its original files. |

The earlier implementation is preserved under the [historical tag](https://github.com/TianciGao/DiffPPO/tree/legacy-pre-cleanroom-main). The current implementation was developed from the paper independently of that earlier code.

## Citation

If you use this work, please cite:

```bibtex
@article{gao2024ppodap,
  title   = {Enhancing Sample Efficiency and Exploration in Reinforcement Learning through the Integration of Diffusion Models and Proximal Policy Optimization},
  author  = {Gao, Tianci and Neusypin, Konstantin A. and Dmitriev, Dmitry D. and Yang, Bo and Rao, Shengren},
  journal = {arXiv preprint arXiv:2409.01427},
  year    = {2024},
  doi     = {10.48550/arXiv.2409.01427},
  url     = {https://arxiv.org/abs/2409.01427v6}
}
```

See [CITATION.cff](CITATION.cff) for citation metadata. When reporting software results, also record the tag or commit you used.

## License

[MIT](LICENSE).
