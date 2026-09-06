# Experiment setup

This directory contains the initial configuration tools for experiments based on [PPO-DAP paper version 6](https://arxiv.org/abs/2409.01427v6). It uses the fixed [`v0.1.0` library release](https://github.com/TianciGao/DiffPPO/releases/tag/v0.1.0).

## Available on this branch

- Validation of experiment settings, with required values supplied explicitly.
- Records identifying datasets and runs, including file checksums.
- Reproducible seeds and separate random-number streams.
- A message-format specification for a simulator running in a separate process.

This branch does not yet contain an environment connection, training or evaluation loop, or dataset downloader. More developed tooling is available on the [experiment branch](https://github.com/TianciGao/DiffPPO/tree/experiment/paper-v6-e1/experiments/paper-v6). Full experiment reproduction remains unfinished.

## Settings needed before training

The experiment protocol must specify:

| Area | Required choices |
| --- | --- |
| Repeated runs | Random seeds, number of runs, and matching of seeds across methods. |
| Evaluation | Evaluation frequency, episode count, time limits, and learning-curve measurement points. |
| Prior pretraining | Model size, number of training passes, learning rate, numeric precision, and noise settings. |
| Actor and critic | Network sizes, initialization, policy standard-deviation limits, discount factor, and update settings. |

See the [configuration schema](schemas/protocol-config-v1.json) and [configuration implementation](src/ppo_dap_paper_v6/config.py) for exact field names. Missing settings are rejected. Test configurations are examples for software checks, not recommended scientific settings.

## Run the checks

Use a checkout of `main` with **Python 3.12.3** and **uv 0.12.0** installed. From the repository root:

```bash
cd experiments/paper-v6
uv sync --frozen --all-groups
uv run pytest
```

These tests check configuration, dataset handling, seeds, and the pinned library version. They do not run environment interaction or PPO-DAP training.

The library is pinned to commit [`31dac8148a84204b9db506909edd8fb92822fcba`](https://github.com/TianciGao/DiffPPO/commit/31dac8148a84204b9db506909edd8fb92822fcba). Keep this version identifier with any experiment records.
