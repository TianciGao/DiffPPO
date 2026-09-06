# Experiment tools

This directory provides configuration, environment interfaces, evaluation, and reporting tools for [PPO-DAP paper version 6](https://arxiv.org/abs/2409.01427v6). It uses the fixed [`v0.1.0` algorithm library](https://github.com/TianciGao/DiffPPO/releases/tag/v0.1.0).

**The tools have been developed and tested with a simulated test backend. Real-environment training and full reproduction of the paper's results remain unfinished.**

## Available tools

| Area | What is implemented |
| --- | --- |
| Configuration and records | Required experiment settings, dataset and run records, and reproducible random-number streams. |
| Environment connection | A separate-process communication layer, an environment adapter, and checkpoint-state transfer. |
| Training integration | Input builders for prior pretraining and online learning, plus checks for caller-supplied actor and critic models. |
| Evaluation | Deterministic evaluation using the policy's mean action. |
| Metrics | Area under the learning curve over the first 40 epochs (ALC@40), Student-t 95% confidence intervals, and paired Wilcoxon comparisons. |
| Reporting | Resource measurements and output records protected against accidental overwriting. |

See the [source directory](src/ppo_dap_paper_v6/) and the [separate-process environment documentation](sidecar/README.md).

## What is needed for a real experiment

A real Gym, D4RL, or MuJoCo backend is not included. The current implementation record reports no real-environment execution, dataset download, prior pretraining, online training, or GPU workload.

Before training, provide a compatible environment and dataset, then document these settings:

| Area | Required choices |
| --- | --- |
| Repeated runs | Random seeds, number of runs, and matching of seeds across methods. |
| Evaluation | Evaluation frequency, episode count, time limits, and learning-curve measurement points. |
| Prior pretraining | Model size, number of training passes, learning rate, numeric precision, and noise settings. |
| Actor and critic | Network sizes, initialization, policy standard-deviation limits, discount factor, and update settings. |

The [configuration schema](schemas/protocol-config-v1.json) and [configuration code](src/ppo_dap_paper_v6/config.py) define the exact fields. Missing settings are rejected; the test recipes do not supply scientific defaults. The dependency set for the real environment must also be fixed and recorded.

## Run the checks

Use a checkout of `experiment/paper-v6-e1` with **Python 3.12.3** and **uv 0.12.0** installed. From the repository root:

```bash
cd experiments/paper-v6
uv sync --frozen --all-groups
uv run pytest
```

The tests use only the deterministic test backend. They check interfaces and library compatibility without running a real simulator, GPU workload, or PPO-DAP training experiment. The label `fixture_only_non_scientific` in test files means exactly that: software-test data only.

The algorithm dependency is pinned to commit [`31dac8148a84204b9db506909edd8fb92822fcba`](https://github.com/TianciGao/DiffPPO/commit/31dac8148a84204b9db506909edd8fb92822fcba). Pin the experiment branch to a commit as well when recording a run.
