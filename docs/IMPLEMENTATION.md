# Finding your way around the code

The `ppo_dap` Python package contains the PPO-DAP algorithm and runtime components. Release `v0.1.0` is a research library: using it for a full experiment requires environment integration, a dataset, model configuration, and an evaluation protocol.

Start with the [algorithm guide](ALGORITHM.md) for the method, then use the paths below to find its implementation.

## Source map

| Component | Source | What it does |
| --- | --- | --- |
| Actions | [`actions/`](../src/ppo_dap/actions/) | Represents actions and converts between model and environment action spaces. |
| Policy distributions | [`distributions/`](../src/ppo_dap/distributions/) | Implements Gaussian policy quantities and standard-deviation constraints. |
| PPO and value estimates | [`estimators/`](../src/ppo_dap/estimators/) | Computes advantages, PPO inputs, value estimates, and training targets. |
| Training losses | [`objectives/`](../src/ppo_dap/objectives/) | Defines actor, critic, and prior-adaptation objectives. |
| Diffusion prior | [`prior/`](../src/ppo_dap/prior/) | Implements denoising, prior training, and action sampling. |
| Value guidance | [`value_guidance/`](../src/ppo_dap/value_guidance/) | Weights candidate actions, applies denoising guidance, and constructs the Gaussian approximation. |
| Model integration | [`interfaces/`](../src/ppo_dap/interfaces/) | Connects components and checks which parameters each update may change. |
| Training iteration | [`algorithm/`](../src/ppo_dap/algorithm/) | Defines the order of operations and the data needed by one iteration. |
| Repeated execution | [`runtime/`](../src/ppo_dap/runtime/) | Handles environment interfaces, random-number streams, and checkpoint/resume. |
| Initialization | [`warm_start/`](../src/ppo_dap/warm_start/) | Provides explicitly configured initialization from recorded data. |
| Diagnostics | [`audit.py`](../src/ppo_dap/audit.py) | Reports training measurements without changing the training procedure. |

## Following a training update

[`algorithm/iteration.py`](../src/ppo_dap/algorithm/iteration.py) is the starting point for reading the training loop. It coordinates rollout collection, PPO preparation, action guidance, model updates, and reporting.

The modules in `objectives/` and `interfaces/` check data provenance and gradient flow. For example, generated action suggestions may contribute to an auxiliary actor loss, while PPO and the critic continue to use fresh environment data.

The `runtime/` modules repeat this process across iterations. An experiment must supply an environment adapter implementing the required operations, including state restoration if exact resume is needed.

## Running the tests

After following the [installation instructions](../README.md#install-and-check), run from the repository root:

```bash
uv run pytest
```

The original `v0.1.0` release record reports **350 tests passed, 0 failed**. The root command runs the algorithm suite; the experiment subproject has its own test command and dependencies.

| Test directory | Coverage area |
| --- | --- |
| [`tests/g3/`](../tests/g3/) | PPO, rollout data, and initialization. |
| [`tests/g4/`](../tests/g4/) | Diffusion-prior operations and model handoff. |
| [`tests/g5/`](../tests/g5/) | Guidance and integrated actor, critic, and prior updates. |
| [`tests/g6/`](../tests/g6/) | Diagnostics. |
| [`tests/g7/`](../tests/g7/) | Environment interfaces, random-number state, repeated execution, and resume. |

Numbered source filenames and test directories reflect development stages, not different PPO-DAP methods. They remain in place so existing imports and tests keep working. The tables above describe their actual purpose.

## Experiment development

The [experiment branch](https://github.com/TianciGao/DiffPPO/tree/experiment/paper-v6-e1/experiments/paper-v6) adds configuration, environment interfaces, evaluation, and result reporting around the fixed `v0.1.0` library. Its documentation distinguishes implemented tooling from the remaining work needed for real training runs.

The library's low-level interfaces may evolve in future releases. Pin a release or commit when building an experiment, and record the environment, dataset, configuration, and seeds alongside your results.

## Release records

- [Validation and limitations](THEORY_CONFORMANCE.md)
- [Release notes](releases/v0.1.0.md)
- [Original release file checksums and source history](PROVENANCE.json)

The checksum record describes the original `v0.1.0` files. Later documentation revisions are recorded in Git history.
