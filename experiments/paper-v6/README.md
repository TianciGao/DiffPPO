# PPO-DAP paper-v6 experiment harness

This subproject is the fail-closed harness for prospective paper-v6
experiments. Its algorithm authority remains the immutable public release
`TianciGao/DiffPPO@v0.1.0`, commit
`31dac8148a84204b9db506909edd8fb92822fcba`.

The completed E1 S1/S2/S3 harness provides:

- fail-closed protocol configuration, canonical run and dataset manifests,
  and deterministic nonalias RNG stream identities and seed derivation;
- a versioned, digest-bound, non-pickle legacy sidecar transport with a
  deterministic `fixture_only_non_scientific` backend;
- a G7 environment adapter and exact opaque checkpoint bridge;
- a Stage-I public-carrier builder and explicit launcher boundary;
- a Stage-II all-required public-authority builder;
- validation and ownership binding for caller-supplied actor and critic
  modules, without a default scientific architecture;
- deterministic mean-action evaluation;
- ALC@40, Student-t 95% confidence intervals, and matched-pair Wilcoxon
  metrics;
- report-only resource monitoring and immutable artifact publication.

E1 does not include a real Gym, D4RL, MuJoCo, or other scientific environment
backend. No real environment has been constructed or executed, no dataset has
been downloaded, no Stage-I or Stage-II training has run, and no GPU workload
has run. The harness supplies no scientific defaults, and every test recipe or
backend value is marked `fixture_only_non_scientific`.

The unresolved scientific choices D01, D02, D05, and D06 must be supplied in
an explicit protocol document before the corresponding real experiment.
Missing values are contract violations; they are never inferred.

This harness is not evidence that the paper experiments have been completed:
`empirical_reproduction=false`.

## Bounded harness checks

From this directory, with Python 3.12.3 and uv 0.12.0:

```console
uv sync --frozen --all-groups
uv run pytest
```

These checks are bounded unit and release-identity tests. They use only the
fixture sidecar backend; they do not construct a real environment, consume real
environment steps, use a GPU, or run PPO-DAP training.
