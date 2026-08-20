# PPO-DAP paper-v6 experiment foundation

This subproject is the fail-closed foundation for prospective paper-v6
experiments. The algorithm authority is the immutable public release
`TianciGao/DiffPPO@v0.1.0`, commit
`31dac8148a84204b9db506909edd8fb92822fcba`.

This slice contains only:

- all-required protocol configuration;
- canonical run and dataset manifests;
- deterministic RNG stream identities and seed derivation;
- the versioned schema for a future legacy-environment sidecar.

It contains no environment client/server, Stage-I or Stage-II execution,
evaluation loop, dataset download, or scientific configuration defaults.
Every test value is marked `fixture_only_non_scientific`.

The unresolved scientific choices D01, D02, D05, and D06 must be supplied in
an explicit protocol document before the corresponding real experiment.
Missing values are contract violations; they are never inferred.

## Foundation checks

From this directory, with Python 3.12.3 and uv 0.12.0:

```console
uv sync --frozen --all-groups
uv run pytest
```

These checks are bounded unit and identity tests. They do not construct an
environment, consume environment steps, use a GPU, or run PPO-DAP training.
