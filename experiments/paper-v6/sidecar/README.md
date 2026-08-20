# PPO-DAP paper-v6 legacy sidecar

This isolated subproject supplies only the versioned, digest-bound subprocess
protocol and a deterministic test backend.  It has no Gym, D4RL, MuJoCo,
Torch, or CUDA dependency.

The bundled backend can start only with the explicit
`--fixture-only-non-scientific` flag.  It is test infrastructure, not a
scientific environment and not a default for an experiment run.  A real
paper-era environment and its independently frozen dependency lock belong to
E2.

Messages contain canonical JSON metadata and named, SHA256-bound byte
payloads.  Pickle, runtime object graphs, shell commands, and serialized
remote exception objects are outside the protocol.
