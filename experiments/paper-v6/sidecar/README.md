# Running an environment in a separate process

This subproject defines how the PPO-DAP experiment tools communicate with an environment running in another process. The code calls that process a **sidecar**. Keeping it separate allows an older simulator to use its own Python environment and dependencies.

The included backend is a deterministic simulator for software tests. It requires the explicit `--fixture-only-non-scientific` flag and does not represent a MuJoCo task. A real environment backend and its fixed dependency versions still need to be supplied.

## Message format

Messages use JSON metadata and named byte payloads, with SHA-256 checksums to check their integrity. They do not transfer executable Python objects or shell commands. The protocol does not use Python pickle serialization.

This subproject itself has no Gym, D4RL, MuJoCo, PyTorch, or CUDA dependency. Its integration tests run through the [parent experiment project](../README.md#run-the-checks).
