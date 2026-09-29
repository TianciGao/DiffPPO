"""Legacy-sidecar protocol with no scientific backend bundled."""

from ppo_dap_legacy_sidecar.backend import FixtureOnlyNonScientificBackend
from ppo_dap_legacy_sidecar.protocol import SCHEMA_VERSION, WIRE_PROTOCOL

__all__ = [
    "FixtureOnlyNonScientificBackend",
    "SCHEMA_VERSION",
    "WIRE_PROTOCOL",
]
