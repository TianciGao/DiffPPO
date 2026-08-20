"""Canonical G3.13 incomplete actor-objective interface guard."""

import pytest

from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.estimators.ppo import ppo_loss
from ppo_dap.interfaces.actor_objective import (
    require_complete_actor_objective_dependencies,
)
from tests.g3.test_ppo_component import _live_gaussian, _ppo_setup
from tests.g3.test_sealed_batch import _DEVICE, _DTYPE


def test_g3_actor_full_objective_interface_guard() -> None:
    setup = _ppo_setup()
    distribution, _, _ = _live_gaussian(setup["sealed"])
    component = ppo_loss(
        setup["view"],  # type: ignore[arg-type]
        distribution,
        live_state_ids=setup["view"].state_ids,  # type: ignore[union-attr]
        actor_reference_id="actor-interface",
        actor_reference_version="epoch-1",
        dtype=_DTYPE,
        device=_DEVICE,
    )
    with pytest.raises(
        ContractViolation,
        match="actor_objective.incomplete_dependencies",
    ):
        require_complete_actor_objective_dependencies(component)
    with pytest.raises(ContractViolation, match="actor_objective.component"):
        require_complete_actor_objective_dependencies({"ppo": component})  # type: ignore[arg-type]
    for forbidden_name in (
        "actor_loss",
        "auxiliary_loss",
        "prior_kl",
        "optimizer",
        "backward",
        "step",
    ):
        assert not hasattr(component, forbidden_name)
