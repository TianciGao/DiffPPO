"""Canonical G3.13 incomplete critic-composition interface guard."""

import pytest
import torch

from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.estimators.v_core import value_loss
from ppo_dap.interfaces.critic_composition import require_complete_critic_composition
from tests.g3.test_sealed_batch import _DEVICE, _DTYPE
from tests.g3.test_v_core import _v_core_setup


def test_g3_critic_composition_interface_guard() -> None:
    setup = _v_core_setup()
    live_tensors = tuple(
        torch.tensor(float(index), dtype=_DTYPE, requires_grad=True) for index in range(6)
    )
    live_values = tuple(
        zip(setup["sealed"].state_ids, live_tensors, strict=True)  # type: ignore[union-attr]
    )
    component = value_loss(
        setup["sealed"],  # type: ignore[arg-type]
        setup["targets"],  # type: ignore[arg-type]
        live_values,
        critic_reference_id="critic-interface",
        critic_reference_version="epoch-1",
        dtype=_DTYPE,
        device=_DEVICE,
    )
    with pytest.raises(
        ContractViolation,
        match="critic_composition.incomplete_dependencies",
    ):
        require_complete_critic_composition(component)
    with pytest.raises(ContractViolation, match="critic_composition.component"):
        require_complete_critic_composition({"v_core": component})  # type: ignore[arg-type]
    for forbidden_name in (
        "q_head",
        "shared_trunk",
        "critic_loss",
        "optimizer",
        "backward",
        "step",
    ):
        assert not hasattr(component, forbidden_name)
