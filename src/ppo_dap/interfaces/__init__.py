"""Fail-closed boundaries for intentionally incomplete G3 compositions."""

from ppo_dap.interfaces.actor_composition import (
    ActorParameterManifest,
    ActorThetaOwner,
)
from ppo_dap.interfaces.actor_objective import (
    require_complete_actor_objective_dependencies,
)
from ppo_dap.interfaces.critic_composition import (
    EntryBoundQSnapshot,
    SharedPhiCriticOwner,
    require_complete_critic_composition,
)
from ppo_dap.interfaces.current_batch_safety import require_later_batch_only_response
from ppo_dap.interfaces.pet_authority import (
    CommittedPETStateAuthority,
    PETConfigId,
    PETInitializationAuthority,
    PETOwnerAuthorityId,
    bind_committed_pet_state_authority,
    bind_pet_config_id,
    bind_pet_owner_authority_id,
    initialize_pet_lora_authority,
)

__all__ = [
    "require_complete_actor_objective_dependencies",
    "ActorParameterManifest",
    "ActorThetaOwner",
    "require_complete_critic_composition",
    "SharedPhiCriticOwner",
    "EntryBoundQSnapshot",
    "require_later_batch_only_response",
    "PETOwnerAuthorityId",
    "PETConfigId",
    "PETInitializationAuthority",
    "CommittedPETStateAuthority",
    "bind_pet_owner_authority_id",
    "bind_pet_config_id",
    "initialize_pet_lora_authority",
    "bind_committed_pet_state_authority",
]
