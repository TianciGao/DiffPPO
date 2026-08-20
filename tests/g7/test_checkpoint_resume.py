"""Focused native-Linux evidence for exact same-run G7.S5 checkpoint/resume."""

from __future__ import annotations

import base64
import gc
import hashlib
import json
import os
import subprocess
import sys
import threading
import types
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path

import pytest
import torch

from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.runtime import g7_checkpoint as _checkpoint
from ppo_dap.runtime.g7_checkpoint import (
    G7CheckpointResumeDependencies,
    _restore_tensor,
    _tensor,
    checkpoint_g7_stage_ii_trainer,
    resume_g7_stage_ii_trainer,
)
from ppo_dap.runtime.g7_environment import G7EnvironmentCheckpointState
from ppo_dap.runtime.g7_stage_ii import G7StageIINextIterationInput
from tests.g7.test_bundle_factory import _runtime_inputs
from tests.g7.test_stage_ii_outer_loop import (
    _align_auxiliary_forbidden_projection,
    _align_no_vg_actor_profile,
    _initial_trainer,
    _next_input,
)


def _environment_state(env: object) -> G7EnvironmentCheckpointState:
    opaque = json.dumps(
        {
            "episodes": env._episodes,
            "global_step": env._global_step,
            "reset_counts": env._reset_counts,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    schema = "g7-test-exact-environment-v1"
    config = env.environment_configuration_id.encode()
    instance = env.environment_instance_id.encode()
    digest = hashlib.sha256(
        b"PPO_DAP_G7_ENVIRONMENT_CHECKPOINT_V1\x00"
        + len(schema.encode()).to_bytes(8, "big")
        + schema.encode()
        + len(config).to_bytes(8, "big")
        + config
        + len(instance).to_bytes(8, "big")
        + instance
        + len(opaque).to_bytes(8, "big")
        + opaque
    ).digest()
    return G7EnvironmentCheckpointState(
        schema_version=schema,
        environment_configuration_id=env.environment_configuration_id,
        environment_instance_id=env.environment_instance_id,
        opaque_state=opaque,
        canonical_digest=digest,
    )


def _checkpoint_environment(env: object, *, after_capture: object | None = None) -> None:
    lock = threading.RLock()
    captures = {"count": 0}

    @contextmanager
    def checkpoint_guard(self):
        with lock:
            yield

    def capture_checkpoint_state(self):
        captures["count"] += 1
        state = _environment_state(self)
        if callable(after_capture):
            after_capture(captures["count"])
        return state

    def restore_checkpoint_state(self, state):
        if type(state) is not G7EnvironmentCheckpointState:
            raise ContractViolation("test.environment", "foreign checkpoint state")
        values = json.loads(state.opaque_state)
        self._episodes = values["episodes"]
        self._global_step = values["global_step"]
        self._reset_counts = values["reset_counts"]

    env.checkpoint_guard = types.MethodType(checkpoint_guard, env)
    env.capture_checkpoint_state = types.MethodType(capture_checkpoint_state, env)
    env.restore_checkpoint_state = types.MethodType(restore_checkpoint_state, env)
    env._checkpoint_capture_count = captures


def _dependencies(trainer, fields, environment) -> G7CheckpointResumeDependencies:
    actor = trainer._actor_owner
    critic = trainer._critic_owner
    pet = trainer._current._pet
    return G7CheckpointResumeDependencies(
        config=trainer._config,
        environment=environment,
        adapter=trainer._adapter,
        actor_module=actor._module,
        actor_density_config_id=actor.density_config_id,
        actor_function_identity=actor.function_identity,
        actor_parameter_manifest=actor.parameter_manifest,
        actor_forbidden_parameter_objects=actor._forbidden_parameter_objects,
        critic_module=critic._module,
        critic_function_identity=critic.function_identity,
        critic_shared_parameter_manifest=critic.shared_parameter_manifest,
        critic_value_parameter_manifest=critic.value_parameter_manifest,
        critic_q_parameter_manifest=critic.q_parameter_manifest,
        pet_training_noise_spec=pet._training_noise_spec,
        pet_module=pet._module,
        pet_architecture_spec=pet._architecture_spec,
        pet_instance_id=pet._instance_id,
        pet_parameter_manifest=pet._parameter_manifest,
        pet_target_manifest=pet._pet_target_manifest,
        pet_parameter_view=pet._pet_parameter_view,
        monitoring_recipe=trainer._monitoring_recipe,
        lambda_q=trainer._lambda_q,
        proxy_owner_identity=trainer._proxy_owner_identity,
    )


def _next_after_resume(trainer, fixture, environment, profile: str):
    owner = trainer._production_rng_owner
    fixture["stack"]["raw_rng"] = owner._raw._generator
    fixture["stack"]["eq7_rng"] = owner._eq7._generator
    fixture["stack"]["raw_binding"]._reverse_sampler_rng_binding = owner._raw._current_binding
    fixture["pet"] = trainer._resume_graph[5]
    legacy_state = fixture["stack"]["legacy_reverse_rng"].get_state()
    restored_legacy = tuple(
        generator
        for generator in fixture["pet"]._forbidden_generators
        if torch.equal(generator.get_state(), legacy_state)
    )
    assert len(restored_legacy) == 1
    fixture["stack"]["legacy_reverse_rng"] = restored_legacy[0]
    if owner._guided is not None:
        fixture["stack"]["guided_rng"] = owner._guided._generator
        fixture["stack"][
            "proposal"
        ]._proposal_binding._guided_reverse_rng_binding = owner._guided._current_binding
    boundary = trainer._checkpoint_boundary
    source = trainer.current_state
    request, _ = _runtime_inputs(
        fixture=fixture,
        environment=environment,
        source_state=source,
        mode="successor",
        current_environment_binding=trainer._resume_graph[0],
        current_pet_binding=trainer._resume_graph[5],
        completed_report=boundary,
        profile=profile,
    )
    fields = dict(request._fields)
    _align_no_vg_actor_profile(fields, profile)
    _align_auxiliary_forbidden_projection(fields)
    return G7StageIINextIterationInput(
        expected_current_state=source,
        expected_production_rng_generation=owner.generation,
        on_policy_batch_id=fields["plan"].batch_id,
        plan=fields["plan"],
        slot_schedule=fields["slot_schedule"],
        raw_state_owner_identity=fields["raw_state_owner_identity"],
        guided_state_owner_identity=fields["guided_state_owner_identity"],
        raw_spec=fields["raw_spec"],
        pet_snapshot=fields["pet_snapshot"],
        prior_inference_snapshot=fields["prior_inference_snapshot"],
        eq7_config=fields["eq7_config"],
        eq8_config=fields["eq8_config"],
        actor_config=fields["actor_config"],
        auxiliary_selection_rng=fields["auxiliary_selection_rng"],
        production_forbidden_generators=fields["production_forbidden_generators"],
        g6_request=fields["g6_request"],
        g6_raw_rng=fields["g6_raw_rng"],
        g6_raw_binding=fields["g6_raw_binding"],
        g6_guided_rng=fields["g6_guided_rng"],
        g6_guided_binding=fields["g6_guided_binding"],
        g6_eq7_binding=fields["g6_eq7_binding"],
        g6_auxiliary_binding=fields["g6_auxiliary_binding"],
        g6_forbidden_generators=fields["g6_forbidden_generators"],
    )


def test_checkpoint_boundary_and_atomic_generation_publication(tmp_path: Path) -> None:
    trainer, _, environment, _ = _initial_trainer("no_vg")
    _checkpoint_environment(environment)
    with pytest.raises(ContractViolation, match="ready_for_successor"):
        checkpoint_g7_stage_ii_trainer(trainer, checkpoint_root=tmp_path)
    trainer.run_initial()
    assert checkpoint_g7_stage_ii_trainer(trainer, checkpoint_root=tmp_path) == 0
    first = (tmp_path / "CURRENT").read_bytes()
    assert checkpoint_g7_stage_ii_trainer(trainer, checkpoint_root=tmp_path) == 1
    assert first != (tmp_path / "CURRENT").read_bytes()
    assert trainer.lifecycle == "ready_for_successor"
    assert trainer._production_rng_owner.lifecycle == "ready"
    assert trainer._production_rng_owner._active is None
    assert environment._checkpoint_capture_count["count"] == 4
    image = json.loads((tmp_path / "generations" / f"{1:020d}" / "image.json").read_text())
    assert set(image) == {
        "schema",
        "runtime_fingerprint",
        "generation",
        "parent_generation",
        "run_id",
        "completed_iteration",
        "completed_batch",
        "training_state",
        "config_evidence",
        "monitoring_evidence",
        "static_root",
        "lambda_q",
        "proxy_owner_identity",
        "actor",
        "critic",
        "pet",
        "production_rng",
        "behavior_rng",
        "s1",
        "environment",
        "g6_generation",
        "environment_s1_root",
    }
    manifest = json.loads((tmp_path / "generations" / f"{1:020d}" / "manifest.json").read_text())
    assert set(manifest["component_digests"]) == set(_checkpoint._COMPONENT_DIGEST_DOMAINS)
    assert manifest["static_dependency_digest"] == image["static_root"]
    assert manifest["root_digest"]
    assert not any("bundle" in path.name for path in tmp_path.rglob("*"))


@pytest.mark.parametrize(
    "target",
    [
        "actor_content",
        "actor_gradient",
        "critic_content",
        "pet_content",
        "pet_remainder",
        "production_rng",
        "behavior_rng",
        "environment",
    ],
)
def test_capture_coherence_drift_fails_without_publication(
    tmp_path: Path,
    target: str,
) -> None:
    trainer, _, environment, _ = _initial_trainer("full_default")
    trainer.run_initial()
    actor = next(trainer._actor_owner._module.parameters())
    critic = next(trainer._critic_owner._module.parameters())
    pet_binding = trainer._current._pet
    pet = pet_binding._pet_parameter_view.ordered_parameters[0]
    raw = trainer._production_rng_owner._raw._generator
    behavior = trainer._current._environment._owner._rng._generator
    actor_content = actor.detach().clone()
    critic_content = critic.detach().clone()
    pet_content = pet.detach().clone()
    pet_remainder = pet_binding._credit_remainder
    raw_state = raw.get_state().clone()
    behavior_state = behavior.get_state().clone()
    environment_step = environment._global_step

    def drift(count: int) -> None:
        trigger = 1 if target == "environment" else 2
        if count != trigger:
            return
        with torch.no_grad():
            if target == "actor_content":
                actor.add_(1.0)
            elif target == "actor_gradient":
                actor.grad = torch.zeros_like(actor)
            elif target == "critic_content":
                critic.add_(1.0)
            elif target == "pet_content":
                pet.add_(1.0)
            elif target == "pet_remainder":
                pet_binding._credit_remainder = pet_remainder + Fraction(1, 7)
            elif target == "production_rng":
                torch.rand((), generator=raw)
            elif target == "behavior_rng":
                torch.rand((), generator=behavior)
            elif target == "environment":
                environment._global_step += 1

    _checkpoint_environment(environment, after_capture=drift)
    try:
        with pytest.raises(ContractViolation):
            checkpoint_g7_stage_ii_trainer(trainer, checkpoint_root=tmp_path)
    finally:
        with torch.no_grad():
            actor.copy_(actor_content)
            actor.grad = None
            critic.copy_(critic_content)
            pet.copy_(pet_content)
            pet_binding._credit_remainder = pet_remainder
            raw.set_state(raw_state)
            behavior.set_state(behavior_state)
            environment._global_step = environment_step
    assert trainer.lifecycle == "ready_for_successor"
    assert not (tmp_path / "CURRENT").exists()


def test_same_process_live_claim_fails_closed(tmp_path: Path) -> None:
    from ppo_dap.interfaces import actor_composition, pet_authority
    from ppo_dap.prior import noise as noise_module

    trainer, _, environment, fields = _initial_trainer("no_vg")
    _checkpoint_environment(environment)
    trainer.run_initial()
    checkpoint_g7_stage_ii_trainer(trainer, checkpoint_root=tmp_path)
    dependencies = _dependencies(trainer, fields, environment)
    actor_registry = tuple(actor_composition._PARAMETER_OWNER_REGISTRY)
    pet_registry = pet_authority._AUTHORITY_STATE
    forward = tuple(noise_module._FORWARD_REGISTRY.items())
    reverse = tuple(noise_module._REVERSE_REGISTRY)
    seals = tuple(noise_module._BINDING_SEALS.items())
    with pytest.raises(ContractViolation):
        resume_g7_stage_ii_trainer(
            checkpoint_root=tmp_path,
            dependencies=dependencies,
        )
    gc.collect()
    assert tuple(actor_composition._PARAMETER_OWNER_REGISTRY) == actor_registry
    assert pet_authority._AUTHORITY_STATE is pet_registry
    assert tuple(noise_module._FORWARD_REGISTRY.items()) == forward
    assert tuple(noise_module._REVERSE_REGISTRY) == reverse
    assert tuple(noise_module._BINDING_SEALS.items()) == seals
    assert trainer.lifecycle == "ready_for_successor"


def _producer(root: str, profile: str) -> None:
    trainer, fixture, environment, _ = _initial_trainer(profile)
    _checkpoint_environment(environment)
    report = trainer.run_initial()
    checkpoint_g7_stage_ii_trainer(trainer, checkpoint_root=root)
    next_iteration, _ = _next_input(trainer, fixture, environment, profile)
    uninterrupted = trainer.run_next(next_iteration)
    reference_root = Path(root, "uninterrupted_reference")
    checkpoint_g7_stage_ii_trainer(trainer, checkpoint_root=reference_root)
    Path(root, "producer.json").write_text(
        json.dumps(
            {
                "iteration": report.committed_state.iteration_index,
                "reference_iteration": uninterrupted.committed_state.iteration_index,
            }
        )
    )


def _resume_process(root: str, profile: str) -> None:
    from ppo_dap.algorithm import state as algorithm_state
    from ppo_dap.interfaces import actor_composition, pet_authority
    from ppo_dap.prior import noise as noise_module

    actor_baseline = list(actor_composition._PARAMETER_OWNER_REGISTRY)
    pet_baseline = pet_authority._AUTHORITY_STATE
    committed_baseline = dict(algorithm_state._COMMITTED_PET_STATE_INSTANCES)
    forward_baseline = tuple(noise_module._FORWARD_REGISTRY.items())
    reverse_baseline = dict(noise_module._REVERSE_REGISTRY)
    seals_baseline = tuple(noise_module._BINDING_SEALS.items())
    template, fixture, environment, fields = _initial_trainer(profile)
    _checkpoint_environment(environment)
    dependencies = _dependencies(template, fields, environment)
    actor_composition._PARAMETER_OWNER_REGISTRY = actor_baseline
    pet_authority._AUTHORITY_STATE = pet_baseline
    algorithm_state._COMMITTED_PET_STATE_INSTANCES = committed_baseline
    noise_module._FORWARD_REGISTRY.clear()
    noise_module._FORWARD_REGISTRY.update(forward_baseline)
    noise_module._REVERSE_REGISTRY.clear()
    noise_module._REVERSE_REGISTRY.update(reverse_baseline)
    noise_module._BINDING_SEALS.clear()
    noise_module._BINDING_SEALS.update(seals_baseline)
    del template, fields
    gc.collect()
    current = json.loads(Path(root, "CURRENT").read_text())
    image = json.loads(
        Path(root, "generations", f"{current['generation']:020d}", "image.json").read_text()
    )
    global_entry = torch.default_generator.get_state().clone()
    trainer = resume_g7_stage_ii_trainer(
        checkpoint_root=root,
        dependencies=dependencies,
    )
    assert torch.equal(torch.default_generator.get_state(), global_entry)
    assert trainer.lifecycle == "ready_for_successor"
    assert trainer._runner._phase == "admitted"
    assert trainer.reports == ()
    assert trainer._checkpoint_boundary is not None
    restored_pet = trainer._resume_graph[5]
    current_pet = restored_pet._borrow_current_committed_state_for_snapshot()
    assert current_pet.initialization_authority.canonical_evidence == base64.b64decode(
        image["pet"]["initialization_evidence"]
    )
    assert current_pet.canonical_evidence == base64.b64decode(image["pet"]["committed_evidence"])
    topology = image["production_rng"]["topology"]
    all_generators = {
        trainer._production_rng_owner._raw._generator,
        trainer._production_rng_owner._eq7._generator,
        trainer._resume_graph[0]._owner._rng._generator,
        restored_pet._sigma_rng,
        restored_pet._epsilon_rng,
        *trainer._production_rng_owner._forbidden_generators,
        *trainer._resume_graph[0]._owner._rng._forbidden_generators,
        *restored_pet._forbidden_generators,
    }
    assert len(all_generators) == len(topology["generators"])
    unmatched = list(all_generators)
    for record in topology["generators"].values():
        expected = _restore_tensor(record["state"], expected_device=torch.device("cpu"))
        match = next(
            (generator for generator in unmatched if torch.equal(generator.get_state(), expected)),
            None,
        )
        assert match is not None
        unmatched.remove(match)
    assert unmatched == []
    assert checkpoint_g7_stage_ii_trainer(trainer, checkpoint_root=root) == 1
    restored_image = json.loads(Path(root, "generations", f"{1:020d}", "image.json").read_text())
    for section in (
        "run_id",
        "completed_iteration",
        "completed_batch",
        "training_state",
        "static_root",
        "actor",
        "critic",
        "pet",
        "production_rng",
        "behavior_rng",
        "s1",
        "environment",
        "g6_generation",
    ):
        assert restored_image[section] == image[section]
    reset_before = len(environment.reset_slots)
    report = trainer.run_next(_next_after_resume(trainer, fixture, environment, profile))
    assert report.commit_succeeded
    assert trainer.lifecycle == "ready_for_successor"
    assert trainer._checkpoint_boundary is None
    assert len(environment.reset_slots) == reset_before
    assert checkpoint_g7_stage_ii_trainer(trainer, checkpoint_root=root) == 2
    resumed_image = json.loads(Path(root, "generations", f"{2:020d}", "image.json").read_text())
    reference_image = json.loads(
        Path(
            root,
            "uninterrupted_reference",
            "generations",
            f"{0:020d}",
            "image.json",
        ).read_text()
    )
    for section in (
        "runtime_fingerprint",
        "run_id",
        "completed_iteration",
        "completed_batch",
        "training_state",
        "config_evidence",
        "monitoring_evidence",
        "static_root",
        "lambda_q",
        "proxy_owner_identity",
        "actor",
        "critic",
        "pet",
        "production_rng",
        "behavior_rng",
        "s1",
        "environment",
        "g6_generation",
    ):
        assert resumed_image[section] == reference_image[section]
    second, _ = _next_input(trainer, fixture, environment, profile)
    second_report = trainer.run_next(second)
    assert (
        second_report.committed_state.iteration_index == report.committed_state.iteration_index + 1
    )
    assert trainer._runner._phase == "admitted"
    assert len(environment.reset_slots) == reset_before
    Path(root, "resume.json").write_text(
        json.dumps(
            {
                "iteration": report.committed_state.iteration_index,
                "second_iteration": second_report.committed_state.iteration_index,
                "runner_phase": trainer._runner._phase,
                "pet_phase": trainer._current._pet._phase,
                "global_rng_unchanged": torch.equal(
                    torch.default_generator.get_state(), global_entry
                ),
            }
        )
    )


def _failed_restore_process(root: str, profile: str) -> None:
    from ppo_dap.algorithm import state as algorithm_state
    from ppo_dap.interfaces import actor_composition, pet_authority
    from ppo_dap.prior import noise as noise_module

    actor_baseline = list(actor_composition._PARAMETER_OWNER_REGISTRY)
    pet_baseline = pet_authority._AUTHORITY_STATE
    committed_baseline = dict(algorithm_state._COMMITTED_PET_STATE_INSTANCES)
    forward_baseline = tuple(noise_module._FORWARD_REGISTRY.items())
    reverse_baseline = dict(noise_module._REVERSE_REGISTRY)
    seals_baseline = tuple(noise_module._BINDING_SEALS.items())
    template, fixture, environment, fields = _initial_trainer(profile)
    _checkpoint_environment(environment)
    dependencies = _dependencies(template, fields, environment)
    actor_composition._PARAMETER_OWNER_REGISTRY = actor_baseline
    pet_authority._AUTHORITY_STATE = pet_baseline
    algorithm_state._COMMITTED_PET_STATE_INSTANCES = committed_baseline
    noise_module._FORWARD_REGISTRY.clear()
    noise_module._FORWARD_REGISTRY.update(forward_baseline)
    noise_module._REVERSE_REGISTRY.clear()
    noise_module._REVERSE_REGISTRY.update(reverse_baseline)
    noise_module._BINDING_SEALS.clear()
    noise_module._BINDING_SEALS.update(seals_baseline)
    del template, fixture, fields
    gc.collect()
    restore = environment.restore_checkpoint_state

    def mismatching_restore(self, state):
        restore(state)
        self._global_step += 1

    environment.restore_checkpoint_state = types.MethodType(mismatching_restore, environment)
    try:
        resume_g7_stage_ii_trainer(checkpoint_root=root, dependencies=dependencies)
    except ContractViolation:
        pass
    else:
        raise AssertionError("mismatching environment restore was published")
    gc.collect()
    assert actor_composition._PARAMETER_OWNER_REGISTRY == actor_baseline
    assert pet_authority._AUTHORITY_STATE is pet_baseline
    assert algorithm_state._COMMITTED_PET_STATE_INSTANCES == committed_baseline
    assert tuple(noise_module._FORWARD_REGISTRY.items()) == forward_baseline
    assert noise_module._REVERSE_REGISTRY == reverse_baseline
    assert tuple(noise_module._BINDING_SEALS.items()) == seals_baseline
    Path(root, "failed_restore.json").write_text(json.dumps({"clean": True}))


@pytest.mark.parametrize("profile", ["no_vg", "full_default"])
def test_fresh_process_resume_executes_real_successor(tmp_path: Path, profile: str) -> None:
    environment = dict(os.environ)
    repo = Path(__file__).resolve().parents[2]
    environment["PYTHONPATH"] = os.pathsep.join((str(repo / "src"), str(repo)))
    for mode in ("produce", "resume"):
        completed = subprocess.run(
            [sys.executable, __file__, mode, str(tmp_path), profile],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
    produced = json.loads((tmp_path / "producer.json").read_text())
    resumed = json.loads((tmp_path / "resume.json").read_text())
    assert resumed["iteration"] == produced["iteration"] + 1
    assert resumed["iteration"] == produced["reference_iteration"]
    assert resumed["second_iteration"] == produced["iteration"] + 2
    assert resumed["runner_phase"] == "admitted"
    assert resumed["pet_phase"] == "active"
    assert resumed["global_rng_unchanged"] is True


def test_corrupt_current_fails_closed_without_fallback(tmp_path: Path) -> None:
    trainer, _, environment, fields = _initial_trainer("no_vg")
    _checkpoint_environment(environment)
    trainer.run_initial()
    checkpoint_g7_stage_ii_trainer(trainer, checkpoint_root=tmp_path)
    checkpoint_g7_stage_ii_trainer(trainer, checkpoint_root=tmp_path)
    current = json.loads((tmp_path / "CURRENT").read_text())
    assert current["generation"] == 1
    manifest = tmp_path / "generations" / f"{1:020d}" / "manifest.json"
    manifest.write_text("{}")
    with pytest.raises(ContractViolation, match="manifest/digest differs"):
        resume_g7_stage_ii_trainer(
            checkpoint_root=tmp_path,
            dependencies=_dependencies(trainer, fields, environment),
        )
    assert (tmp_path / "generations" / f"{0:020d}" / "manifest.json").is_file()
    assert json.loads((tmp_path / "CURRENT").read_text())["generation"] == 1
    assert trainer.lifecycle == "ready_for_successor"


def test_failed_restore_before_final_claim_leaves_registries_clean(tmp_path: Path) -> None:
    environment = dict(os.environ)
    repo = Path(__file__).resolve().parents[2]
    environment["PYTHONPATH"] = os.pathsep.join((str(repo / "src"), str(repo)))
    for mode in ("produce", "failed_restore"):
        completed = subprocess.run(
            [sys.executable, __file__, mode, str(tmp_path), "full_default"],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads((tmp_path / "failed_restore.json").read_text()) == {"clean": True}


def test_tensor_checkpoint_preserves_noncontiguous_stride() -> None:
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4).transpose(0, 1)
    assert source.is_contiguous() is False
    restored = _restore_tensor(_tensor(source), expected_device=torch.device("cpu"))
    assert tuple(restored.shape) == tuple(source.shape)
    assert tuple(restored.stride()) == tuple(source.stride())
    assert torch.equal(restored, source)


@pytest.mark.parametrize(
    "domain",
    ["manifest", "incomplete", "actor", "critic", "pet", "rng", "run", "environment"],
)
def test_corrupt_checkpoint_domains_fail_before_restore_claim(
    tmp_path: Path,
    domain: str,
) -> None:
    trainer, _, environment, fields = _initial_trainer("no_vg")
    _checkpoint_environment(environment)
    trainer.run_initial()
    checkpoint_g7_stage_ii_trainer(trainer, checkpoint_root=tmp_path)
    generation = tmp_path / "generations" / f"{0:020d}"
    image_path = generation / "image.json"
    manifest_path = generation / "manifest.json"
    if domain == "manifest":
        manifest_path.write_text("{}")
    elif domain == "incomplete":
        image_path.unlink()
    else:
        image = json.loads(image_path.read_text())
        if domain == "actor":
            image["actor"]["parameters"][0][1]["content"] = "AA=="
        elif domain == "critic":
            image["critic"]["parameters"][0][1]["content"] = "AA=="
        elif domain == "pet":
            image["pet"]["current_content"][0]["content"] = "AA=="
        elif domain == "rng":
            image["production_rng"]["raw"]["ordinal"] += 1
        elif domain == "run":
            image["run_id"] += "-foreign"
        elif domain == "environment":
            image["environment"]["configuration_id"] += "-foreign"
        image_path.write_text(json.dumps(image, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ContractViolation):
        resume_g7_stage_ii_trainer(
            checkpoint_root=tmp_path,
            dependencies=_dependencies(trainer, fields, environment),
        )
    assert trainer.lifecycle == "ready_for_successor"


def test_forbidden_checkpoint_phases_fail_without_claiming_input(tmp_path: Path) -> None:
    trainer, _, environment, _ = _initial_trainer("no_vg")
    _checkpoint_environment(environment)
    trainer.run_initial()
    for phase in ("running", "committed_pending_rng_ack", "failed_terminal", "ready_initial"):
        object.__setattr__(trainer, "_lifecycle", phase)
        with pytest.raises(ContractViolation, match="ready_for_successor"):
            checkpoint_g7_stage_ii_trainer(trainer, checkpoint_root=tmp_path)
        assert not (tmp_path / "CURRENT").exists()
    object.__setattr__(trainer, "_lifecycle", "ready_for_successor")
    assert trainer._production_rng_owner._active is None


def test_written_manifest_is_revalidated_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, _, environment, _ = _initial_trainer("no_vg")
    _checkpoint_environment(environment)
    trainer.run_initial()
    read_bytes = Path.read_bytes

    def corrupt_written_manifest(value: Path) -> bytes:
        payload = read_bytes(value)
        return b"{}" if value.name == "manifest.json" else payload

    with monkeypatch.context() as context:
        context.setattr(Path, "read_bytes", corrupt_written_manifest)
        with pytest.raises(ContractViolation, match="manifest/digest differs"):
            checkpoint_g7_stage_ii_trainer(trainer, checkpoint_root=tmp_path)
    assert not (tmp_path / "CURRENT").exists()
    assert trainer.lifecycle == "ready_for_successor"


def test_failed_publication_preserves_previous_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, _, environment, _ = _initial_trainer("no_vg")
    _checkpoint_environment(environment)
    trainer.run_initial()
    assert checkpoint_g7_stage_ii_trainer(trainer, checkpoint_root=tmp_path) == 0
    previous = (tmp_path / "CURRENT").read_bytes()
    replace = _checkpoint.os.replace

    def fail_current(source, destination):
        if Path(destination).name == "CURRENT":
            raise OSError("injected current publication failure")
        return replace(source, destination)

    with monkeypatch.context() as context:
        context.setattr(_checkpoint.os, "replace", fail_current)
        with pytest.raises(OSError, match="publication failure"):
            checkpoint_g7_stage_ii_trainer(trainer, checkpoint_root=tmp_path)
    assert (tmp_path / "CURRENT").read_bytes() == previous
    assert trainer.lifecycle == "ready_for_successor"
    assert checkpoint_g7_stage_ii_trainer(trainer, checkpoint_root=tmp_path) == 2
    current = json.loads((tmp_path / "CURRENT").read_text())
    manifest = json.loads(
        (tmp_path / "generations" / f"{current['generation']:020d}" / "manifest.json").read_text()
    )
    assert current["generation"] == 2
    assert manifest["parent_generation"] == 0


if __name__ == "__main__":
    command, root, profile = sys.argv[1:]
    if command == "produce":
        _producer(root, profile)
    elif command == "resume":
        _resume_process(root, profile)
    elif command == "failed_restore":
        _failed_restore_process(root, profile)
    else:
        raise SystemExit(2)
