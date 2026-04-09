from __future__ import annotations

from agenttree.core.registry import NodeRegistry
from agenttree.schemas.nodes import (
    BindExecutorRequest,
    CreateAgentRequest,
    MoveNodeRequest,
    NodeKind,
    RegisterExecutorRequest,
    TransferExecutorRequest,
    TriggerSpec,
    UpdateNodeRequest,
)


def test_registry_creates_agent_and_registers_executor() -> None:
    registry = NodeRegistry()
    agent = registry.create_agent(
        CreateAgentRequest(
            parent_path="/",
            name="root_manager",
            prompt="manage subtree",
        )
    )
    executor = registry.register_executor(
        RegisterExecutorRequest(
            path="/temperature.executor",
            executor_kind="demo.temperature",
            capabilities=["event_push"],
            metadata={"external": True},
        )
    )

    assert agent.kind == NodeKind.AGENT
    assert executor.kind == NodeKind.EXECUTOR
    assert executor.owner_path is None
    assert executor.path == "/temperature.executor"


def test_registry_binds_and_transfers_executor_owner() -> None:
    registry = NodeRegistry()
    parent = registry.create_agent(
        CreateAgentRequest(parent_path="/", name="root_manager", prompt="manage subtree")
    )
    new_owner = registry.create_agent(
        CreateAgentRequest(parent_path=parent.path, name="operator", prompt="operate")
    )
    executor = registry.register_executor(
        RegisterExecutorRequest(path="/temperature.executor", executor_kind="demo.temperature")
    )

    bound = registry.bind_executor(BindExecutorRequest(executor_path=executor.path, owner_path=parent.path))
    assert bound.owner_path == parent.path

    updated = registry.transfer_executor(
        TransferExecutorRequest(executor_path=executor.path, new_owner_path=new_owner.path)
    )

    assert updated.owner_path == new_owner.path
    assert updated.path == "/temperature.executor"


def test_registry_updates_node_and_triggers() -> None:
    registry = NodeRegistry()
    agent = registry.create_agent(
        CreateAgentRequest(parent_path="/", name="root_manager", prompt="manage subtree")
    )

    updated = registry.update_node(
        agent.path,
        UpdateNodeRequest(prompt="new prompt", description="updated description", metadata={"tier": 1}),
    )
    with_trigger = registry.upsert_trigger(
        agent.path,
        TriggerSpec(trigger_id="manual-demo", trigger_type="manual", config={"enabled": True}),
    )

    assert updated.prompt == "new prompt"
    assert updated.description == "updated description"
    assert updated.metadata["tier"] == 1
    assert len(with_trigger.triggers) == 1

    removed = registry.remove_trigger(agent.path, "manual-demo")
    assert removed.triggers == []


def test_registry_moves_subtree() -> None:
    registry = NodeRegistry()
    root = registry.create_agent(
        CreateAgentRequest(parent_path="/", name="root_manager", prompt="manage subtree")
    )
    child = registry.create_agent(
        CreateAgentRequest(parent_path=root.path, name="operator", prompt="operate")
    )
    moved = registry.move_node(
        MoveNodeRequest(path=child.path, new_parent_path="/", new_name="operator_renamed")
    )

    assert moved[0][0] == "/root_manager/operator"
    moved_child = registry.get_node("/operator_renamed")
    assert moved_child.parent_path == "/"
