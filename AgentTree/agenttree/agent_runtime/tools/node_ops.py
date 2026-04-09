from __future__ import annotations

import json

import httpx

from agenttree.agent_runtime.tools.common import ToolTraceHook, run_tool_action
from agenttree.config import AgentTreeSettings


def build_node_tools(settings: AgentTreeSettings, self_path: str, trace_hook: ToolTraceHook | None = None) -> list:
    def parse_json_object(text: str) -> dict:
        """Parse a JSON object string and return an empty dict on invalid input."""
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    async def create_child_agent(name: str, prompt: str, description: str = "") -> str:
        """Create a child agent under the current node with a prompt and optional description."""
        async def action() -> str:
            async with httpx.AsyncClient(base_url=settings.base_url, timeout=30.0) as client:
                response = await client.post(
                    f"{settings.api_prefix}/agents",
                    json={
                        "parent_path": self_path,
                        "name": name,
                        "prompt": prompt,
                        "description": description,
                    },
                )
                response.raise_for_status()
                return response.text

        return await run_tool_action(
            tool_name="create_child_agent",
            args={"name": name, "prompt": prompt, "description": description},
            action=action,
            trace_hook=trace_hook,
        )

    async def create_child_agent_from_template(name: str, prompt_template: str, description: str = "") -> str:
        """Create a child agent using a prompt template defined in const_prompt.py."""
        async def action() -> str:
            async with httpx.AsyncClient(base_url=settings.base_url, timeout=30.0) as client:
                response = await client.post(
                    f"{settings.api_prefix}/agents",
                    json={
                        "parent_path": self_path,
                        "name": name,
                        "prompt_template": prompt_template,
                        "description": description,
                    },
                )
                response.raise_for_status()
                return response.text

        return await run_tool_action(
            tool_name="create_child_agent_from_template",
            args={"name": name, "prompt_template": prompt_template, "description": description},
            action=action,
            trace_hook=trace_hook,
        )

    async def send_command(target_path: str, text: str) -> str:
        """Send a command event from the current node to another node path."""
        async def action() -> str:
            async with httpx.AsyncClient(base_url=settings.base_url, timeout=30.0) as client:
                response = await client.post(
                    f"{settings.api_prefix}/messages",
                    json={
                        "kind": "command",
                        "source_path": self_path,
                        "target_path": target_path,
                        "text": text,
                    },
                )
                response.raise_for_status()
                return response.text

        return await run_tool_action(
            tool_name="send_command",
            args={"target_path": target_path, "text": text},
            action=action,
            trace_hook=trace_hook,
        )

    async def send_message(target_path: str, text: str) -> str:
        """Send a normal message event from the current node to another node path."""
        async def action() -> str:
            async with httpx.AsyncClient(base_url=settings.base_url, timeout=30.0) as client:
                response = await client.post(
                    f"{settings.api_prefix}/messages",
                    json={
                        "kind": "message",
                        "source_path": self_path,
                        "target_path": target_path,
                        "text": text,
                    },
                )
                response.raise_for_status()
                return response.text

        return await run_tool_action(
            tool_name="send_message",
            args={"target_path": target_path, "text": text},
            action=action,
            trace_hook=trace_hook,
        )

    async def update_self_prompt(prompt: str, description: str = "") -> str:
        """Update the current node prompt and optional description."""
        async def action() -> str:
            async with httpx.AsyncClient(base_url=settings.base_url, timeout=30.0) as client:
                response = await client.patch(
                    f"{settings.api_prefix}/nodes/{self_path.strip('/')}",
                    json={"prompt": prompt, "description": description},
                )
                response.raise_for_status()
                return response.text

        return await run_tool_action(
            tool_name="update_self_prompt",
            args={"prompt": prompt, "description": description},
            action=action,
            trace_hook=trace_hook,
        )

    async def delete_child_node(path: str) -> str:
        """Delete a child node by path."""
        async def action() -> str:
            async with httpx.AsyncClient(base_url=settings.base_url, timeout=30.0) as client:
                response = await client.delete(f"{settings.api_prefix}/nodes/{path.strip('/')}")
                response.raise_for_status()
                return response.text

        return await run_tool_action(
            tool_name="delete_child_node",
            args={"path": path},
            action=action,
            trace_hook=trace_hook,
        )

    async def create_channel(channel_id: str, members_csv: str) -> str:
        """Create a channel and ensure the current node is included as a member."""
        members = [member.strip() for member in members_csv.split(",") if member.strip()]
        if self_path not in members:
            members.append(self_path)
        async def action() -> str:
            async with httpx.AsyncClient(base_url=settings.base_url, timeout=30.0) as client:
                response = await client.post(
                    f"{settings.api_prefix}/channels",
                    json={"channel_id": channel_id, "members": members, "metadata": {}},
                )
                response.raise_for_status()
                return response.text

        return await run_tool_action(
            tool_name="create_channel",
            args={"channel_id": channel_id, "members": members},
            action=action,
            trace_hook=trace_hook,
        )

    async def broadcast_channel(channel_id: str, text: str) -> str:
        """Broadcast a text message to all members of a channel."""
        async def action() -> str:
            async with httpx.AsyncClient(base_url=settings.base_url, timeout=30.0) as client:
                response = await client.post(
                    f"{settings.api_prefix}/channels/{channel_id}/broadcast",
                    json={"source_path": self_path, "text": text},
                )
                response.raise_for_status()
                return response.text

        return await run_tool_action(
            tool_name="broadcast_channel",
            args={"channel_id": channel_id, "text": text},
            action=action,
            trace_hook=trace_hook,
        )

    async def upsert_trigger(trigger_id: str, trigger_type: str, config_json: str = "{}") -> str:
        """Create or update a trigger on the current node using a JSON config object."""
        config = parse_json_object(config_json)

        async def action() -> str:
            async with httpx.AsyncClient(base_url=settings.base_url, timeout=30.0) as client:
                response = await client.post(
                    f"{settings.api_prefix}/triggers/upsert",
                    json={
                        "path": self_path,
                        "trigger": {
                            "trigger_id": trigger_id,
                            "trigger_type": trigger_type,
                            "config": config,
                        },
                    },
                )
                response.raise_for_status()
                return response.text

        return await run_tool_action(
            tool_name="upsert_trigger",
            args={"trigger_id": trigger_id, "trigger_type": trigger_type, "config": config},
            action=action,
            trace_hook=trace_hook,
        )

    async def remove_trigger(trigger_id: str) -> str:
        """Remove a trigger from the current node by trigger id."""
        async def action() -> str:
            async with httpx.AsyncClient(base_url=settings.base_url, timeout=30.0) as client:
                response = await client.request(
                    "DELETE",
                    f"{settings.api_prefix}/triggers",
                    json={"path": self_path, "trigger_id": trigger_id},
                )
                response.raise_for_status()
                return response.text

        return await run_tool_action(
            tool_name="remove_trigger",
            args={"trigger_id": trigger_id},
            action=action,
            trace_hook=trace_hook,
        )

    async def get_tree_structure() -> str:
        """Fetch the current AgentTree topology, including nodes, child relations, and channels."""
        async def action() -> str:
            async with httpx.AsyncClient(base_url=settings.base_url, timeout=30.0) as client:
                response = await client.get(f"{settings.api_prefix}/tree")
                response.raise_for_status()
                return response.text

        return await run_tool_action(
            tool_name="get_tree_structure",
            args={},
            action=action,
            trace_hook=trace_hook,
        )

    async def get_node_detail(path: str) -> str:
        """Fetch the detailed record for a specific node path."""
        normalized_path = path.strip("/")

        async def action() -> str:
            async with httpx.AsyncClient(base_url=settings.base_url, timeout=30.0) as client:
                response = await client.get(f"{settings.api_prefix}/nodes/{normalized_path}")
                response.raise_for_status()
                return response.text

        return await run_tool_action(
            tool_name="get_node_detail",
            args={"path": path},
            action=action,
            trace_hook=trace_hook,
        )

    async def get_current_node() -> str:
        """Fetch the detailed record for the current agent node."""
        normalized_path = self_path.strip("/")

        async def action() -> str:
            async with httpx.AsyncClient(base_url=settings.base_url, timeout=30.0) as client:
                response = await client.get(f"{settings.api_prefix}/nodes/{normalized_path}")
                response.raise_for_status()
                return response.text

        return await run_tool_action(
            tool_name="get_current_node",
            args={"path": self_path},
            action=action,
            trace_hook=trace_hook,
        )

    async def list_prompt_templates() -> str:
        """List available prompt templates defined in const_prompt.py."""
        async def action() -> str:
            async with httpx.AsyncClient(base_url=settings.base_url, timeout=30.0) as client:
                response = await client.get(f"{settings.api_prefix}/prompts")
                response.raise_for_status()
                return response.text

        return await run_tool_action(
            tool_name="list_prompt_templates",
            args={},
            action=action,
            trace_hook=trace_hook,
        )

    async def get_prompt_template(prompt_name: str) -> str:
        """Get the content of a prompt template defined in const_prompt.py."""
        normalized_name = prompt_name.strip()

        async def action() -> str:
            async with httpx.AsyncClient(base_url=settings.base_url, timeout=30.0) as client:
                response = await client.get(f"{settings.api_prefix}/prompts/{normalized_name}")
                response.raise_for_status()
                return response.text

        return await run_tool_action(
            tool_name="get_prompt_template",
            args={"prompt_name": prompt_name},
            action=action,
            trace_hook=trace_hook,
        )

    async def export_prompts_to_knowledge(target_root_path: str, prompt_names_csv: str) -> str:
        """Export selected prompt templates into the knowledge base under a target root path."""
        prompt_names = [item.strip() for item in prompt_names_csv.split(",") if item.strip()]

        async def action() -> str:
            async with httpx.AsyncClient(base_url=settings.base_url, timeout=30.0) as client:
                response = await client.post(
                    f"{settings.api_prefix}/prompts/export-to-knowledge",
                    json={
                        "owner_node_path": self_path,
                        "target_root_path": target_root_path,
                        "prompt_names": prompt_names,
                    },
                )
                response.raise_for_status()
                return response.text

        return await run_tool_action(
            tool_name="export_prompts_to_knowledge",
            args={"target_root_path": target_root_path, "prompt_names": prompt_names},
            action=action,
            trace_hook=trace_hook,
        )

    async def list_manual_knowledge_templates() -> str:
        """List manual knowledge templates defined in const_prompt.py."""
        async def action() -> str:
            async with httpx.AsyncClient(base_url=settings.base_url, timeout=30.0) as client:
                response = await client.get(f"{settings.api_prefix}/knowledge-templates")
                response.raise_for_status()
                return response.text

        return await run_tool_action(
            tool_name="list_manual_knowledge_templates",
            args={},
            action=action,
            trace_hook=trace_hook,
        )

    async def get_manual_knowledge_template(template_name: str) -> str:
        """Get one manual knowledge template defined in const_prompt.py."""
        normalized_name = template_name.strip()

        async def action() -> str:
            async with httpx.AsyncClient(base_url=settings.base_url, timeout=30.0) as client:
                response = await client.get(f"{settings.api_prefix}/knowledge-templates/{normalized_name}")
                response.raise_for_status()
                return response.text

        return await run_tool_action(
            tool_name="get_manual_knowledge_template",
            args={"template_name": template_name},
            action=action,
            trace_hook=trace_hook,
        )

    async def inject_manual_knowledge_templates(target_root_path: str, template_names_csv: str) -> str:
        """Inject selected manual knowledge templates from const_prompt.py into the knowledge base."""
        knowledge_names = [item.strip() for item in template_names_csv.split(",") if item.strip()]

        async def action() -> str:
            async with httpx.AsyncClient(base_url=settings.base_url, timeout=30.0) as client:
                response = await client.post(
                    f"{settings.api_prefix}/knowledge-templates/export-to-knowledge",
                    json={
                        "owner_node_path": self_path,
                        "target_root_path": target_root_path,
                        "knowledge_names": knowledge_names,
                    },
                )
                response.raise_for_status()
                return response.text

        return await run_tool_action(
            tool_name="inject_manual_knowledge_templates",
            args={"target_root_path": target_root_path, "knowledge_names": knowledge_names},
            action=action,
            trace_hook=trace_hook,
        )

    return [
        create_child_agent,
        create_child_agent_from_template,
        send_command,
        send_message,
        update_self_prompt,
        delete_child_node,
        create_channel,
        broadcast_channel,
        upsert_trigger,
        remove_trigger,
        get_tree_structure,
        get_node_detail,
        get_current_node,
        list_prompt_templates,
        get_prompt_template,
        export_prompts_to_knowledge,
        list_manual_knowledge_templates,
        get_manual_knowledge_template,
        inject_manual_knowledge_templates,
    ]