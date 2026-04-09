from __future__ import annotations

import argparse
import asyncio
import random
from dataclasses import dataclass
from typing import Any

import httpx

from agenttree.agent_runtime.client import RuntimeClient
from agenttree.config import get_settings
from agenttree.schemas.events import EventEnvelope, EventKind
from agenttree.schemas.nodes import NodeKind
from agenttree.schemas.protocol import RuntimeHello, RuntimeMessage, RuntimeMessageType


@dataclass(slots=True)
class TemperatureState:
    current_temperature: float
    target_temperature: float
    ambient_temperature: float
    fan_speed: int = 1


class TemperatureControllerExecutor:
    def __init__(
        self,
        *,
        path: str,
        initial_temperature: float = 72.0,
        target_temperature: float = 68.0,
        ambient_temperature: float = 75.0,
        publish_interval_seconds: float = 30.0,
    ) -> None:
        self.settings = get_settings()
        self.path = path
        self.client = RuntimeClient(self.settings)
        self.owner_path: str | None = None
        self.state = TemperatureState(
            current_temperature=initial_temperature,
            target_temperature=target_temperature,
            ambient_temperature=ambient_temperature,
        )
        self.publish_interval_seconds = publish_interval_seconds
        self._random = random.Random()

    async def register(self) -> None:
        async with httpx.AsyncClient(base_url=self.settings.base_url, timeout=30.0) as client:
            response = await client.post(
                f"{self.settings.api_prefix}/executors/register",
                json={
                    "path": self.path,
                    "executor_kind": "demo.temperature",
                    "description": "External temperature controller demo executor",
                    "capabilities": ["event_push", "temperature_control", "set_target_temperature"],
                    "metadata": {
                        "publish_interval_seconds": self.publish_interval_seconds,
                        "initial_temperature": self.state.current_temperature,
                        "target_temperature": self.state.target_temperature,
                        "ambient_temperature": self.state.ambient_temperature,
                    },
                },
            )
            response.raise_for_status()
            payload = response.json()
            node = payload.get("node", {})
            owner_path = node.get("owner_path")
            self.owner_path = str(owner_path) if owner_path else None

    async def run(self) -> None:
        await self.register()
        hello = RuntimeHello(
            path=self.path,
            kind=NodeKind.EXECUTOR,
            capabilities=["event_push", "temperature_control", "set_target_temperature"],
            metadata={"executor_kind": "demo.temperature", "publish_interval_seconds": self.publish_interval_seconds},
        )
        websocket = await self.client.connect(hello)
        await self.log(websocket, "runtime", "temperature controller connected", self.snapshot_payload())

        heartbeat_task = asyncio.create_task(self.client.heartbeat_loop(websocket, self.path))
        telemetry_task = asyncio.create_task(self.telemetry_loop(websocket))

        try:
            while True:
                message = await self.client.receive(websocket)
                if message.message_type == RuntimeMessageType.LOG:
                    continue
                if message.message_type == RuntimeMessageType.WAKE:
                    await self.client.send(
                        websocket,
                        RuntimeMessage(message_type=RuntimeMessageType.REQUEST_EVENT, path=self.path),
                    )
                    continue
                if message.message_type == RuntimeMessageType.EVENT and message.event is not None:
                    await self.handle_event(websocket, message.event)
        finally:
            heartbeat_task.cancel()
            telemetry_task.cancel()

    async def telemetry_loop(self, websocket) -> None:
        while True:
            await asyncio.sleep(self.publish_interval_seconds)
            self.advance_temperature()
            payload = self.snapshot_payload()
            payload.update(
                {
                    "event_name": "temperature_update",
                    "status": self.temperature_status(),
                    "source": "periodic_push",
                }
            )
            target = self.owner_path or "/supervisor"
            await self.publish_event(websocket, EventKind.EVENT, payload, target_path=target)
            await self.log(websocket, "telemetry", f"published periodic temperature update to {target}", payload)

    def advance_temperature(self) -> None:
        drift = (self.state.ambient_temperature - self.state.current_temperature) * 0.04
        control = (self.state.target_temperature - self.state.current_temperature) * (0.16 + 0.03 * self.state.fan_speed)
        noise = self._random.uniform(-0.9, 0.9)
        next_temperature = self.state.current_temperature + drift + control + noise
        self.state.current_temperature = round(next_temperature, 2)

    def temperature_status(self) -> str:
        delta = self.state.current_temperature - self.state.target_temperature
        if delta >= 8:
            return "critical_high"
        if delta >= 4:
            return "high"
        if delta <= -6:
            return "low"
        return "stable"

    async def handle_event(self, websocket, event: EventEnvelope) -> None:
        await self.log(
            websocket,
            "event_received",
            f"temperature controller received {event.kind.value} from {event.source_path}",
            {"event": event.model_dump(mode="json")},
        )

        if event.kind == EventKind.STRUCT and event.payload.get("action") == "executor_bound":
            owner_path = event.payload.get("owner_path")
            self.owner_path = str(owner_path) if owner_path else None
            await self.log(websocket, "ownership", f"temperature controller bound to {self.owner_path}", {"owner_path": self.owner_path})
            await self.client.send(
                websocket,
                RuntimeMessage(message_type=RuntimeMessageType.ACK_EVENT, path=self.path, event_id=event.event_id),
            )
            return

        command = str(event.payload.get("command") or event.payload.get("action") or "").strip()
        result_payload: dict[str, Any]

        if command in {"adjust_parameters", "set_target_temperature", "update_target_temperature"}:
            result_payload = self.apply_temperature_update(event.payload)
        elif command in {"status", "read_temperature", "snapshot"}:
            result_payload = {
                "executor_path": self.path,
                "handled": True,
                "command": command or "status",
                "summary": "temperature controller status snapshot",
                **self.snapshot_payload(),
            }
        else:
            result_payload = {
                "executor_path": self.path,
                "handled": False,
                "command": command,
                "summary": f"unsupported command: {command or 'unknown'}",
                **self.snapshot_payload(),
            }

        reply_event = EventEnvelope(
            kind=EventKind.EVENT,
            source_path=self.path,
            target_path=self.owner_path or "/supervisor",
            payload=result_payload,
            trace_id=event.trace_id,
            metadata={"reply_to": event.event_id},
        )
        await self.client.send(
            websocket,
            RuntimeMessage(message_type=RuntimeMessageType.PUBLISH_EVENT, path=self.path, event=reply_event),
        )
        await self.log(websocket, "reply_sent", "temperature controller sent command result", {"reply": reply_event.model_dump(mode="json")})

        await self.client.send(
            websocket,
            RuntimeMessage(message_type=RuntimeMessageType.ACK_EVENT, path=self.path, event_id=event.event_id),
        )

    def apply_temperature_update(self, payload: dict[str, Any]) -> dict[str, Any]:
        previous_target = self.state.target_temperature
        previous_fan_speed = self.state.fan_speed

        if "target_temperature" in payload:
            self.state.target_temperature = float(payload["target_temperature"])
        if "fan_speed" in payload:
            self.state.fan_speed = max(1, min(5, int(payload["fan_speed"])))
        if "ambient_temperature" in payload:
            self.state.ambient_temperature = float(payload["ambient_temperature"])

        self.advance_temperature()

        return {
            "executor_path": self.path,
            "handled": True,
            "command": str(payload.get("command") or "set_target_temperature"),
            "summary": (
                f"target temperature updated from {previous_target:.1f}C to {self.state.target_temperature:.1f}C; "
                f"fan speed {previous_fan_speed} -> {self.state.fan_speed}"
            ),
            "previous_target_temperature": previous_target,
            "previous_fan_speed": previous_fan_speed,
            **self.snapshot_payload(),
        }

    def snapshot_payload(self) -> dict[str, Any]:
        return {
            "executor_path": self.path,
            "current_temperature": self.state.current_temperature,
            "target_temperature": self.state.target_temperature,
            "ambient_temperature": self.state.ambient_temperature,
            "fan_speed": self.state.fan_speed,
            "unit": "C",
        }

    async def publish_event(self, websocket, kind: EventKind, payload: dict[str, Any], target_path: str) -> None:
        outgoing = EventEnvelope(
            kind=kind,
            source_path=self.path,
            target_path=target_path,
            payload=payload,
        )
        await self.client.send(
            websocket,
            RuntimeMessage(message_type=RuntimeMessageType.PUBLISH_EVENT, path=self.path, event=outgoing),
        )

    async def log(self, websocket, category: str, message: str, payload: dict[str, Any] | None = None) -> None:
        await self.client.send(
            websocket,
            RuntimeMessage(
                message_type=RuntimeMessageType.LOG,
                path=self.path,
                payload={"category": category, "message": message, **({"payload": payload} if payload else {})},
            ),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo temperature controller executor")
    parser.add_argument("--path", required=True)
    parser.add_argument("--initial-temperature", type=float, default=72.0)
    parser.add_argument("--target-temperature", type=float, default=68.0)
    parser.add_argument("--ambient-temperature", type=float, default=75.0)
    parser.add_argument("--publish-interval-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = TemperatureControllerExecutor(
        path=args.path,
        initial_temperature=args.initial_temperature,
        target_temperature=args.target_temperature,
        ambient_temperature=args.ambient_temperature,
        publish_interval_seconds=args.publish_interval_seconds,
    )
    asyncio.run(runtime.run())


if __name__ == "__main__":
    main()
