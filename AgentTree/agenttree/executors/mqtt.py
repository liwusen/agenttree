from __future__ import annotations

import argparse
import asyncio
from typing import Any

from paho.mqtt import client as mqtt_client

from agenttree.executors.base import ExternalExecutorBase, build_settings_for_core, operation_field, operation_spec
from agenttree.schemas.events import EventEnvelope, EventKind


class MQTTClientExecutor(ExternalExecutorBase):
    def __init__(
        self,
        *,
        core: str,
        path: str,
        broker: str,
        username: str | None = None,
        password: str | None = None,
        client_id: str | None = None,
    ) -> None:
        host, sep, port_text = broker.rpartition(":")
        if not sep or not host or not port_text:
            raise ValueError(f"invalid mqtt broker address: {broker}")
        self.mqtt_host = host
        self.mqtt_port = int(port_text)
        self.mqtt_username = username
        self.mqtt_password = password
        self.client_id = client_id or f"agenttree-{path.strip('/').replace('/', '-') or 'mqtt'}"
        self.subscriptions: dict[str, int] = {}
        self.mqtt_client: mqtt_client.Client | None = None
        super().__init__(
            settings=build_settings_for_core(core),
            path=path,
            executor_kind="mqtt.client",
            description="MQTT client executor for subscribe/publish workflows.",
            capabilities=["event_push", "mqtt_subscribe", "mqtt_unsubscribe", "mqtt_publish", "mqtt_status"],
            metadata={"mqtt_host": self.mqtt_host, "mqtt_port": self.mqtt_port},
            operations=[
                operation_spec(
                    "mqtt_subscribe",
                    summary="订阅 MQTT Topic。",
                    description="建立对某个 topic 的订阅，后续消息会作为 event 回推给 owner 节点。",
                    aliases=["subscribe"],
                    payload_schema=[
                        operation_field("topic", field_type="string", description="要订阅的 MQTT topic。", required=True),
                        operation_field("qos", field_type="integer", description="QoS 等级，默认 0。", required=False, default=0),
                    ],
                    returns=[
                        operation_field("handled", field_type="boolean", description="是否订阅成功。", required=True),
                        operation_field("summary", field_type="string", description="结果摘要。", required=True),
                        operation_field("subscriptions", field_type="object", description="当前订阅 topic 到 qos 的映射。", required=True),
                    ],
                ),
                operation_spec(
                    "mqtt_unsubscribe",
                    summary="取消订阅 MQTT Topic。",
                    description="停止订阅某个 topic。",
                    aliases=["unsubscribe"],
                    payload_schema=[
                        operation_field("topic", field_type="string", description="要取消订阅的 MQTT topic。", required=True),
                    ],
                    returns=[
                        operation_field("handled", field_type="boolean", description="是否取消成功。", required=True),
                        operation_field("summary", field_type="string", description="结果摘要。", required=True),
                        operation_field("subscriptions", field_type="object", description="当前剩余订阅映射。", required=True),
                    ],
                ),
                operation_spec(
                    "mqtt_publish",
                    summary="向 MQTT Topic 发布消息。",
                    description="把字符串消息发往指定 topic。适合向外部设备或服务发送控制命令。",
                    aliases=["publish"],
                    payload_schema=[
                        operation_field("topic", field_type="string", description="目标 MQTT topic。", required=True),
                        operation_field("message", field_type="string", description="要发布的字符串消息。", required=False),
                        operation_field("payload", field_type="string", description="message 的别名字段。", required=False),
                        operation_field("qos", field_type="integer", description="QoS 等级，默认 0。", required=False, default=0),
                        operation_field("retain", field_type="boolean", description="是否 retain，默认 false。", required=False, default=False),
                    ],
                    returns=[
                        operation_field("handled", field_type="boolean", description="是否发布请求成功提交。", required=True),
                        operation_field("summary", field_type="string", description="结果摘要。", required=True),
                        operation_field("mid", field_type="integer", description="MQTT publish message id。", required=False),
                        operation_field("topic", field_type="string", description="发布目标 topic。", required=False),
                    ],
                ),
                operation_spec(
                    "mqtt_status",
                    summary="查询 MQTT 执行器状态。",
                    description="返回 broker 地址、client_id、当前 owner 和订阅列表。",
                    aliases=["status"],
                    payload_schema=[],
                    returns=[
                        operation_field("handled", field_type="boolean", description="是否查询成功。", required=True),
                        operation_field("mqtt_host", field_type="string", description="MQTT broker 主机。", required=True),
                        operation_field("mqtt_port", field_type="integer", description="MQTT broker 端口。", required=True),
                        operation_field("client_id", field_type="string", description="当前 MQTT client id。", required=True),
                        operation_field("subscriptions", field_type="object", description="当前订阅映射。", required=True),
                    ],
                ),
            ],
        )

    async def on_started(self) -> None:
        client = mqtt_client.Client(client_id=self.client_id)
        if self.mqtt_username:
            client.username_pw_set(self.mqtt_username, self.mqtt_password)
        client.on_connect = self._on_connect
        client.on_message = self._on_message
        client.on_disconnect = self._on_disconnect
        client.connect(self.mqtt_host, self.mqtt_port)
        client.loop_start()
        self.mqtt_client = client
        await self.log("mqtt", f"connecting to MQTT broker {self.mqtt_host}:{self.mqtt_port}", self.describe_state())

    async def on_shutdown(self) -> None:
        if self.mqtt_client is not None:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()

    async def handle_event(self, event: EventEnvelope) -> None:
        command = str(event.payload.get("command") or "").strip().lower()
        if command in {"subscribe", "mqtt_subscribe"}:
            result = self._subscribe(event.payload)
        elif command in {"unsubscribe", "mqtt_unsubscribe"}:
            result = self._unsubscribe(event.payload)
        elif command in {"publish", "mqtt_publish"}:
            result = self._publish(event.payload)
        elif command in {"status", "mqtt_status"}:
            result = self._status_payload()
        else:
            result = {
                "handled": False,
                "summary": f"unsupported mqtt command: {command or 'unknown'}",
                "available_commands": ["mqtt_subscribe", "mqtt_unsubscribe", "mqtt_publish", "mqtt_status"],
            }
        await self.reply_to_event(event, {"executor_path": self.path, **result})

    def _subscribe(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.mqtt_client is None:
            return {"handled": False, "summary": "mqtt client is not connected"}
        topic = str(payload.get("topic") or "").strip()
        qos = int(payload.get("qos", 0) or 0)
        if not topic:
            return {"handled": False, "summary": "topic is required"}
        self.mqtt_client.subscribe(topic, qos=qos)
        self.subscriptions[topic] = qos
        return {"handled": True, "summary": f"subscribed to {topic}", "subscriptions": self.subscriptions}

    def _unsubscribe(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.mqtt_client is None:
            return {"handled": False, "summary": "mqtt client is not connected"}
        topic = str(payload.get("topic") or "").strip()
        if not topic:
            return {"handled": False, "summary": "topic is required"}
        self.mqtt_client.unsubscribe(topic)
        self.subscriptions.pop(topic, None)
        return {"handled": True, "summary": f"unsubscribed from {topic}", "subscriptions": self.subscriptions}

    def _publish(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.mqtt_client is None:
            return {"handled": False, "summary": "mqtt client is not connected"}
        topic = str(payload.get("topic") or "").strip()
        message = str(payload.get("message") or payload.get("payload") or "")
        qos = int(payload.get("qos", 0) or 0)
        retain = bool(payload.get("retain", False))
        if not topic:
            return {"handled": False, "summary": "topic is required"}
        info = self.mqtt_client.publish(topic, payload=message, qos=qos, retain=retain)
        return {
            "handled": True,
            "summary": f"published message to {topic}",
            "mid": info.mid,
            "topic": topic,
            "qos": qos,
            "retain": retain,
        }

    def _status_payload(self) -> dict[str, Any]:
        return {
            "handled": True,
            "mqtt_host": self.mqtt_host,
            "mqtt_port": self.mqtt_port,
            "client_id": self.client_id,
            "subscriptions": self.subscriptions,
            "owner_path": self.owner_path,
        }

    def _on_connect(self, client: mqtt_client.Client, userdata, flags, reason_code, properties=None) -> None:
        self.schedule_coroutine(
            self.log(
                "mqtt",
                f"connected to MQTT broker {self.mqtt_host}:{self.mqtt_port}",
                {"reason_code": str(reason_code), "subscriptions": self.subscriptions},
            )
        )
        for topic, qos in self.subscriptions.items():
            client.subscribe(topic, qos=qos)

    def _on_disconnect(self, client: mqtt_client.Client, userdata, disconnect_flags, reason_code, properties=None) -> None:
        self.schedule_coroutine(
            self.log("mqtt", "mqtt client disconnected", {"reason_code": str(reason_code)})
        )

    def _on_message(self, client: mqtt_client.Client, userdata, message: mqtt_client.MQTTMessage) -> None:
        payload = {
            "event_name": "mqtt_message",
            "executor_path": self.path,
            "topic": message.topic,
            "payload": message.payload.decode(errors="replace"),
            "qos": message.qos,
            "retain": bool(message.retain),
        }
        self.schedule_coroutine(self.publish_event(EventKind.EVENT, payload))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentTree MQTT client executor")
    parser.add_argument("--core", required=True, help="Core address in host:port form")
    parser.add_argument("--path", required=True, help="Executor path in the Agent tree")
    parser.add_argument("--mqtt-broker", required=True, help="MQTT broker host:port")
    parser.add_argument("--mqtt-user", help="MQTT username")
    parser.add_argument("--mqtt-pwd", help="MQTT password")
    parser.add_argument("--mqtt-client-id", help="MQTT client id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = MQTTClientExecutor(
        core=args.core,
        path=args.path,
        broker=args.mqtt_broker,
        username=args.mqtt_user,
        password=args.mqtt_pwd,
        client_id=args.mqtt_client_id,
    )
    asyncio.run(runtime.run())


if __name__ == "__main__":
    main()