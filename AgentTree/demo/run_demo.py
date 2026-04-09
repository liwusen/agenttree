from __future__ import annotations

import argparse
import asyncio
import json
from multiprocessing import Process

import httpx
import uvicorn
import websockets

from agenttree.config import get_settings
from agenttree.core.app import create_app
from demo.temp_exe import TemperatureControllerExecutor

import os

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentTree demo runner")
    parser.add_argument("--model", help="model name to push through /api/model-config")
    parser.add_argument("--api-key", help="OpenAI-compatible API key to push through /api/model-config")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL to push through /api/model-config")
    parser.add_argument("--temperature", type=float, help="model temperature to push through /api/model-config")
    parser.add_argument("--observe-seconds", type=float, default=20.0, help="observer runtime duration")
    return parser.parse_args()


def run_core() -> None:
    settings = get_settings()
    uvicorn.run(create_app(), host=settings.host, port=settings.port)


def run_temperature_executor() -> None:
    runtime = TemperatureControllerExecutor(
        path="/temperature.executor",
        initial_temperature=72.0,
        target_temperature=68.0,
        ambient_temperature=75.0,
        publish_interval_seconds=30.0,
    )
    asyncio.run(runtime.run())


async def bootstrap_demo() -> None:
    settings = get_settings()
    await wait_for_core(settings)
    async with httpx.AsyncClient(base_url=settings.base_url, timeout=30.0) as client:
        await client.post(
            f"{settings.api_prefix}/agents",
            json={
                "parent_path": "/",
                "name": "root_manager",
                "prompt": "你负责创建并协调 turbine_operator 子 Agent。",
                "description": "Demo root manager",
            },
        )
        await asyncio.sleep(1.5)
        await client.post(
            f"{settings.api_prefix}/agents",
            json={
                "parent_path": "/root_manager",
                "name": "turbine_operator",
                "prompt": "你负责涡轮温度监控与汇报。",
                "description": "Demo turbine operator",
            },
        )
        await asyncio.sleep(1.5)
        await client.post(
            f"{settings.api_prefix}/knowledge/upsert",
            json={
                "owner_node_path": "/root_manager/turbine_operator",
                "doc_path": "/root_manager/turbine_operator/INTRO.md",
                "text": "涡轮操作员负责监控温度，接收执行器事件，并向 root_manager 汇报。",
            },
        )
        await client.post(
            f"{settings.api_prefix}/channels",
            json={
                "channel_id": "ops-room",
                "members": ["/root_manager", "/root_manager/turbine_operator"],
                "metadata": {"purpose": "demo"},
            },
        )
        await client.post(
            f"{settings.api_prefix}/triggers/upsert",
            json={
                "path": "/root_manager/turbine_operator",
                "trigger": {
                    "trigger_id": "temperature_watch",
                    "trigger_type": "manual",
                    "config": {"description": "demo manual trigger"},
                },
            },
        )
        await client.post(
            f"{settings.api_prefix}/channels/ops-room/broadcast",
            json={
                "source_path": "/root_manager",
                "text": "请汇报最新的温度状态与执行器结果。",
            },
        )
        await client.post(
            f"{settings.api_prefix}/supervisor/command",
            json={
                "target_path": "/root_manager",
                "text": "发现新的外部执行器 /temperature.executor 后，将它绑定到 /root_manager/turbine_operator，并把目标温度设置为 68C，然后汇报状态。",
            },
        )


async def ensure_model_config(settings, args: argparse.Namespace) -> None:
    async with httpx.AsyncClient(base_url=settings.base_url, timeout=30.0) as client:
        current = await client.get(f"{settings.api_prefix}/model-config")
        current.raise_for_status()
        current_payload = current.json()

        request_payload = {
            "model": args.model,
            "openai_api_key": args.api_key,
            "openai_base_url": args.base_url,
            "model_temperature": args.temperature,
        }
        request_payload = {key: value for key, value in request_payload.items() if value is not None}

        if request_payload:
            response = await client.put(f"{settings.api_prefix}/model-config", json=request_payload)
            response.raise_for_status()
            configured = response.json()
            print(f"model config updated via API: {json.dumps(configured, ensure_ascii=False)}")
            return

        if current_payload.get("openai_api_key_configured"):
            print(f"using existing model config: {json.dumps(current_payload, ensure_ascii=False)}")
            return

        prompt_payload = prompt_for_model_config(current_payload)
        response = await client.put(f"{settings.api_prefix}/model-config", json=prompt_payload)
        response.raise_for_status()
        configured = response.json()
        print(f"model config updated via API: {json.dumps(configured, ensure_ascii=False)}")


def prompt_for_model_config(current_payload: dict) -> dict:
    default_model = str(current_payload.get("model") or "deepseek-chat")
    default_base_url = str(current_payload.get("openai_base_url") or "https://api.deepseek.com")
    default_temperature = str(current_payload.get("model_temperature") if current_payload.get("model_temperature") is not None else 0.0)

    print("model config is not set, please input values for /api/model-config")
    model = input(f"model [{default_model}]: ").strip() or default_model
    api_key = input("api key: ").strip()
    while not api_key:
        api_key = input("api key is required, api key: ").strip()
    base_url = input(f"base url [{default_base_url}]: ").strip() or default_base_url
    temperature_text = input(f"temperature [{default_temperature}]: ").strip() or default_temperature
    try:
        temperature = float(temperature_text)
    except ValueError:
        temperature = 0.0
    return {
        "model": model,
        "openai_api_key": api_key,
        "openai_base_url": base_url,
        "model_temperature": temperature,
    }


async def observe_runtime(settings, duration_seconds: float = 20.0) -> None:
    deadline = asyncio.get_running_loop().time() + duration_seconds
    async with websockets.connect(f"ws://{settings.host}:{settings.port}/ws/observe") as websocket:
        while asyncio.get_running_loop().time() < deadline:
            timeout = max(0.1, deadline - asyncio.get_running_loop().time())
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                continue
            payload = json.loads(raw)
            if payload.get("type") == "history":
                for entry in payload.get("entries", []):
                    print_trace(entry)
                continue
            entry = payload.get("entry")
            if entry:
                print_trace(entry)


def print_trace(entry: dict) -> None:
    timestamp = str(entry.get("timestamp", ""))
    source = str(entry.get("source", ""))
    category = str(entry.get("category", ""))
    message = str(entry.get("message", ""))
    payload = entry.get("payload", {}) or {}
    print(f"[{timestamp}] {source} {category}: {message}")
    if "thought_summary" in payload:
        print(f"  thought: {payload.get('thought_summary')}")
    if "text" in payload:
        print(f"  output: {payload.get('text')}")
    if "tree" in payload:
        print("  tree:")
        for line in str(payload.get("tree", "")).splitlines():
            print(f"    {line}")
    elif payload:
        print(f"  payload: {json.dumps(payload, ensure_ascii=False)}")


async def wait_for_core(settings, retries: int = 40, delay: float = 0.25) -> None:
    async with httpx.AsyncClient(base_url=settings.base_url, timeout=5.0) as client:
        last_error = None
        for _ in range(retries):
            try:
                response = await client.get("/")
                response.raise_for_status()
                return
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(delay)
        raise RuntimeError(f"core not ready: {last_error}")


def main() -> None:
    args = parse_args()
    process = Process(target=run_core, daemon=True)
    process.start()
    try:
        settings = get_settings()
        asyncio.run(run_demo_session(settings, args))
    finally:
        os.system("pause")
        if process.is_alive():
            process.terminate()


async def run_demo_session(settings, args: argparse.Namespace) -> None:
    await wait_for_core(settings)
    await ensure_model_config(settings, args)
    temperature_executor_process = Process(target=run_temperature_executor, daemon=True)
    temperature_executor_process.start()
    observer_task = asyncio.create_task(observe_runtime(settings, duration_seconds=args.observe_seconds))
    try:
        await asyncio.sleep(2.0)
        await bootstrap_demo()
        await asyncio.sleep(8.0)
    finally:
        observer_task.cancel()
        try:
            await observer_task
        except asyncio.CancelledError:
            pass
        if temperature_executor_process.is_alive():
            temperature_executor_process.terminate()


if __name__ == "__main__":
    main()