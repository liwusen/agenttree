from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from agenttree.executors.base import ExternalExecutorBase, build_settings_for_core, operation_field, operation_spec
from agenttree.schemas.events import EventEnvelope, EventKind


@dataclass(slots=True)
class CommandJob:
    job_id: str
    command: str
    cwd: str | None
    process: asyncio.subprocess.Process
    stdout_lines: list[str] = field(default_factory=list)
    stderr_lines: list[str] = field(default_factory=list)
    returncode: int | None = None
    status: str = "running"


class SystemCommandExecutor(ExternalExecutorBase):
    def __init__(self, *, core: str, path: str, working_dir: str | None = None) -> None:
        self.working_dir = str(Path(working_dir).resolve()) if working_dir else None
        self.jobs: dict[str, CommandJob] = {}
        super().__init__(
            settings=build_settings_for_core(core),
            path=path,
            executor_kind="system.command",
            description="Execute system commands through subprocess with concurrent jobs and IO capture.",
            capabilities=["event_push", "run_command", "terminate_command", "command_status"],
            metadata={"working_dir": self.working_dir},
            operations=[
                operation_spec(
                    "run_command",
                    summary="异步启动一个或多个系统命令。",
                    description="适用于需要执行 shell 命令、脚本或程序的场景。返回通常表示任务已启动，最终 stdout/stderr 和结束状态会通过事件回推。",
                    aliases=["run", "exec", "execute"],
                    payload_schema=[
                        operation_field("shell_command", field_type="string", description="单条 shell 命令。与 commands 二选一。", required=False),
                        operation_field("commands", field_type="array[string]", description="批量执行的命令列表。与 shell_command 二选一。", required=False),
                        operation_field("cwd", field_type="string", description="命令执行目录，默认使用执行器 working_dir。", required=False),
                    ],
                    returns=[
                        operation_field("handled", field_type="boolean", description="是否成功接受调用。", required=True),
                        operation_field("summary", field_type="string", description="对执行结果的简要说明。", required=True),
                        operation_field("jobs", field_type="array[object]", description="已启动任务列表，包含 job_id、command、cwd。", required=False),
                    ],
                    examples=[
                        {
                            "command": "run_command",
                            "payload": {"shell_command": "python --version"},
                        }
                    ],
                ),
                operation_spec(
                    "command_status",
                    summary="查询命令任务状态。",
                    description="用于轮询某个 job_id 或查看全部命令任务当前状态。",
                    aliases=["status"],
                    payload_schema=[
                        operation_field("job_id", field_type="string", description="要查询的任务 ID；省略时返回全部任务。", required=False),
                    ],
                    returns=[
                        operation_field("handled", field_type="boolean", description="是否成功处理查询。", required=True),
                        operation_field("job", field_type="object", description="单个任务详情。", required=False),
                        operation_field("jobs", field_type="array[object]", description="全部任务详情列表。", required=False),
                    ],
                ),
                operation_spec(
                    "terminate",
                    summary="终止一个正在运行的命令任务。",
                    description="用于停止指定 job_id 的系统命令任务。",
                    aliases=["kill", "stop", "terminate_command"],
                    payload_schema=[
                        operation_field("job_id", field_type="string", description="要终止的任务 ID。", required=True),
                    ],
                    returns=[
                        operation_field("handled", field_type="boolean", description="是否成功终止。", required=True),
                        operation_field("job", field_type="object", description="终止后的任务状态详情。", required=False),
                        operation_field("summary", field_type="string", description="终止结果摘要。", required=False),
                    ],
                ),
            ],
        )

    async def handle_event(self, event: EventEnvelope) -> None:
        command = str(event.payload.get("command") or "").strip().lower()
        if command in {"run", "run_command", "exec", "execute"}:
            result = await self._start_commands(event)
        elif command in {"status", "command_status"}:
            result = self._status_payload(event.payload.get("job_id"))
        elif command in {"terminate", "kill", "stop"}:
            result = await self._terminate_job(str(event.payload.get("job_id") or ""))
        else:
            result = {
                "handled": False,
                "summary": f"unsupported command operation: {command or 'unknown'}",
                "available_commands": ["run_command", "command_status", "terminate"],
            }
        await self.reply_to_event(event, {"executor_path": self.path, **result})

    async def _start_commands(self, event: EventEnvelope) -> dict[str, Any]:
        commands = event.payload.get("commands")
        if isinstance(commands, list):
            command_list = [str(item) for item in commands if str(item).strip()]
        else:
            single = str(event.payload.get("shell_command") or event.payload.get("command_line") or "").strip()
            command_list = [single] if single else []
        if not command_list:
            return {"handled": False, "summary": "no command provided"}
        cwd = str(event.payload.get("cwd") or self.working_dir or "").strip() or None
        started_jobs = []
        for command in command_list:
            job = await self._spawn_job(command, cwd=cwd, trace_id=event.trace_id)
            started_jobs.append({"job_id": job.job_id, "command": job.command, "cwd": job.cwd})
        return {"handled": True, "summary": f"started {len(started_jobs)} command job(s)", "jobs": started_jobs}

    async def _spawn_job(self, command: str, *, cwd: str | None, trace_id: str) -> CommandJob:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )
        job = CommandJob(job_id=uuid4().hex, command=command, cwd=cwd, process=process)
        self.jobs[job.job_id] = job
        asyncio.create_task(self._watch_stream(job, process.stdout, "stdout", trace_id))
        asyncio.create_task(self._watch_stream(job, process.stderr, "stderr", trace_id))
        asyncio.create_task(self._wait_for_job(job, trace_id))
        await self.log("job_started", f"started command job {job.job_id}", {"job_id": job.job_id, "command": command, "cwd": cwd})
        return job

    async def _watch_stream(self, job: CommandJob, stream, stream_name: str, trace_id: str) -> None:
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                return
            text = line.decode(errors="replace").rstrip()
            if stream_name == "stdout":
                job.stdout_lines.append(text)
            else:
                job.stderr_lines.append(text)
            await self.publish_event(
                EventKind.EVENT,
                {
                    "event_name": f"command_{stream_name}",
                    "executor_path": self.path,
                    "job_id": job.job_id,
                    "command": job.command,
                    "line": text,
                },
                trace_id=trace_id,
            )

    async def _wait_for_job(self, job: CommandJob, trace_id: str) -> None:
        job.returncode = await job.process.wait()
        job.status = "completed" if job.returncode == 0 else "failed"
        await self.publish_event(
            EventKind.EVENT,
            {
                "event_name": "command_completed",
                "executor_path": self.path,
                "job_id": job.job_id,
                "command": job.command,
                "status": job.status,
                "returncode": job.returncode,
                "stdout": job.stdout_lines,
                "stderr": job.stderr_lines,
            },
            trace_id=trace_id,
        )

    def _status_payload(self, job_id: Any) -> dict[str, Any]:
        if job_id:
            job = self.jobs.get(str(job_id))
            if job is None:
                return {"handled": False, "summary": f"job not found: {job_id}"}
            return {"handled": True, "job": self._serialize_job(job)}
        return {"handled": True, "jobs": [self._serialize_job(job) for job in self.jobs.values()]}

    async def _terminate_job(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job is None:
            return {"handled": False, "summary": f"job not found: {job_id}"}
        if job.process.returncode is None:
            job.process.terminate()
            await job.process.wait()
        job.returncode = job.process.returncode
        job.status = "terminated"
        return {"handled": True, "job": self._serialize_job(job)}

    def _serialize_job(self, job: CommandJob) -> dict[str, Any]:
        return {
            "job_id": job.job_id,
            "command": job.command,
            "cwd": job.cwd,
            "status": job.status,
            "returncode": job.returncode,
            "stdout": job.stdout_lines,
            "stderr": job.stderr_lines,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentTree system command executor")
    parser.add_argument("--core", required=True, help="Core address in host:port form")
    parser.add_argument("--path", required=True, help="Executor path in the Agent tree")
    parser.add_argument("--working-dir", help="Optional default working directory for commands")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = SystemCommandExecutor(core=args.core, path=args.path, working_dir=args.working_dir)
    asyncio.run(runtime.run())


if __name__ == "__main__":
    main()