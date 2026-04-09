from __future__ import annotations

import argparse
import asyncio

from agenttree.agent_runtime.runtime import AgentRuntime
from agenttree.config import get_settings
from agenttree.schemas.nodes import NodeKind


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentTree agent runtime")
    parser.add_argument("--path", required=True)
    parser.add_argument("--kind", default="agent")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    runtime = AgentRuntime(settings=settings, path=args.path, kind=NodeKind(args.kind))
    asyncio.run(runtime.run())


if __name__ == "__main__":
    main()