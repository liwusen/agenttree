from __future__ import annotations

import argparse
import asyncio

from agenttree.config import get_settings
from agenttree.executor_runtime.runtime import ExecutorRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentTree executor runtime")
    parser.add_argument("--path", required=True)
    parser.add_argument("--executor-kind", default="external.generic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = ExecutorRuntime(get_settings(), args.path, executor_kind=args.executor_kind)
    asyncio.run(runtime.run())


if __name__ == "__main__":
    main()