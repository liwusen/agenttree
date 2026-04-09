from __future__ import annotations

import uvicorn

from agenttree.config import get_settings
from agenttree.core.app import create_app


def main() -> None:
    settings = get_settings()
    uvicorn.run(create_app(), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()