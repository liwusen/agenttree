from __future__ import annotations

from pathlib import Path


_CURRENT_DIR = Path(__file__).resolve().parent
_NESTED_PACKAGE_DIR = _CURRENT_DIR.parent / "AgentTree" / "agenttree"

__path__ = [str(_CURRENT_DIR)]
if _NESTED_PACKAGE_DIR.exists():
    __path__.append(str(_NESTED_PACKAGE_DIR))
