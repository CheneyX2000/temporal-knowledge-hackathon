"""Contract types: the harness's classes when available, otherwise identical local copies
(the harness accepts any object with the same fields)."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Optional
import sys


try:  # pragma: no cover - depends on where the folder is mounted
    _h = next((d / "harness" for d in Path(__file__).resolve().parents
               if (d / "harness" / "contract.py").exists()), None)
    if _h is not None and str(_h) not in sys.path:
        sys.path.insert(0, str(_h))
    from contract import Answer, Conflict, Source  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover
    @dataclass
    class Source:  # type: ignore[no-redef]
        url: str
        snapshot_id: Optional[str] = None
        quote: Optional[str] = None

    @dataclass
    class Answer:  # type: ignore[no-redef]
        text: str
        sources: list = field(default_factory=list)
        confident: bool = True
        conflict_note: Optional[str] = None

    @dataclass
    class Conflict:  # type: ignore[no-redef]
        topic: str
        values: list
        sources: list
        resolution: Optional[str] = None
        reason: Optional[str] = None
