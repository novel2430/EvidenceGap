from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from evidencegap_backend.common import atomic_write_json


class AgentTraceWriter:
    def __init__(self, agent_dir: Path) -> None:
        self.agent_dir = agent_dir.resolve()
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.agent_dir / "action_trace.jsonl"
        self.path.touch(exist_ok=True)

    def append(self, event: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(dict(event), ensure_ascii=False, sort_keys=True) + "\n"
            )

    def write_workspace(self, workspace: Mapping[str, Any]) -> Path:
        path = self.agent_dir / "workspace.json"
        atomic_write_json(path, dict(workspace))
        return path
