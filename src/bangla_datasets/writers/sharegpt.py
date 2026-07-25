"""ShareGPT-format writer for compatibility with common training frameworks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bangla_datasets.schema import Role, Trajectory

_ROLE_MAP = {
    Role.SYSTEM: "system",
    Role.USER: "human",
    Role.ASSISTANT: "gpt",
    Role.TOOL: "tool",
}


def to_sharegpt(trajectory: Trajectory) -> dict[str, Any]:
    conversations = []
    for m in trajectory.messages:
        if m.role is Role.SYSTEM:
            conversations.append({"from": "system", "value": m.content or ""})
        elif m.role is Role.TOOL:
            conversations.append(
                {"from": "tool", "value": m.content or "", "tool_call_id": m.tool_call_id}
            )
        else:
            entry = {"from": _ROLE_MAP[m.role], "value": m.content or ""}
            if m.tool_calls:
                entry["tool_calls"] = [tc.model_dump() for tc in m.tool_calls]
            conversations.append(entry)
    return {
        "id": trajectory.id,
        "domain": trajectory.domain,
        "tools": [t.model_dump() for t in trajectory.tools],
        "conversations": conversations,
    }


def write_sharegpt(trajectories: list[Trajectory], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for t in trajectories:
            f.write(json.dumps(to_sharegpt(t), ensure_ascii=False) + "\n")
