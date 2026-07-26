"""Zero-shot single-turn tool-call eval task.

Given a Bangla system prompt + first user turn + tool catalog, the model must
emit the correct first tool call with valid arguments. The gold label is the
trajectory's actual first assistant tool call.

Register mapping (paper terminology):
    tui   = intimate (তুই/তোর)
    tumi  = familiar (তুমি/তোমাকে)
    apni  = formal/respectful (আপনি)
"""
from __future__ import annotations

from typing import Any

from bangla_datasets.schema import Role, ToolCall, ToolDef, Trajectory


def sanitize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip null-valued entries from a tool's parameter schema.

    The dataset's tool schemas merge all known parameters across all tools and
    null out the ones a given tool doesn't use (e.g. ``"specialty": null``).
    Gemini tolerates these nulls, but strict JSON-Schema validators used by some
    OpenAI-compatible providers (Groq) reject them (additionalProperties must be
    object or boolean, not null). This sanitizer produces a clean schema that
    passes both, without modifying the source data.
    """
    if not isinstance(schema, dict):
        return schema
    cleaned: dict[str, Any] = {}
    for key, val in schema.items():
        if val is None:
            continue
        if isinstance(val, dict):
            val = sanitize_schema(val)
        elif isinstance(val, list):
            val = [sanitize_schema(v) if isinstance(v, dict) else v for v in val]
        cleaned[key] = val
    return cleaned


def tool_to_openai_function(tool: ToolDef) -> dict[str, Any]:
    """Convert a ToolDef to an OpenAI function-calling tool definition.

    Schema is sanitized (nulls stripped) so strict validators accept it.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": sanitize_schema(tool.parameters_json_schema),
        },
    }


def build_chat_messages(traj: Trajectory) -> list[dict[str, Any]]:
    """System prompt + original first user turn."""
    first_user = next(
        (m.content for m in traj.messages if m.role is Role.USER and m.content), ""
    )
    return [
        {"role": "system", "content": traj.system_prompt},
        {"role": "user", "content": first_user},
    ]


def build_tools_param(traj: Trajectory) -> list[dict[str, Any]]:
    """OpenAI-style tools list for providers that support function-calling."""
    return [tool_to_openai_function(t) for t in traj.tools]


def extract_gold(traj: Trajectory) -> ToolCall | None:
    """The trajectory's first assistant tool call, or None."""
    for m in traj.messages:
        if m.tool_calls:
            return m.tool_calls[0]
    return None
