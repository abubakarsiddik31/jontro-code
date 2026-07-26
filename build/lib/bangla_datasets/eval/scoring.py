"""Score predictions vs gold: metrics, sociolinguistic breakdowns, errors.

Metrics:
- tool-match rate: fraction of examples where the predicted tool name == gold.
- arg-validity rate: among tool-matches, fraction whose args conform to the
  gold tool's JSON schema.
Breakdowns by persona.register and persona.script are the empirical heart of
the paper's axes-first claim (does accuracy degrade on apni / banglish?).
"""
from __future__ import annotations

import jsonschema
from dataclasses import dataclass

from bangla_datasets.eval.task import sanitize_schema
from bangla_datasets.schema import ToolCall, Trajectory


def _args_valid(pred: ToolCall | None, traj: Trajectory) -> bool:
    """True if predicted args conform to the gold tool's parameter schema.

    We validate against the *gold tool's* schema (looked up from traj.tools).
    The schema is sanitized first (null entries stripped) because the dataset's
    tool schemas merge all params per-tool and null out unused ones, which
    strict validators reject. If we can't find the schema (unknown tool name),
    treat as valid since we can't check — the tool-name mismatch is already
    counted elsewhere.
    """
    if pred is None:
        return False
    schema = None
    for t in traj.tools:
        if t.name == pred.name:
            schema = sanitize_schema(t.parameters_json_schema)
            break
    if schema is None:
        return True  # unknown tool name; can't schema-check
    try:
        jsonschema.validate(instance=pred.arguments, schema=schema)
        return True
    except (jsonschema.ValidationError, jsonschema.SchemaError):
        return False


@dataclass
class Bucket:
    n: int = 0
    tool_match: int = 0
    arg_valid: int = 0

    def rate(self, attr: str) -> float:
        return getattr(self, attr) / self.n if self.n else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "n": float(self.n),
            "tool_match_rate": self.rate("tool_match"),
            "arg_validity_rate": self.rate("arg_valid"),
        }


@dataclass
class ModelScore:
    overall: dict[str, float]
    by_register: dict[str, dict[str, float]]
    by_script: dict[str, dict[str, float]]
    by_domain: dict[str, dict[str, float]]
    error_taxonomy: dict[str, int]


def classify_error(pred: ToolCall | None, traj: Trajectory) -> str:
    """Classify a prediction's outcome (only meaningful when not correct)."""
    gold = next(
        (tc for m in traj.messages if m.tool_calls for tc in m.tool_calls), None
    )
    if pred is None:
        return "no_tool_call"
    if gold is None:
        return "spurious_tool_call"
    if pred.name != gold.name:
        return "wrong_tool"
    if not _args_valid(pred, traj):
        return "right_tool_bad_args"
    return "correct"


def score(
    predictions: list[ToolCall | None], trajectories: list[Trajectory]
) -> ModelScore:
    assert len(predictions) == len(trajectories)
    overall = Bucket()
    by_register: dict[str, Bucket] = {}
    by_script: dict[str, Bucket] = {}
    by_domain: dict[str, Bucket] = {}
    errors: dict[str, int] = {}

    for pred, traj in zip(predictions, trajectories, strict=True):
        gold = next(
            (tc for m in traj.messages if m.tool_calls for tc in m.tool_calls), None
        )
        gold_name = gold.name if gold else None
        matched = gold_name is not None and pred is not None and pred.name == gold_name
        argv = matched and _args_valid(pred, traj)
        for store, key in (
            (overall, "_"),
            (by_register.setdefault(traj.persona.register, Bucket()), traj.persona.register),
            (by_script.setdefault(traj.persona.script, Bucket()), traj.persona.script),
            (by_domain.setdefault(traj.domain, Bucket()), traj.domain),
        ):
            store.n += 1
            if matched:
                store.tool_match += 1
            if argv:
                store.arg_valid += 1
        kind = classify_error(pred, traj)
        if kind != "correct":
            errors[kind] = errors.get(kind, 0) + 1

    return ModelScore(
        overall=overall.as_dict(),
        by_register={k: v.as_dict() for k, v in by_register.items()},
        by_script={k: v.as_dict() for k, v in by_script.items()},
        by_domain={k: v.as_dict() for k, v in by_domain.items()},
        error_taxonomy=errors,
    )
