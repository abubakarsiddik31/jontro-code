"""Stage 1: schema & structural checks (pure Python)."""
from bangla_datasets.schema import Role, Trajectory, Verdict


def check_schema(trajectory: Trajectory) -> Verdict:
    """Validate structural integrity. Pydantic already enforced most on construction."""
    msgs = trajectory.messages
    if not msgs:
        return Verdict(trajectory_id=trajectory.id, passed=False, stage_failed="schema",
                       reasons=["empty messages"])
    non_system = [m for m in msgs if m.role is not Role.SYSTEM]
    if not non_system or non_system[0].role is not Role.USER:
        return Verdict(trajectory_id=trajectory.id, passed=False, stage_failed="schema",
                       reasons=["first non-system message must be user"])
    # tool_call_ids must be unique
    ids = [tc.id for m in msgs if m.tool_calls for tc in m.tool_calls]
    if len(ids) != len(set(ids)):
        return Verdict(trajectory_id=trajectory.id, passed=False, stage_failed="schema",
                       reasons=["duplicate tool_call ids"])
    # every tool message must reference an existing tool_call_id
    emitted = set(ids)
    for m in msgs:
        if m.role is Role.TOOL and m.tool_call_id not in emitted:
            return Verdict(trajectory_id=trajectory.id, passed=False, stage_failed="schema",
                           reasons=[f"tool message references unknown tool_call_id "
                                    f"{m.tool_call_id}"])
    return Verdict(trajectory_id=trajectory.id, passed=True)
