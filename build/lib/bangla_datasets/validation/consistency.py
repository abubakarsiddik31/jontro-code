"""Stage 2: simulator consistency + language-boundary check (pure Python).

This is the anti-hallucination gate. For every tool message we recompute the
expected output via the *same* deterministic simulator the orchestrator used
(`bangla_datasets.tools.simulators.simulate`), seeded by the per-call seed
stored in `trajectory.metadata["sim_seeds"]`, and byte-compare it against the
tool message content. A mismatch means the trajectory was tampered with or the
orchestrator had a bug. We also re-check the language boundary (tool names and
argument keys must be ASCII identifiers) as defense-in-depth — we do not rely
solely on the Pydantic validator that ran at construction time.
"""
from __future__ import annotations

from bangla_datasets.schema import Role, Trajectory, Verdict
from bangla_datasets.tools.simulators import simulate
from bangla_datasets.utils.script import has_bangla_key, is_ascii_identifier


def check_consistency(trajectory: Trajectory) -> Verdict:
    reasons: list[str] = []
    allowed_tools = {t.name for t in trajectory.tools}
    sim_seeds: dict = trajectory.metadata.get("sim_seeds", {})

    for m in trajectory.messages:
        if m.role is Role.TOOL:
            # find the corresponding tool call
            tc = next(
                (c for msg in trajectory.messages if msg.tool_calls for c in msg.tool_calls
                 if c.id == m.tool_call_id),
                None,
            )
            if tc is None:
                reasons.append(f"orphan tool message {m.tool_call_id}")
                continue
            # tool must be in subset
            if tc.name not in allowed_tools:
                reasons.append(f"phantom tool not in subset: {tc.name}")
                continue
            # tool name + arg keys must be ASCII (language boundary)
            if not is_ascii_identifier(tc.name):
                reasons.append(f"non-ASCII tool name: {tc.name}")
            for k in tc.arguments:
                if isinstance(k, str) and (
                    has_bangla_key(k) or not k.replace("_", "").isascii()
                ):
                    reasons.append(f"non-ASCII arg key in {tc.name}: {k}")
            # output must equal simulator recompute (anti-hallucination)
            seed = sim_seeds.get(tc.id)
            if seed is None:
                reasons.append(f"missing sim_seed for {tc.id}")
                continue
            expected = simulate(tc.name, tc.arguments, seed)
            if m.content != expected:
                reasons.append(
                    f"simulator mismatch for {tc.id}: tool output does not match recompute"
                )

    if reasons:
        return Verdict(trajectory_id=trajectory.id, passed=False, stage_failed="consistency",
                       reasons=reasons)
    return Verdict(trajectory_id=trajectory.id, passed=True)
