"""Stage 3: heuristic filters (pure Python).

Cheap pre-filters: turn-length sanity, repeated identical tool calls, and
script-mix detection.

The script check is *persona.script-aware*: a ``bengali`` trajectory must have
Bengali-script conversation turns; a ``banglish`` trajectory must have Latin
(Romanized) turns. A turn matching neither (e.g. English prose where Bangla or
Banglish is expected) is flagged as a language flip.
"""
from __future__ import annotations

import json
from collections import Counter

from bangla_datasets.schema import Role, Trajectory, Verdict
from bangla_datasets.utils.script import has_bengali, has_latin, register_markers

MIN_TURN_CHARS = 3
# Turns shorter than this are not script-checked — too short to be meaningful.
_MIN_SCRIPT_CHECK_LEN = 20


def run_heuristics(trajectory: Trajectory) -> Verdict:
    reasons: list[str] = []

    # length sanity
    for m in trajectory.messages:
        if (m.content is not None
                and len(m.content.strip()) < MIN_TURN_CHARS
                and m.role in (Role.USER, Role.ASSISTANT)):
            reasons.append(f"turn too short: {m.role.value}")

    # repeated identical tool calls (>=3 same name+args). Arguments are serialized
    # to a stable JSON string so list/dict-valued args (e.g. place_order's items
    # array) are hashable and comparable.
    calls = [
        (tc.name, json.dumps(tc.arguments, sort_keys=True, ensure_ascii=False))
        for m in trajectory.messages if m.tool_calls for tc in m.tool_calls
    ]
    counts = Counter(calls)
    for call, n in counts.items():
        if n >= 3:
            reasons.append(f"repeated identical tool call x{n}: {call[0]}")

    # script-mix: enforce the persona's expected script on conversation turns.
    # Old/golden data without an explicit script defaults to "bengali".
    script = getattr(trajectory.persona, "script", "bengali") or "bengali"
    for m in trajectory.messages:
        if m.role in (Role.USER, Role.ASSISTANT) and m.content:
            if len(m.content) <= _MIN_SCRIPT_CHECK_LEN:
                continue
            if script == "banglish":
                # Romanized Bangla: must have Latin letters. A pure-Bengali or
                # pure-English-prose turn here is a script flip.
                if not has_latin(m.content):
                    reasons.append(
                        f"{m.role.value} turn has no Latin script "
                        "(possible script flip; banglish expected)"
                    )
            else:  # "bengali" — the original behavior
                if not has_bengali(m.content):
                    reasons.append(
                        f"{m.role.value} turn has no Bangla script (possible language flip)"
                    )

    # register-conformance: the persona's declared honorific level must match
    # the verb morphology of the user turns. This is the gate that was missing
    # during original generation — it let tui verbs (দিবি, রে) ship under a
    # `tumi` label. We check user turns only (assistant turns are register-
    # neutral service prose). Only flag confident cross-register contradictions
    # — turns with no decisive markers are not penalized.
    declared = getattr(trajectory.persona, "register", "apni") or "apni"
    for m in trajectory.messages:
        if m.role != Role.USER or not m.content:
            continue
        if len(m.content) <= _MIN_SCRIPT_CHECK_LEN:
            continue
        markers = register_markers(m.content)
        # a cross-register pronoun (e.g. আপনি in a tui turn) is a hard signal.
        for reg in ("tui", "tumi", "apni"):
            if reg == declared:
                continue
            cross_pronouns = [t for t in markers[reg] if t in {
                "tui", "tor", "toke", "তুই", "তোর", "তোকে",
                "tumi", "tomar", "tomake", "তুমি", "তোমার", "তোমাকে",
                "apni", "apnar", "আপনি", "আপনার", "মহাশয়", "মহাশয়",
            }]
            if cross_pronouns:
                reasons.append(
                    f"user turn uses {reg} pronoun(s) {cross_pronouns} "
                    f"but declared register is '{declared}'"
                )

    if reasons:
        return Verdict(trajectory_id=trajectory.id, passed=False, stage_failed="heuristics",
                       reasons=reasons)
    return Verdict(trajectory_id=trajectory.id, passed=True)
