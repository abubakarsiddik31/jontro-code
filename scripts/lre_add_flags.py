"""Attach validation flags to the released corpus, and write the v2 deposit.

Section 4.3 of the LRE manuscript states that released records carry the
validation flags produced by the deterministic gates. They did not: the shipped
metadata keys were only `turns, tools_used, seed, sim_seeds, terminated,
persona_goal`, and a downstream user had nothing to filter on. This script makes
that sentence true.

For every trajectory it records, under `metadata.validation`:

  schema_pass        deterministic language-boundary gate
  consistency_pass   simulator replay, byte-compared against the stored result
  heuristics_pass    shipped heuristics (script, turn length, cross-register pronoun)
  heuristic_reasons  the reasons, when it fails
  register_verb_flag conformance including verb morphology, which the shipped
                     gate omits (Section 4.4); advisory, an upper bound on
                     mislabelling, not an error count
  sadhu_bhasha       request carries literary-register verb morphology (Section 3.4)
  reference_call     whether the reference response makes a tool call
  defect_class       for the 468 records with no reference call, which defect
                     class of Section 5.5 it falls in
  n_candidate_tools  candidate-set size k, the quantity Section 6.5 reports against
  judge_scored       always false: the judge stage never ran on this release

It also writes the goal-disjoint split manifests and both sets of Parquet files,
so the deposit matches what the paper describes.

No model calls, no network, no LLM judge. Deterministic and re-runnable.

Usage:  python scripts/lre_add_flags.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bangla_datasets.schema import Trajectory  # noqa: E402
from bangla_datasets.utils.script import register_markers  # noqa: E402
from bangla_datasets.validation.consistency import check_consistency  # noqa: E402
from bangla_datasets.validation.heuristics import run_heuristics  # noqa: E402
from bangla_datasets.validation.schema_check import check_schema  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from lre_analysis import (  # noqa: E402
    SADHU_TOKENS,
    classify_no_gold,
    gold_call,
    goal_disjoint_split,
    load_corpus,
)

SRC = ROOT / "outputs/dataset/bangla_agentic.jsonl"
DEST = ROOT / "outputs/release_v2"

CROSS_PRONOUNS = {
    "tui": {"তুই", "তোর", "তোকে", "tui", "tor", "toke"},
    "tumi": {"তুমি", "তোমার", "তোমাকে", "tumi", "tomar", "tomake"},
    "apni": {"আপনি", "আপনার", "apni", "apnar", "মহাশয়"},
}
def verb_flag(rec: dict) -> bool:
    """Conformance including verb endings, which the shipped gate discards."""
    declared = rec["persona"]["register"]
    for m in rec["messages"]:
        if m.get("role") != "user" or not m.get("content") or len(m["content"]) <= 20:
            continue
        markers = register_markers(m["content"])
        for reg in ("tui", "tumi", "apni"):
            if reg != declared and markers[reg]:
                return True
    return False


def sadhu_flag(rec: dict) -> bool:
    text = " ".join(
        m.get("content") or "" for m in rec["messages"] if m.get("role") == "user"
    )
    toks = re.sub(r"[।\?\!\,\.\;\:]", " ", text).split()
    return any(t in SADHU_TOKENS for t in toks)


def defect_class(rec: dict) -> str | None:
    """Which no-reference-call class this record falls in, or None if it has a call."""
    if gold_call(rec) is not None:
        return None
    return classify_no_gold(rec)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    records = [json.loads(l) for l in SRC.open()]

    counts: dict[str, int] = {}
    out_path = DEST / "bangla_agentic.jsonl"
    with out_path.open("w") as fh:
        for rec in records:
            traj = Trajectory.model_validate(rec)
            sv = check_schema(traj)
            cv = check_consistency(traj)
            hv = run_heuristics(traj)
            flags = {
                "schema_pass": bool(sv.passed),
                "consistency_pass": bool(cv.passed),
                "heuristics_pass": bool(hv.passed),
                "heuristic_reasons": list(hv.reasons or []) if not hv.passed else [],
                "register_verb_flag": verb_flag(rec),
                "sadhu_bhasha": sadhu_flag(rec),
                "reference_call": gold_call(rec) is not None,
                "defect_class": defect_class(rec),
                "n_candidate_tools": len(rec.get("tools") or []),
                "judge_scored": False,
            }
            rec.setdefault("metadata", {})["validation"] = flags
            for k, v in flags.items():
                if isinstance(v, bool) and v:
                    counts[k] = counts.get(k, 0) + 1
            if flags["defect_class"]:
                counts[flags["defect_class"]] = counts.get(flags["defect_class"], 0) + 1
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # split manifests: goal-disjoint (new) alongside the original, kept so that
    # results published against the original remain reproducible.
    train_ids, test_ids, _ = goal_disjoint_split(corpus)
    for name, ids in (("train", train_ids), ("test", test_ids)):
        with (DEST / f"goal_disjoint_{name}_ids.jsonl").open("w") as fh:
            for i in ids:
                fh.write(json.dumps({"id": i}) + "\n")
    for name in ("train", "test"):
        src = ROOT / f"outputs/splits/{name}_ids.jsonl"
        (DEST / f"stratified_{name}_ids.jsonl").write_text(src.read_text())

    # parquet, for both partitions
    from bangla_datasets.splits.io_parquet import write_split_parquet

    trajs = [Trajectory.model_validate(json.loads(l)) for l in out_path.open()]
    pq = DEST / "parquet"
    pq.mkdir(exist_ok=True)
    partitions = {
        "goal_disjoint": (set(train_ids), set(test_ids)),
        "stratified": (
            {json.loads(l)["id"] for l in (DEST / "stratified_train_ids.jsonl").open()},
            {json.loads(l)["id"] for l in (DEST / "stratified_test_ids.jsonl").open()},
        ),
    }
    for part, (tr, te) in partitions.items():
        for split, ids in (("train", tr), ("test", te)):
            for fmt, suffix in (("canonical", ""), ("sharegpt", "_sharegpt")):
                write_split_parquet(
                    trajs, ids, pq / f"jontro_{part}_{split}{suffix}.parquet", fmt=fmt
                )

    print(f"wrote {out_path} ({len(records):,} records, flags attached)")
    print("flag counts:", json.dumps(counts, indent=1, ensure_ascii=False))
    print("\nparquet:")
    for p in sorted(pq.glob("*.parquet")):
        print(f"  {p.name}: {p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
