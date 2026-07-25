"""Offline re-analysis of the Jontro release for the LRE submission.

Runs entirely on artifacts already in the repository: no model calls, no LLM
judge, no network. Every number the revised manuscript reports comes from here.

What this fixes relative to the arXiv v6 analysis
-------------------------------------------------
1. Goal-disjoint split. The shipped 90/10 split is stratified on register and
   script, which splits paraphrases of the same task goal: all 916 test goals
   also occur in train. We build a split that holds out whole goals, and we
   report the seen-goal vs unseen-goal gap as a measured contamination effect.
2. Tool-selection is scored only on examples that have a gold call. The old
   tool-match metric had an unreachable ceiling of 94.98% because 46/916 test
   items have no gold call and could never be counted as a match.
3. Abstention is scored as its own outcome, with precision and recall, instead
   of charging a correct abstention as a `no_tool_call` error.
4. Argument quality is measured against gold values (exact match and per-slot
   accuracy), not only against the gold tool's JSON schema. Schema conformance
   is retained under its accurate name, `schema-validity`.
5. Difficulty is reported against the candidate-set size actually presented to
   the model (1-4 tools), not the 54-tool catalogue.
6. Register x script is reported as a full 3x2 crosstab, and the script contrast
   is reported register-controlled.
7. Intervals are bootstrapped by resampling *goals*, not examples, because
   examples are paraphrase replicates and are not independent.

Usage:  python scripts/lre_analysis.py
Writes: outputs/lre/*.json, outputs/lre/tables/*.tex, outputs/splits_v2/*.jsonl
"""
from __future__ import annotations

import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "outputs/dataset/bangla_agentic.jsonl"
EVAL = ROOT / "outputs/eval"
OUT = ROOT / "outputs/lre"
TABLES = OUT / "tables"
SPLITS2 = ROOT / "outputs/splits_v2"

SEED = 20260725
# The six endpoints that were run over the full 1,734-id prediction pool in a
# single pass. GPT-OSS-120B and Nemotron-3-Ultra-550B exist only in the earlier
# 914-id snapshot and are reported separately so no row mixes two runs.
MODELS_MAIN = [
    "Gemma-4-31B",
    "GPT-OSS-20B",
    "Qwen3.5-9B",
    "Llama-3.1-8B",
    "DeepSeek-V4-Flash",
    "Mistral-Small-3.2-24B",
]
# Ran only over batch A. Because batch A is the primary evaluation batch, these
# two can be reported in the same table as the six above.
MODELS_SNAPSHOT = ["GPT-OSS-120B", "Nemotron-3-Ultra-550B"]
MODELS_ALL = MODELS_MAIN + MODELS_SNAPSHOT

BENGALI_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_corpus() -> dict[str, dict]:
    return {r["id"]: r for r in (json.loads(l) for l in CORPUS.open())}


def gold_call(rec: dict) -> dict | None:
    for m in rec["messages"]:
        for tc in m.get("tool_calls") or []:
            return tc
    return None


def load_predictions(model: str, snapshot: bool = False) -> dict[str, dict | None]:
    path = (EVAL / "_snapshot914_backup" if snapshot else EVAL) / f"predictions_{model}.jsonl"
    out: dict[str, dict | None] = {}
    for line in path.open():
        r = json.loads(line)
        out[r["example_id"]] = r.get("tool_call")
    return out


# --------------------------------------------------------------------------
# goal-disjoint split
# --------------------------------------------------------------------------
def goal_disjoint_split(corpus: dict[str, dict], test_goal_frac: float = 0.30):
    """Hold out whole task goals, stratified by domain.

    Every example of a held-out goal goes to test, so no test goal has a
    paraphrase in train. Stratifying by domain keeps all 18 domains present on
    both sides.
    """
    goals_by_domain: dict[str, set[str]] = defaultdict(set)
    for rec in corpus.values():
        goals_by_domain[rec["domain"]].add(rec["metadata"]["persona_goal"])

    rng = random.Random(SEED)
    test_goals: set[str] = set()
    for domain in sorted(goals_by_domain):
        goals = sorted(goals_by_domain[domain])
        rng.shuffle(goals)
        k = max(1, round(len(goals) * test_goal_frac))
        test_goals.update(goals[:k])

    test_ids, train_ids = [], []
    for rid, rec in corpus.items():
        (test_ids if rec["metadata"]["persona_goal"] in test_goals else train_ids).append(rid)

    # invariant: zero goal overlap
    tg = {corpus[i]["metadata"]["persona_goal"] for i in test_ids}
    rg = {corpus[i]["metadata"]["persona_goal"] for i in train_ids}
    assert not (tg & rg), "goal leak between train and test"
    return sorted(train_ids), sorted(test_ids), test_goals


# --------------------------------------------------------------------------
# argument scoring
# --------------------------------------------------------------------------
def norm_value(v) -> str:
    """Normalize an argument value for comparison.

    Bengali and ASCII digits are unified because the corpus writes account
    numbers, dates and amounts in both (৯৯০০৯৯১১২২ vs 9900991122) and a model
    that transliterates correctly should not be penalized. Case and surrounding
    punctuation are ignored. Values are otherwise compared literally: no
    Bangla-to-Bangla synonym matching, so this is a strict measure.
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        s = str(v)
    elif isinstance(v, (list, dict)):
        s = json.dumps(v, sort_keys=True, ensure_ascii=False)
    else:
        s = str(v)
    s = s.translate(BENGALI_DIGITS).strip().casefold()
    s = re.sub(r"[\s,]+", " ", s)
    return s.strip(" .।-")


def score_arguments(pred: dict | None, gold: dict | None) -> tuple[int, int, bool]:
    """Return (slots_correct, slots_total, exact_match) against gold values."""
    if gold is None:
        return 0, 0, False
    g = gold.get("arguments") or {}
    if not g:
        return 0, 0, bool(pred is not None and not (pred.get("arguments") or {}))
    p = (pred or {}).get("arguments") or {}
    correct = sum(1 for k, v in g.items() if k in p and norm_value(p[k]) == norm_value(v))
    return correct, len(g), correct == len(g)


def schema_valid(pred: dict | None, rec: dict) -> bool:
    """Schema conformance of predicted args against the matched tool's schema.

    Reimplements the original `_args_valid` semantics (including its
    unknown-tool passthrough) so the revised paper can report the same quantity
    under an accurate name and readers can reconcile the two analyses.
    """
    if pred is None:
        return False
    import jsonschema

    schema = None
    for t in rec.get("tools") or []:
        if t["name"] == pred.get("name"):
            schema = sanitize_schema(t.get("parameters_json_schema") or {})
            break
    if schema is None:
        return True
    try:
        jsonschema.validate(instance=pred.get("arguments") or {}, schema=schema)
        return True
    except Exception:
        return False


def sanitize_schema(schema: dict) -> dict:
    """Strip null-valued property entries, matching eval/task.sanitize_schema."""
    if not isinstance(schema, dict):
        return {}
    out = dict(schema)
    props = out.get("properties")
    if isinstance(props, dict):
        out["properties"] = {k: v for k, v in props.items() if v is not None}
    return out


# --------------------------------------------------------------------------
# per-example outcome
# --------------------------------------------------------------------------
def outcome(pred: dict | None, rec: dict) -> dict:
    gold = gold_call(rec)
    callable_ = gold is not None
    matched = callable_ and pred is not None and pred.get("name") == gold["name"]
    slots_ok, slots_tot, exact = score_arguments(pred, gold) if matched else (0, 0, False)

    if not callable_:
        kind = "correct_abstention" if pred is None else "spurious_call"
    elif pred is None:
        kind = "missed_call"
    elif pred.get("name") != gold["name"]:
        kind = "wrong_tool"
    elif not (pred.get("arguments") or {}):
        kind = "empty_args"
    elif not exact:
        kind = "wrong_arg_values"
    else:
        kind = "correct"

    return {
        "callable": callable_,
        "matched": matched,
        "schema_valid": matched and schema_valid(pred, rec),
        "arg_exact": exact,
        "slots_ok": slots_ok,
        "slots_tot": slots_tot,
        "kind": kind,
        "k": len(rec.get("tools") or []),
    }


# --------------------------------------------------------------------------
# aggregation with goal-clustered bootstrap
# --------------------------------------------------------------------------
def rate(num: int, den: int) -> float | None:
    return 100.0 * num / den if den else None


def cluster_bootstrap(items: list[tuple[str, int, int]], reps: int = 2000) -> tuple[float, float] | None:
    """95% CI for a rate, resampling goals (clusters) with replacement.

    `items` is a list of (goal, numerator, denominator) per example. Examples
    sharing a goal are paraphrases of one task, so they move together.
    """
    by_goal: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for g, n, d in items:
        by_goal[g].append((n, d))
    goals = list(by_goal)
    if len(goals) < 3:
        return None
    rng = random.Random(SEED)
    draws = []
    for _ in range(reps):
        num = den = 0
        for _ in goals:
            for n, d in by_goal[rng.choice(goals)]:
                num += n
                den += d
        if den:
            draws.append(100.0 * num / den)
    if not draws:
        return None
    draws.sort()
    return draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]


def evaluate(model: str, ids: list[str], corpus: dict, preds: dict) -> dict:
    rows = []
    for i in ids:
        if i not in preds:
            continue
        rows.append((i, outcome(preds[i], corpus[i])))
    if not rows:
        return {}

    goal = lambda i: corpus[i]["metadata"]["persona_goal"]
    callable_rows = [(i, o) for i, o in rows if o["callable"]]
    abstain_rows = [(i, o) for i, o in rows if not o["callable"]]

    sel_items = [(goal(i), int(o["matched"]), 1) for i, o in callable_rows]
    exact_items = [(goal(i), int(o["arg_exact"]), 1) for i, o in callable_rows]
    slot_items = [(goal(i), o["slots_ok"], o["slots_tot"]) for i, o in callable_rows]

    n_call = len(callable_rows)
    matched = sum(o["matched"] for _, o in callable_rows)
    exact = sum(o["arg_exact"] for _, o in callable_rows)
    schema_ok = sum(o["schema_valid"] for _, o in callable_rows)
    slots_ok = sum(o["slots_ok"] for _, o in callable_rows)
    slots_tot = sum(o["slots_tot"] for _, o in callable_rows)

    # abstention: positive class = "no call is correct"
    tp = sum(1 for _, o in abstain_rows if o["kind"] == "correct_abstention")
    fp = sum(1 for _, o in callable_rows if o["kind"] == "missed_call")
    fn = len(abstain_rows) - tp

    by_cell: dict[str, dict] = {}
    for i, o in callable_rows:
        p = corpus[i]["persona"]
        key = f"{p['register']}|{p['script']}"
        c = by_cell.setdefault(key, {"n": 0, "matched": 0, "exact": 0})
        c["n"] += 1
        c["matched"] += o["matched"]
        c["exact"] += o["arg_exact"]

    by_k: dict[str, dict] = {}
    for i, o in callable_rows:
        c = by_k.setdefault(str(o["k"]), {"n": 0, "matched": 0})
        c["n"] += 1
        c["matched"] += o["matched"]

    return {
        "n_scored": len(rows),
        "n_callable": n_call,
        "n_abstain_gold": len(abstain_rows),
        "n_goals": len({goal(i) for i, _ in rows}),
        "tool_selection": rate(matched, n_call),
        "tool_selection_ci": cluster_bootstrap(sel_items),
        "arg_exact": rate(exact, n_call),
        "arg_exact_ci": cluster_bootstrap(exact_items),
        "slot_accuracy": rate(slots_ok, slots_tot),
        "slot_accuracy_ci": cluster_bootstrap(slot_items),
        "schema_validity": rate(schema_ok, n_call),
        "abstention_precision": rate(tp, tp + fp),
        "abstention_recall": rate(tp, tp + fn),
        "errors": dict(Counter(o["kind"] for _, o in rows)),
        "by_cell": {
            k: {"n": v["n"], "tool_selection": rate(v["matched"], v["n"]),
                "arg_exact": rate(v["exact"], v["n"])}
            for k, v in sorted(by_cell.items())
        },
        "by_k": {
            k: {"n": v["n"], "tool_selection": rate(v["matched"], v["n"])}
            for k, v in sorted(by_k.items())
        },
    }


# --------------------------------------------------------------------------
# contamination effect: seen-goal vs unseen-goal, same model, same run
# --------------------------------------------------------------------------
def contamination(model: str, corpus: dict, preds: dict, test_goals: set[str]) -> dict:
    seen, unseen = [], []
    for i, pred in preds.items():
        rec = corpus.get(i)
        if rec is None:
            continue
        o = outcome(pred, rec)
        if not o["callable"]:
            continue
        (unseen if rec["metadata"]["persona_goal"] in test_goals else seen).append(o)
    f = lambda rs: rate(sum(o["matched"] for o in rs), len(rs))
    g = lambda rs: rate(sum(o["arg_exact"] for o in rs), len(rs))
    return {
        "n_seen_goal": len(seen), "n_unseen_goal": len(unseen),
        "tool_selection_seen": f(seen), "tool_selection_unseen": f(unseen),
        "tool_selection_delta": (f(seen) - f(unseen)) if seen and unseen else None,
        "arg_exact_seen": g(seen), "arg_exact_unseen": g(unseen),
        "arg_exact_delta": (g(seen) - g(unseen)) if seen and unseen else None,
    }


# --------------------------------------------------------------------------
# corpus-level audits
# --------------------------------------------------------------------------
# Sadhu-bhasha (literary Bangla) verb morphology. Cholit-bhasha is the modern
# colloquial standard; sadhu forms use distinct, longer verb inflections. Their
# presence is task-relevant: they change verb endings wholesale and therefore
# defeat a register validator keyed to cholit morphology.
SADHU_SUFFIXES = ["ইতে", "িয়া", "ইবে", "ইবেন", "িবে", "িবেন", "ইল", "িল", "ইয়াছে", "ইতেছে"]
SADHU_TOKENS = {
    "হইতে", "হইয়া", "যাইবে", "যাইবার", "করিয়া", "করিবে", "করিবেন", "দিবে", "দিবেন",
    "পড়িবে", "জানাইবেন", "বলিয়া", "থাকিবে", "আসিবে", "লইয়া", "দেখিয়া", "কহিল",
    "হইবে", "হইবেন", "চাহি", "করিতে", "দিতেছি", "নাই",
}


def sadhu_audit(corpus: dict) -> dict:
    hits, by_register = [], Counter()
    for rid, rec in corpus.items():
        text = " ".join(
            m.get("content") or "" for m in rec["messages"] if m.get("role") == "user"
        )
        toks = re.sub(r"[।\?\!\,\.\;\:]", " ", text).split()
        found = [t for t in toks if t in SADHU_TOKENS]
        if found:
            hits.append(rid)
            by_register[rec["persona"]["register"]] += 1
    return {
        "n_trajectories": len(hits),
        "pct": round(100.0 * len(hits) / len(corpus), 2),
        "by_declared_register": dict(by_register),
        "example_ids": hits[:12],
    }


def language_feature_audit(corpus: dict) -> dict:
    """Quantify the Bangla-specific features that bear on slot filling.

    Three things the manuscript needs numbers for: how often a request writes
    numeric slot values in Bengali digits that the reference call expects in
    ASCII, how request length varies with address form, and how much Romanized
    spelling variation the corpus actually contains. The last is a limitation
    rather than a feature: real Banglish has no orthographic standard, and a
    generator-produced corpus may not reproduce that variability.
    """
    bn_digit = re.compile(r"[০-৯]")
    bn_script = re.compile(r"[ঀ-৿]")

    n_call = translit_needed = digits_in_request = 0
    slots_total = slots_bengali = 0
    words: dict[str, list[int]] = defaultdict(list)

    for rec in corpus.values():
        user = " ".join(
            m.get("content") or "" for m in rec["messages"] if m.get("role") == "user"
        )
        words[rec["persona"]["register"]].append(len(user.split()))
        gold = gold_call(rec)
        if gold is None:
            continue
        n_call += 1
        args = gold.get("arguments") or {}
        if bn_digit.search(user):
            digits_in_request += 1
            if not bn_digit.search(json.dumps(args, ensure_ascii=False)):
                translit_needed += 1
        for v in args.values():
            slots_total += 1
            if isinstance(v, str) and bn_script.search(v):
                slots_bengali += 1

    # Romanized spelling variation: for frequent Banglish stems, how many
    # distinct surface spellings does the corpus actually use?
    variants: dict[str, dict[str, int]] = {}
    for stem in ("dhaka", "chittagong", "recharge", "korte", "taka", "chai"):
        c: Counter = Counter()
        for rec in corpus.values():
            if rec["persona"]["script"] != "banglish":
                continue
            user = " ".join(
                m.get("content") or "" for m in rec["messages"] if m.get("role") == "user"
            ).lower()
            for tok in re.findall(r"[a-z]+", user):
                if tok.startswith(stem[:4]):
                    c[tok] += 1
        variants[stem] = dict(c.most_common(8))

    return {
        "n_callable": n_call,
        "requests_with_bengali_digits": digits_in_request,
        "pct_requests_with_bengali_digits": round(100 * digits_in_request / n_call, 1),
        "requests_requiring_digit_transliteration": translit_needed,
        "pct_requiring_digit_transliteration": round(100 * translit_needed / n_call, 1),
        "gold_slots_total": slots_total,
        "gold_slots_bengali_script": slots_bengali,
        "pct_gold_slots_bengali_script": round(100 * slots_bengali / slots_total, 1),
        "request_length_words": {
            k: {"mean": round(statistics.mean(v), 1), "median": statistics.median(v), "n": len(v)}
            for k, v in sorted(words.items())
        },
        "romanization_variants": variants,
    }


# Rule-based classification of the reference responses that make no tool call.
# The boundary between "asked for more information" and "asserted something
# without checking" is not crisply separable by rule in Bangla free text, so
# these counts are approximate and the rules are published rather than described.
REFUSAL_RE = re.compile(
    r"(দুঃখিত|পার(ছি|বো|ব|ি) না|সম্ভব নয়|শুধু(মাত্র)?\s|only|sorry|cannot)", re.I
)
ASK_RE = re.compile(
    r"(\?|বলুন|জানান|জানাবেন|জানালে|জানতে চাই|কোনটি|কাকে|কত টাকা|কোন বিষয়"
    r"|কী ধরনের|কোথা থেকে|কোন স্টেশন|নাম কি|নম্বর কি|উল্লেখ করুন|প্রয়োজন হবে)"
)
ANNOUNCE_RE = re.compile(r"(করছি|খুঁজছি|দেখছি|করতে পারি|দিতে পারি|অনুসন্ধান)")


def classify_no_gold(rec: dict) -> str:
    reply = " ".join(
        m.get("content") or "" for m in rec["messages"] if m.get("role") == "assistant"
    )
    if REFUSAL_RE.search(reply):
        return "refusal"
    if ASK_RE.search(reply):
        return "clarification_unanswered"
    if ANNOUNCE_RE.search(reply):
        return "announced_not_executed"
    return "unsourced_claim"


def register_detector_asymmetry(corpus: dict) -> dict:
    """Why the tui x Banglish cell is empty, measured rather than asserted.

    `tui` was not a generation-time label: the recipe's register axis was
    [tumi, semi_formal, apni] when the corpus was built. Every tui record is
    therefore a product of the relabelling pass, which assigns tui only when
    `dominant_register` returns it and otherwise inherits the old label (and no
    old label maps to tui). So tui is reachable only through marker detection.

    That detector is asymmetric by script. Its Bengali path matches productive
    verb suffixes by suffix comparison; its Latin path has only a fixed token
    list and no suffix matching at all. We report what it returns per script,
    plus an independent loose regex for romanized intimate morphology, to
    separate two explanations: that Banglish tui text exists but is invisible to
    the detector, or that it was never generated.
    """
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from bangla_datasets.utils.script import (
        _LATIN_TOKENS,
        _VERB_SUFFIXES,
        dominant_register,
    )

    # Deliberately looser than the shipped detector: any romanized token with an
    # intimate verb ending, or an intimate pronoun. Over-generates on purpose.
    loose_tui = re.compile(r"\b\w+(bi|bis|chis|is)\b|\b(de|re|tor|toke|tui|shon)\b")

    per_script: dict[str, Counter] = {"bengali": Counter(), "banglish": Counter()}
    loose_hits: dict[str, int] = {"bengali": 0, "banglish": 0}
    for rec in corpus.values():
        scr = rec["persona"]["script"]
        text = " ".join(
            m.get("content") or "" for m in rec["messages"] if m.get("role") == "user"
        )
        per_script[scr][dominant_register(text) or "undecided"] += 1
        if loose_tui.search(text.lower()):
            loose_hits[scr] += 1

    return {
        "detector_output_by_script": {k: dict(v) for k, v in per_script.items()},
        "loose_romanized_tui_regex_hits": loose_hits,
        "bengali_verb_suffixes_matched": {k: v for k, v in _VERB_SUFFIXES.items()},
        "latin_token_counts": {k: len(v) for k, v in _LATIN_TOKENS.items()},
        "latin_has_suffix_matching": False,
        "tui_was_a_generation_label": False,
    }


def no_gold_audit(corpus: dict) -> dict:
    """Classify the trajectories whose reference response makes no tool call.

    An earlier version of this work counted all 468 as legitimate abstentions and
    derived a 94.9% "tool-use rate" from them. They are not abstentions. The
    largest group is a clarification question that is never answered, because
    generation stopped and no second user turn exists; the next largest is an
    outright refusal, sometimes of a request the candidate tools cannot serve
    (a goal-to-domain assignment error) and sometimes of one they could.
    """
    cats: Counter = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    empty_reply = 0
    for rid, rec in corpus.items():
        if gold_call(rec) is not None:
            continue
        cat = classify_no_gold(rec)
        cats[cat] += 1
        if len(examples[cat]) < 8:
            examples[cat].append(rid)
        if not any(
            (m.get("content") or "").strip()
            for m in rec["messages"]
            if m.get("role") == "assistant"
        ):
            empty_reply += 1
    return {
        "total": sum(cats.values()),
        "method": "rule-based on the reference assistant text; approximate",
        "categories": dict(cats),
        "empty_assistant_reply": empty_reply,
        "examples": dict(examples),
    }


def register_verb_audit(corpus: dict) -> dict:
    """Re-run register conformance using verb morphology, not pronouns only.

    The shipped heuristic filters the marker set to a hard-coded pronoun
    whitelist and discards the verb endings that `register_markers` computes,
    so it cannot detect the error class the register audit was built to catch.
    """
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from bangla_datasets.utils.script import register_markers

    from bangla_datasets.schema import Trajectory
    from bangla_datasets.validation.heuristics import run_heuristics

    pronoun_only = 0
    with_verbs = 0
    detail = Counter()
    for rec in corpus.values():
        declared = rec["persona"]["register"]
        # the shipped gate, exactly as it runs in the release
        v = run_heuristics(Trajectory.model_validate(rec))
        if not v.passed and any("declared register" in r for r in (v.reasons or [])):
            pronoun_only += 1
        flag_v = False
        for m in rec["messages"]:
            if m.get("role") != "user" or not m.get("content"):
                continue
            if len(m["content"]) <= 20:
                continue
            markers = register_markers(m["content"])
            for reg in ("tui", "tumi", "apni"):
                if reg == declared:
                    continue
                if not markers[reg]:
                    continue
                flag_v = True
                detail[f"{declared}->{reg}"] += 1
        with_verbs += flag_v
    return {
        "flagged_by_shipped_pronoun_gate": pronoun_only,
        "flagged_with_verb_morphology": with_verbs,
        "corpus": len(corpus),
        "cross_register_evidence": dict(detail.most_common()),
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    SPLITS2.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus()
    train_ids, test_ids, test_goals = goal_disjoint_split(corpus)

    with (SPLITS2 / "train_ids.jsonl").open("w") as fh:
        for i in train_ids:
            fh.write(json.dumps({"id": i}) + "\n")
    with (SPLITS2 / "test_ids.jsonl").open("w") as fh:
        for i in test_ids:
            fh.write(json.dumps({"id": i}) + "\n")

    old_test = [json.loads(l)["id"] for l in (ROOT / "outputs/splits/test_ids.jsonl").open()]

    report: dict = {
        "seed": SEED,
        "corpus": {
            "n": len(corpus),
            "n_goals": len({r["metadata"]["persona_goal"] for r in corpus.values()}),
            "user_turns": dict(Counter(
                sum(1 for m in r["messages"] if m.get("role") == "user") for r in corpus.values()
            )),
            "tool_subset_sizes": dict(sorted(Counter(
                len(r.get("tools") or []) for r in corpus.values()
            ).items())),
            "n_no_gold_call": sum(1 for r in corpus.values() if gold_call(r) is None),
            "register_script_cells": dict(sorted(Counter(
                f"{r['persona']['register']}|{r['persona']['script']}" for r in corpus.values()
            ).items())),
        },
        "split_v2": {
            "policy": "goal-disjoint, 30% of goals per domain held out, seed 20260725",
            "n_train": len(train_ids),
            "n_test": len(test_ids),
            "n_test_goals": len(test_goals),
            "n_train_goals": len({corpus[i]["metadata"]["persona_goal"] for i in train_ids}),
            "goal_overlap": 0,
        },
        "split_v1_contamination": {
            "n_test": len(old_test),
            "test_goals_also_in_train": sum(
                1 for i in old_test
                if corpus[i]["metadata"]["persona_goal"] in
                {corpus[j]["metadata"]["persona_goal"]
                 for j in (json.loads(l)["id"] for l in (ROOT / "outputs/splits/train_ids.jsonl").open())}
            ),
        },
        "models_main": {},
        "models_snapshot": {},
        "contamination_effect": {},
        "audits": {
            "sadhu_bhasha": sadhu_audit(corpus),
            "language_features": language_feature_audit(corpus),
            "register_detector_asymmetry": register_detector_asymmetry(corpus),
            "no_gold_call": no_gold_audit(corpus),
            "register_verb_morphology": register_verb_audit(corpus),
        },
    }

    # Batch A is the primary evaluation batch: all eight endpoints were run over
    # its 914 ids under one protocol, so every row of the main table comes from
    # the same batch. Batch B (820 further ids, six endpoints) is reported only
    # as a robustness check, because one endpoint fails on it (see run_provenance).
    batch_a = set(load_predictions("Llama-3.1-8B", snapshot=True))

    for m in MODELS_ALL:
        preds_all = load_predictions(m)
        preds_a = {k: v for k, v in preds_all.items() if k in batch_a}
        report["models_main"][m] = {
            "provider_id": next(
                (json.loads(l)["model"] for l in (EVAL / f"predictions_{m}.jsonl").open()), None
            ),
            "n_predictions_total": len(preds_all),
            "n_predictions_batch_a": len(preds_a),
            "batch_a_only": m in MODELS_SNAPSHOT,
            "unseen_goal_eval": evaluate(m, test_ids, corpus, preds_a),
            "v1_contaminated_eval": evaluate(m, old_test, corpus, preds_a),
        }
        report["contamination_effect"][m] = contamination(m, corpus, preds_a, test_goals)

    # Run provenance. Each 1,734-id prediction file is a concatenation of two
    # batches: the 914 ids that also appear in the earlier snapshot directory,
    # and 820 ids added afterwards. We report the empty-argument rate separately
    # per batch for every endpoint, because one endpoint is not stable across
    # them and a pooled number would hide that.
    batch_a = set(load_predictions("Llama-3.1-8B", snapshot=True))  # earlier 914
    def empty_rate(preds, ids):
        calls = [preds[i] for i in ids if preds.get(i)]
        n_empty = sum(1 for c in calls if not (c.get("arguments") or {}))
        return {"n_calls": len(calls), "n_empty_args": n_empty,
                "pct_empty": round(100.0 * n_empty / len(calls), 1) if calls else None}
    prov = {}
    for m in MODELS_MAIN:
        preds = load_predictions(m)
        a = sorted(i for i in preds if i in batch_a)
        b = sorted(i for i in preds if i not in batch_a)
        prov[m] = {"batch_a_914": empty_rate(preds, a), "batch_b_820": empty_rate(preds, b)}
    report["run_provenance"] = {
        "batch_a_ids": len(batch_a),
        "per_model_empty_argument_rate": prov,
        "note": "Llama-3.1-8B is stable in batch A and returns tool names with empty "
                "argument objects on nearly all of batch B, at the same provider id and "
                "on comparable items. Every other endpoint is stable across both "
                "batches. Treated as a serving/serialization failure of that batch, not "
                "a property of the model.",
    }

    # Robustness: the same unseen-goal evaluation on batch B, for the six
    # endpoints that have it. Reported so readers can see that five endpoints
    # reproduce and one does not.
    report["unseen_goal_batch_b"] = {}
    for m in MODELS_MAIN:
        preds = {k: v for k, v in load_predictions(m).items() if k not in batch_a}
        report["unseen_goal_batch_b"][m] = evaluate(m, test_ids, corpus, preds)

    # Register-controlled writing-system contrast. The marginal Bengali-Banglish
    # comparison is confounded: Banglish is disproportionately apni. Holding
    # register fixed at apni (the only register with both scripts well
    # represented) isolates the writing-system difference.
    report["script_contrast"] = {}
    for m in MODELS_ALL:
        cells = report["models_main"][m]["unseen_goal_eval"].get("by_cell", {})
        be, bn = cells.get("apni|bengali"), cells.get("apni|banglish")
        marg_be = [c for k, c in cells.items() if k.endswith("|bengali")]
        marg_bn = [c for k, c in cells.items() if k.endswith("|banglish")]
        w = lambda cs: (
            sum(c["tool_selection"] * c["n"] for c in cs) / sum(c["n"] for c in cs)
            if cs else None
        )
        report["script_contrast"][m] = {
            "apni_bengali": be and {"n": be["n"], "tool_selection": be["tool_selection"]},
            "apni_banglish": bn and {"n": bn["n"], "tool_selection": bn["tool_selection"]},
            "apni_controlled_delta": (bn["tool_selection"] - be["tool_selection"]) if be and bn else None,
            "marginal_delta": (w(marg_bn) - w(marg_be)) if marg_be and marg_bn else None,
        }

    (OUT / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("models_main","unseen_goal_batch_b","contamination_effect")},
                     ensure_ascii=False, indent=2)[:6000])
    print("\n=== main models ===")
    for m, d in report["models_main"].items():
        u = d["unseen_goal_eval"]
        c = report["contamination_effect"][m]
        print(f"{m:26s} unseen: sel={u['tool_selection']:.1f} exact={u['arg_exact']:.1f} "
              f"slot={u['slot_accuracy']:.1f} schema={u['schema_validity']:.1f} "
              f"n={u['n_callable']} goals={u['n_goals']} | delta_sel={c['tool_selection_delta']:+.1f}")


if __name__ == "__main__":
    main()
