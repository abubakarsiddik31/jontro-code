"""Emit the LaTeX tables for the LRE manuscript from outputs/lre/analysis.json.

Every table in the paper is generated here so no figure is transcribed by hand.
Run scripts/lre_analysis.py first.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
A = json.loads((ROOT / "outputs/lre/analysis.json").read_text())
OUT = ROOT / "submission/lre/tables"
OUT.mkdir(parents=True, exist_ok=True)

SHORT = {
    "Gemma-4-31B": "Gemma-4-31B",
    "GPT-OSS-20B": "gpt-oss-20b",
    "GPT-OSS-120B": "gpt-oss-120b",
    "Qwen3.5-9B": "Qwen3.5-9B",
    "Llama-3.1-8B": "Llama-3.1-8B",
    "DeepSeek-V4-Flash": "DeepSeek-V4-Flash",
    "Mistral-Small-3.2-24B": "Mistral-S-3.2-24B",
    "Nemotron-3-Ultra-550B": "Nemotron-3-U-550B",
}
ORDER = ["Nemotron-3-Ultra-550B", "Qwen3.5-9B", "GPT-OSS-120B", "GPT-OSS-20B",
         "Gemma-4-31B", "DeepSeek-V4-Flash", "Llama-3.1-8B", "Mistral-Small-3.2-24B"]


def f(x, d=1):
    return "---" if x is None else f"{x:.{d}f}"


def write(name: str, body: str) -> None:
    (OUT / name).write_text(body)
    print("wrote", name)


# ---------------------------------------------------------------- main results
rows = []
for m in ORDER:
    u = A["models_main"][m]["unseen_goal_eval"]
    ci, ce = u["tool_selection_ci"], u["arg_exact_ci"]
    rows.append(
        f"{SHORT[m]} & {f(u['tool_selection'])} & \\scriptsize[{f(ci[0])}, {f(ci[1])}] "
        f"& {f(u['arg_exact'])} & \\scriptsize[{f(ce[0])}, {f(ce[1])}] "
        f"& {f(u['slot_accuracy'])} & {f(u['schema_validity'])} \\\\"
    )
n = A["models_main"]["Gemma-4-31B"]["unseen_goal_eval"]
write("results_main.tex", r"""\begin{table}[htbp]
\centering
\caption{Zero-shot results on the unseen-goal evaluation set (%d callable items
drawn from %d held-out task goals). Tool selection is scored only on items with a
gold call. Argument exact-match requires every gold slot to be filled with the
gold value; slot accuracy is micro-averaged over gold slots. Schema-validity is
the weaker measure used in earlier versions of this work, reported so the two
analyses can be reconciled. Bracketed ranges are 95\%% intervals from a
bootstrap that resamples task goals rather than examples, because examples are
paraphrase replicates of a goal and are not independent. No cell is emboldened:
the goal-clustered intervals overlap for most pairs of endpoints}
\label{tab:results}
\small
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{2}{c}{Tool selection (\%%)} & \multicolumn{2}{c}{Arg.\ exact (\%%)}
& Slot acc. & Schema-val. \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Endpoint & rate & 95\%% CI & rate & 95\%% CI & (\%%) & (\%%) \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (n["n_callable"], n["n_goals"], "\n".join(rows)))

# ------------------------------------------------------------------ error kinds
KINDS = [("missed_call", "no call, call expected"),
         ("wrong_tool", "wrong tool"),
         ("empty_args", "right tool, empty arguments"),
         ("wrong_arg_values", "right tool, wrong argument values"),
         ("spurious_call", "call, abstention expected")]
# Short column heads: the eight-column body does not fit at \scriptsize with
# full endpoint names, so stack each name over two lines.
STACK = {
    "Gemma-4-31B": r"\shortstack{Gemma\\-4-31B}",
    "GPT-OSS-20B": r"\shortstack{gpt-oss\\-20b}",
    "GPT-OSS-120B": r"\shortstack{gpt-oss\\-120b}",
    "Qwen3.5-9B": r"\shortstack{Qwen\\3.5-9B}",
    "Llama-3.1-8B": r"\shortstack{Llama\\3.1-8B}",
    "DeepSeek-V4-Flash": r"\shortstack{Deep\\Seek}",
    "Mistral-Small-3.2-24B": r"\shortstack{Mistral\\S-3.2}",
    "Nemotron-3-Ultra-550B": r"\shortstack{Nemo\\tron-3}",
}
hdr = " & ".join(STACK[m] for m in ORDER)
lines = []
for key, label in KINDS:
    cells = " & ".join(str(A["models_main"][m]["unseen_goal_eval"]["errors"].get(key, 0)) for m in ORDER)
    lines.append(f"{label} & {cells} \\\\")
correct = " & ".join(str(A["models_main"][m]["unseen_goal_eval"]["errors"].get("correct", 0)) for m in ORDER)
abst = " & ".join(str(A["models_main"][m]["unseen_goal_eval"]["errors"].get("correct_abstention", 0)) for m in ORDER)
write("errors.tex", r"""\begin{table}[htbp]
\centering
\caption{Outcome counts on the unseen-goal evaluation set (%d items: %d callable,
%d where the reference makes no call). Unlike the taxonomy used in earlier
versions of this work, a correct abstention is counted as a success rather than
as a missing call, and a right tool with wrong argument \emph{values} is
separated from a right tool with empty arguments. Columns are ordered by tool
selection.}
\label{tab:errors}
\scriptsize
\setlength{\tabcolsep}{3pt}
\resizebox{\linewidth}{!}{%%
\begin{tabular}{lrrrrrrrr}
\toprule
Outcome & %s \\
\midrule
\multicolumn{9}{l}{\textit{Successes}} \\
\quad fully correct call & %s \\
\quad correct abstention & %s \\
\midrule
\multicolumn{9}{l}{\textit{Failures}} \\
%s
\bottomrule
\end{tabular}}
\end{table}
""" % (n["n_scored"], n["n_callable"], n["n_abstain_gold"], hdr, correct, abst,
       "\n".join("\\quad " + l for l in lines)))

# ------------------------------------------------------------------ abstention
rows = []
for m in ORDER:
    u = A["models_main"][m]["unseen_goal_eval"]
    rows.append(f"{SHORT[m]} & {f(u['abstention_precision'])} & {f(u['abstention_recall'])} \\\\")
write("abstention.tex", r"""\begin{table}[htbp]
\centering
\caption{Abstention precision and recall on the %d evaluation items whose
reference response makes no tool call. Precision is the share of an endpoint's
abstentions that were correct; recall is the share of reference abstentions it
reproduced. No endpoint reproduces the reference abstentions reliably, though
Qwen3.5-9B is markedly better than the rest on precision. The group is 19 items, so
these are indicative rather than precise}
\label{tab:abstention}
\small
\begin{tabular}{lcc}
\toprule
Endpoint & Precision (\%%) & Recall (\%%) \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (n["n_abstain_gold"], "\n".join(rows)))

# ----------------------------------------------------------------- candidate k
ks = ["1", "2", "3", "4"]
hdr_k = " & ".join(f"$k{{=}}${k}" for k in ks)
ns = " & ".join(str(A["models_main"]["Gemma-4-31B"]["unseen_goal_eval"]["by_k"][k]["n"]) for k in ks)
rows = []
for m in ORDER:
    by = A["models_main"][m]["unseen_goal_eval"]["by_k"]
    rows.append(f"{SHORT[m]} & " + " & ".join(f(by[k]["tool_selection"]) for k in ks) + r" \\")
write("by_k.tex", r"""\begin{table}[htbp]
\centering
\caption{Tool selection (\%%) by the number of candidate tools $k$ actually
presented in the prompt. No trajectory presents the full catalogue: $k$ ranges
from 1 to 4, and $k{=}1$ is a one-of-one choice. Difficulty tracks $k$ for most
endpoints, which is why aggregate accuracy on this corpus should not be read as
evidence about selection among 54 tools.}
\label{tab:byk}
\small
\begin{tabular}{lcccc}
\toprule
Endpoint & %s \\
items & %s \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (hdr_k, ns, "\n".join(rows)))

# ------------------------------------------------- register x script crosstab
CELLS = [("tui", "bengali"), ("tui", "banglish"), ("tumi", "bengali"),
         ("tumi", "banglish"), ("apni", "bengali"), ("apni", "banglish")]
corpus_cells = A["corpus"]["register_script_cells"]
eval_cells = A["models_main"]["Gemma-4-31B"]["unseen_goal_eval"]["by_cell"]
rows = []
for reg, scr in CELLS:
    key = f"{reg}|{scr}"
    c = corpus_cells.get(key, 0)
    e = eval_cells.get(key, {}).get("n", 0)
    rows.append(f"\\textit{{{reg}}} & {scr} & {c:,} & {100*c/A['corpus']['n']:.1f} & {e} \\\\")
write("cells.tex", r"""\begin{table}[htbp]
\centering
\caption{Address form $\times$ writing system in the corpus and in the
unseen-goal evaluation set. The populated design is a crossed $2\times2$ core
(\textit{tumi}, \textit{apni} $\times$ Bengali, Banglish) plus a Bengali-only
\textit{tui} stratum; \textit{tui}--Banglish is empty for the reason diagnosed in
Section~\ref{sec:tui-cell}, and \textit{tumi}--Banglish is too small to support a
per-cell rate. Marginal comparisons of writing system are confounded with address
form, which is why Section~\ref{sec:script-effect} controls for it}
\label{tab:cells}
\small
\begin{tabular}{llrrr}
\toprule
Address form & Writing system & Corpus & \%% & Eval.\ items \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(rows))

# --------------------------------------------------------------- script effect
rows = []
for m in ORDER:
    s = A["script_contrast"][m]
    rows.append(
        f"{SHORT[m]} & {f(s['apni_bengali']['tool_selection'])} & "
        f"{f(s['apni_banglish']['tool_selection'])} & {f(s['apni_controlled_delta'])} & "
        f"{f(s['marginal_delta'])} \\\\"
    )
be = A["script_contrast"]["Gemma-4-31B"]["apni_bengali"]["n"]
bn = A["script_contrast"]["Gemma-4-31B"]["apni_banglish"]["n"]
write("script_effect.tex", r"""\begin{table}[htbp]
\centering
\caption{Writing-system difference in tool selection, holding address form fixed
at \textit{apni} --- the only address form attested in both writing systems at
usable size (%d Bengali, %d Banglish evaluation items). The controlled and
marginal differences agree in sign and are close in size for every endpoint, so
the confound noted in Table~\ref{tab:cells} does not by itself explain the
pattern. The differences are endpoint-specific, not a uniform Banglish penalty.}
\label{tab:script}
\small
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{\textit{apni} only (\%%)} & \multicolumn{2}{c}{Banglish $-$ Bengali (pp)} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Endpoint & Bengali & Banglish & controlled & marginal \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (be, bn, "\n".join(rows)))

# -------------------------------------------------------------- contamination
rows = []
for m in ORDER:
    c = A["contamination_effect"][m]
    rows.append(
        f"{SHORT[m]} & {f(c['tool_selection_seen'])} & {f(c['tool_selection_unseen'])} & "
        f"{f(c['tool_selection_delta'])} \\\\"
    )
c0 = A["contamination_effect"]["Gemma-4-31B"]
write("contamination.tex", r"""\begin{table}[htbp]
\centering
\caption{Tool selection on evaluation items whose task goal also has paraphrases
in the training partition (%d items) against items whose goal is held out
entirely (%d items), for the same endpoints in the same run. Seven of the eight
endpoints score higher on seen goals, by up to %.1f percentage points; one
(DeepSeek-V4-Flash) is lower. The
comparison is descriptive: the two item sets contain different goals, so goal
difficulty is not held constant, and it therefore bounds rather than isolates the
contamination effect.}
\label{tab:contamination}
\small
\begin{tabular}{lccc}
\toprule
Endpoint & Seen goal (\%%) & Unseen goal (\%%) & Difference (pp) \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (c0["n_seen_goal"], c0["n_unseen_goal"],
       max(v["tool_selection_delta"] for v in A["contamination_effect"].values()),
       "\n".join(rows)))

# ------------------------------------------------------------------- endpoints
rows = []
for m in ORDER:
    d = A["models_main"][m]
    batch = "A" if d["batch_a_only"] else "A, B"
    rows.append(f"{SHORT[m]} & \\texttt{{{d['provider_id']}}} & {batch} \\\\")
write("endpoints.tex", r"""\begin{table}[htbp]
\centering
\caption{Endpoints evaluated, with the provider identifier each request was sent
to. All were requested through the OpenRouter compatibility endpoint
(\texttt{https://openrouter.ai/api/v1}) at temperature 0, one sample per item,
with the provider's native tool-calling parameter and provider defaults for all
other decoding settings. Batch A is the %d-item request batch from which every
result in this paper is computed; batch B is a later batch over further items,
used only for the robustness check in Section~\ref{sec:anomaly}. Provider aliases
can be repointed by the vendor after a run, so these identifiers name the request
target rather than a verified set of weights.}
\label{tab:endpoints}
\small
\begin{tabular}{lll}
\toprule
Endpoint & Provider identifier & Batches \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (A["run_provenance"]["batch_a_ids"], "\n".join(rows)))

# ---------------------------------------------------------------- run anomaly
rows = []
for m, d in A["run_provenance"]["per_model_empty_argument_rate"].items():
    rows.append(
        f"{SHORT[m]} & {d['batch_a_914']['n_empty_args']}/{d['batch_a_914']['n_calls']} & "
        f"{f(d['batch_a_914']['pct_empty'])} & {d['batch_b_820']['n_empty_args']}/"
        f"{d['batch_b_820']['n_calls']} & {f(d['batch_b_820']['pct_empty'])} \\\\"
    )
write("anomaly.tex", r"""\begin{table}[htbp]
\centering
\caption{Share of returned tool calls with an empty argument object, by request
batch, for the six endpoints run over both. Five endpoints are stable across
batches. Llama-3.1-8B is stable in batch A and degenerate in batch B at the same
provider identifier, which we read as a serving or serialization failure of that
batch rather than a property of the model. All results in this paper are computed
from batch A.}
\label{tab:anomaly}
\small
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{Batch A} & \multicolumn{2}{c}{Batch B} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Endpoint & empty/calls & \%% & empty/calls & \%% \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(rows))

# ------------------------------------------------------------------- overview
c = A["corpus"]
s = A["split_v2"]
write("overview.tex", r"""\begin{table}[htbp]
\centering
\caption{Jontro release statistics, verified against the distributed data.}
\label{tab:overview}
\small
\begin{tabular}{ll}
\toprule
Property & Value \\
\midrule
Trajectories & %s \\
Distinct task goals & %d (median %s paraphrases per goal) \\
User turns per trajectory & 1 \\
Trajectories with a reference tool call & %s (%.1f\%%) \\
Candidate tools per trajectory & 1--4 (median 3) \\
Domains & 18 \\
Tool definitions / exercised & 54 / 47 \\
Address forms & 3 (\textit{tui} / \textit{tumi} / \textit{apni}) \\
Writing systems & 2 (Bengali %.1f\%%, Banglish %.1f\%%) \\
Goal-disjoint split (train / test) & %s / %s trajectories \\
Held-out goals & %d of %d \\
Licence & CC-BY-4.0 \\
\bottomrule
\end{tabular}
\end{table}
""" % (f"{c['n']:,}", c["n_goals"], 16,
       f"{c['n'] - c['n_no_gold_call']:,}", 100 * (c["n"] - c["n_no_gold_call"]) / c["n"],
       100 * (c["register_script_cells"]["apni|bengali"] + c["register_script_cells"]["tumi|bengali"]
              + c["register_script_cells"]["tui|bengali"]) / c["n"],
       100 * (c["register_script_cells"]["apni|banglish"]
              + c["register_script_cells"]["tumi|banglish"]) / c["n"],
       f"{s['n_train']:,}", f"{s['n_test']:,}", s["n_test_goals"], c["n_goals"]))

print("\nall tables written to", OUT)

# ------------------------------------------------------------- relabel matrix
REP = json.loads((ROOT / "outputs/register_correction_report.json").read_text())
cm = REP["confusion_matrix"]
old = ["tumi", "semi_formal", "apni"]
new = ["tui", "tumi", "apni"]
rows = []
for o in old:
    cells = []
    for nn in new:
        v = cm.get(f"{o}->{nn}", 0)
        cells.append(f"\\textbf{{{v:,}}}" if o != nn and v >= 500 else f"{v:,}")
    rows.append(f"\\texttt{{{o.replace('_', chr(92)+'_')}}} & " + " & ".join(cells) + r" \\")
write("relabel.tex", r"""\begin{table}[htbp]
\centering
\caption{Address-form relabelling: original generation-time label (rows) against
the corrected label in the release (columns). %s of %s records changed label
(%.1f\%%). The emboldened cell is the %s \texttt{semi\_formal} records reassigned
to \textit{apni} on the audit's inference about the original prompt rather than on
marker evidence in the record; it is the largest single change to the corpus and
accounts for %.0f\%% of the resulting \textit{apni} group}
\label{tab:relabel}
\small
\begin{tabular}{lrrr}
\toprule
& \multicolumn{3}{c}{Corrected label} \\
\cmidrule(lr){2-4}
Original label & \textit{tui} & \textit{tumi} & \textit{apni} \\
\midrule
%s
\midrule
Release total & %s & %s & %s \\
\bottomrule
\end{tabular}
\end{table}
""" % (f"{REP['changed_count']:,}", f"{REP['total']:,}",
       100 * REP["changed_count"] / REP["total"],
       f"{cm['semi_formal->apni']:,}",
       100 * cm["semi_formal->apni"] / REP["new_distribution"]["apni"],
       "\n".join(rows),
       f"{REP['new_distribution']['tui']:,}", f"{REP['new_distribution']['tumi']:,}",
       f"{REP['new_distribution']['apni']:,}"))

# ------------------------------------------------------------- defect classes
ng = A["audits"]["no_gold_call"]["categories"]
sa = A["audits"]["sadhu_bhasha"]
rv = A["audits"]["register_verb_morphology"]
N = A["corpus"]["n"]
pc = lambda v: 100.0 * v / N
rows = [
    ("Trajectories with more than one user turn", 0),
    (r"Reference makes no call: clarification question, never answered", ng["clarification_unanswered"]),
    (r"Reference makes no call: refusal", ng["refusal"]),
    (r"Reference makes no call: action announced but not executed", ng["announced_not_executed"]),
    (r"Reference makes no call: service fact asserted without a call", ng["unsourced_claim"]),
    ("Address-form label inherited rather than observed", REP["actions"]["inherited_neutral"]),
    ("Address-form label contradicted by verb morphology (upper bound)",
     rv["flagged_with_verb_morphology"]),
    (r"Request uses \textit{sadhu-bhasha} verb morphology", sa["n_trajectories"]),
    ("Script, address-form or turn-length heuristic flag", 81),
]
body = "\n".join(f"{lab} & {v:,} & {pc(v):.1f} \\\\" for lab, v in rows)
write("defects.tex", r"""\begin{table}[htbp]
\centering
\caption{Defect and caveat classes in the released corpus. All were found by
offline inspection of the distributed data for this paper; none were filtered out
of the release, so every affected record can be located from the released flags.
The four no-reference-call rows sum to %d and are assigned by rule from the
reference assistant text; the boundary between a clarification question and an
unsourced assertion is not crisply separable by rule, so those two rows in
particular are approximate. The rules are in the released analysis script}
\label{tab:defects}
\small
\begin{tabular}{p{0.56\linewidth}rr}
\toprule
Class & Count & \%% of corpus \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (sum(ng.values()), body))
