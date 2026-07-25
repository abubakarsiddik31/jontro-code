# Jontro — code accompanying the paper

Code for **"Jontro: a Bangla tool-use corpus and a first-call diagnostic for
Bangla-language agents"**.

The corpus itself is a separate deposit:
<https://huggingface.co/datasets/abubakar-siddik/Jontro> (CC-BY-4.0).
This repository is the code needed to **use the corpus, verify every number in the
paper, and add your own model to the evaluation**.

## What this contains

| Path | What it is |
|---|---|
| `src/bangla_datasets/schema.py` | The trajectory schema. Load the corpus with this. |
| `src/bangla_datasets/tools/catalog.py` | All 54 tool definitions and their JSON schemas. |
| `src/bangla_datasets/tools/simulators.py` | The deterministic simulators. Every tool result in the corpus is recomputable byte-for-byte from its stored seed with these. |
| `src/bangla_datasets/validation/` | The three deterministic gates whose pass rates are reported in the paper: schema, consistency (simulator replay), heuristics. |
| `src/bangla_datasets/utils/script.py` | Bangla script and address-form marker detection. Section 3.2.1 of the paper analyses this module's behaviour directly. |
| `src/bangla_datasets/eval/` | The evaluation harness: prompt construction, the endpoint client, and scoring. Use this to add a model. |
| `src/bangla_datasets/splits/` | Partitioning and Parquet I/O for both released splits. |
| `src/bangla_datasets/gemini/prompts.py` | The prompt templates used to generate the corpus, including the persona, address-form and judge-rubric text. |
| `recipes/agentic_v1.yaml` | The generation recipe: domain weights, persona axes, goal templates. |
| `scripts/lre_analysis.py` | Produces every number in the paper. Offline, no network. |
| `scripts/lre_tables.py` | Renders those numbers into the paper's LaTeX tables. |
| `scripts/lre_add_flags.py` | Attaches the validation flags to the corpus and writes the release bundle. |

## What this deliberately does not contain

The **generation orchestrator** — the multi-agent loop that drives the persona,
user and assistant roles, its seed sampler, its provider client and its
checkpointing — is not included. Neither is the LLM judge implementation, which
was never executed on the released corpus (see §4.3 of the paper).

This is a choice, not a constraint, and the paper states it. The consequence is
scoped: nothing in the paper's results depends on the withheld code, because the
corpus is released, the simulators that produced its tool results are here, and the
analysis that produced every table is here. What the withheld code would let you do
is cheaply regenerate or extend the corpus. The prompt templates and the recipe
*are* included, so the data's provenance can be assessed even though the harness
around them is not.

## Reproducing the paper

Everything runs offline against the public corpus. No model calls, no API keys.

```bash
pip install -e .

# fetch the corpus into outputs/dataset/ and outputs/splits/,
# and the raw predictions into outputs/eval/ (both from the data deposit)

python scripts/lre_analysis.py    # -> outputs/lre/analysis.json, outputs/splits_v2/
python scripts/lre_tables.py      # -> the paper's LaTeX tables
python scripts/lre_add_flags.py   # -> outputs/release_v2/ with validation flags
```

`lre_tables.py` regenerates every data table in the paper. They are generated
rather than transcribed, so a mismatch between the paper and the corpus is
detectable by running this.

## Adding a model to the evaluation

`eval/client.py` speaks the OpenAI Chat Completions shape, so any
OpenAI-compatible provider works. Set `EVAL_API_KEY` and `EVAL_BASE_URL`, then
score with `eval/scoring.py`. The paper's runs used temperature 0, one sample per
item, and the provider's native tool-calling parameter.

Please score on the **goal-disjoint** split. The corpus is a paraphrase expansion
of 317 task goals, and the older address-form-stratified split places paraphrases
of every test goal in the training partition (§5.2). It is retained only so that
previously published numbers remain reproducible.

## Known defects

The corpus is unfiltered and contains documented defect classes — 468 reference
responses that make no tool call, address-form labels that are partly inferred
rather than observed, and an empty *tui*×Banglish cell. Section 5.5 and Section 8
of the paper enumerate them with counts, and `scripts/lre_add_flags.py` writes a
flag onto every affected record so you can filter them. Read those before training
on this.

## Citation

```bibtex
@article{siddik2026jontro,
  title  = {Jontro: a {Bangla} tool-use corpus and a first-call diagnostic for
            {Bangla}-language agents},
  author = {Siddik, Abu Bakar},
  year   = {2026},
  note   = {Under review}
}
```

## Licence

MIT, see `LICENSE`. The corpus is licensed separately under CC-BY-4.0.
