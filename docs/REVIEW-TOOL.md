# Preliminary trajectory review

The Jontro source repository contains `scripts/validate_dataset.py`, a local
Flask review panel for inspecting release trajectories and recording `good`,
`bad`, or `flag` judgments in append-only JSONL.

![Review panel](review-tool.png)

Run the tool from the source repository with:

```bash
uv run --with flask python scripts/validate_dataset.py \
  --input outputs/release_v2/bangla_agentic.jsonl \
  --labels outputs/validation/labels.jsonl
```

This is deliberately described as a preliminary manual audit, not systematic
independent human validation. A formal audit needs fixed stratified sampling, two
independent native Bangla annotators, an adjudication procedure, and agreement and
quality estimates reported by rubric dimension.
