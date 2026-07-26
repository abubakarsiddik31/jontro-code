"""Validation gates. Each stage returns a Verdict; stages are independent."""
from bangla_datasets.validation.consistency import check_consistency
from bangla_datasets.validation.heuristics import run_heuristics
from bangla_datasets.validation.schema_check import check_schema

__all__ = ["check_consistency", "check_schema", "run_heuristics"]
