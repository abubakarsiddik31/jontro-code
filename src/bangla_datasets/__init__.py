"""Jontro: a Bangla tool-use corpus toolkit.

Code accompanying "Jontro: a Bangla tool-use corpus and a first-call diagnostic
for Bangla-language agents". This package provides the trajectory schema, the
deterministic tool simulators and catalogue, the generation orchestrator, the
validation gates, the evaluation harness, and the partitioning utilities used to
build and evaluate the corpus.

The public API re-exports the most-used names so that callers can write
``from bangla_datasets import Trajectory, SelfPlayOrchestrator`` rather than
reaching into submodules. Submodule paths remain stable and may also be imported
directly.
"""
from bangla_datasets.eval.client import EvalClient
from bangla_datasets.eval.scoring import score
from bangla_datasets.gemini.client import GeminiClient
from bangla_datasets.generation.orchestrator import SelfPlayOrchestrator
from bangla_datasets.schema import (
    Message,
    Persona,
    Role,
    ToolCall,
    ToolDef,
    Trajectory,
    Verdict,
)
from bangla_datasets.splits.partition import partition
from bangla_datasets.tools.registry import ToolRegistry
from bangla_datasets.validation.consistency import check_consistency
from bangla_datasets.validation.heuristics import run_heuristics
from bangla_datasets.validation.judge import run_judge
from bangla_datasets.validation.schema_check import check_schema
from bangla_datasets.writers.sharegpt import to_sharegpt

__version__ = "1.1.0"

__all__ = [
    # schema
    "Message",
    "Persona",
    "Role",
    "ToolCall",
    "ToolDef",
    "Trajectory",
    "Verdict",
    # generation
    "SelfPlayOrchestrator",
    # tools
    "ToolRegistry",
    # gemini client
    "GeminiClient",
    # validation
    "check_consistency",
    "check_schema",
    "run_heuristics",
    "run_judge",
    # eval
    "EvalClient",
    "score",
    # splits
    "partition",
    # writers
    "to_sharegpt",
    "__version__",
]
