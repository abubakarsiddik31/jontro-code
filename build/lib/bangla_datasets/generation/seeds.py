"""Seed-plan sampling from a recipe. Drives diversity at seed time."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from bangla_datasets.generation.orchestrator import SeedPlan
from bangla_datasets.schema import Persona
from bangla_datasets.tools.registry import ToolRegistry
from bangla_datasets.utils.seeding import SeededRNG


@dataclass
class Recipe:
    name: str
    target_count: int
    max_turns: int
    pass_threshold: int
    domains: dict[str, Any]
    goal_complexity_mix: dict[str, float]
    persona_axes: dict[str, Any]
    goal_templates: dict[str, list[str]]

    @property
    def domain_names(self) -> list[str]:
        return list(self.domains)


def load_recipe(path: str | Path) -> Recipe:
    data = yaml.safe_load(Path(path).read_text())
    recipe = Recipe(
        name=data["name"],
        target_count=data["target_count"],
        max_turns=data["max_turns"],
        pass_threshold=data["pass_threshold"],
        domains=data["domains"],
        goal_complexity_mix=data["goal_complexity_mix"],
        persona_axes=data["persona_axes"],
        goal_templates=data["goal_templates"],
    )
    # Key-set consistency: every domain must have goal templates (and vice versa).
    # A divergence would otherwise surface as a confusing KeyError mid-generation.
    domain_keys = set(recipe.domains)
    template_keys = set(recipe.goal_templates)
    if domain_keys != template_keys:
        missing_templates = domain_keys - template_keys
        missing_domains = template_keys - domain_keys
        details = []
        if missing_templates:
            details.append(f"domains without goal_templates: {sorted(missing_templates)}")
        if missing_domains:
            details.append(f"goal_templates without domains: {sorted(missing_domains)}")
        raise ValueError(
            "Recipe key-set mismatch — domains and goal_templates must match "
            f"({'; '.join(details)})"
        )
    return recipe


def _sample_axis(rng: SeededRNG, pa: dict, key: str, default: Any) -> Any:
    """Sample a persona axis that may be a uniform ``list`` or weighted ``dict``.

    Lets recipes express ratios like ``{english: 0.6, bangla: 0.4}`` directly.
    Missing keys fall back to *default*.
    """
    val = pa.get(key)
    if val is None:
        return default
    if isinstance(val, dict):
        opts = list(val.keys())
        weights = list(val.values())
        return rng.weighted_choice(opts, weights)
    return rng.choice(val)  # list → uniform


def sample_seed(
    idx: int,
    rng: SeededRNG,
    recipe: Recipe,
    registry: ToolRegistry | None = None,
) -> SeedPlan:
    """Sample one SeedPlan. Deterministic given (idx, rng state)."""
    registry = registry or ToolRegistry()
    base_seed = idx  # aligns trajectory id space with CLI checkpoint ids

    # Domain by weight.
    domains = list(recipe.domains.keys())
    weights = [recipe.domains[d]["weight"] for d in domains]
    domain = rng.weighted_choice(domains, weights)

    # Tool subset.
    lo, hi = recipe.domains[domain]["tool_count_range"]
    count = rng.integers(lo, hi + 1)
    tool_subset = registry.subset(domain, count, rng)

    # Goal complexity.
    complexities = list(recipe.goal_complexity_mix.keys())
    cw = list(recipe.goal_complexity_mix.values())
    complexity = rng.weighted_choice(complexities, cw)

    # Goal template.
    template = rng.choice(recipe.goal_templates[domain])

    # Persona axes — supports both list (uniform) and dict (weighted) values.
    pa = recipe.persona_axes
    persona = Persona(
        age=rng.integers(pa["age_range"][0], pa["age_range"][1] + 1),
        location=rng.choice(pa["locations"]),
        region=rng.choice(pa["regions"]),
        profession=_sample_axis(rng, pa, "professions", None),
        register=_sample_axis(rng, pa, "registers", "apni"),
        tech_literacy=rng.choice(pa["tech_literacy"]),
        script=_sample_axis(rng, pa, "scripts", "bengali"),
        system_prompt_language=_sample_axis(rng, pa, "system_prompt_languages", "bangla"),
    )

    goal = f"[{complexity}] {template}"
    return SeedPlan(
        seed=base_seed,
        domain=domain,
        tool_subset=tool_subset,
        persona=persona,
        goal=goal,
    )
