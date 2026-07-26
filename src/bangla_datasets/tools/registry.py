"""Tool registry: lookup, simulate, and select domain subsets."""

from bangla_datasets.schema import ToolDef
from bangla_datasets.tools.catalog import ALL_TOOLS, DOMAIN_TOOLS
from bangla_datasets.tools.simulators import simulate
from bangla_datasets.utils.seeding import SeededRNG


class ToolRegistry:
    """Lookup tools by name, execute via simulator, select domain subsets."""

    def __init__(self) -> None:
        self._by_name = {t.name: t for t in ALL_TOOLS}
        self._by_domain = DOMAIN_TOOLS

    def get(self, name: str) -> ToolDef:
        if name not in self._by_name:
            raise KeyError(f"Unknown tool: {name}")
        return self._by_name[name]

    def simulate(self, name: str, args: dict[str, object], seed: int) -> str:
        # Validate the tool exists before simulating.
        self.get(name)
        return simulate(name, args, seed)

    def subset(self, domain: str, count: int, rng: SeededRNG) -> list[ToolDef]:
        pool = self._by_domain[domain]
        n = min(count, len(pool))
        return rng.sample(pool, k=n)
