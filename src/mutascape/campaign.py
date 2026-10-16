"""Budgeted, round-based campaigns for scoring a search method against an oracle.

A directed evolution campaign measures a plate at a time. It proposes a batch,
waits for the assay, retrains, and proposes again. Scoring a method against a
landscape therefore needs a budget, a round structure, and an assay that answers
only what the campaign paid for.

A campaign never exposes the oracle or the optimum while it runs. The score is
available afterwards through ``result``, which reads the true fitness of the
variant the campaign would ship rather than the read that selected it. A method
that chases assay noise is penalised for it.
"""

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mutascape.mutant import Mutant, VariantLike, parse_variant

DEFAULT_BATCH_SIZE = 96
"""One microplate. A suggested plate size for a method to read off
``campaign.batch_size``; the campaign does not enforce it, because a method is
free to spend its budget in whatever shape it likes."""


def _noise_draw(label: str, seed: int) -> float:
    """Standard normal pinned to a variant, so a repeated read never changes."""
    digest = hashlib.blake2b(f"{seed!r}|{label}".encode(), digest_size=16).digest()
    return float(np.random.default_rng(int.from_bytes(digest[:8], "little")).normal())


@dataclass(frozen=True)
class CampaignResult:
    """What one method achieved against one landscape."""

    best_found: float
    best_observed: float
    optimum: float
    worst: float
    simple_regret: float
    queries: int
    rounds: int
    history: pd.DataFrame

    def to_dict(self) -> dict[str, float | int]:
        """Flatten the scalar metrics, dropping the per-query history."""
        return {
            "best_found": self.best_found,
            "best_observed": self.best_observed,
            "optimum": self.optimum,
            "worst": self.worst,
            "simple_regret": self.simple_regret,
            "queries": self.queries,
            "rounds": self.rounds,
        }


class SearchMethod(Protocol):
    """A campaign strategy.

    Spend the budget however you like. ``campaign.measure`` is the only way to
    learn a fitness, and it costs one query for each variant not measured
    before. Return nothing. The campaign records everything.
    """

    def __call__(self, campaign: Campaign) -> None: ...


class Campaign:
    """A budgeted, round-based view of one landscape.

    The oracle stays private. A method reaches fitness only through ``measure``,
    so the budget is the whole story and no method can read a value it did not
    pay for.
    """

    def __init__(
        self,
        oracle,
        candidates: Sequence[VariantLike],
        *,
        budget: int,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seed: int = 0,
        noise: float = 0.0,
    ) -> None:
        if budget <= 0:
            raise ValueError("budget must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if noise < 0:
            raise ValueError("noise must be non-negative")
        if len(candidates) == 0:
            raise ValueError("a campaign needs at least one candidate")

        self._oracle = oracle
        self._candidates = tuple(self._parse(variant) for variant in candidates)
        self._index = {mutant.label: position for position, mutant in enumerate(self._candidates)}
        if len(self._index) != len(self._candidates):
            raise ValueError("candidates contain a duplicate variant")

        self.budget = int(budget)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.noise = float(noise)

        self._truth: dict[str, float] = {}
        self._read: dict[str, float] = {}
        self._order: list[str] = []
        self._rounds: list[int] = []
        self._round = 0
        self._noise_scale: float | None = None
        self._design: NDArray[np.float32] | None = None
        self._columns: dict[tuple[int, str], int] | None = None

    def _parse(self, variant: VariantLike) -> Mutant:
        domain = getattr(self._oracle, "domain", None)
        if domain is not None:
            return domain.parse(variant)
        return parse_variant(variant)

    # Candidate view -----------------------------------------------------

    @property
    def candidates(self) -> tuple[Mutant, ...]:
        """Every variant this campaign may measure."""
        return self._candidates

    @property
    def unmeasured(self) -> tuple[Mutant, ...]:
        return tuple(mutant for mutant in self._candidates if mutant.label not in self._read)

    def encode(self, variants: Iterable[VariantLike] | None = None) -> NDArray[np.float32]:
        """One-hot design over the substitutions the candidate set contains."""
        if self._design is None:
            columns: dict[tuple[int, str], int] = {}
            for mutant in self._candidates:
                for position, amino_acid in sorted(mutant.items()):
                    columns.setdefault((int(position), amino_acid), len(columns))
            design = np.zeros((len(self._candidates), len(columns)), dtype=np.float32)
            for row, mutant in enumerate(self._candidates):
                for position, amino_acid in mutant.items():
                    design[row, columns[(int(position), amino_acid)]] = 1.0
            self._columns = columns
            self._design = design
        if variants is None:
            return self._design
        rows = [self._index[self._parse(variant).label] for variant in variants]
        return self._design[np.asarray(rows, dtype=np.int64)]

    def sample(self, n: int, *, seed: int | None = None) -> list[Mutant]:
        """Draw distinct unmeasured candidates uniformly."""
        pool = self.unmeasured
        rng = np.random.default_rng(self.seed if seed is None else seed)
        take = min(int(n), len(pool))
        if take <= 0:
            return []
        chosen = rng.choice(len(pool), size=take, replace=False)
        return [pool[int(position)] for position in sorted(chosen.tolist())]

    # Measurement --------------------------------------------------------

    @property
    def spent(self) -> int:
        return len(self._order)

    @property
    def remaining(self) -> int:
        return self.budget - self.spent

    @property
    def round(self) -> int:
        return self._round

    def measure(self, variants: VariantLike | Iterable[VariantLike]) -> NDArray[np.float64]:
        """Assay a batch and return its reads, spending one query per new variant.

        A repeat is free and returns the original read, because a campaign that
        already holds a measurement does not buy it twice. A variant the budget
        cannot reach returns ``nan``, so a caller taking an argmax can never
        select a variant it did not pay for.
        """
        requested = self._as_list(variants)
        reads: list[float] = []
        for mutant in requested:
            label = mutant.label
            if label not in self._read:
                if self.remaining <= 0:
                    reads.append(float("nan"))
                    continue
                truth = float(self._oracle.evaluate(mutant))
                self._truth[label] = truth
                self._read[label] = truth + self._noise_for(label)
                self._order.append(label)
                self._rounds.append(self._round)
            reads.append(self._read[label])
        self._round += 1
        return np.asarray(reads, dtype=np.float64)

    def _as_list(self, variants: VariantLike | Iterable[VariantLike]) -> list[Mutant]:
        # A single variant may itself be a Mapping, which is also an iterable of
        # its own positions, so it has to be matched before the iterable branch.
        if isinstance(variants, str | Mapping):
            return [self._parse(cast(VariantLike, variants))]
        return [self._parse(variant) for variant in variants]

    def _noise_for(self, label: str) -> float:
        if self.noise <= 0:
            return 0.0
        if self._noise_scale is None:
            # Scale noise by the spread of the landscape itself, so one setting
            # means the same thing across assays with different units.
            values = self._oracle.evaluate_batch(list(self._candidates))
            finite = np.asarray(values, dtype=np.float64)
            finite = finite[np.isfinite(finite)]
            self._noise_scale = float(np.std(finite)) if finite.size else 1.0
        return self.noise * self._noise_scale * _noise_draw(label, self.seed)

    # Observations -------------------------------------------------------

    @property
    def observed(self) -> pd.DataFrame:
        """Every measurement so far, in the order the campaign made it."""
        return pd.DataFrame(
            {
                "variant": list(self._order),
                "fitness": [self._read[label] for label in self._order],
                "round": list(self._rounds),
            }
        )

    @property
    def best(self) -> float:
        """Highest read so far."""
        if not self._order:
            return float("nan")
        return max(self._read[label] for label in self._order)

    # Scoring ------------------------------------------------------------

    def result(self) -> CampaignResult:
        """Score the campaign against the whole candidate set.

        This evaluates every candidate, so call it only after the run. The
        shipped variant is the one with the highest read, and its score is the
        true fitness underneath that read.
        """
        values = np.asarray(self._oracle.evaluate_batch(list(self._candidates)), dtype=np.float64)
        finite = values[np.isfinite(values)]
        optimum = float(finite.max()) if finite.size else float("nan")
        worst = float(finite.min()) if finite.size else float("nan")
        if self._order:
            shipped = max(self._order, key=lambda label: self._read[label])
            best_found = self._truth[shipped]
            best_observed = self._read[shipped]
        else:
            best_found = float("nan")
            best_observed = float("nan")
        span = optimum - worst
        regret = float("nan") if span <= 0 else (optimum - best_found) / span
        return CampaignResult(
            best_found=best_found,
            best_observed=best_observed,
            optimum=optimum,
            worst=worst,
            simple_regret=regret,
            queries=self.spent,
            rounds=self._round,
            history=self.observed,
        )


def run_campaign(
    method: SearchMethod,
    oracle,
    candidates: Sequence[VariantLike],
    *,
    budget: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 0,
    noise: float = 0.0,
) -> CampaignResult:
    """Run one method against one landscape and score it."""
    campaign = Campaign(
        oracle,
        candidates,
        budget=budget,
        batch_size=batch_size,
        seed=seed,
        noise=noise,
    )
    method(campaign)
    return campaign.result()
