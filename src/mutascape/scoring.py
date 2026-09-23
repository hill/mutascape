"""Protein language model scoring for single-site fitness effect prediction.

Provides masked marginal log-likelihood ratios as a proxy for mutational
fitness. For each position in a protein, mask the residue, run the pLM
forward pass, and compare the predicted probability of the mutant amino
acid vs the wild-type amino acid:

    score(pos, aa) = log P(aa | context) - log P(wt_aa | context)

Negative scores indicate the model predicts the substitution is
deleterious; positive scores indicate it is tolerated or beneficial.
These scores correlate well with experimental DMS fitness measurements
(Meier et al., 2021).

The output format matches mutascape's single_site_effects structure,
so pLM-predicted effects plug directly into Landscape construction.
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from mutascape.epistasis import AMINO_ACIDS

logger = logging.getLogger(__name__)

try:
    import esm
    import torch
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class PLMScorer(ABC):
    """Base class for protein language model fitness scoring.

    Subclasses implement masked marginal scoring for a specific pLM.
    The output is a dict matching mutascape's single_site_effects format.
    """

    @abstractmethod
    def score(
        self,
        sequence: str,
        positions: list[int] | None = None,
    ) -> dict[int, dict[str, float]]:
        """Compute masked marginal log-likelihood ratios.

        Parameters
        sequence : str
            Wild-type protein sequence.
        positions : list[int] or None
            1-indexed positions to score. None scores all positions.

        Returns
        dict[int, dict[str, float]]
            Maps position -> {amino_acid: log_ratio}.
            Wild-type amino acid is always 0.0.
        """
        ...


# ESM2 model configurations (same as mutantopt)
_ESM2_MODELS = {
    "esm2_t6_8M_UR50D": {"layers": 6, "dim": 320},
    "esm2_t12_35M_UR50D": {"layers": 12, "dim": 480},
    "esm2_t30_150M_UR50D": {"layers": 30, "dim": 640},
    "esm2_t33_650M_UR50D": {"layers": 33, "dim": 1280},
    "esm2_t36_3B_UR50D": {"layers": 36, "dim": 2560},
    "esm2_t48_15B_UR50D": {"layers": 48, "dim": 5120},
}


class ESM2Scorer(PLMScorer):
    """Masked marginal scoring using ESM2.

    Parameters
    model_name : str
        ESM2 model variant.
    device : str
        "auto", "cuda", "mps", or "cpu".
    batch_size : int
        Number of masked positions per forward pass.
    cache_dir : Path or None
        Directory for cached scores. Defaults to ~/.cache/mutascape/scores/.
    """

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        device: str = "auto",
        batch_size: int = 32,
        cache_dir: Path | None = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "ESM2Scorer requires torch and fair-esm. Install project dependencies with: uv sync"
            )

        if model_name not in _ESM2_MODELS:
            available = ", ".join(_ESM2_MODELS)
            raise ValueError(f"Unknown model: {model_name}. Available: {available}")

        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_dir = cache_dir or Path.home() / ".cache" / "mutascape" / "scores"

        self._model = None
        self._alphabet = None
        self._batch_converter = None

        # Device selection (same logic as mutantopt)
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

    def _load_model(self) -> None:
        if self._model is not None:
            return

        logger.info(f"Loading {self.model_name} on {self.device}")
        self._model, self._alphabet = esm.pretrained.load_model_and_alphabet(self.model_name)
        self._model = self._model.to(self.device)
        self._model.eval()
        self._batch_converter = self._alphabet.get_batch_converter()

    def score(
        self,
        sequence: str,
        positions: list[int] | None = None,
    ) -> dict[int, dict[str, float]]:
        if positions is None:
            positions = list(range(1, len(sequence) + 1))

        # Check cache (always cache all positions if scoring all)
        cached = self._load_cache(sequence)
        if cached is not None:
            if set(positions).issubset(cached.keys()):
                return {pos: cached[pos] for pos in positions}

        self._load_model()
        assert self._alphabet is not None
        assert self._batch_converter is not None

        # Build AA token index mapping
        aa_token_indices = {aa: self._alphabet.get_idx(aa) for aa in AMINO_ACIDS}

        # Tokenize the WT sequence
        _, _, wt_tokens = self._batch_converter([("wt", sequence)])
        wt_tokens = wt_tokens.to(self.device)  # shape: (1, seq_len + 2)

        mask_idx = self._alphabet.mask_idx
        results: dict[int, dict[str, float]] = {}

        # Process positions in batches
        for batch_start in range(0, len(positions), self.batch_size):
            batch_positions = positions[batch_start : batch_start + self.batch_size]

            # Build batch: one copy of the sequence per position, each with
            # a different residue masked
            batch_tokens = wt_tokens.repeat(len(batch_positions), 1)
            for i, pos in enumerate(batch_positions):
                # Position is 1-indexed; token index is pos (BOS at 0)
                batch_tokens[i, pos] = mask_idx

            with torch.no_grad():
                logits = self._model(batch_tokens)["logits"]
                # logits shape: (batch, seq_len + 2, vocab_size)

            for i, pos in enumerate(batch_positions):
                wt_aa = sequence[pos - 1]
                log_probs = F.log_softmax(logits[i, pos], dim=-1)

                wt_log_prob = log_probs[aa_token_indices[wt_aa]].item()

                effects: dict[str, float] = {}
                for aa in AMINO_ACIDS:
                    if aa == wt_aa:
                        effects[aa] = 0.0
                    else:
                        effects[aa] = log_probs[aa_token_indices[aa]].item() - wt_log_prob

                results[pos] = effects

        self._save_cache(sequence, results)
        return results

    def _cache_path(self, sequence: str) -> Path:
        seq_hash = hashlib.sha256(sequence.encode()).hexdigest()[:16]
        return self.cache_dir / f"{self.model_name}_{seq_hash}.parquet"

    def _save_cache(self, sequence: str, results: dict[int, dict[str, float]]) -> None:
        path = self._cache_path(sequence)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for pos, effects in results.items():
            for aa, score in effects.items():
                rows.append({"position": pos, "amino_acid": aa, "score": score})

        pd.DataFrame(rows).to_parquet(path, compression="snappy")
        logger.info(f"Cached scores to {path.name}")

    def _load_cache(self, sequence: str) -> dict[int, dict[str, float]] | None:
        path = self._cache_path(sequence)
        if not path.exists():
            return None

        logger.info(f"Loading cached scores from {path.name}")
        df = pd.read_parquet(path)
        results: dict[int, dict[str, float]] = {}
        for _, row in df.iterrows():
            pos = int(row["position"])
            if pos not in results:
                results[pos] = {}
            results[pos][row["amino_acid"]] = float(row["score"])
        return results
