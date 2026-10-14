"""Search methods with matched evaluation budgets for MLDE and ALDE benchmarks.

Model methods start with 32 new observations. Known measurements cost no budget.
Local searches start from the supplied starting row. A recorded random fallback
spends any remaining budget when a method stops before exhausting the library.
"""

from dataclasses import dataclass
from functools import wraps
from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm, rankdata
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from mutascape.benchmarks.library import BudgetedOracle, SearchSpace

INITIAL_DESIGN = 32
BATCH_SIZE = 32
SEARCH_PROTOCOL_VERSION = "new-observations-v2"
# Scoring every genotype of a 160,000-row cube at every round dominates runtime
# and changes no ranking, so acquisition draws a fresh pool each round. The
# incumbent's neighbours are always included, which keeps the peak reachable.
ACQUISITION_POOL = 20_000


@dataclass
class _Features:
    """One-hot design built once per landscape and shared by every method."""

    space: SearchSpace
    _one_hot: NDArray[np.float32] | None = None

    def one_hot(self, rows: NDArray[np.int64]) -> NDArray[np.float32]:
        if self._one_hot is None:
            self._one_hot = self.space.one_hot()
        return self._one_hot[rows]


def initial_rows(
    space: SearchSpace, seed: int, size: int, *, exclude: set[int] | None = None
) -> NDArray[np.int64]:
    """Sample initial rows without replacement, excluding known observations."""
    rng = np.random.default_rng(seed)
    if exclude:
        available = np.setdiff1d(np.arange(space.size), np.fromiter(exclude, dtype=int))
        return rng.choice(available, size=min(size, len(available)), replace=False)
    take = min(size, space.size)
    return rng.choice(space.size, size=take, replace=False)


def _pool(
    space: SearchSpace,
    rng: np.random.Generator,
    seen: set[int],
    incumbent: int | None,
) -> NDArray[np.int64]:
    """Unqueried rows to score this round."""
    if space.size <= ACQUISITION_POOL:
        rows = np.arange(space.size)
    else:
        rows = rng.choice(space.size, size=ACQUISITION_POOL, replace=False)
        if incumbent is not None:
            rows = np.unique(np.concatenate([rows, np.asarray(space.neighbours(incumbent))]))
    mask = np.fromiter((int(row) not in seen for row in rows), dtype=bool, count=len(rows))
    return rows[mask]


def _top_rows(pool: NDArray[np.int64], scores: NDArray[np.float64], take: int) -> NDArray[np.int64]:
    take = min(take, len(pool))
    if take <= 0:
        return np.empty(0, dtype=np.int64)
    offsets = np.argpartition(scores, -take)[-take:]
    return pool[offsets]


class _PairwiseLinear:
    """Ridge over site and site-pair indicators, scored by table lookup.

    A dense pairwise design over a 160,000-row cube needs gigabytes. Every
    feature is an indicator, so a prediction is a sum of learned table entries
    and the design never has to exist outside the training rows.
    """

    def __init__(self, sizes: tuple[int, ...], alpha: float = 1.0, max_order: int = 2) -> None:
        self.sizes = sizes
        self.alpha = alpha
        self.max_order = max_order
        self.pairs = list(combinations(range(len(sizes)), 2)) if max_order >= 2 else []
        self.intercept = 0.0
        self.singles: list[NDArray[np.float64]] = []
        self.couples: dict[tuple[int, int], NDArray[np.float64]] = {}

    def _design(self, codes: NDArray[np.int16]) -> NDArray[np.float32]:
        blocks = []
        for site, size in enumerate(self.sizes):
            block = np.zeros((len(codes), size), dtype=np.float32)
            block[np.arange(len(codes)), codes[:, site].astype(np.int64)] = 1.0
            blocks.append(block)
        for first, second in self.pairs:
            size = self.sizes[first] * self.sizes[second]
            block = np.zeros((len(codes), size), dtype=np.float32)
            flat = codes[:, first].astype(np.int64) * self.sizes[second] + codes[:, second]
            block[np.arange(len(codes)), flat] = 1.0
            blocks.append(block)
        return np.concatenate(blocks, axis=1)

    def fit(self, codes: NDArray[np.int16], values: NDArray[np.float64]) -> _PairwiseLinear:
        feature_count = sum(self.sizes) + sum(
            self.sizes[first] * self.sizes[second] for first, second in self.pairs
        )
        if feature_count > 4096:
            return self._fit_dual(codes, values)
        model = Ridge(alpha=self.alpha).fit(self._design(codes), values)
        weights = model.coef_
        self.intercept = float(model.intercept_)
        cursor = 0
        self.singles = []
        for size in self.sizes:
            self.singles.append(weights[cursor : cursor + size])
            cursor += size
        self.couples = {}
        for first, second in self.pairs:
            size = self.sizes[first] * self.sizes[second]
            self.couples[(first, second)] = weights[cursor : cursor + size].reshape(
                self.sizes[first], self.sizes[second]
            )
            cursor += size
        return self

    def _fit_dual(self, codes: NDArray[np.int16], values: NDArray[np.float64]) -> _PairwiseLinear:
        """Fit the same indicator regression without a large feature matrix.

        Two rows share one site indicator per matching site. Each pair of
        matching sites adds one shared pair indicator. Centring this kernel
        gives the unpenalised intercept used by Ridge.
        """
        matches = np.zeros((len(codes), len(codes)), dtype=np.float64)
        for site in range(codes.shape[1]):
            matches += codes[:, site, None] == codes[None, :, site]
        kernel = matches
        if self.pairs:
            kernel = matches * (matches + 1.0) / 2.0
        column_mean = kernel.mean(axis=0)
        kernel = kernel - column_mean[None, :] - column_mean[:, None] + kernel.mean()
        kernel[np.diag_indices_from(kernel)] += self.alpha
        dual = np.linalg.solve(kernel, values - np.mean(values))
        # Reconstruct coefficients for fast lookup during acquisition.
        dual -= dual.mean()
        self.intercept = float(np.mean(values))
        self.singles = []
        for site, size in enumerate(self.sizes):
            indices = codes[:, site].astype(np.int64)
            weights = np.bincount(indices, weights=dual, minlength=size)
            self.singles.append(weights)
            self.intercept -= float(np.mean(weights[indices]))
        self.couples = {}
        for first, second in self.pairs:
            size = self.sizes[first] * self.sizes[second]
            indices = codes[:, first].astype(np.int64) * self.sizes[second] + codes[:, second]
            weights = np.bincount(indices, weights=dual, minlength=size)
            self.couples[first, second] = weights.reshape(self.sizes[first], self.sizes[second])
            self.intercept -= float(np.mean(weights[indices]))
        return self

    def predict(self, codes: NDArray[np.int16]) -> NDArray[np.float64]:
        out = np.full(len(codes), self.intercept, dtype=np.float64)
        for site, table in enumerate(self.singles):
            out += table[codes[:, site].astype(np.int64)]
        for (first, second), table in self.couples.items():
            out += table[codes[:, first].astype(np.int64), codes[:, second].astype(np.int64)]
        return out


class _OverlapGP:
    """Gaussian process with a categorical overlap kernel.

    The kernel counts matching sites, which is the natural similarity on a
    mutation library and needs no embedding.
    """

    def __init__(self, lengthscale: float = 2.0, noise: float = 0.05) -> None:
        self.lengthscale = lengthscale
        self.noise = noise
        self.codes: NDArray[np.int16] | None = None

    def _kernel(self, left: NDArray[np.int16], right: NDArray[np.int16]) -> NDArray[np.float64]:
        distance = np.zeros((len(left), len(right)), dtype=np.float32)
        for site in range(left.shape[1]):
            distance += left[:, site][:, None] != right[:, site][None, :]
        return np.exp(-distance / self.lengthscale).astype(np.float64)

    def fit(self, codes: NDArray[np.int16], values: NDArray[np.float64]) -> _OverlapGP:
        self.codes = codes
        self.mean = float(np.mean(values))
        self.amplitude = float(np.var(values)) or 1.0
        kernel = self.amplitude * self._kernel(codes, codes)
        kernel[np.diag_indices_from(kernel)] += self.noise * self.amplitude + 1e-8
        self.cholesky = np.linalg.cholesky(kernel)
        self.alpha = np.linalg.solve(
            self.cholesky.T, np.linalg.solve(self.cholesky, values - self.mean)
        )
        return self

    def predict(self, codes: NDArray[np.int16]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        means = np.empty(len(codes), dtype=np.float64)
        deviations = np.empty(len(codes), dtype=np.float64)
        for start in range(0, len(codes), 8192):
            block = codes[start : start + 8192]
            cross = self.amplitude * self._kernel(block, self.codes)
            means[start : start + 8192] = self.mean + cross @ self.alpha
            solved = np.linalg.solve(self.cholesky, cross.T)
            variance = self.amplitude - np.sum(solved**2, axis=0)
            deviations[start : start + 8192] = np.sqrt(np.maximum(variance, 1e-12))
        return means, deviations


def _run_model_loop(
    space: SearchSpace,
    oracle: BudgetedOracle,
    features: _Features,
    seed: int,
    acquire,
) -> None:
    """Random start, then repeated fit-and-select until the budget is spent."""
    rng = np.random.default_rng(seed + 7919)
    oracle.query(
        initial_rows(space, seed, min(INITIAL_DESIGN, oracle.remaining), exclude=oracle.seen())
    )
    for _ in range(oracle.budget):
        if oracle.remaining <= 0:
            break
        rows = oracle.rows
        values = oracle.observed
        incumbent = int(rows[int(np.argmax(values))])
        pool = _pool(space, rng, oracle.seen(), incumbent)
        if len(pool) == 0:
            break
        take = min(BATCH_SIZE, oracle.remaining)
        selected = acquire(pool, rows, values, features, rng, take)
        if len(selected) == 0:
            break
        oracle.query(selected)


def random_search(space, oracle, features, seed) -> None:
    """Uniform sample, the floor every other method has to beat."""
    oracle.query(initial_rows(space, seed, oracle.remaining, exclude=oracle.seen()))


def greedy_search(space, oracle, features, seed) -> None:
    """Steepest ascent over single substitutions, restarting when stuck."""
    rng = np.random.default_rng(seed)
    current = space.start_row(rng)
    current_value = float(oracle.query([current])[0])
    while oracle.remaining > 0:
        seen = oracle.seen()
        neighbours = [row for row in space.neighbours(current) if row not in seen]
        climbed = False
        if neighbours:
            values = oracle.query(neighbours)
            best = int(np.argmax(values))
            if values[best] > current_value:
                current, current_value = neighbours[best], float(values[best])
                climbed = True
        if climbed:
            continue
        if oracle.remaining <= 0:
            break
        restart = int(rng.integers(space.size))
        for _ in range(64):
            if restart not in seen:
                break
            restart = int(rng.integers(space.size))
        else:
            break
        current = restart
        current_value = float(oracle.query([current])[0])


def annealing_search(space, oracle, features, seed) -> None:
    """Metropolis walk over single substitutions with a geometric cooling."""
    rng = np.random.default_rng(seed)
    current = space.start_row(rng)
    current_value = float(oracle.query([current])[0])
    # A Metropolis walk revisits, and a revisit is free, so the budget alone
    # would not end the loop once the walk settles into a basin.
    idle = 0
    for step in range(20 * oracle.budget):
        if oracle.remaining <= 0:
            break
        if idle >= 50:
            # The walk has settled and every proposal is a revisit, which costs
            # nothing and would leave the budget unspent. Restart it.
            current = int(rng.integers(space.size))
            current_value = float(oracle.query([current])[0])
            idle = 0
            continue
        neighbours = space.neighbours(current)
        if not neighbours:
            break
        proposal = int(rng.choice(neighbours))
        before = oracle.spent
        value = float(oracle.query([proposal])[0])
        idle = 0 if oracle.spent > before else idle + 1
        if value == -np.inf:
            break
        gap = value - current_value
        observed_scale = float(np.std(oracle.observed))
        scale = max(observed_scale, abs(gap), np.finfo(float).eps)
        temperature = max(0.5 * (0.997**step), np.finfo(float).eps)
        if gap >= 0 or rng.random() < np.exp(gap / (scale * temperature)):
            current, current_value = proposal, value


def genetic_search(space, oracle, features, seed) -> None:
    """Directed evolution: keep the best, recombine sites, mutate one at a time."""
    rng = np.random.default_rng(seed)
    population_size = 24
    rows = list(
        initial_rows(space, seed, min(population_size, oracle.remaining), exclude=oracle.seen())
    )
    oracle.query(rows)
    while oracle.remaining > 0:
        history = oracle.rows
        order = np.argsort(oracle.observed)[::-1]
        parents = history[order[:population_size]]
        children: list[int] = []
        attempts = 0
        seen = oracle.seen()
        while len(children) < min(BATCH_SIZE, oracle.remaining) and attempts < 400:
            attempts += 1
            first, second = rng.choice(parents, size=2, replace=True)
            code = np.where(
                rng.random(space.n_sites) < 0.5, space.codes[first], space.codes[second]
            )
            site = int(rng.integers(space.n_sites))
            code[site] = rng.integers(space.alphabet_sizes[site])
            row = space.lookup.get(tuple(int(value) for value in code))
            if row is not None and row not in seen:
                children.append(row)
                seen.add(row)
        if not children:
            break
        oracle.query(children)


def _linear_acquire(explore: float):
    def acquire(pool, rows, values, features, rng, take):
        codes = features.space.codes
        if explore <= 0:
            model = _PairwiseLinear(features.space.alphabet_sizes).fit(codes[rows], values)
            return _top_rows(pool, model.predict(codes[pool]), take)
        predictions = []
        for _ in range(8):
            draw = rng.integers(len(rows), size=len(rows))
            model = _PairwiseLinear(features.space.alphabet_sizes).fit(
                codes[rows][draw], values[draw]
            )
            predictions.append(model.predict(codes[pool]))
        stacked = np.stack(predictions)
        score = stacked.mean(axis=0) + explore * stacked.std(axis=0)
        return _top_rows(pool, score, take)

    return acquire


def _gp_acquire(kind: str):
    def acquire(pool, rows, values, features, rng, take):
        codes = features.space.codes
        model = _OverlapGP().fit(codes[rows], values)
        mean, deviation = model.predict(codes[pool])
        if kind == "ucb":
            return _top_rows(pool, mean + 2.0 * deviation, take)
        best = float(np.max(values))
        gap = mean - best
        z = gap / deviation
        return _top_rows(pool, gap * norm.cdf(z) + deviation * norm.pdf(z), take)

    return acquire


def _forest_acquire(kind: str):
    def acquire(pool, rows, values, features, rng, take):
        # Equal observations must receive equal targets on fitness plateaus.
        targets = (rankdata(values, method="average") - 1) / max(len(values) - 1, 1)
        forest = RandomForestRegressor(
            n_estimators=32, min_samples_leaf=2, n_jobs=1, random_state=int(rng.integers(1 << 31))
        ).fit(features.one_hot(rows), targets)
        design = features.one_hot(pool)
        if kind == "greedy":
            scores = np.round(forest.predict(design), decimals=12)
            return _top_rows(pool, scores, take)
        # Thompson sampling: each pick follows a different tree's belief.
        chosen: list[int] = []
        trees = rng.choice(len(forest.estimators_), size=take, replace=True)
        available = set(range(len(pool)))
        for index in trees:
            scores = np.round(forest.estimators_[int(index)].predict(design), decimals=12)
            order = np.argsort(scores)[::-1]
            for offset in order:
                if int(offset) in available:
                    available.discard(int(offset))
                    chosen.append(int(pool[offset]))
                    break
        return np.asarray(chosen, dtype=np.int64)

    return acquire


def _mlp_acquire(pool, rows, values, features, rng, take):
    """Ensemble of small networks with an upper confidence bound. This is MLDE."""
    design = features.one_hot(rows)
    candidates = features.one_hot(pool)
    target_scale = float(np.std(values)) or 1.0
    targets = (values - float(np.mean(values))) / target_scale
    predictions = []
    for member in range(4):
        network = MLPRegressor(
            hidden_layer_sizes=(64,),
            max_iter=1200,
            random_state=int(rng.integers(1 << 31)) + member,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
            tol=1e-3,
        ).fit(design, targets)
        predictions.append(network.predict(candidates))
    stacked = np.stack(predictions)
    return _top_rows(pool, stacked.mean(axis=0) + 1.5 * stacked.std(axis=0), take)


def _adalead_acquire(pool, rows, values, features, rng, take):
    """Model-guided evolution from the measured elite, following AdaLead."""
    space = features.space
    codes = space.codes
    model = _PairwiseLinear(space.alphabet_sizes).fit(codes[rows], values)
    threshold = float(np.max(values)) - 0.2 * float(np.std(values) or 1.0)
    elite = rows[values >= threshold]
    if len(elite) == 0:
        elite = rows[np.argsort(values)[-4:]]
    proposals: set[int] = set()
    measured = set(int(row) for row in rows)
    for _ in range(take * 4):
        parent = int(rng.choice(elite))
        code = codes[parent].copy()
        parent_score = float(model.predict(code[None, :])[0])
        for _ in range(8):
            site = int(rng.integers(space.n_sites))
            trial = code.copy()
            trial[site] = rng.integers(space.alphabet_sizes[site])
            row = space.lookup.get(tuple(int(value) for value in trial))
            if row is None:
                continue
            score = float(model.predict(trial[None, :])[0])
            if score <= parent_score:
                continue
            code, parent_score = trial, score
            if row not in measured:
                proposals.add(row)
    candidates = np.asarray(sorted(proposals), dtype=np.int64)
    if len(candidates) == 0:
        return _top_rows(pool, model.predict(codes[pool]), take)
    return _top_rows(candidates, model.predict(codes[candidates]), take)


def _model_method(acquire):
    def method(space, oracle, features, seed) -> None:
        _run_model_loop(space, oracle, features, seed, acquire)

    return method


def _complete_budget(method):
    """Record random queries when the search cannot spend its remaining budget."""

    @wraps(method)
    def completed(space, oracle, features, seed):
        method(space, oracle, features, seed)
        if oracle.remaining > 0:
            rows = initial_rows(space, seed + 79001, oracle.remaining, exclude=oracle.seen())
            before = oracle.spent
            oracle.query(rows)
            oracle.fallback_queries += oracle.spent - before

    return completed


METHODS = {
    "random": random_search,
    "greedy": greedy_search,
    "annealing": annealing_search,
    "genetic": genetic_search,
    "ridge_greedy": _model_method(_linear_acquire(0.0)),
    "ridge_ucb": _model_method(_linear_acquire(1.5)),
    "gp_ei": _model_method(_gp_acquire("ei")),
    "gp_ucb": _model_method(_gp_acquire("ucb")),
    "forest_greedy": _model_method(_forest_acquire("greedy")),
    "forest_thompson": _model_method(_forest_acquire("thompson")),
    "mlp_ucb": _model_method(_mlp_acquire),
    "adalead": _model_method(_adalead_acquire),
}
METHODS = {name: _complete_budget(method) for name, method in METHODS.items()}


def run_method(
    name: str,
    space: SearchSpace,
    values: NDArray[np.float64],
    features: _Features,
    *,
    budget: int,
    seed: int,
) -> float:
    """Spend the budget with one method and return the best fitness it found."""
    oracle = BudgetedOracle(values, budget)
    METHODS[name](space, oracle, features, seed)
    return oracle.best()


def make_features(space: SearchSpace) -> _Features:
    return _Features(space)
