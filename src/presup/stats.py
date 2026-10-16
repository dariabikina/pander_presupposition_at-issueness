import random
import statistics

PERMUTATIONS = 20_000
Q = 0.05


def as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bootstrap(values: list[float], draws: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of the mean, resampling items."""
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(statistics.fmean(sample))
    means.sort()
    return (means[int(0.025 * draws)], means[int(0.975 * draws)])


def bootstrap_statistic(pairs: list[tuple[float, float]], statistic,
                        draws: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """Same, for a statistic of paired (x, y) data, e.g. a correlation."""
    if len(pairs) < 3:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    values = []
    for _ in range(draws):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        value = statistic([x for x, _ in sample], [y for _, y in sample])
        if value == value:  # skip nan draws
            values.append(value)
    if len(values) < 2:
        return (float("nan"), float("nan"))
    values.sort()
    return (values[int(0.025 * len(values))], values[int(0.975 * len(values))])


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation, average ranks for ties."""
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            shared = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = shared
            i = j + 1
        return out

    if len(xs) < 3:
        return float("nan")
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def ols_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else float("nan")


def permutation_p(differences: list[float], draws: int = PERMUTATIONS, seed: int = 0) -> float:
    """Two-sided sign-flip test on paired per-item differences."""
    if len(differences) < 2:
        return float("nan")
    observed = abs(statistics.fmean(differences))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(draws):
        flipped = statistics.fmean(d if rng.random() < 0.5 else -d for d in differences)
        if abs(flipped) >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (draws + 1)


def permutation_p_rho(xs: list[float], ys: list[float],
                      draws: int = PERMUTATIONS, seed: int = 0) -> float:
    """Two-sided p for Spearman rho, shuffling y against x."""
    if len(xs) < 4:
        return float("nan")
    observed = abs(spearman(xs, ys))
    rng = random.Random(seed)
    shuffled = list(ys)
    extreme = 0
    for _ in range(draws):
        rng.shuffle(shuffled)
        if abs(spearman(xs, shuffled)) >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (draws + 1)


def benjamini_hochberg(ps: list[float], q: float = Q) -> list[bool]:
    """Step-up FDR; for each p (nan skipped), whether it is rejected."""
    usable = sorted((p, i) for i, p in enumerate(ps) if p == p)
    rejected = [False] * len(ps)
    cutoff = 0
    for rank, (p, _) in enumerate(usable, start=1):
        if p <= q * rank / len(usable):
            cutoff = rank
    for rank, (_, index) in enumerate(usable, start=1):
        if rank <= cutoff:
            rejected[index] = True
    return rejected
