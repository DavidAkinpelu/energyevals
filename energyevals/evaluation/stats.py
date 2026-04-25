import math

from .models import ModelComparison, ScoreStatistics


def compute_score_statistics(
    scores: list[float],
    confidence_level: float = 0.95,
) -> ScoreStatistics:
    """Compute mean, std, and confidence interval from *scores*.

    Uses the *t*-distribution for the CI when ``n >= 2``.  For ``n == 1`` the
    CI collapses to the single observation.
    """
    n = len(scores)
    if n == 0:
        return ScoreStatistics(mean=0.0, std=0.0, ci_lower=0.0, ci_upper=0.0, n=0)

    mean = sum(scores) / n
    if n == 1:
        return ScoreStatistics(mean=mean, std=0.0, ci_lower=mean, ci_upper=mean, n=1)

    variance = sum((s - mean) ** 2 for s in scores) / (n - 1)
    std = math.sqrt(variance)

    from scipy.stats import t as t_dist  # type: ignore[import-untyped]

    alpha = 1.0 - confidence_level
    t_crit = t_dist.ppf(1.0 - alpha / 2, df=n - 1)
    margin = t_crit * std / math.sqrt(n)

    return ScoreStatistics(
        mean=mean,
        std=std,
        ci_lower=mean - margin,
        ci_upper=mean + margin,
        n=n,
    )


def compare_models_paired(
    scores_a: list[float],
    scores_b: list[float],
    alpha: float = 0.05,
) -> ModelComparison:
    """Paired significance test between two sets of scores.

    Uses the Wilcoxon signed-rank test when ``n >= 6``; falls back to a
    paired *t*-test for smaller samples.

    Args:
        scores_a: Scores for model A (one per question, averaged across trials).
        scores_b: Scores for model B (same ordering).
        alpha: Significance threshold.

    Returns:
        ModelComparison with test results (model names left blank for caller to fill).
    """
    n = len(scores_a)
    if n != len(scores_b):
        raise ValueError("Score lists must have the same length")
    if n == 0:
        return ModelComparison(
            model_a="",
            model_b="",
            dimension="",
            test_name="none",
            p_value=1.0,
            significant=False,
            effect_size=0.0,
            direction="none",
        )

    mean_a = sum(scores_a) / n
    mean_b = sum(scores_b) / n
    direction = "a>b" if mean_a > mean_b else ("b>a" if mean_b > mean_a else "equal")

    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    nonzero_diffs = [d for d in diffs if d != 0.0]

    if len(nonzero_diffs) == 0:
        return ModelComparison(
            model_a="",
            model_b="",
            dimension="",
            test_name="exact_tie",
            p_value=1.0,
            significant=False,
            effect_size=0.0,
            direction="equal",
        )

    if n >= 6 and len(nonzero_diffs) >= 6:
        from scipy.stats import wilcoxon  # type: ignore[import-untyped]

        stat, p_value = wilcoxon(scores_a, scores_b, alternative="two-sided")
        test_name = "wilcoxon"
        effect_size = float(stat) / (n * (n + 1) / 2) if n > 0 else 0.0
    else:
        from scipy.stats import ttest_rel  # type: ignore[import-untyped]

        stat, p_value = ttest_rel(scores_a, scores_b)
        test_name = "paired_t"
        pooled_std = math.sqrt(
            sum(d ** 2 for d in diffs) / n - (sum(diffs) / n) ** 2
        ) if n > 1 else 1.0
        effect_size = abs(mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0

    return ModelComparison(
        model_a="",
        model_b="",
        dimension="",
        test_name=test_name,
        p_value=float(p_value),
        significant=float(p_value) < alpha,
        effect_size=effect_size,
        direction=direction,
    )
