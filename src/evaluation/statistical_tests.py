"""
Statistical Testing Utilities for KAN-Fine++ Experiments

Implements:
- Wilcoxon signed-rank test (paired, non-parametric)
- Cohen's d effect size
- Bootstrap 95% confidence intervals
- Bonferroni correction for multiple comparisons
- Full comparison report generation

Reference:
- "Reinterpreting Confidence Intervals in FSL" (arXiv:2409.02850)
- "Benchmarking Few-Shot Transferability" (arXiv:2603.00478)
"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import itertools


@dataclass
class ComparisonResult:
    """Result of a pairwise statistical comparison."""
    model_a: str
    model_b: str
    mean_a: float
    mean_b: float
    ci_a: Tuple[float, float]  # 95% CI for model A
    ci_b: Tuple[float, float]  # 95% CI for model B
    p_value: float
    p_value_corrected: float  # After Bonferroni
    cohens_d: float
    significant: bool  # p_corrected < 0.05
    practically_significant: bool  # |d| >= 0.2
    direction: str  # "A > B", "B > A", or "A ≈ B"


@dataclass
class ExperimentReport:
    """Full statistical report for an experiment configuration."""
    setting: str  # e.g., "5-way 5-shot, near-domain"
    comparisons: List[ComparisonResult] = field(default_factory=list)
    model_stats: Dict[str, dict] = field(default_factory=dict)


def bootstrap_ci(
    data: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        data: 1D array of episode accuracies
        confidence: Confidence level (default 0.95)
        n_bootstrap: Number of bootstrap resamples
        seed: Random seed for reproducibility

    Returns:
        (lower, upper) bounds of the CI
    """
    rng = np.random.RandomState(seed)
    n = len(data)
    bootstrap_means = np.array([
        rng.choice(data, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return (lower, upper)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cohen's d for paired samples.

    Uses the pooled standard deviation as denominator.
    Interpretation:
        |d| < 0.2: negligible
        0.2 ≤ |d| < 0.5: small
        0.5 ≤ |d| < 0.8: medium
        |d| ≥ 0.8: large

    Args:
        x, y: 1D arrays of paired observations (same episodes)

    Returns:
        Cohen's d value
    """
    n = len(x)
    diff = x - y
    mean_diff = diff.mean()

    # Pooled std
    s_pooled = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)

    if s_pooled < 1e-10:
        return 0.0

    return mean_diff / s_pooled


def wilcoxon_test(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Wilcoxon signed-rank test for paired samples.

    Non-parametric alternative to paired t-test. Appropriate because:
    - Episode accuracies can be skewed (not normally distributed)
    - Same episodes are evaluated for all models (paired)

    Args:
        x, y: 1D arrays of paired observations

    Returns:
        p-value (two-sided)
    """
    diff = x - y
    # Remove zeros (ties)
    nonzero_diff = diff[diff != 0]

    if len(nonzero_diff) < 10:
        # Too few non-tied observations for reliable test
        return 1.0

    statistic, p_value = stats.wilcoxon(nonzero_diff, alternative='two-sided')
    return p_value


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[float]:
    """
    Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of uncorrected p-values
        alpha: Family-wise error rate

    Returns:
        List of corrected p-values (capped at 1.0)
    """
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


def pairwise_comparison(
    results: Dict[str, np.ndarray],
    setting_name: str = "",
    alpha: float = 0.05,
) -> ExperimentReport:
    """
    Perform all pairwise statistical comparisons between models.

    Args:
        results: Dict mapping model_name → array of per-episode accuracies.
                 All arrays must have the same length (same episodes).
        setting_name: Description of the experimental setting
        alpha: Significance level

    Returns:
        ExperimentReport with all pairwise comparisons and per-model stats
    """
    report = ExperimentReport(setting=setting_name)

    # Per-model statistics
    for name, accs in results.items():
        accs = np.array(accs)
        ci = bootstrap_ci(accs)
        report.model_stats[name] = {
            'mean': accs.mean(),
            'std': accs.std(),
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'n_episodes': len(accs),
        }

    # Pairwise comparisons
    model_names = list(results.keys())
    pairs = list(itertools.combinations(model_names, 2))
    raw_p_values = []
    comparisons_data = []

    for name_a, name_b in pairs:
        accs_a = np.array(results[name_a])
        accs_b = np.array(results[name_b])

        p_val = wilcoxon_test(accs_a, accs_b)
        d = cohens_d(accs_a, accs_b)
        ci_a = bootstrap_ci(accs_a)
        ci_b = bootstrap_ci(accs_b)

        raw_p_values.append(p_val)
        comparisons_data.append((name_a, name_b, accs_a, accs_b, p_val, d, ci_a, ci_b))

    # Apply Bonferroni correction
    corrected_p = bonferroni_correction(raw_p_values, alpha)

    for i, (name_a, name_b, accs_a, accs_b, p_val, d, ci_a, ci_b) in enumerate(comparisons_data):
        p_corrected = corrected_p[i]
        sig = p_corrected < alpha
        prac_sig = abs(d) >= 0.2

        if d > 0.05:
            direction = f"{name_a} > {name_b}"
        elif d < -0.05:
            direction = f"{name_b} > {name_a}"
        else:
            direction = f"{name_a} ≈ {name_b}"

        comparison = ComparisonResult(
            model_a=name_a,
            model_b=name_b,
            mean_a=accs_a.mean(),
            mean_b=accs_b.mean(),
            ci_a=ci_a,
            ci_b=ci_b,
            p_value=p_val,
            p_value_corrected=p_corrected,
            cohens_d=d,
            significant=sig,
            practically_significant=prac_sig,
            direction=direction,
        )
        report.comparisons.append(comparison)

    return report


def format_report(report: ExperimentReport) -> str:
    """Format an ExperimentReport as a readable string."""
    lines = [f"\n{'='*70}"]
    lines.append(f"Statistical Report: {report.setting}")
    lines.append(f"{'='*70}\n")

    # Per-model stats
    lines.append("Model Performance:")
    lines.append(f"{'Model':<25} {'Mean':>8} {'Std':>8} {'95% CI':>20}")
    lines.append("-" * 65)
    for name, s in report.model_stats.items():
        ci_str = f"[{s['ci_lower']:.4f}, {s['ci_upper']:.4f}]"
        lines.append(f"{name:<25} {s['mean']:>8.4f} {s['std']:>8.4f} {ci_str:>20}")

    # Pairwise comparisons
    lines.append(f"\n{'Pairwise Comparisons:'}")
    lines.append(f"{'Comparison':<35} {'p-val':>8} {'p-corr':>8} {'d':>8} {'Sig?':>6} {'Prac?':>6}")
    lines.append("-" * 75)

    for c in report.comparisons:
        sig_mark = "✓" if c.significant else "✗"
        prac_mark = "✓" if c.practically_significant else "✗"
        comp_str = f"{c.model_a} vs {c.model_b}"
        lines.append(
            f"{comp_str:<35} {c.p_value:>8.4f} {c.p_value_corrected:>8.4f} "
            f"{c.cohens_d:>8.3f} {sig_mark:>6} {prac_mark:>6}"
        )
        lines.append(f"  → {c.direction}")

    # Warnings
    lines.append("")
    for c in report.comparisons:
        if c.significant and not c.practically_significant:
            lines.append(
                f"⚠️  {c.model_a} vs {c.model_b}: statistically significant "
                f"(p={c.p_value_corrected:.4f}) but NOT practically significant "
                f"(d={c.cohens_d:.3f} < 0.2). The difference may not be meaningful."
            )

    return "\n".join(lines)
