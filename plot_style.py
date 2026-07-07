"""Shared publication-quality matplotlib style for the RecycleBot figures.

The style mimics LaTeX typesetting without requiring a TeX installation:
Computer Modern math fonts via mathtext, serif text, thin spines, and
colorblind-safe colors (Okabe-Ito). Figure widths match common journal
column sizes (single column ~3.4 in, double column ~7.0 in).

All aggregation helpers treat one random seed as one independent run and
report the mean with a 95% confidence interval computed from the
t-distribution over seeds.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SINGLE_COL_IN = 3.4
DOUBLE_COL_IN = 7.0

# Okabe-Ito colorblind-safe palette.
OKABE_ITO = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

POLICY_LABELS: Dict[str, str] = {
    "random": "Random",
    "human": r"Imitation $\mu_H$",
    "imitation_muH": r"Imitation $\mu_H$",
    "greedy": r"Greedy (no explore)",
    "greedy_eps": r"Greedy $\varepsilon$=0.04",
    "thompson": "Thompson (BLR)",
    "greedy_bad_Q": r"Greedy (bad $\widehat{Q}$)",
    "kl_human_ghost": r"Operator-prior (KL)",
    "KL_bad_Q": r"Op.-prior KL (bad $\widehat{Q}$)",
    "KL_oracle_noise_Q": r"Op.-prior KL (oracle+noise $\widehat{Q}$)",
    "lp": "LP + rounding",
    "greedy_density": "Greedy density",
}


def policy_label(name: str) -> str:
    return POLICY_LABELS.get(name, name)


def apply_style(usetex: bool = False) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "text.usetex": usetex,
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "lines.linewidth": 1.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.30,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,  # embed TrueType so text stays editable
            "ps.fonttype": 42,
        }
    )


def mean_ci(values: np.ndarray, axis: int = 0, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and half-width of the t-based confidence interval along ``axis``."""
    arr = np.asarray(values, dtype=float)
    n = arr.shape[axis]
    mean = np.nanmean(arr, axis=axis)
    if n < 2:
        return mean, np.zeros_like(mean)
    sem = np.nanstd(arr, axis=axis, ddof=1) / np.sqrt(n)
    tval = stats.t.ppf(0.5 + confidence / 2.0, df=n - 1)
    return mean, tval * sem


def seed_matrix(df: pd.DataFrame, value_col: str, index_col: str, seed_col: str = "seed") -> Tuple[np.ndarray, np.ndarray]:
    """Pivot a long dataframe into an (n_seeds, n_steps) matrix.

    Returns (x, matrix) where x is the sorted index values common to all seeds.
    """
    pivot = df.pivot_table(index=index_col, columns=seed_col, values=value_col)
    pivot = pivot.dropna(axis=0, how="any")
    return pivot.index.to_numpy(), pivot.to_numpy().T


def plot_band(
    ax: plt.Axes,
    x: np.ndarray,
    matrix: np.ndarray,
    label: str,
    color: str,
    confidence: float = 0.95,
    smooth: Optional[int] = None,
    linestyle: str = "-",
) -> None:
    """Plot mean line with a shaded confidence band over seeds."""
    mean, half = mean_ci(matrix, axis=0, confidence=confidence)
    if smooth is not None and smooth > 1:
        kernel = np.ones(smooth) / smooth
        mean = np.convolve(mean, kernel, mode="same")
        half = np.convolve(half, kernel, mode="same")
    ax.plot(x, mean, label=label, color=color, linestyle=linestyle)
    ax.fill_between(x, mean - half, mean + half, color=color, alpha=0.18, linewidth=0)


def bar_with_ci(
    ax: plt.Axes,
    labels: Sequence[str],
    per_seed_values: Sequence[np.ndarray],
    colors: Sequence[str],
    confidence: float = 0.95,
    show_points: bool = True,
) -> None:
    """Bar chart of per-seed means with CI error bars and per-seed dots."""
    x = np.arange(len(labels))
    means, halves = [], []
    for vals in per_seed_values:
        m, h = mean_ci(np.asarray(vals, dtype=float))
        means.append(float(m))
        halves.append(float(h))
    ax.bar(x, means, yerr=halves, capsize=2.5, color=colors, alpha=0.85,
           error_kw={"linewidth": 0.9}, width=0.62, zorder=2)
    if show_points:
        rng = np.random.default_rng(0)
        for i, vals in enumerate(per_seed_values):
            jitter = rng.uniform(-0.10, 0.10, size=len(vals))
            ax.scatter(np.full(len(vals), x[i]) + jitter, vals, s=6, color="black",
                       alpha=0.55, zorder=3, linewidths=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")


def panel_label(ax: plt.Axes, letter: str, x: float = -0.16, y: float = 1.06) -> None:
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top")


def save_figure(fig: plt.Figure, out_dir: Path, name: str, formats: Sequence[str] = ("pdf", "png")) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
