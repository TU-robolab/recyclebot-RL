"""Skill x beta sweep: how good must the operator be, and how tightly
should the robot bind to them?

Run:
    python sweep_skill_beta.py --seeds 10

For every combination of operator skill and KL temperature beta, the
operator-prior (KL) policy is run for the full bandit horizon; pure imitation
of the same operator is run once per skill level as the beta-independent
reference. Outputs:

    figures/fig8_skill_beta_sweep.{pdf,png}   two heatmaps:
        (a) final cumulative regret of the KL policy;
        (b) paired improvement in recovered value/min over imitation.
    figures/tables/sweep_summary.tex          booktabs table (mean +/- 95% CI)
    figures/data/sweep_runs.csv               raw per-run results (cache for --reuse-data)

One seed = one run; panel (b) uses per-seed paired differences under common
random numbers. The interesting structure: at high skill, large beta (bind to
the human) is cheap; at low skill, large beta locks in human error and the
value model must be trusted instead.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from demo_contextual_bandit import run_policy
from plot_style import DOUBLE_COL_IN, apply_style, mean_ci, panel_label, save_figure
from recyclebot_core import load_config

SKILLS = [0.60, 0.75, 0.85, 0.95]
BETAS = [0.2, 0.65, 2.0, 5.0]


def _job(args: Tuple[str, float, float, int, int, dict]) -> dict:
    policy, skill, beta, steps, seed, base_config = args
    config = copy.deepcopy(base_config)
    config["human_model"]["skill"] = float(skill)
    config["learning"]["beta_kl"] = float(beta)
    df = run_policy(policy, config=config, steps=steps, seed=seed)
    return {
        "policy": policy,
        "skill": skill,
        "beta": beta,
        "seed": seed,
        "final_regret": float(df["cum_regret"].iloc[-1]),
        "value_per_min": float(df["recovered_value_per_min"].iloc[-1]),
        "interventions": int(df["intervention"].sum()),
        "correct_rate": float(df["correct_bin_decision"].tail(200).mean()),
    }


def run_sweep(config: dict, seeds: int, steps: int, jobs: int) -> pd.DataFrame:
    seed_list = [42 + 1000 * i for i in range(seeds)]
    tasks = [("kl_human_ghost", s, b, steps, sd, config) for s in SKILLS for b in BETAS for sd in seed_list]
    # Imitation reference is beta-independent: run once per skill (beta value irrelevant).
    tasks += [("human", s, BETAS[0], steps, sd, config) for s in SKILLS for sd in seed_list]
    print(f"Running {len(tasks)} runs on {jobs} workers ...")
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        rows = list(pool.map(_job, tasks))
    return pd.DataFrame(rows)


def heatmap(ax: plt.Axes, grid: np.ndarray, cmap: str, fmt: str, center_zero: bool = False) -> plt.cm.ScalarMappable:
    if center_zero:
        vmax = float(np.nanmax(np.abs(grid)))
        im = ax.imshow(grid, cmap=cmap, aspect="auto", origin="lower", vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(grid, cmap=cmap, aspect="auto", origin="lower")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            lum = im.cmap(im.norm(grid[i, j]))
            text_color = "white" if 0.299 * lum[0] + 0.587 * lum[1] + 0.114 * lum[2] < 0.5 else "black"
            ax.text(j, i, format(grid[i, j], fmt), ha="center", va="center",
                    fontsize=7.5, color=text_color)
    ax.set_xticks(range(len(BETAS)))
    ax.set_xticklabels([str(b) for b in BETAS])
    ax.set_yticks(range(len(SKILLS)))
    ax.set_yticklabels([f"{s:.2f}" for s in SKILLS])
    ax.set_xlabel(r"KL temperature $\beta$")
    ax.grid(False)
    return im


def make_figure(df: pd.DataFrame, fig_dir: Path, table_dir: Path) -> None:
    kl = df[df["policy"] == "kl_human_ghost"]
    imit = df[df["policy"] == "human"].set_index(["skill", "seed"])

    regret = np.zeros((len(SKILLS), len(BETAS)))
    delta_value = np.zeros_like(regret)
    delta_half = np.zeros_like(regret)
    for i, s in enumerate(SKILLS):
        for j, b in enumerate(BETAS):
            cell = kl[(kl["skill"] == s) & (kl["beta"] == b)].set_index("seed")
            regret[i, j] = cell["final_regret"].mean()
            paired = cell["value_per_min"] - imit.loc[s]["value_per_min"].reindex(cell.index)
            m, h = mean_ci(paired.to_numpy())
            delta_value[i, j] = m
            delta_half[i, j] = h

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_IN, 2.7))
    im0 = heatmap(axes[0], regret, cmap="viridis_r", fmt=".0f")
    axes[0].set_ylabel("Operator skill")
    axes[0].set_title("Final cumulative regret (KL policy)", fontsize=9)
    panel_label(axes[0], "a", x=-0.22)
    fig.colorbar(im0, ax=axes[0], shrink=0.85)

    im1 = heatmap(axes[1], delta_value, cmap="RdBu_r", fmt="+.1f", center_zero=True)
    axes[1].set_title(r"$\Delta$ value/min vs. imitation (paired)", fontsize=9)
    panel_label(axes[1], "b", x=-0.16)
    fig.colorbar(im1, ax=axes[1], shrink=0.85)
    fig.tight_layout(pad=0.5)
    save_figure(fig, fig_dir, "fig8_skill_beta_sweep")

    # Booktabs table: one row per (skill, beta) with regret and paired delta.
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\small",
        "  \\setlength{\\tabcolsep}{4pt}",
        "  \\caption{Skill $\\times$ $\\beta$ sweep for the operator-prior (KL) policy "
        "(mean $\\pm$ 95\\% CI over seeds). $\\Delta$ value/min is the paired per-seed "
        "improvement over pure imitation of the same operator.}",
        "  \\label{tab:sweep}",
        "  \\resizebox{\\linewidth}{!}{%",
        "  \\begin{tabular}{llccc}",
        "    \\toprule",
        "    Skill & $\\beta$ & Final regret & $\\Delta$ value/min & Interventions \\\\",
        "    \\midrule",
    ]
    for i, s in enumerate(SKILLS):
        for j, b in enumerate(BETAS):
            cell = kl[(kl["skill"] == s) & (kl["beta"] == b)]
            mr, hr = mean_ci(cell["final_regret"].to_numpy())
            mi, hi = mean_ci(cell["interventions"].to_numpy().astype(float))
            skill_col = f"{s:.2f}" if j == 0 else ""
            lines.append(
                f"    {skill_col} & {b} & ${mr:.0f} \\pm {hr:.0f}$ & "
                f"${delta_value[i, j]:+.1f} \\pm {delta_half[i, j]:.1f}$ & "
                f"${mi:.0f} \\pm {hi:.0f}$ \\\\"
            )
        if i < len(SKILLS) - 1:
            lines.append("    \\addlinespace")
    lines += ["    \\bottomrule", "  \\end{tabular}}", "\\end{table}", ""]
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "sweep_summary.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="RecycleBot skill x beta sweep")
    parser.add_argument("--config", type=str, default=str(Path(__file__).with_name("recyclebot_config.json")))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 4, 8))
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).with_name("figures")))
    parser.add_argument("--reuse-data", action="store_true")
    args = parser.parse_args()

    apply_style()
    out_dir = Path(args.output_dir)
    data_path = out_dir / "data" / "sweep_runs.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)

    if args.reuse_data and data_path.exists():
        print("Reusing cached sweep data:", data_path)
        df = pd.read_csv(data_path)
    else:
        df = run_sweep(load_config(args.config), seeds=args.seeds, steps=args.steps, jobs=args.jobs)
        df.to_csv(data_path, index=False)

    make_figure(df, out_dir, out_dir / "tables")
    print("Wrote:")
    print(f"  {out_dir / 'fig8_skill_beta_sweep.pdf'}")
    print(f"  {out_dir / 'tables' / 'sweep_summary.tex'}")


if __name__ == "__main__":
    main()
