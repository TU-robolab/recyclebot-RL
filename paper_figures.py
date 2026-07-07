"""Publication-quality, multi-seed figures and LaTeX tables for the RecycleBot demos.

Run:
    python paper_figures.py --seeds 10

Outputs (under figures/):
    fig1_bandit_learning.{pdf,png}      learning curves with 95% CI bands
    fig2_bandit_convergence.{pdf,png}   epistemic variance, entropy, Q-model error
    fig3_bandit_final.{pdf,png}         final performance bars with error bars
    fig4_bandit_calibration.{pdf,png}   value-model reliability diagram
    fig5_lp_assignment.{pdf,png}        LP vs greedy, rounding gap, solve-time scaling
    fig6_ghost_diagnostics.{pdf,png}    human-ghost prior diagnostics
    tables/bandit_summary.tex           booktabs summary (mean +/- 95% CI over seeds)
    tables/lp_summary.tex
    tables/ghost_summary.tex
    data/*.csv                          raw per-seed runs (reused with --reuse-data)

Every experiment is repeated over independent seeds; one seed = one run.
Shaded bands and error bars are 95% confidence intervals over seeds
(t-distribution). Convergence time is the first step at which the rolling
reward reaches 95% of its final plateau and stays there.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from demo_contextual_bandit import run_policy as bandit_run_policy
from demo_convex_assignment import run_demo as lp_run_demo
from demo_human_ghost import run_experiment as ghost_run_experiment
from plot_style import (
    DOUBLE_COL_IN,
    OKABE_ITO,
    SINGLE_COL_IN,
    apply_style,
    bar_with_ci,
    mean_ci,
    panel_label,
    plot_band,
    policy_label,
    save_figure,
    seed_matrix,
)
from recyclebot_core import load_config

BANDIT_POLICIES = ["random", "human", "greedy", "greedy_eps", "thompson", "kl_human_ghost"]
BANDIT_LEARNING_POLICIES = ["greedy_eps", "thompson", "kl_human_ghost"]
GHOST_POLICIES = ["greedy_bad_Q", "imitation_muH", "KL_bad_Q", "KL_oracle_noise_Q"]
ROLL = 100  # rolling window (control steps) for smoothed curves

POLICY_COLORS: Dict[str, str] = {
    "random": OKABE_ITO[7],
    "human": OKABE_ITO[2],
    "greedy": OKABE_ITO[1],
    "greedy_eps": OKABE_ITO[4],
    "thompson": OKABE_ITO[5],
    "kl_human_ghost": OKABE_ITO[0],
    "greedy_bad_Q": OKABE_ITO[1],
    "imitation_muH": OKABE_ITO[2],
    "KL_bad_Q": OKABE_ITO[4],
    "KL_oracle_noise_Q": OKABE_ITO[0],
}


# ---------------------------------------------------------------------------
# Experiment runners (multi-seed, parallel over processes)
# ---------------------------------------------------------------------------

def _bandit_job(args: Tuple[str, dict, int, int]) -> pd.DataFrame:
    policy, config, steps, seed = args
    df = bandit_run_policy(policy, config=config, steps=steps, seed=seed)
    df["seed"] = seed
    return df


def _lp_job(args: Tuple[dict, int, int]) -> pd.DataFrame:
    config, windows, seed = args
    df, _snapshot = lp_run_demo(config=config, windows=windows, seed=seed)
    df["seed"] = seed
    return df


def _ghost_job(args: Tuple[dict, int, int, float]) -> pd.DataFrame:
    config, steps, seed, beta = args
    df = ghost_run_experiment(config, steps=steps, seed=seed, beta=beta)
    df["seed"] = seed
    return df


def run_all(
    config: dict,
    seeds: Sequence[int],
    bandit_steps: int,
    lp_windows: int,
    ghost_steps: int,
    ghost_beta: float,
    jobs: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bandit_jobs = [(p, config, bandit_steps, s) for p in BANDIT_POLICIES for s in seeds]
    lp_jobs = [(config, lp_windows, s) for s in seeds]
    ghost_jobs = [(config, ghost_steps, s, ghost_beta) for s in seeds]

    with ProcessPoolExecutor(max_workers=jobs) as pool:
        print(f"Running {len(bandit_jobs)} bandit runs, {len(lp_jobs)} LP runs, "
              f"{len(ghost_jobs)} ghost runs on {jobs} workers ...")
        bandit_frames = list(pool.map(_bandit_job, bandit_jobs))
        lp_frames = list(pool.map(_lp_job, lp_jobs))
        ghost_frames = list(pool.map(_ghost_job, ghost_jobs))

    return (
        pd.concat(bandit_frames, ignore_index=True),
        pd.concat(lp_frames, ignore_index=True),
        pd.concat(ghost_frames, ignore_index=True),
    )


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------

def add_rolling(df: pd.DataFrame, group_cols: Sequence[str], t_col: str, specs: Dict[str, str], window: int = ROLL) -> pd.DataFrame:
    """Add rolling-mean columns; specs maps new_col -> source_col."""
    df = df.sort_values(list(group_cols) + [t_col]).copy()
    g = df.groupby(list(group_cols), sort=False)
    for new_col, src in specs.items():
        df[new_col] = g[src].transform(lambda s: s.rolling(window, min_periods=10).mean())
    return df


def convergence_step(t: np.ndarray, rolling_reward: np.ndarray, frac: float = 0.95) -> float:
    """First step at which the rolling reward reaches ``frac`` of its final
    plateau (mean of the last 20% of steps, measured from the initial level)
    and never drops below that threshold again. NaN if never reached."""
    mask = ~np.isnan(rolling_reward)
    t, y = t[mask], rolling_reward[mask]
    if len(y) < 20:
        return float("nan")
    plateau = float(np.mean(y[int(0.8 * len(y)):]))
    initial = float(np.mean(y[: max(int(0.05 * len(y)), 1)]))
    if plateau <= initial:  # no measurable learning
        return 0.0
    threshold = initial + frac * (plateau - initial)
    below = y < threshold
    if below.all():
        return float("nan")
    # last index where the curve is below threshold; convergence is the step after
    last_below = int(np.where(below)[0][-1]) if below.any() else -1
    if last_below + 1 >= len(t):
        return float("nan")
    return float(t[last_below + 1])


def fmt_pm(mean: float, half: float, digits: int = 2) -> str:
    if np.isnan(mean):
        return "--"
    return f"${mean:.{digits}f} \\pm {half:.{digits}f}$"


def write_booktabs(path: Path, caption: str, label: str, header: Sequence[str], rows: Sequence[Sequence[str]], align: Optional[str] = None) -> None:
    align = align or ("l" + "c" * (len(header) - 1))
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{align}}}",
        "    \\toprule",
        "    " + " & ".join(header) + " \\\\",
        "    \\midrule",
    ]
    for row in rows:
        lines.append("    " + " & ".join(row) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Figure 1-4: contextual bandit
# ---------------------------------------------------------------------------

def make_bandit_figures(df: pd.DataFrame, fig_dir: Path, table_dir: Path) -> None:
    df = add_rolling(
        df,
        group_cols=["policy", "seed"],
        t_col="t",
        specs={
            "roll_reward": "reward",
            "roll_correct": "correct_bin_decision",
            "roll_intervention": "intervention",
            "roll_qvar": "max_epistemic_q_var",
            "roll_entropy": "policy_entropy",
        },
    )
    df["abs_q_error"] = (df["q_mean_selected"] - df["reward"]).abs()
    df = add_rolling(df, ["policy", "seed"], "t", {"roll_q_error": "abs_q_error"})

    # ---- Fig 1: learning curves -------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, 4.6), sharex=True)
    panels = [
        ("roll_reward", "Reward per decision", "a"),
        ("cum_regret", "Cumulative regret", "b"),
        ("roll_correct", "Correct-bin decision rate", "c"),
        ("roll_intervention", "Human intervention rate", "d"),
    ]
    for ax, (col, ylabel, letter) in zip(axes.flat, panels):
        for policy in BANDIT_POLICIES:
            sub = df[df["policy"] == policy]
            x, mat = seed_matrix(sub, col, "t")
            plot_band(ax, x, mat, policy_label(policy), POLICY_COLORS[policy])
        ax.set_ylabel(ylabel)
        panel_label(ax, letter)
    for ax in axes[1]:
        ax.set_xlabel("Control step $t$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.955))
    save_figure(fig, fig_dir, "fig1_bandit_learning")

    # ---- Fig 2: convergence diagnostics ------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_IN, 1.95), sharex=True)
    for policy in BANDIT_POLICIES:
        sub = df[df["policy"] == policy]
        color = POLICY_COLORS[policy]
        x, mat = seed_matrix(sub, "roll_qvar", "t")
        plot_band(axes[0], x, mat, policy_label(policy), color)
        x, mat = seed_matrix(sub, "roll_entropy", "t")
        plot_band(axes[1], x, mat, policy_label(policy), color)
    for policy in BANDIT_LEARNING_POLICIES:
        sub = df[df["policy"] == policy]
        x, mat = seed_matrix(sub, "roll_q_error", "t")
        plot_band(axes[2], x, mat, policy_label(policy), POLICY_COLORS[policy])
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"Epistemic $\mathrm{Var}[\widehat{Q}]$")
    axes[1].set_ylabel(r"Policy entropy $H(\pi_t)$ [nats]")
    axes[2].set_ylabel(r"$|\widehat{Q} - R|$ (selected)")
    for ax, letter in zip(axes, "abc"):
        ax.set_xlabel("Control step $t$")
        panel_label(ax, letter, x=-0.30, y=1.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.88))
    save_figure(fig, fig_dir, "fig2_bandit_convergence")

    # ---- Fig 3: final performance bars -------------------------------------
    last = df.groupby(["policy", "seed"]).apply(
        lambda g: pd.Series(
            {
                "value_per_min": g["recovered_value_per_min"].iloc[-1],
                "correct_rate": g["correct_bin_decision"].tail(200).mean(),
                "cum_regret": g["cum_regret"].iloc[-1],
            }
        ),
        include_groups=False,
    ).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_IN, 2.1))
    metrics = [
        ("value_per_min", "Recovered value [units/min]", "a"),
        ("correct_rate", "Correct-bin decisions (last 200)", "b"),
        ("cum_regret", "Final cumulative regret", "c"),
    ]
    labels = [policy_label(p) for p in BANDIT_POLICIES]
    colors = [POLICY_COLORS[p] for p in BANDIT_POLICIES]
    for ax, (col, ylabel, letter) in zip(axes, metrics):
        vals = [last.loc[last["policy"] == p, col].to_numpy() for p in BANDIT_POLICIES]
        bar_with_ci(ax, labels, vals, colors)
        ax.set_ylabel(ylabel)
        panel_label(ax, letter)
    fig.tight_layout(pad=0.4)
    save_figure(fig, fig_dir, "fig3_bandit_final")

    # ---- Fig 4: value-model calibration ------------------------------------
    fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, SINGLE_COL_IN * 0.9))
    tail_frac = 0.4  # calibration measured after learning has mostly settled
    for policy in ["greedy_eps", "kl_human_ghost"]:
        sub = df[(df["policy"] == policy) & (df["t"] >= df["t"].max() * (1 - tail_frac))]
        pred = sub["q_mean_selected"].to_numpy()
        realized = sub["reward"].to_numpy()
        edges = np.quantile(pred, np.linspace(0, 1, 9))
        edges[-1] += 1e-9
        centers, means, halves = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (pred >= lo) & (pred < hi)
            if m.sum() < 10:
                continue
            centers.append(pred[m].mean())
            mu, half = mean_ci(realized[m])
            means.append(mu)
            halves.append(half)
        ax.errorbar(centers, means, yerr=halves, fmt="o-", ms=3.5, capsize=2,
                    linewidth=1.0, color=POLICY_COLORS[policy], label=policy_label(policy))
    lims = ax.get_xlim()
    grid = np.linspace(lims[0], lims[1], 10)
    ax.plot(grid, grid, "--", color="gray", linewidth=0.8, label="Perfect calibration")
    ax.set_xlabel(r"Predicted value $\widehat{Q}(S_t, a_t)$")
    ax.set_ylabel(r"Realized reward $R_t$")
    ax.legend(loc="upper left")
    fig.tight_layout(pad=0.4)
    save_figure(fig, fig_dir, "fig4_bandit_calibration")

    # ---- Table: bandit summary ---------------------------------------------
    conv = (
        df.groupby(["policy", "seed"])
        .apply(lambda g: convergence_step(g["t"].to_numpy(), g["roll_reward"].to_numpy()), include_groups=False)
        .rename("t_conv")
        .reset_index()
    )
    rows = []
    for policy in BANDIT_POLICIES:
        lp = last[last["policy"] == policy]
        cv = conv.loc[conv["policy"] == policy, "t_conv"].to_numpy()
        interventions = df[df["policy"] == policy].groupby("seed")["intervention"].sum().to_numpy()
        m1, h1 = mean_ci(lp["value_per_min"].to_numpy())
        m2, h2 = mean_ci(lp["correct_rate"].to_numpy())
        m3, h3 = mean_ci(lp["cum_regret"].to_numpy())
        m4, h4 = mean_ci(cv[~np.isnan(cv)]) if np.any(~np.isnan(cv)) else (float("nan"), 0.0)
        m5, h5 = mean_ci(interventions.astype(float))
        rows.append([
            policy_label(policy),
            fmt_pm(m1, h1), fmt_pm(m2, h2), fmt_pm(m3, h3, 1),
            fmt_pm(m4, h4, 0), fmt_pm(m5, h5, 1),
        ])
    # Paired per-seed differences (seeds are common across policies).
    pivot = last.pivot(index="seed", columns="policy")
    for a, b in [("kl_human_ghost", "human"), ("kl_human_ghost", "thompson")]:
        dv = (pivot[("value_per_min", a)] - pivot[("value_per_min", b)]).to_numpy()
        dr = (pivot[("cum_regret", a)] - pivot[("cum_regret", b)]).to_numpy()
        mv, hv = mean_ci(dv)
        mr, hr = mean_ci(dr)
        rows.append([
            f"\\quad {policy_label(a)} $-$ {policy_label(b)} (paired)",
            fmt_pm(mv, hv), "--", fmt_pm(mr, hr, 1), "--", "--",
        ])
    write_booktabs(
        table_dir / "bandit_summary.tex",
        caption=("Contextual-bandit results (mean $\\pm$ 95\\% CI over seeds; paired rows "
                 "use per-seed differences under common random numbers). "
                 "Convergence step $t_{\\mathrm{conv}}$ is the first step at which the "
                 f"rolling reward (window {ROLL}) reaches 95\\% of its final plateau."),
        label="tab:bandit",
        header=["Policy", "Value/min", "Correct-bin", "Cum.\\ regret", "$t_{\\mathrm{conv}}$", "Interventions"],
        rows=rows,
    )


# ---------------------------------------------------------------------------
# Figure 7: off-policy evaluation (SNIPS) from the KL-ghost logs
# ---------------------------------------------------------------------------

def make_ope_figure(df: pd.DataFrame, fig_dir: Path, table_dir: Path, epsilon: float = 0.04) -> None:
    """Self-normalized IPS estimates of the human and random policies computed
    from the KL-ghost behavior logs, validated against on-policy ground truth.

    Uses the environment reward (before intervention cost) and the corrected
    behavior probability mu_t (1.0 on deterministic override steps). Target
    probabilities are reconstructed from logged quantities: the human target is
    the epsilon-mixed human prior, exactly the sampling rule of the on-policy
    ``human`` runs; the random target is uniform over legal actions.
    """
    reward_col = "reward_env" if "reward_env" in df.columns else "reward"
    logs = df[df["policy"] == "kl_human_ghost"]

    snips: Dict[str, List[float]] = {"human": [], "random": []}
    ess: Dict[str, List[float]] = {"human": [], "random": []}
    for _seed, g in logs.groupby("seed"):
        mu = g["behavior_policy_probability_mu_t"].to_numpy()
        n_actions = g["n_actions"].to_numpy()
        r = g[reward_col].to_numpy()
        targets = {
            "human": (1.0 - epsilon) * g["human_prior_selected"].to_numpy() + epsilon / n_actions,
            "random": 1.0 / n_actions,
        }
        for name, pi in targets.items():
            w = pi / np.maximum(mu, 1e-12)
            snips[name].append(float(np.sum(w * r) / max(np.sum(w), 1e-12)))
            ess[name].append(float(np.sum(w) ** 2 / max(np.sum(w**2), 1e-12)))

    truth: Dict[str, np.ndarray] = {}
    for name in ("human", "random"):
        on_policy = df[df["policy"] == name]
        truth[name] = on_policy.groupby("seed")[reward_col].mean().to_numpy()

    fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, 2.4))
    labels = ["Human\n(on-policy)", "Human\n(SNIPS)", "Random\n(on-policy)", "Random\n(SNIPS)"]
    values = [truth["human"], np.asarray(snips["human"]), truth["random"], np.asarray(snips["random"])]
    colors = [OKABE_ITO[2], OKABE_ITO[2], OKABE_ITO[7], OKABE_ITO[7]]
    bar_with_ci(ax, labels, values, colors)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_ylabel("Mean reward per decision")
    fig.tight_layout(pad=0.4)
    save_figure(fig, fig_dir, "fig7_ope_snips")

    rows = []
    for name in ("human", "random"):
        mt, ht = mean_ci(truth[name])
        ms, hs = mean_ci(np.asarray(snips[name]))
        me, _ = mean_ci(np.asarray(ess[name]))
        n_steps = int(logs.groupby("seed").size().mean())
        rows.append([
            policy_label(name) if name != "random" else "Random",
            fmt_pm(mt, ht, 3), fmt_pm(ms, hs, 3), f"{me:.0f} / {n_steps}",
        ])
    write_booktabs(
        table_dir / "ope_summary.tex",
        caption=("Off-policy evaluation from KL-ghost behavior logs: SNIPS estimates vs.\\ "
                 "on-policy ground truth (mean $\\pm$ 95\\% CI over seeds); ESS is the "
                 "effective sample size of the importance weights."),
        label="tab:ope",
        header=["Target policy", "On-policy value", "SNIPS estimate", "ESS / steps"],
        rows=rows,
    )


# ---------------------------------------------------------------------------
# Figure 5: LP assignment
# ---------------------------------------------------------------------------

def make_lp_figures(df: pd.DataFrame, fig_dir: Path, table_dir: Path) -> None:
    df = df.sort_values(["seed", "window"]).copy()
    for col, src in [("roll_lp_value", "lp_true_value"), ("roll_greedy_value", "greedy_true_value")]:
        df[col] = df.groupby("seed")[src].transform(lambda s: s.rolling(25, min_periods=5).mean())

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, 4.4))

    # (a) rolling true value per window
    ax = axes[0, 0]
    for col, name, color in [
        ("roll_lp_value", policy_label("lp"), OKABE_ITO[0]),
        ("roll_greedy_value", policy_label("greedy_density"), OKABE_ITO[1]),
    ]:
        x, mat = seed_matrix(df, col, "window")
        plot_band(ax, x, mat, name, color)
    ax.set_xlabel("Planning window")
    ax.set_ylabel("True value per window")
    ax.legend(loc="lower right")
    panel_label(ax, "a")

    # (b) paired per-seed improvement of LP over greedy
    ax = axes[0, 1]
    per_seed = df.groupby("seed").agg(
        lp=("lp_true_value", "mean"),
        greedy=("greedy_true_value", "mean"),
        lp_correct=("lp_correct_bin_rate", "mean"),
        greedy_correct=("greedy_correct_bin_rate", "mean"),
        lp_util=("lp_robot_utilization", "mean"),
        greedy_util=("greedy_robot_utilization", "mean"),
    )
    diff = (per_seed["lp"] - per_seed["greedy"]).to_numpy()
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.06, 0.06, size=len(diff))
    ax.scatter(jitter, diff, s=14, color="black", alpha=0.6, linewidths=0, zorder=3)
    m, h = mean_ci(diff)
    ax.errorbar([0.25], [m], yerr=[h], fmt="o", ms=5, capsize=3.5,
                color=OKABE_ITO[0], zorder=4)
    ax.annotate("mean $\\pm$ 95% CI", xy=(0.25, m), xytext=(0.33, m),
                va="center", fontsize=7.5, color=OKABE_ITO[0])
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlim(-0.5, 0.85)
    ax.set_xticks([])
    ax.set_ylabel("Paired difference per seed\n(LP $-$ greedy) value/window")
    panel_label(ax, "b")

    # (c) LP rounding optimality gap (ECDF over all windows, all seeds)
    ax = axes[1, 0]
    gap = np.sort(df["lp_rounding_gap_rel"].to_numpy() * 100.0)
    ecdf = np.arange(1, len(gap) + 1) / len(gap)
    ax.plot(gap, ecdf, color=OKABE_ITO[0])
    ax.axvline(float(np.median(gap)), color="gray", linestyle="--", linewidth=0.8)
    ax.text(float(np.median(gap)), 0.06, f"  median {np.median(gap):.1f}%", fontsize=7.5)
    ax.set_xlabel("Rounding gap vs. LP upper bound [%]")
    ax.set_ylabel("Empirical CDF")
    ax.set_ylim(0, 1.02)
    panel_label(ax, "c")

    # (d) solve time vs problem size
    ax = axes[1, 1]
    ax.scatter(df["n_actions"], df["lp_solve_ms"], s=5, alpha=0.35,
               color=OKABE_ITO[0], linewidths=0)
    coef = np.polyfit(df["n_actions"], df["lp_solve_ms"], deg=1)
    xs = np.linspace(df["n_actions"].min(), df["n_actions"].max(), 50)
    ax.plot(xs, np.polyval(coef, xs), color=OKABE_ITO[1],
            label=f"linear fit: {coef[0]*1000:.1f} $\\mu$s/action")
    ax.set_xlabel("Number of legal actions $|\\mathcal{A}_t|$")
    ax.set_ylabel("LP solve time [ms]")
    ax.legend(loc="upper left")
    panel_label(ax, "d")

    fig.tight_layout(pad=0.4)
    save_figure(fig, fig_dir, "fig5_lp_assignment")

    # ---- Table -------------------------------------------------------------
    rows = []
    for name, val_col, corr_col, util_col in [
        (policy_label("lp"), "lp", "lp_correct", "lp_util"),
        (policy_label("greedy_density"), "greedy", "greedy_correct", "greedy_util"),
    ]:
        m1, h1 = mean_ci(per_seed[val_col].to_numpy())
        m2, h2 = mean_ci(per_seed[corr_col].to_numpy())
        m3, h3 = mean_ci(per_seed[util_col].to_numpy())
        rows.append([name, fmt_pm(m1, h1), fmt_pm(m2, h2), fmt_pm(m3, h3)])
    md, hd = mean_ci(diff)
    rows.append(["Paired difference (LP $-$ greedy)", fmt_pm(md, hd), "--", "--"])
    write_booktabs(
        table_dir / "lp_summary.tex",
        caption="Rolling LP assignment vs.\\ greedy value-density planning (mean $\\pm$ 95\\% CI over seeds).",
        label="tab:lp",
        header=["Planner", "Value/window", "Correct-bin", "Utilization"],
        rows=rows,
    )


# ---------------------------------------------------------------------------
# Figure 6: human-ghost diagnostics
# ---------------------------------------------------------------------------

def make_ghost_figures(df: pd.DataFrame, fig_dir: Path, table_dir: Path) -> None:
    df = df.sort_values(["policy", "seed", "step"]).copy()
    g = df.groupby(["policy", "seed"], sort=False)
    df["roll_reward"] = g["reward"].transform(lambda s: s.rolling(ROLL, min_periods=10).mean())
    df["cum_reward"] = g["reward"].cumsum()
    df["cum_sorted"] = g["sorted_item"].cumsum()
    df["cum_correct"] = g["correct_bin"].cumsum()
    df["correct_rate"] = df["cum_correct"] / np.maximum(df["cum_sorted"], 1)
    df["cum_corrections"] = g["correction_needed_proxy"].cumsum()

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, 4.6), sharex=True)
    panels = [
        ("roll_reward", "Reward per decision", "a"),
        ("cum_reward", "Cumulative reward", "b"),
        ("correct_rate", "Correct-bin rate", "c"),
        ("cum_corrections", "Cumulative human corrections", "d"),
    ]
    for ax, (col, ylabel, letter) in zip(axes.flat, panels):
        for policy in GHOST_POLICIES:
            sub = df[df["policy"] == policy]
            x, mat = seed_matrix(sub, col, "step")
            plot_band(ax, x, mat, policy_label(policy), POLICY_COLORS[policy])
        ax.set_ylabel(ylabel)
        panel_label(ax, letter)
    for ax in axes[1]:
        ax.set_xlabel("Control step $t$")
    axes[0, 1].legend(loc="upper left")
    fig.tight_layout(pad=0.4)
    save_figure(fig, fig_dir, "fig6_ghost_diagnostics")

    # ---- Table -------------------------------------------------------------
    per_seed = df.groupby(["policy", "seed"]).apply(
        lambda gg: pd.Series(
            {
                "total_reward": gg["reward"].sum(),
                "correct_rate": gg["correct_rate"].iloc[-1],
                "contamination": gg["contamination_event"].mean(),
                "corrections": gg["correction_needed_proxy"].sum(),
            }
        ),
        include_groups=False,
    ).reset_index()
    rows = []
    for policy in GHOST_POLICIES:
        sub = per_seed[per_seed["policy"] == policy]
        m1, h1 = mean_ci(sub["total_reward"].to_numpy())
        m2, h2 = mean_ci(sub["correct_rate"].to_numpy())
        m3, h3 = mean_ci(sub["contamination"].to_numpy())
        m4, h4 = mean_ci(sub["corrections"].to_numpy())
        rows.append([policy_label(policy), fmt_pm(m1, h1, 1), fmt_pm(m2, h2, 3), fmt_pm(m3, h3, 3), fmt_pm(m4, h4, 1)])
    write_booktabs(
        table_dir / "ghost_summary.tex",
        caption="Human-ghost prior diagnostics (mean $\\pm$ 95\\% CI over seeds).",
        label="tab:ghost",
        header=["Policy", "Total reward", "Correct-bin", "Contam.\\ rate", "Corrections"],
        rows=rows,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RecycleBot publication figures")
    parser.add_argument("--config", type=str, default=str(Path(__file__).with_name("recyclebot_config.json")))
    parser.add_argument("--seeds", type=int, default=10, help="Number of independent seeds per experiment.")
    parser.add_argument("--bandit-steps", type=int, default=1500)
    parser.add_argument("--lp-windows", type=int, default=300)
    parser.add_argument("--ghost-steps", type=int, default=1200)
    parser.add_argument("--ghost-beta", type=float, default=0.55)
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 4, 8))
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).with_name("figures")))
    parser.add_argument("--reuse-data", action="store_true", help="Reuse cached per-seed CSVs in figures/data/ instead of re-running experiments.")
    parser.add_argument("--usetex", action="store_true", help="Render text with a local LaTeX installation.")
    args = parser.parse_args()

    apply_style(usetex=args.usetex)
    config = load_config(args.config)
    out_dir = Path(args.output_dir)
    fig_dir = out_dir
    table_dir = out_dir / "tables"
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    paths = {k: data_dir / f"{k}_runs.csv" for k in ("bandit", "lp", "ghost")}
    if args.reuse_data and all(p.exists() for p in paths.values()):
        print("Reusing cached per-seed data in", data_dir)
        bandit_df = pd.read_csv(paths["bandit"])
        lp_df = pd.read_csv(paths["lp"])
        ghost_df = pd.read_csv(paths["ghost"])
    else:
        seeds = [42 + 1000 * i for i in range(args.seeds)]
        bandit_df, lp_df, ghost_df = run_all(
            config,
            seeds,
            bandit_steps=args.bandit_steps,
            lp_windows=args.lp_windows,
            ghost_steps=args.ghost_steps,
            ghost_beta=args.ghost_beta,
            jobs=args.jobs,
        )
        bandit_df.to_csv(paths["bandit"], index=False)
        lp_df.to_csv(paths["lp"], index=False)
        ghost_df.to_csv(paths["ghost"], index=False)

    print("Building figures ...")
    make_bandit_figures(bandit_df, fig_dir, table_dir)
    make_ope_figure(bandit_df, fig_dir, table_dir)
    make_lp_figures(lp_df, fig_dir, table_dir)
    make_ghost_figures(ghost_df, fig_dir, table_dir)

    print("Wrote figures to", fig_dir)
    for p in sorted(fig_dir.glob("fig*.pdf")):
        print(f"  {p}")
    for p in sorted(table_dir.glob("*.tex")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
