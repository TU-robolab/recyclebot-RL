"""Demo 3: human-ghost prior diagnostics.

Run:
    python demo_human_ghost.py --steps 1200

Outputs:
    outputs/human_ghost_metrics.csv
    outputs/human_ghost_results.html
    outputs/human_ghost_summary.json

This demo compares four policies:
  1. greedy_bad_Q: a deliberately incomplete value model,
  2. imitation_muH: pure human-ghost imitation,
  3. KL_bad_Q: human ghost plus a degraded Q estimate,
  4. KL_oracle_noise_Q: human ghost plus an oracle-plus-noise Q estimate
     (the true expected reward with N(0, 0.10) noise; an upper-bound diagnostic,
     not a fitted model).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from recyclebot_core import Action, RecycleBotSimulator, load_config


def bad_q_estimate(sim: RecycleBotSimulator, action: Action) -> float:
    """A deliberately incomplete Q model.

    It over-trusts raw item value and target-bin probability, and underweights
    pickability, placement probability, hazard risk, and motion cost. This is
    useful for showing why a human prior can stabilize early learning.
    """
    if action.is_null:
        return 0.0
    assert action.tau is not None and action.robot_id is not None and action.target_bin is not None
    tau = action.tau
    raw_item_value = sum(
        prob * sim.material_item_value.get(M, 0.0)
        for M, prob in tau.material_posterior.items()
    )
    target_confidence = sim.p_correct_bin_from_posterior(tau, action.target_bin)
    weak_cycle_cost = 0.08 * tau.cycle_time_by_robot[action.robot_id]
    return float(0.95 * raw_item_value + 0.45 * target_confidence - weak_cycle_cost)


def choose_index(probs: Sequence[float], rng: np.random.Generator, sample: bool) -> int:
    p = np.asarray(probs, dtype=float)
    p = p / max(p.sum(), 1e-12)
    if sample:
        return int(rng.choice(np.arange(len(p)), p=p))
    return int(np.argmax(p))


def run_experiment(config: dict, steps: int, seed: int, beta: float = 0.55, sample: bool = False) -> pd.DataFrame:
    """Run the four ghost-diagnostic policies for one seed and return raw rows."""
    sim = RecycleBotSimulator(config=config, seed=seed)
    rng = np.random.default_rng(seed + 909)
    decision_window_s = float(config["temporal_dynamics"]["decision_window_s"])

    rows = []
    for step in range(steps):
        env = sim.sample_environment()
        bin_state = sim.sample_bin_state()
        W_t = sim.sample_waste_set(step=step, env=env)
        actions = sim.actions_for(W_t, include_null=True, only_feasible=True, bin_state=bin_state)
        if not actions:
            continue
        mu_h = sim.human_ghost_policy(actions, env)
        true_q = np.asarray([sim.expected_true_reward(action) for action in actions], dtype=float)
        bad_q = np.asarray([bad_q_estimate(sim, action) for action in actions], dtype=float)
        oracle_noise_q = true_q + rng.normal(0.0, 0.10, size=len(actions))
        degraded_q = 0.35 * true_q + 0.65 * bad_q + rng.normal(0.0, 0.18, size=len(actions))

        policies = {
            "greedy_bad_Q": sim.greedy_policy(bad_q),
            "imitation_muH": mu_h,
            "KL_bad_Q": sim.kl_regularized_policy(mu_h, degraded_q, beta=beta),
            "KL_oracle_noise_Q": sim.kl_regularized_policy(mu_h, oracle_noise_q, beta=beta),
        }
        human_top_idx = int(np.argmax(mu_h))
        human_top_action = actions[human_top_idx]

        for policy_name, probs in policies.items():
            idx = choose_index(probs, rng, sample=sample)
            action = actions[idx]
            reward = float(true_q[idx])
            sorted_item = bool(not action.is_null)
            correct = bool(sorted_item and action.tau is not None and action.target_bin == action.tau.true_bin)
            contamination = bool(sorted_item and not correct)
            disagreement = bool(idx != human_top_idx)
            correction_needed = bool(
                disagreement
                and not action.is_null
                and not human_top_action.is_null
                and float(mu_h[human_top_idx] - mu_h[idx]) > 0.20
            )
            rows.append(
                {
                    "step": step,
                    "elapsed_s": (step + 1) * decision_window_s,
                    "policy": policy_name,
                    "n_visible_tau": len(W_t),
                    "n_actions": len(actions),
                    "lighting_U1": env.lighting_U1,
                    "conveyor_speed_V_mps": env.conveyor_speed_mps,
                    "selected_action": action.label(),
                    "reward": reward,  # expected true reward of the choice (deterministic diagnostic)
                    "sorted_item": sorted_item,
                    "correct_bin": correct,
                    "contamination_event": contamination,
                    "human_top_action": human_top_action.label(),
                    "disagrees_with_human_top": disagreement,
                    "correction_needed_proxy": correction_needed,
                    "selected_policy_prob": float(probs[idx]),
                    "human_prior_prob_selected": float(mu_h[idx]),
                    "true_q_selected": float(true_q[idx]),
                    "bad_q_selected": float(bad_q[idx]),
                    "oracle_noise_q_selected": float(oracle_noise_q[idx]),
                    "degraded_q_selected": float(degraded_q[idx]),
                }
            )
    return pd.DataFrame(rows)


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cum_reward"] = df.groupby("policy")["reward"].cumsum()
    df["recovered_value_per_min"] = df["cum_reward"] / np.maximum(df["elapsed_s"] / 60.0, 1e-12)
    df["cum_sorted"] = df.groupby("policy")["sorted_item"].cumsum()
    df["cum_correct"] = df.groupby("policy")["correct_bin"].cumsum()
    df["correct_bin_rate"] = df["cum_correct"] / np.maximum(df["cum_sorted"], 1)
    df["cum_corrections_proxy"] = df.groupby("policy")["correction_needed_proxy"].cumsum()
    df["rolling_reward_100"] = df.groupby("policy")["reward"].transform(lambda s: s.rolling(100, min_periods=5).mean())
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="RecycleBot human-ghost demo")
    parser.add_argument("--config", type=str, default=str(Path(__file__).with_name("recyclebot_config.json")))
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--beta", type=float, default=0.55)
    parser.add_argument("--sample", action="store_true", help="Sample from each policy instead of taking argmax.")
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).with_name("outputs")))
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = run_experiment(config, steps=args.steps, seed=args.seed, beta=args.beta, sample=args.sample)
    df = add_derived_metrics(df)

    metrics_path = out_dir / "human_ghost_metrics.csv"
    summary_path = out_dir / "human_ghost_summary.json"
    html_path = out_dir / "human_ghost_results.html"
    df.to_csv(metrics_path, index=False)

    summary = (
        df.groupby("policy")
        .agg(
            total_reward=("reward", "sum"),
            final_value_per_min=("recovered_value_per_min", "last"),
            final_correct_bin_rate=("correct_bin_rate", "last"),
            correction_proxy_count=("correction_needed_proxy", "sum"),
            sorted_count=("sorted_item", "sum"),
        )
        .sort_values("total_reward", ascending=False)
    )
    summary_path.write_text(json.dumps(summary.to_dict(orient="index"), indent=2), encoding="utf-8")

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Recovered value per minute",
            "Rolling reward",
            "Correct-bin rate",
            "Proxy for human corrections needed",
        ),
    )
    for policy_name, group in df.groupby("policy"):
        group = group.sort_values("step")
        fig.add_trace(go.Scatter(x=group["step"], y=group["recovered_value_per_min"], mode="lines", name=f"{policy_name} value/min"), row=1, col=1)
        fig.add_trace(go.Scatter(x=group["step"], y=group["rolling_reward_100"], mode="lines", name=f"{policy_name} reward"), row=2, col=1)
        fig.add_trace(go.Scatter(x=group["step"], y=group["correct_bin_rate"], mode="lines", name=f"{policy_name} correct-bin"), row=3, col=1)
        fig.add_trace(go.Scatter(x=group["step"], y=group["cum_corrections_proxy"], mode="lines", name=f"{policy_name} corrections"), row=4, col=1)
    fig.update_layout(title="RecycleBot Demo 3 — human ghost prior diagnostics", height=1100, legend_orientation="h")
    fig.update_yaxes(title_text="units/min", row=1, col=1)
    fig.update_yaxes(title_text="reward", row=2, col=1)
    fig.update_yaxes(title_text="correct-bin / selected", row=3, col=1)
    fig.update_yaxes(title_text="count", row=4, col=1)
    fig.update_xaxes(title_text="control step", row=4, col=1)
    fig.write_html(html_path, include_plotlyjs="cdn")

    print("Wrote:")
    print(f"  {metrics_path}")
    print(f"  {summary_path}")
    print(f"  {html_path}")


if __name__ == "__main__":
    main()
