"""Demo 1: contextual bandit with a KL-regularized human-ghost policy.

Run:
    python demo_contextual_bandit.py --steps 1500

Outputs:
    outputs/contextual_bandit_metrics.csv
    outputs/contextual_bandit_results.html
    outputs/contextual_bandit_summary.json

At each control step the simulator creates W_t, the visible waste-item set.
Each legal action is a(tau, r, b), plus the null action. The model learns
Q_hat(S, a) online from reward observations, while the KL policy stays close to
mu_H(a|S), the human-ghost prior.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from recyclebot_core import (
    LinearValueEnsemble,
    RecycleBotSimulator,
    choose_action,
    entropy,
    load_config,
)


def run_policy(policy: str, config: dict, steps: int, seed: int) -> pd.DataFrame:
    sim = RecycleBotSimulator(config=config, seed=seed)
    ensemble = LinearValueEnsemble(feature_dim=sim.feature_dim, n_models=5, seed=seed + 99)
    action_rng = np.random.default_rng(seed + 202)
    reward_rng = np.random.default_rng(seed + 303)

    learning = config["learning"]
    beta = float(learning["beta_kl"])
    lr = float(learning["model_learning_rate"])
    l2 = float(learning["l2_penalty"])
    q_var_threshold = float(learning["human_epistemic_variance_threshold"])
    value_threshold = float(learning["human_value_threshold"])
    human_cost = float(learning["human_intervention_cost"])
    decision_window_s = float(config["temporal_dynamics"]["decision_window_s"])

    rows: List[dict] = []
    cum_reward = 0.0
    cum_regret = 0.0
    cum_interventions = 0

    for t in range(steps):
        env = sim.sample_environment()
        bin_state = sim.sample_bin_state()
        W_t = sim.sample_waste_set(step=t, env=env)
        actions = sim.actions_for(W_t, robot_ids=["R1"], include_null=True, only_feasible=True, bin_state=bin_state)
        if not actions:
            continue

        features = np.vstack([sim.action_features(action, env) for action in actions])
        q_mean, q_var = ensemble.predict_mean_var(features)
        human_prior = sim.human_ghost_policy(actions, env)
        selected_idx, policy_probs = choose_action(
            action_rng,
            actions,
            q_mean,
            human_prior,
            mode=policy,
            beta=beta,
            epsilon=0.04,
        )

        top_expected_gain = max(sim.expected_action_gain(action, env) for action in actions)
        max_epistemic_var = float(np.max(q_var))
        intervention = False
        chosen_by_human = False

        # Human query rule: ask only when the model is uncertain in a learnable
        # way (epistemic variance) and the value at stake is large enough.
        if policy in {"kl", "kl_human_ghost"}:
            if max_epistemic_var > q_var_threshold and top_expected_gain > value_threshold:
                selected_idx = int(np.argmax(human_prior))
                intervention = True
                chosen_by_human = True
                cum_interventions += 1

        action = actions[selected_idx]
        rr = sim.sample_reward(action, rng=reward_rng)
        reward = rr.reward - (human_cost if intervention else 0.0)
        ensemble.update(features[selected_idx], reward, lr=lr, l2=l2)

        true_values = np.asarray([sim.expected_true_reward(action_i) for action_i in actions], dtype=float)
        best_true = float(np.max(true_values))
        selected_true = float(true_values[selected_idx])
        regret = max(0.0, best_true - selected_true)
        cum_reward += reward
        cum_regret += regret

        if action.is_null:
            tau_id = "null"
            material = "null"
            shape = "null"
            condition = "null"
            target_bin = "null"
            true_bin = "null"
            p_pick = 0.0
            p_place = 0.0
            p_correct = 0.0
            grasp = 0.0
            time_to_exit = 0.0
        else:
            assert action.tau is not None and action.robot_id is not None and action.target_bin is not None
            tau = action.tau
            tau_id = tau.tau_id
            material = tau.material_true_M
            shape = tau.shape_F
            condition = tau.condition_K
            target_bin = action.target_bin
            true_bin = tau.true_bin
            p_pick = tau.pick_success_by_robot[action.robot_id]
            p_place = sim.p_place_success(tau, action.robot_id, action.target_bin)
            p_correct = sim.p_correct_bin_from_posterior(tau, action.target_bin)
            grasp = tau.best_grasp_G_star
            time_to_exit = tau.time_to_exit_s

        rows.append(
            {
                "t": t,
                "elapsed_s": (t + 1) * decision_window_s,
                "policy": policy,
                "reward": reward,
                "cum_reward": cum_reward,
                "recovered_value_per_min": cum_reward / max(((t + 1) * decision_window_s) / 60.0, 1e-12),
                "regret": regret,
                "cum_regret": cum_regret,
                "picked": rr.picked,
                "success_pick": rr.success_pick,
                "success_place": rr.success_place,
                "correct_bin": rr.correct_bin,
                "correct_bin_signal": float(rr.picked and rr.success_pick and rr.success_place and rr.correct_bin),
                "intervention": intervention,
                "chosen_by_human": chosen_by_human,
                "cum_interventions": cum_interventions,
                "policy_entropy": entropy(policy_probs),
                "max_epistemic_q_var": max_epistemic_var,
                "n_visible_tau": len(W_t),
                "n_actions": len(actions),
                "lighting_U1": env.lighting_U1,
                "conveyor_speed_V_mps": env.conveyor_speed_mps,
                "lighting_quality": env.quality,
                "full_bins": ",".join([b for b, status in bin_state.items() if status == "full"]),
                "selected_action": action.label(),
                "selected_tau": tau_id,
                "material_true_M": material,
                "shape_F": shape,
                "condition_K": condition,
                "target_bin": target_bin,
                "true_bin": true_bin,
                "p_pick": p_pick,
                "p_place": p_place,
                "p_correct_target_bin": p_correct,
                "best_grasp_G_star": grasp,
                "time_to_exit_s": time_to_exit,
                "q_mean_selected": float(q_mean[selected_idx]),
                "q_var_selected": float(q_var[selected_idx]),
                "human_prior_selected": float(human_prior[selected_idx]),
                "policy_prob_selected": float(policy_probs[selected_idx]),
                "behavior_policy_probability_mu_t": float(policy_probs[selected_idx]),
                "best_true_expected_reward": best_true,
                "selected_true_expected_reward": selected_true,
            }
        )

    return pd.DataFrame(rows)


def add_rolling_metrics(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    pieces = []
    for policy, group in df.groupby("policy", sort=False):
        g = group.sort_values("t").copy()
        g["rolling_reward"] = g["reward"].rolling(window, min_periods=5).mean()
        g["rolling_correct_bin_rate"] = g["correct_bin_signal"].rolling(window, min_periods=5).mean()
        g["rolling_intervention_rate"] = g["intervention"].astype(float).rolling(window, min_periods=5).mean()
        g["rolling_epistemic_var"] = g["max_epistemic_q_var"].rolling(window, min_periods=5).mean()
        g["rolling_entropy"] = g["policy_entropy"].rolling(window, min_periods=5).mean()
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def build_plot(df: pd.DataFrame, output_html: Path) -> None:
    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        subplot_titles=(
            "Recovered value per minute",
            "Cumulative recovered value",
            "Rolling reward per decision",
            "Rolling correct-bin rate: pick succeeded and correct bin",
            "Rolling human intervention rate",
            "Epistemic Q variance and policy entropy",
        ),
    )
    for policy, g in df.groupby("policy", sort=False):
        fig.add_trace(go.Scatter(x=g["t"], y=g["recovered_value_per_min"], mode="lines", name=f"{policy} value/min"), row=1, col=1)
        fig.add_trace(go.Scatter(x=g["t"], y=g["cum_reward"], mode="lines", name=f"{policy} cumulative"), row=2, col=1)
        fig.add_trace(go.Scatter(x=g["t"], y=g["rolling_reward"], mode="lines", name=f"{policy} reward"), row=3, col=1)
        fig.add_trace(go.Scatter(x=g["t"], y=g["rolling_correct_bin_rate"], mode="lines", name=f"{policy} correct-bin"), row=4, col=1)
        fig.add_trace(go.Scatter(x=g["t"], y=g["rolling_intervention_rate"], mode="lines", name=f"{policy} interventions"), row=5, col=1)
        fig.add_trace(go.Scatter(x=g["t"], y=g["rolling_epistemic_var"], mode="lines", name=f"{policy} Q var"), row=6, col=1)
        fig.add_trace(go.Scatter(x=g["t"], y=g["rolling_entropy"], mode="lines", name=f"{policy} entropy"), row=6, col=1)
    fig.update_layout(
        title="RecycleBot Demo 1 — contextual bandit over a(tau, r, b) with KL human-ghost regularization",
        height=1300,
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="control step", row=6, col=1)
    fig.write_html(output_html, include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser(description="RecycleBot contextual bandit demo")
    parser.add_argument("--config", type=str, default=str(Path(__file__).with_name("recyclebot_config.json")))
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).with_name("outputs")))
    parser.add_argument("--policies", nargs="+", default=["random", "human", "greedy", "kl_human_ghost"])
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for policy in args.policies:
        print(f"Running policy: {policy}")
        frames.append(run_policy(policy, config=config, steps=args.steps, seed=args.seed))
    df = pd.concat(frames, ignore_index=True)
    df = add_rolling_metrics(df, window=100)

    metrics_path = out_dir / "contextual_bandit_metrics.csv"
    html_path = out_dir / "contextual_bandit_results.html"
    summary_path = out_dir / "contextual_bandit_summary.json"
    df.to_csv(metrics_path, index=False)
    build_plot(df, html_path)

    summary: Dict[str, dict] = {}
    for policy, g in df.groupby("policy"):
        g = g.sort_values("t")
        summary[policy] = {
            "final_cumulative_reward": float(g["cum_reward"].iloc[-1]),
            "final_value_per_min": float(g["recovered_value_per_min"].iloc[-1]),
            "mean_reward_last_100": float(g["reward"].tail(100).mean()),
            "mean_correct_bin_rate_last_100": float(g["correct_bin_signal"].tail(100).mean()),
            "interventions_total": int(g["intervention"].sum()),
            "mean_epistemic_var_last_100": float(g["max_epistemic_q_var"].tail(100).mean()),
        }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Wrote:")
    print(f"  {metrics_path}")
    print(f"  {summary_path}")
    print(f"  {html_path}")


if __name__ == "__main__":
    main()
