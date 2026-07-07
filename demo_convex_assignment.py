"""Demo 2: short-horizon LP assignment.

Run:
    python demo_convex_assignment.py --windows 300

Outputs:
    outputs/convex_assignment_metrics.csv
    outputs/convex_assignment_results.html
    outputs/convex_assignment_snapshot.html

The LP relaxation chooses y_{tau,r,b} in [0,1] for a rolling window. It keeps
cycle time in the robot-capacity constraint rather than dividing the objective
by time. The rounded LP plan is compared with a naive greedy value-density plan.

Current simplification: bin state is only empty/full. Bin-quality pressure is
handled through the contamination-cost term in the objective. Correct-bin rate is
reported only as a diagnostic of how often selected actions match the true material bin.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import linprog

from recyclebot_core import Action, RecycleBotSimulator, load_config


def build_lp(
    sim: RecycleBotSimulator,
    actions: Sequence[Action],
    env,
    capacity_s: float,
) -> Tuple[np.ndarray, List[dict], float, bool, float]:
    """Solve LP relaxation for one planning window.

    Feasibility filtering for reachability and full bins happens when actions are
    created. The LP itself enforces item uniqueness and robot cycle-time capacity.
    Returns (fractions, rows, solve_ms, success, lp_fractional_value) where the
    last element is the optimal LP objective, an upper bound on any integral plan.
    """
    if not actions:
        return np.asarray([]), [], 0.0, False, 0.0

    objective = -np.asarray([sim.expected_action_gain(action, env) for action in actions], dtype=float)
    n = len(actions)
    A_ub: List[List[float]] = []
    b_ub: List[float] = []

    tau_ids = sorted({action.tau.tau_id for action in actions if action.tau is not None})
    robot_ids = sorted({action.robot_id for action in actions if action.robot_id is not None})

    # C1: at most one assignment per tau.
    for tau_id in tau_ids:
        row = [1.0 if (action.tau is not None and action.tau.tau_id == tau_id) else 0.0 for action in actions]
        A_ub.append(row)
        b_ub.append(1.0)

    # C2: robot capacity per window.
    for rid in robot_ids:
        row = []
        for action in actions:
            if action.tau is not None and action.robot_id == rid:
                row.append(action.tau.cycle_time_by_robot[rid])
            else:
                row.append(0.0)
        A_ub.append(row)
        b_ub.append(capacity_s)

    t0 = time.perf_counter()
    res = linprog(
        objective,
        A_ub=np.asarray(A_ub, dtype=float) if A_ub else None,
        b_ub=np.asarray(b_ub, dtype=float) if b_ub else None,
        bounds=[(0.0, 1.0)] * n,
        method="highs",
    )
    solve_ms = (time.perf_counter() - t0) * 1000.0
    if not res.success:
        return np.zeros(n, dtype=float), [], solve_ms, False, 0.0

    rows = []
    for j, action in enumerate(actions):
        assert action.tau is not None and action.robot_id is not None and action.target_bin is not None
        tau = action.tau
        rows.append(
            {
                "var_index": j,
                "tau_id": tau.tau_id,
                "robot_id": action.robot_id,
                "target_bin": action.target_bin,
                "expected_gain": -float(objective[j]),
                "lp_fraction": float(res.x[j]),
                "p_target_bin_matches_material": sim.p_correct_bin_from_posterior(tau, action.target_bin),
                "cycle_time_s": tau.cycle_time_by_robot[action.robot_id],
                "material_true_M": tau.material_true_M,
                "true_bin": tau.true_bin,
                "shape_F": tau.shape_F,
                "condition_K": tau.condition_K,
                "x_m": tau.x,
                "y_m": tau.y,
                "time_to_exit_s": tau.time_to_exit_s,
                "best_grasp_G_star": tau.best_grasp_G_star,
            }
        )
    return np.asarray(res.x, dtype=float), rows, solve_ms, True, float(-res.fun)


def round_lp_solution(
    sim: RecycleBotSimulator,
    actions: Sequence[Action],
    fractions: Sequence[float],
    capacity_s: float,
) -> List[int]:
    """Greedy rounding of the LP solution.

    The rounding keeps item uniqueness and robot capacity. Reachability and
    full-bin constraints were already applied when legal actions were created.
    """
    order = sorted(
        range(len(actions)),
        key=lambda j: (float(fractions[j]), sim.expected_action_gain(actions[j], None)),
        reverse=True,
    )
    used_tau = set()
    used_capacity = {rid: 0.0 for rid in sim.robots}
    selected: List[int] = []
    for j in order:
        if fractions[j] <= 1e-7:
            continue
        action = actions[j]
        assert action.tau is not None and action.robot_id is not None and action.target_bin is not None
        tau = action.tau
        rid = action.robot_id
        if tau.tau_id in used_tau:
            continue
        if used_capacity[rid] + tau.cycle_time_by_robot[rid] > capacity_s + 1e-9:
            continue
        selected.append(j)
        used_tau.add(tau.tau_id)
        used_capacity[rid] += tau.cycle_time_by_robot[rid]
    return selected


def greedy_baseline(sim: RecycleBotSimulator, actions: Sequence[Action], env, capacity_s: float) -> List[int]:
    """Naive baseline: value density, robot capacity, and item uniqueness only."""
    order = sorted(
        range(len(actions)),
        key=lambda j: sim.expected_action_gain(actions[j], env) / max(actions[j].tau.cycle_time_by_robot[actions[j].robot_id], 1e-9),  # type: ignore[union-attr,index]
        reverse=True,
    )
    used_tau = set()
    used_capacity = {rid: 0.0 for rid in sim.robots}
    selected: List[int] = []
    for j in order:
        action = actions[j]
        assert action.tau is not None and action.robot_id is not None and action.target_bin is not None
        tau = action.tau
        rid = action.robot_id
        if tau.tau_id in used_tau:
            continue
        if used_capacity[rid] + tau.cycle_time_by_robot[rid] > capacity_s + 1e-9:
            continue
        selected.append(j)
        used_tau.add(tau.tau_id)
        used_capacity[rid] += tau.cycle_time_by_robot[rid]
    return selected


def summarize_selection(sim: RecycleBotSimulator, actions: Sequence[Action], selected: Sequence[int], capacity_s: float) -> dict:
    if not selected:
        return {
            "selected_count": 0,
            "true_value": 0.0,
            "correct_bin_rate": 1.0,
            "contamination_events": 0,
            "robot_utilization": 0.0,
        }
    true_rewards = []
    correct = []
    used_time = 0.0
    contamination_events = 0
    for j in selected:
        action = actions[j]
        assert action.tau is not None and action.robot_id is not None and action.target_bin is not None
        true_rewards.append(sim.expected_true_reward(action))
        is_correct = bool(action.target_bin == action.tau.true_bin)
        correct.append(float(is_correct))
        contamination_events += int(not is_correct)
        used_time += action.tau.cycle_time_by_robot[action.robot_id]
    return {
        "selected_count": int(len(selected)),
        "true_value": float(np.sum(true_rewards)),
        "correct_bin_rate": float(np.mean(correct)) if correct else 1.0,
        "contamination_events": int(contamination_events),
        "robot_utilization": float(used_time / max(capacity_s * len(sim.robots), 1e-9)),
    }


def run_demo(config: dict, windows: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sim = RecycleBotSimulator(config=config, seed=seed)
    capacity_s = float(config["temporal_dynamics"]["decision_window_s"])
    rows = []
    snapshot_rows = []

    for w in range(windows):
        env = sim.sample_environment()
        bin_state = sim.sample_bin_state()
        W_t = sim.sample_waste_set(step=w, env=env)
        actions = sim.actions_for(W_t, include_null=False, only_feasible=True, bin_state=bin_state)
        fractions, lp_rows, solve_ms, success, lp_fractional_value = build_lp(sim, actions, env, capacity_s)
        lp_selected = round_lp_solution(sim, actions, fractions, capacity_s) if success else []
        greedy_selected = greedy_baseline(sim, actions, env, capacity_s) if actions else []
        lp_summary = summarize_selection(sim, actions, lp_selected, capacity_s)
        greedy_summary = summarize_selection(sim, actions, greedy_selected, capacity_s)
        best_possible = float(sum(max(0.0, sim.expected_true_reward(action)) for action in actions))
        lp_rounded_expected = float(sum(sim.expected_action_gain(actions[j], env) for j in lp_selected))
        greedy_expected = float(sum(sim.expected_action_gain(actions[j], env) for j in greedy_selected))

        rows.append(
            {
                "window": w,
                "n_visible_tau": len(W_t),
                "n_actions": len(actions),
                "lp_success": success,
                "lp_solve_ms": solve_ms,
                "lp_fractional_value": lp_fractional_value,
                "lp_rounded_expected_value": lp_rounded_expected,
                "greedy_expected_value": greedy_expected,
                "lp_rounding_gap": max(0.0, lp_fractional_value - lp_rounded_expected),
                "lp_rounding_gap_rel": max(0.0, lp_fractional_value - lp_rounded_expected) / max(abs(lp_fractional_value), 1e-9),
                "lighting_U1": env.lighting_U1,
                "conveyor_speed_V_mps": env.conveyor_speed_mps,
                "full_bins": ",".join([b for b, status in bin_state.items() if status == "full"]),
                "lp_selected_count": lp_summary["selected_count"],
                "lp_true_value": lp_summary["true_value"],
                "lp_correct_bin_rate": lp_summary["correct_bin_rate"],
                "lp_contamination_events": lp_summary["contamination_events"],
                "lp_robot_utilization": lp_summary["robot_utilization"],
                "greedy_selected_count": greedy_summary["selected_count"],
                "greedy_true_value": greedy_summary["true_value"],
                "greedy_correct_bin_rate": greedy_summary["correct_bin_rate"],
                "greedy_contamination_events": greedy_summary["contamination_events"],
                "greedy_robot_utilization": greedy_summary["robot_utilization"],
                "missed_positive_value_lp_proxy": max(0.0, best_possible - lp_summary["true_value"]),
            }
        )

        if w == 0:
            selected_lp_set = set(lp_selected)
            selected_greedy_set = set(greedy_selected)
            for row in lp_rows:
                row["selected_by_lp_rounding"] = int(row["var_index"] in selected_lp_set)
                row["selected_by_greedy"] = int(row["var_index"] in selected_greedy_set)
                snapshot_rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(snapshot_rows)


def build_results_plot(df: pd.DataFrame, output_html: Path) -> None:
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "True value selected per window",
            "Correct-bin rate of selected actions",
            "Robot utilization",
            "LP solve time",
            "Visible items and legal actions",
        ),
    )
    fig.add_trace(go.Scatter(x=df["window"], y=df["lp_true_value"], mode="lines", name="LP rounded value"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["window"], y=df["greedy_true_value"], mode="lines", name="greedy value"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["window"], y=df["lp_correct_bin_rate"], mode="lines", name="LP correct-bin"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["window"], y=df["greedy_correct_bin_rate"], mode="lines", name="greedy correct-bin"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["window"], y=df["lp_robot_utilization"], mode="lines", name="LP utilization"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["window"], y=df["greedy_robot_utilization"], mode="lines", name="greedy utilization"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["window"], y=df["lp_solve_ms"], mode="lines", name="LP solve ms"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df["window"], y=df["n_visible_tau"], mode="lines", name="visible tau"), row=5, col=1)
    fig.add_trace(go.Scatter(x=df["window"], y=df["n_actions"], mode="lines", name="legal actions"), row=5, col=1)
    fig.update_layout(title="RecycleBot Demo 2 — rolling LP assignment vs naive greedy", height=1150, hovermode="x unified")
    fig.update_xaxes(title_text="planning window", row=5, col=1)
    fig.write_html(output_html, include_plotlyjs="cdn")


def build_snapshot_plot(snapshot: pd.DataFrame, output_html: Path) -> None:
    if snapshot.empty:
        return
    # Show one best action row per tau for spatial readability.
    best = snapshot.sort_values("expected_gain", ascending=False).drop_duplicates("tau_id").copy()
    selected_tau = set(snapshot.loc[snapshot["selected_by_lp_rounding"] == 1, "tau_id"])
    best["lp_selected_tau"] = best["tau_id"].isin(selected_tau)
    fig = go.Figure()
    for selected, group in best.groupby("lp_selected_tau"):
        fig.add_trace(
            go.Scatter(
                x=group["x_m"],
                y=group["y_m"],
                mode="markers+text",
                text=group["tau_id"],
                textposition="top center",
                marker=dict(size=np.where(group["lp_selected_tau"], 18, 10), symbol=np.where(group["lp_selected_tau"], "star", "circle")),
                name="LP selected" if selected else "not selected",
                customdata=np.stack(
                    [
                        group["material_true_M"],
                        group["shape_F"],
                        group["condition_K"],
                        group["target_bin"],
                        group["expected_gain"],
                        group["p_target_bin_matches_material"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "tau=%{text}<br>x=%{x:.2f} y=%{y:.2f}<br>"
                    "M=%{customdata[0]}<br>F=%{customdata[1]}<br>K=%{customdata[2]}<br>"
                    "target bin=%{customdata[3]}<br>gain=%{customdata[4]:.3f}<br>p_match=%{customdata[5]:.2f}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title="RecycleBot Demo 2 — first-window spatial snapshot of visible W_t",
        xaxis_title="x position on conveyor (m)",
        yaxis_title="y position on conveyor (m)",
        height=700,
    )
    fig.write_html(output_html, include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser(description="RecycleBot convex assignment demo")
    parser.add_argument("--config", type=str, default=str(Path(__file__).with_name("recyclebot_config.json")))
    parser.add_argument("--windows", type=int, default=300)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).with_name("outputs")))
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df, snapshot = run_demo(config=config, windows=args.windows, seed=args.seed)

    metrics_path = out_dir / "convex_assignment_metrics.csv"
    html_path = out_dir / "convex_assignment_results.html"
    snapshot_path = out_dir / "convex_assignment_snapshot.html"
    df.to_csv(metrics_path, index=False)
    snapshot.to_csv(out_dir / "convex_assignment_snapshot.csv", index=False)
    build_results_plot(df, html_path)
    build_snapshot_plot(snapshot, snapshot_path)

    print("Wrote:")
    print(f"  {metrics_path}")
    print(f"  {html_path}")
    print(f"  {snapshot_path}")


if __name__ == "__main__":
    main()
