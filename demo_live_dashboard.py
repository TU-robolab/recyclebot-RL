"""Optional local live dashboard for the simplified RecycleBot policy.

Run locally with a display:
    python demo_live_dashboard.py --steps 1000 --policy kl_human_ghost

In headless environments, use the HTML outputs from the other demos instead.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from recyclebot_core import LinearValueModel, RecycleBotSimulator, choose_action, entropy, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="RecycleBot live dashboard")
    parser.add_argument("--config", type=str, default=str(Path(__file__).with_name("recyclebot_config.json")))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--policy", type=str, default="kl_human_ghost")
    args = parser.parse_args()

    config = load_config(args.config)
    sim = RecycleBotSimulator(config=config, seed=args.seed)
    model = LinearValueModel(sim.feature_dim, seed=args.seed + 50)
    rng = np.random.default_rng(args.seed + 100)
    beta = float(config["learning"]["beta_kl"])
    lr = float(config["learning"]["model_learning_rate"])
    l2 = float(config["learning"]["l2_penalty"])

    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(9, 8))
    cumulative_rewards = []
    correct_bin_rates = []
    entropies = []
    cumulative_reward = 0.0
    correct_count = 0
    picked_count = 0

    for step in range(args.steps):
        env = sim.sample_environment()
        bin_state = sim.sample_bin_state()
        W_t = sim.sample_waste_set(step=step, env=env)
        actions = sim.actions_for(W_t, robot_ids=["R1"], include_null=True, only_feasible=True, bin_state=bin_state)
        if not actions:
            continue
        features = np.vstack([sim.action_features(a, env) for a in actions])
        q_values = model.predict(features)
        mu_h = sim.human_ghost_policy(actions, env)
        idx, probs = choose_action(rng, actions, q_values, mu_h, mode=args.policy, beta=beta, epsilon=0.03)
        rr = sim.sample_reward(actions[idx], rng=rng)
        model.update(features[idx], rr.reward, lr=lr, l2=l2)
        cumulative_reward += rr.reward
        cumulative_rewards.append(cumulative_reward)
        if rr.picked:
            picked_count += 1
            correct_count += int(rr.success_pick and rr.success_place and rr.correct_bin)
        correct_bin_rates.append(correct_count / max(picked_count, 1))
        entropies.append(entropy(probs))

        if step % 10 == 0 or step == args.steps - 1:
            axes[0].clear()
            axes[1].clear()
            axes[2].clear()
            axes[0].plot(cumulative_rewards)
            axes[0].set_title("Cumulative recovered value")
            axes[1].plot(correct_bin_rates)
            axes[1].set_title("Correct-bin rate")
            axes[2].plot(entropies)
            axes[2].set_title("Policy entropy")
            axes[2].set_xlabel("control step")
            fig.tight_layout()
            plt.pause(0.001)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
