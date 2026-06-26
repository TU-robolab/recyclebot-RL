"""Core simulation primitives for the simplified RecycleBot RL demos.

The code mirrors the readable mathematical formulation:

    W_t = {tau_1(t), ..., tau_n(t)}
    F_tau(t) in R^p
    a_t = a(tau, r, b) or a_t = null

The simplified material set is:
    M = {PET, HDPE, cardboard, metal, glass, organic, hazard}

The simulator is intentionally small. It is not a plant model. It is a demo
that makes the algorithms visible: contextual bandit selection, KL human-ghost
regularization, short-horizon LP assignment, and human-query logic.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-12


def load_config(path: Optional[str | Path] = None) -> dict:
    if path is None:
        path = Path(__file__).with_name("recyclebot_config.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def softmax(scores: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return arr
    temperature = max(float(temperature), 1e-6)
    scaled = arr / temperature
    scaled = scaled - np.max(scaled)
    exp = np.exp(np.clip(scaled, -60.0, 60.0))
    return exp / max(float(np.sum(exp)), EPS)


def entropy(probs: Sequence[float]) -> float:
    p = np.asarray(probs, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def categorical_choice(rng: np.random.Generator, weighted: Dict[str, float]) -> str:
    keys = list(weighted.keys())
    vals = np.asarray([weighted[k] for k in keys], dtype=float)
    vals = vals / max(vals.sum(), EPS)
    return str(rng.choice(keys, p=vals))


@dataclass
class EnvironmentState:
    """Simplified E_t: U1 lighting and conveyor speed only."""

    lighting_U1: str
    conveyor_speed_mps: float
    detection_factor: float
    quality: float

    def as_dict(self) -> dict:
        return {
            "lighting_U1": self.lighting_U1,
            "conveyor_speed_mps": self.conveyor_speed_mps,
            "detection_factor": self.detection_factor,
            "quality": self.quality,
        }


@dataclass
class WasteTau:
    """One tracked trash item tau in W_t."""

    tau_id: str
    material_true_M: str
    true_bin: str
    shape_F: str
    condition_K: str
    pose_xytheta: Tuple[float, float, float]
    dimensions_m: Tuple[float, float, float]
    time_to_exit_s: float
    material_posterior: Dict[str, float]
    grasp_scores_G: List[float]
    best_grasp_G_star: float
    cycle_time_by_robot: Dict[str, float]
    pick_success_by_robot: Dict[str, float]
    reachable_by_robot: Dict[str, bool]
    risk_score: float

    @property
    def x(self) -> float:
        return float(self.pose_xytheta[0])

    @property
    def y(self) -> float:
        return float(self.pose_xytheta[1])

    @property
    def theta(self) -> float:
        return float(self.pose_xytheta[2])

    @property
    def n_grasp_points(self) -> int:
        return len(self.grasp_scores_G)


@dataclass
class Action:
    """Elementary action a(tau, r, b), or null action."""

    tau: Optional[WasteTau]
    robot_id: Optional[str]
    target_bin: Optional[str]

    @property
    def is_null(self) -> bool:
        return self.tau is None

    def label(self) -> str:
        if self.is_null:
            return "null"
        assert self.tau is not None and self.robot_id is not None and self.target_bin is not None
        return f"{self.robot_id}:pick({self.tau.tau_id})->{self.target_bin}"


@dataclass
class RewardResult:
    reward: float
    success_pick: bool
    success_place: bool
    correct_bin: bool
    picked: bool
    item_value: float
    contamination_penalty: float
    cycle_cost: float
    risk_cost: float


class RecycleBotSimulator:
    """Small stochastic simulator for the RecycleBot demos."""

    FEATURE_NAMES = [
        "bias",
        "p_correct_target_bin",
        "max_material_posterior",
        "best_grasp_G_star",
        "p_pick",
        "p_place",
        "urgency",
        "expected_gain_scaled",
        "lighting_quality",
        "is_bottle",
        "is_tray",
        "is_can",
        "is_box",
        "is_crumpled",
        "is_occluded",
        "cycle_time_scaled",
        "conveyor_speed_scaled",
        "target_bin_value_prior",
    ]

    def __init__(self, config: Optional[dict] = None, seed: int = 7):
        self.config = config or load_config()
        self.rng = np.random.default_rng(seed)
        self.materials = list(self.config["materials_M"])
        self.shapes = list(self.config["shapes_F"])
        self.conditions = list(self.config["conditions_K"])
        self.bins = list(self.config["bins_B"])
        self.robots = dict(self.config["robots_R"])
        self.material_to_bin = dict(self.config["material_to_bin_B_of_M"])
        self.material_item_value = dict(self.config["material_item_value"])
        self.contamination_penalty = dict(self.config["contamination_penalty"])

    @property
    def feature_dim(self) -> int:
        return len(self.FEATURE_NAMES)

    def bin_for_material(self, material_M: str) -> str:
        return self.material_to_bin.get(material_M, "hazard")

    def sample_environment(self) -> EnvironmentState:
        lighting_cfg = self.config["environment_state_E"]["U1_lighting"]
        lighting = categorical_choice(self.rng, {k: v["probability"] for k, v in lighting_cfg.items()})
        detection_factor = float(lighting_cfg[lighting]["detection_factor"])
        speed = float(self.rng.choice(self.config["temporal_dynamics"]["V_conveyor_speed_levels_mps"]))
        return EnvironmentState(
            lighting_U1=lighting,
            conveyor_speed_mps=speed,
            detection_factor=detection_factor,
            quality=detection_factor,
        )

    def sample_visible_count(self) -> int:
        dyn = self.config["temporal_dynamics"]
        return int(self.rng.integers(dyn["O_visible_objects_min"], dyn["O_visible_objects_max"] + 1))

    def sample_material(self) -> str:
        return categorical_choice(self.rng, self.config["material_distribution"])

    def sample_shape(self, material_M: str) -> str:
        priors = {
            "PET": {"bottle": 0.65, "tray": 0.15, "can": 0.05, "box": 0.15},
            "HDPE": {"bottle": 0.50, "tray": 0.25, "can": 0.05, "box": 0.20},
            "cardboard": {"box": 0.75, "tray": 0.20, "bottle": 0.02, "can": 0.03},
            "metal": {"can": 0.70, "box": 0.15, "tray": 0.10, "bottle": 0.05},
            "glass": {"bottle": 0.70, "box": 0.15, "tray": 0.08, "can": 0.07},
            "organic": {"tray": 0.50, "box": 0.30, "bottle": 0.10, "can": 0.10},
            "hazard": {"can": 0.30, "bottle": 0.25, "box": 0.25, "tray": 0.20},
        }
        return categorical_choice(self.rng, priors.get(material_M, {s: 1.0 for s in self.shapes}))

    def sample_condition(self) -> str:
        return categorical_choice(self.rng, {"clean": 0.58, "crumpled": 0.25, "occluded": 0.17})

    def sample_bin_state(self) -> Dict[str, str]:
        p_full = float(self.config["bin_state_model"].get("prob_full", 0.0))
        return {b: ("full" if self.rng.random() < p_full else "empty") for b in self.bins}

    def _material_posterior(self, true_M: str, condition_K: str, env: EnvironmentState) -> Dict[str, float]:
        condition_factor = self.config["condition_effects"][condition_K]["detection_factor"]
        quality = clamp(env.detection_factor * condition_factor, 0.05, 1.0)
        true_conf = clamp(0.28 + 0.64 * quality + self.rng.normal(0.0, 0.035), 0.17, 0.97)
        other_count = len(self.materials) - 1
        noise = self.rng.dirichlet(np.ones(other_count))
        out: Dict[str, float] = {}
        j = 0
        for M in self.materials:
            if M == true_M:
                out[M] = float(true_conf)
            else:
                out[M] = float((1.0 - true_conf) * noise[j])
                j += 1
        total = max(sum(out.values()), EPS)
        return {M: float(p / total) for M, p in out.items()}

    def _dimensions_for_shape(self, shape_F: str) -> Tuple[float, float, float]:
        base = self.config["shape_defaults"][shape_F]["dimensions_m"]
        noise = self.rng.lognormal(mean=0.0, sigma=0.14, size=3)
        return tuple(float(max(0.005, float(base[i]) * noise[i])) for i in range(3))

    def _grasp_scores(self, shape_F: str, condition_K: str) -> List[float]:
        shape_cfg = self.config["shape_defaults"][shape_F]
        cond_cfg = self.config["condition_effects"][condition_K]
        mean_g = max(1, int(round(shape_cfg["grasp_points_mean"])))
        n_g = max(1, int(self.rng.poisson(max(mean_g - 0.25, 0.1)) + 1))
        base = float(shape_cfg["base_grasp"]) * float(cond_cfg["grasp_factor"])
        return [float(clamp(base + self.rng.normal(0.0, 0.13), 0.03, 0.98)) for _ in range(n_g)]

    def _reachable_and_success(
        self,
        pose_xytheta: Tuple[float, float, float],
        best_grasp: float,
        shape_F: str,
        condition_K: str,
        time_to_exit_s: float,
    ) -> Tuple[Dict[str, bool], Dict[str, float], Dict[str, float]]:
        x, y, _ = pose_xytheta
        shape_cfg = self.config["shape_defaults"][shape_F]
        cond_cfg = self.config["condition_effects"][condition_K]
        reachable: Dict[str, bool] = {}
        p_pick: Dict[str, float] = {}
        cycle: Dict[str, float] = {}
        for rid, rcfg in self.robots.items():
            cycle_time = float(shape_cfg["cycle_time_s"]) * float(rcfg["cycle_time_factor"])
            in_workspace = bool(rcfg["x_min"] <= x <= rcfg["x_max"] and rcfg["y_min"] <= y <= rcfg["y_max"])
            time_feasible = bool(cycle_time <= time_to_exit_s + 0.10)
            reachable[rid] = bool(in_workspace and time_feasible)
            p = 0.10 + 0.82 * best_grasp * float(rcfg["grasp_skill"])
            p -= 0.28 * float(cond_cfg["risk"])
            if not reachable[rid]:
                p *= 0.05
            p_pick[rid] = float(clamp(p, 0.01, 0.98))
            cycle[rid] = float(cycle_time)
        return reachable, p_pick, cycle

    def sample_waste_set(self, step: int = 0, env: Optional[EnvironmentState] = None, n: Optional[int] = None) -> List[WasteTau]:
        if env is None:
            env = self.sample_environment()
        if n is None:
            n = self.sample_visible_count()
        dyn = self.config["temporal_dynamics"]
        length = float(dyn["workspace_length_m"])
        width = float(dyn["workspace_width_m"])
        W_t: List[WasteTau] = []
        for k in range(n):
            material = self.sample_material()
            true_bin = self.bin_for_material(material)
            shape = self.sample_shape(material)
            condition = self.sample_condition()
            x = float(self.rng.uniform(0.0, length))
            y = float(self.rng.uniform(-width / 2.0, width / 2.0))
            theta = float(self.rng.uniform(-math.pi, math.pi))
            pose = (x, y, theta)
            distance_to_exit = max(0.05, length - x)
            time_to_exit = float(distance_to_exit / max(env.conveyor_speed_mps, 0.05))
            time_to_exit = max(0.05, time_to_exit + float(self.rng.normal(0.0, 0.08)))
            posterior = self._material_posterior(material, condition, env)
            grasp_scores = self._grasp_scores(shape, condition)
            best_grasp = float(max(grasp_scores))
            reachable, p_pick, cycle = self._reachable_and_success(pose, best_grasp, shape, condition, time_to_exit)
            risk = float(self.config["condition_effects"][condition]["risk"])
            if material == "hazard":
                risk = min(1.0, risk + 0.45)
            tau = WasteTau(
                tau_id=f"tau_{step:05d}_{k:02d}",
                material_true_M=material,
                true_bin=true_bin,
                shape_F=shape,
                condition_K=condition,
                pose_xytheta=pose,
                dimensions_m=self._dimensions_for_shape(shape),
                time_to_exit_s=time_to_exit,
                material_posterior=posterior,
                grasp_scores_G=grasp_scores,
                best_grasp_G_star=best_grasp,
                cycle_time_by_robot=cycle,
                pick_success_by_robot=p_pick,
                reachable_by_robot=reachable,
                risk_score=risk,
            )
            W_t.append(tau)
        return W_t

    # Backward-compatible alias used by a few old notebooks/scripts.
    def sample_items(self, env: EnvironmentState, n: Optional[int] = None) -> List[WasteTau]:
        return self.sample_waste_set(step=0, env=env, n=n)

    def actions_for(
        self,
        W_t: Sequence[WasteTau],
        robot_ids: Optional[Sequence[str]] = None,
        include_null: bool = True,
        only_feasible: bool = True,
        bin_state: Optional[Dict[str, str]] = None,
    ) -> List[Action]:
        if robot_ids is None:
            robot_ids = list(self.robots.keys())
        actions: List[Action] = []
        for tau in W_t:
            for rid in robot_ids:
                if only_feasible and not tau.reachable_by_robot.get(rid, False):
                    continue
                for b in self.bins:
                    if bin_state is not None and bin_state.get(b, "empty") == "full":
                        continue
                    actions.append(Action(tau=tau, robot_id=rid, target_bin=b))
        if include_null:
            actions.append(Action(tau=None, robot_id=None, target_bin=None))
        return actions

    # Backward-compatible alias.
    def make_actions(self, items: Sequence[WasteTau], robot_ids: Optional[Sequence[str]] = None, include_null: bool = True, only_feasible: bool = True) -> List[Action]:
        return self.actions_for(items, robot_ids=robot_ids, include_null=include_null, only_feasible=only_feasible)

    def p_correct_bin_from_posterior(self, tau: WasteTau, target_bin: str) -> float:
        return float(sum(prob for M, prob in tau.material_posterior.items() if self.bin_for_material(M) == target_bin))

    # Backward-compatible alias.
    def p_item_belongs_to_bin(self, tau: WasteTau, target_bin: str) -> float:
        return self.p_correct_bin_from_posterior(tau, target_bin)

    def p_place_success(self, tau: WasteTau, robot_id: str, target_bin: str) -> float:
        p_correct = self.p_correct_bin_from_posterior(tau, target_bin)
        cycle = tau.cycle_time_by_robot[robot_id]
        p = 0.70 + 0.22 * p_correct - 0.12 * tau.risk_score - 0.03 * max(cycle - 0.75, 0.0)
        return float(clamp(p, 0.35, 0.98))

    def expected_gain_G(self, tau: WasteTau, target_bin: str) -> float:
        """G_{tau,b,t}: expected material/bin gain using per-item value.

        G = sum_M P(M_tau=M|z) [1[b=B(M)] V_M_item - C_contamination_{M,b}]
        """
        total = 0.0
        for M, prob in tau.material_posterior.items():
            correct = self.bin_for_material(M) == target_bin
            value = float(self.material_item_value[M]) if correct else 0.0
            contamination = float(self.contamination_penalty[M][target_bin])
            total += float(prob) * (value - contamination)
        return float(total)

    # Backward-compatible alias.
    def expected_material_gain(self, tau: WasteTau, target_bin: str) -> float:
        return self.expected_gain_G(tau, target_bin)

    def expected_action_gain(self, action: Action, env: Optional[EnvironmentState] = None) -> float:
        if action.is_null:
            return 0.0
        assert action.tau is not None and action.robot_id is not None and action.target_bin is not None
        tau = action.tau
        p_pick = tau.pick_success_by_robot[action.robot_id]
        p_place = self.p_place_success(tau, action.robot_id, action.target_bin)
        G = self.expected_gain_G(tau, action.target_bin)
        C_risk = 0.32 * tau.risk_score
        C_motion = 0.055 * tau.cycle_time_by_robot[action.robot_id]
        return float(p_pick * p_place * G - C_risk - C_motion)

    def expected_true_reward(self, action: Action) -> float:
        if action.is_null:
            return 0.0
        assert action.tau is not None and action.robot_id is not None and action.target_bin is not None
        tau = action.tau
        M = tau.material_true_M
        correct = self.bin_for_material(M) == action.target_bin
        p_pick = tau.pick_success_by_robot[action.robot_id]
        p_place = self.p_place_success(tau, action.robot_id, action.target_bin)
        value = float(self.material_item_value[M]) if correct else 0.0
        contamination = 0.0 if correct else float(self.contamination_penalty[M][action.target_bin])
        C_risk = 0.32 * tau.risk_score
        C_motion = 0.055 * tau.cycle_time_by_robot[action.robot_id]
        return float(p_pick * p_place * (value - contamination) - C_risk - C_motion)

    # Backward-compatible alias used by old scripts.
    def true_expected_reward(self, action: Action) -> RewardResult:
        r = self.expected_true_reward(action)
        if action.is_null:
            return RewardResult(r, False, False, False, False, 0.0, 0.0, 0.0, 0.0)
        assert action.tau is not None and action.robot_id is not None and action.target_bin is not None
        M = action.tau.material_true_M
        correct = self.bin_for_material(M) == action.target_bin
        return RewardResult(
            reward=r,
            success_pick=action.tau.pick_success_by_robot[action.robot_id] > 0.5,
            success_place=self.p_place_success(action.tau, action.robot_id, action.target_bin) > 0.5,
            correct_bin=correct,
            picked=True,
            item_value=float(self.material_item_value[M]) if correct else 0.0,
            contamination_penalty=0.0 if correct else float(self.contamination_penalty[M][action.target_bin]),
            cycle_cost=0.055 * action.tau.cycle_time_by_robot[action.robot_id],
            risk_cost=0.32 * action.tau.risk_score,
        )

    def sample_reward(self, action: Action, rng: Optional[np.random.Generator] = None) -> RewardResult:
        if rng is None:
            rng = self.rng
        if action.is_null:
            return RewardResult(0.0, False, False, False, False, 0.0, 0.0, 0.0, 0.0)
        assert action.tau is not None and action.robot_id is not None and action.target_bin is not None
        tau = action.tau
        M = tau.material_true_M
        correct = self.bin_for_material(M) == action.target_bin
        p_pick = tau.pick_success_by_robot[action.robot_id]
        p_place = self.p_place_success(tau, action.robot_id, action.target_bin)
        success_pick = bool(rng.random() < p_pick)
        success_place = bool(success_pick and rng.random() < p_place)
        value = float(self.material_item_value[M]) if correct else 0.0
        contamination = 0.0 if correct else float(self.contamination_penalty[M][action.target_bin])
        cycle_cost = 0.055 * tau.cycle_time_by_robot[action.robot_id]
        risk_cost = 0.32 * tau.risk_score
        reward = 0.0
        if success_pick and success_place and correct:
            reward += value
        elif success_pick and success_place and not correct:
            reward -= contamination
        elif not success_pick:
            # Failed grasp costs time and can lose the object.
            reward -= 0.08 * max(value, 0.0)
        else:
            # Picked but failed placement.
            reward -= 0.05 + 0.15 * tau.risk_score
        reward -= cycle_cost + risk_cost
        return RewardResult(
            reward=float(reward),
            success_pick=success_pick,
            success_place=success_place,
            correct_bin=correct,
            picked=True,
            item_value=value,
            contamination_penalty=contamination,
            cycle_cost=cycle_cost,
            risk_cost=risk_cost,
        )

    def action_features(self, action: Action, env: EnvironmentState) -> np.ndarray:
        if action.is_null:
            x = np.zeros(self.feature_dim, dtype=float)
            x[self.FEATURE_NAMES.index("bias")] = 1.0
            x[self.FEATURE_NAMES.index("lighting_quality")] = env.quality
            x[self.FEATURE_NAMES.index("conveyor_speed_scaled")] = env.conveyor_speed_mps / 1.0
            return x
        assert action.tau is not None and action.robot_id is not None and action.target_bin is not None
        tau = action.tau
        p_correct = self.p_correct_bin_from_posterior(tau, action.target_bin)
        max_p = max(tau.material_posterior.values())
        p_pick = tau.pick_success_by_robot[action.robot_id]
        p_place = self.p_place_success(tau, action.robot_id, action.target_bin)
        urgency = clamp(1.0 / max(tau.time_to_exit_s, 0.10), 0.0, 5.0) / 5.0
        expected_gain_scaled = self.expected_action_gain(action, env) / 2.5
        target_value_prior = sum(
            max(0.0, float(self.material_item_value[M]))
            for M in self.materials
            if self.bin_for_material(M) == action.target_bin
        )
        target_value_prior = clamp(target_value_prior / 2.5, 0.0, 1.5)
        x = np.asarray(
            [
                1.0,
                p_correct,
                max_p,
                tau.best_grasp_G_star,
                p_pick,
                p_place,
                urgency,
                expected_gain_scaled,
                env.quality,
                1.0 if tau.shape_F == "bottle" else 0.0,
                1.0 if tau.shape_F == "tray" else 0.0,
                1.0 if tau.shape_F == "can" else 0.0,
                1.0 if tau.shape_F == "box" else 0.0,
                1.0 if tau.condition_K == "crumpled" else 0.0,
                1.0 if tau.condition_K == "occluded" else 0.0,
                clamp(tau.cycle_time_by_robot[action.robot_id] / 1.5, 0.0, 2.0),
                env.conveyor_speed_mps / 1.0,
                target_value_prior,
            ],
            dtype=float,
        )
        return x

    def human_ghost_policy(self, actions: Sequence[Action], env: EnvironmentState, temperature: Optional[float] = None) -> np.ndarray:
        if temperature is None:
            temperature = float(self.config["learning"]["softmax_temperature"])
        scores: List[float] = []
        for action in actions:
            if action.is_null:
                scores.append(-0.20)
                continue
            assert action.tau is not None and action.robot_id is not None and action.target_bin is not None
            tau = action.tau
            p_correct = self.p_correct_bin_from_posterior(tau, action.target_bin)
            p_pick = tau.pick_success_by_robot[action.robot_id]
            p_place = self.p_place_success(tau, action.robot_id, action.target_bin)
            urgency = clamp(1.0 / max(tau.time_to_exit_s, 0.10), 0.0, 5.0) / 5.0
            expected = self.expected_action_gain(action, env)
            true_match = 1.0 if action.target_bin == tau.true_bin else 0.0
            score = (
                1.25 * expected
                + 0.65 * p_correct
                + 0.45 * p_pick
                + 0.20 * p_place
                + 0.35 * urgency
                + 0.25 * true_match
                - 0.65 * tau.risk_score
                - 0.08 * tau.cycle_time_by_robot[action.robot_id]
            )
            scores.append(float(score))
        return softmax(scores, temperature=temperature)

    def greedy_policy(self, q_values: Sequence[float]) -> np.ndarray:
        q = np.asarray(q_values, dtype=float)
        p = np.zeros_like(q, dtype=float)
        p[int(np.argmax(q))] = 1.0
        return p

    def kl_regularized_policy(self, human_prior: Sequence[float], q_values: Sequence[float], beta: float) -> np.ndarray:
        return kl_regularized_policy(q_values, human_prior, beta)

    def item_table_rows(self, W_t: Sequence[WasteTau]) -> List[dict]:
        rows = []
        for tau in W_t:
            row = {
                "tau_id": tau.tau_id,
                "material_true_M": tau.material_true_M,
                "true_bin_B_of_M": tau.true_bin,
                "shape_F": tau.shape_F,
                "condition_K": tau.condition_K,
                "x_m": tau.x,
                "y_m": tau.y,
                "theta_rad": tau.theta,
                "time_to_exit_s": tau.time_to_exit_s,
                "n_grasp_points_G": tau.n_grasp_points,
                "best_grasp_G_star": tau.best_grasp_G_star,
                "risk_score": tau.risk_score,
                "length_m": tau.dimensions_m[0],
                "width_m": tau.dimensions_m[1],
                "height_m": tau.dimensions_m[2],
            }
            for rid in self.robots:
                row[f"reachable_{rid}"] = tau.reachable_by_robot.get(rid, False)
                row[f"p_pick_{rid}"] = tau.pick_success_by_robot.get(rid, 0.0)
                row[f"cycle_time_{rid}_s"] = tau.cycle_time_by_robot.get(rid, 0.0)
            for M in self.materials:
                row[f"P_M_{M}"] = tau.material_posterior[M]
            rows.append(row)
        return rows


class LinearValueModel:
    """Small online linear model for Q_hat(S, action)."""

    def __init__(self, feature_dim: int, seed: int = 123):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.03, size=feature_dim)
        self.w[0] = 0.0

    def predict_one(self, features: np.ndarray) -> float:
        return float(np.dot(self.w, features))

    def predict(self, feature_matrix: np.ndarray) -> np.ndarray:
        return np.asarray(feature_matrix @ self.w, dtype=float)

    def update(self, features: np.ndarray, reward: float, lr: float = 0.03, l2: float = 0.0005) -> float:
        pred = self.predict_one(features)
        err = float(reward - pred)
        self.w += lr * (err * features - l2 * self.w)
        self.w = np.clip(self.w, -10.0, 10.0)
        return err


class LinearValueEnsemble:
    """Tiny bootstrap-style ensemble to expose epistemic variance."""

    def __init__(self, feature_dim: int, n_models: int = 5, seed: int = 123):
        self.models = [LinearValueModel(feature_dim, seed=seed + 31 * i) for i in range(n_models)]
        self.rng = np.random.default_rng(seed + 999)

    def predict_matrix(self, feature_matrix: np.ndarray) -> np.ndarray:
        return np.vstack([m.predict(feature_matrix) for m in self.models])

    def predict_mean_var(self, feature_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        preds = self.predict_matrix(feature_matrix)
        return preds.mean(axis=0), preds.var(axis=0)

    def update(self, features: np.ndarray, reward: float, lr: float = 0.03, l2: float = 0.0005) -> None:
        for model in self.models:
            # Bootstrap mask plus tiny target noise gives visible disagreement early on.
            if self.rng.random() < 0.85:
                noisy_reward = float(reward + self.rng.normal(0.0, 0.03))
                model.update(features, noisy_reward, lr=lr, l2=l2)


def kl_regularized_policy(q_values: Sequence[float], human_prior: Sequence[float], beta: float) -> np.ndarray:
    q = np.asarray(q_values, dtype=float)
    mu = np.asarray(human_prior, dtype=float)
    mu = np.maximum(mu, 1e-9)
    mu = mu / max(mu.sum(), EPS)
    beta = max(float(beta), 1e-6)
    scores = np.log(mu) + q / beta
    return softmax(scores, temperature=1.0)


def reverse_kl_policy(q_values: Sequence[float], human_prior: Sequence[float], beta: float) -> np.ndarray:
    """Reverse-KL policy: pi_a = beta * mu_a / (nu - Q_a).

    The normalizer nu is found by bisection. This is included for experiments;
    the demos use the simpler forward-KL policy by default.
    """
    q = np.asarray(q_values, dtype=float)
    mu = np.asarray(human_prior, dtype=float)
    mu = np.maximum(mu, 1e-9)
    mu = mu / max(mu.sum(), EPS)
    beta = max(float(beta), 1e-6)
    lo = float(np.max(q) + 1e-9)
    hi = lo + beta + np.max(np.abs(q)) + 10.0

    def f(nu: float) -> float:
        return float(np.sum(beta * mu / np.maximum(nu - q, 1e-12)) - 1.0)

    while f(hi) > 0:
        hi = hi * 2.0 + 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    nu = 0.5 * (lo + hi)
    pi = beta * mu / np.maximum(nu - q, 1e-12)
    return pi / max(float(pi.sum()), EPS)


def choose_action(
    rng: np.random.Generator,
    actions: Sequence[Action],
    q_values: Sequence[float],
    human_prior: Sequence[float],
    mode: str,
    beta: float,
    epsilon: float = 0.03,
) -> Tuple[int, np.ndarray]:
    mode = mode.lower()
    n = len(actions)
    if n == 0:
        raise ValueError("No actions available.")
    if mode == "random":
        probs = np.ones(n, dtype=float) / n
    elif mode in {"human", "imitation", "imitation_muh"}:
        probs = np.asarray(human_prior, dtype=float)
        probs = probs / max(probs.sum(), EPS)
    elif mode == "greedy":
        probs = np.zeros(n, dtype=float)
        probs[int(np.argmax(q_values))] = 1.0
    elif mode in {"kl", "kl_human_ghost"}:
        probs = kl_regularized_policy(q_values, human_prior, beta=beta)
    elif mode in {"reverse_kl", "rkl"}:
        probs = reverse_kl_policy(q_values, human_prior, beta=beta)
    else:
        raise ValueError(f"Unknown policy mode: {mode}")

    if epsilon > 0 and mode not in {"greedy"}:
        probs = (1.0 - epsilon) * probs + epsilon * (np.ones(n) / n)
        probs = probs / max(probs.sum(), EPS)
    idx = int(rng.choice(np.arange(n), p=probs))
    return idx, probs
