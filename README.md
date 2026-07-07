# RecycleBot RL demo package

This package contains a simplified RecycleBot RL formulation and three runnable demos.



RecycleBot starts as a contextual bandit over \(a(\tau,r,b)\) actions, regularized toward a human-ghost prior and planned through a rolling LP assignment, with value driven by pick success, placement success, per-item recovery value, contamination cost, motion cost, and risk cost.



The current version uses:

- trash item notation \(\tau\);
- visible waste set \(\mathcal{W}_t\);
- feature vector \(\mathbf{F}_{\tau(t)}\in\mathbb{R}^p\);
- elementary action \(a_t=a(\tau,r,b)\);
- null action \(\varnothing\);
- per-item value instead of value per kilogram;
- seven material classes: PET, HDPE, cardboard, metal, glass, organic, hazard;
- four shapes: bottle, tray, can, box;
- three item conditions: clean, crumpled, occluded;
- environment state with only lighting and conveyor speed;
- bin state with only empty/full;
- contamination costs inside the expected gain formula.

There is no separate bin-quality constraint in this simplified version. Quality loss is handled directly through the expected value formula: wrong-bin placements receive no recovered item value and pay a contamination penalty.

---

## Files

| File | Purpose |
|---|---|
| `RECYCLEBOT_RL_MATHEMATICAL_FORMULATION.md` | Expanded mathematical formulation. This is the current source of truth. |
| `RecycleBot RL.md` | Summary mathematical overview of the algorithms from scratch. |
| `recyclebot_config.json` | Editable simulator configuration. |
| `recyclebot_core.py` | Core simulator, value model, policies, and helper functions. |
| `demo_contextual_bandit.py` | Demo 1: contextual bandit — random / imitation / greedy / greedy-ε / Thompson / KL operator-prior. |
| `demo_convex_assignment.py` | Demo 2: rolling LP assignment vs greedy baseline. |
| `demo_human_ghost.py` | Demo 3: human-ghost prior diagnostics. |
| `demo_live_dashboard.py` | Optional local matplotlib dashboard. |
| `paper_figures.py` | Multi-seed publication figures (PDF/PNG) and LaTeX summary tables. |
| `plot_style.py` | Shared publication matplotlib style (Computer Modern, Okabe-Ito colors, CI helpers). |
| `requirements.txt` | Python dependencies. |

The package intentionally keeps only one mathematical formulation file to avoid version confusion.

---

## Install

From this folder:

```bash
pip install -r requirements.txt
```

---

## Run the demos

```bash
python demo_contextual_bandit.py --steps 1500
python demo_convex_assignment.py --windows 300
python demo_human_ghost.py --steps 1200
```

Optional live dashboard:

```bash
python demo_live_dashboard.py --steps 1000 --policy kl_human_ghost
```

Generated files are written to `outputs/`.

---

## Publication figures

```bash
python paper_figures.py --seeds 10
```

Runs every experiment over independent seeds (in parallel) and writes vector
figures plus LaTeX tables to `figures/`:

| Output | Content |
|---|---|
| `fig1_bandit_learning.pdf` | Learning curves (reward, cumulative regret, correct-bin rate, intervention rate) with 95% CI bands. |
| `fig2_bandit_convergence.pdf` | Convergence diagnostics: epistemic Q-variance, policy entropy, value-model prediction error. |
| `fig3_bandit_final.pdf` | Final performance bars with 95% CI error bars and per-seed points. |
| `fig4_bandit_calibration.pdf` | Value-model reliability diagram (predicted Q vs realized reward). |
| `fig5_lp_assignment.pdf` | LP vs greedy value, paired per-seed difference, LP rounding optimality gap, solve-time scaling. |
| `fig6_ghost_diagnostics.pdf` | Human-ghost prior diagnostics with CI bands. |
| `fig7_ope_snips.pdf` | Off-policy evaluation: SNIPS estimates from the KL-ghost logs vs on-policy ground truth. |
| `tables/*.tex` | Booktabs summary tables (mean +/- 95% CI over seeds, plus paired per-seed differences), ready to `\input`. |
| `data/*_runs.csv` | Raw per-seed runs; reuse with `--reuse-data` to re-plot without re-running. |

Every band and error bar is a 95% confidence interval over seeds
(t-distribution, one seed = one run). PNG copies are written next to each PDF.
Use `--usetex` if a local LaTeX installation should render the text instead of
mathtext. Reported metrics include cumulative regret, convergence step
(first step at which the rolling reward reaches 95% of its final plateau),
value-model calibration, human-intervention counts, LP rounding gap versus the
LP fractional upper bound, and LP solve-time scaling.

To support the multi-seed pipeline, two demos were lightly refactored (behavior
unchanged, both still run standalone):

- `demo_human_ghost.py` now exposes `run_experiment(config, steps, seed, beta, sample)`
  and `add_derived_metrics(df)` instead of doing everything in `main()`.
- `demo_convex_assignment.py`'s `build_lp(...)` now also returns the optimal LP
  fractional objective, and the per-window metrics CSV gains
  `lp_fractional_value`, `lp_rounded_expected_value`, `greedy_expected_value`,
  `lp_rounding_gap`, and `lp_rounding_gap_rel` columns.

Note when loading the metrics CSVs with pandas: the literal string `null`
(null actions) is parsed as NaN by default — use
`pd.read_csv(path, keep_default_na=False, na_values=[""])`.

---

##  MVP demos

1. **Contextual bandit** — compares random, imitation, greedy (with/without ε), Thompson sampling, and the KL operator-prior policy.
2. **LP assignment** — compares rolling LP assignment with greedy value-density planning.
3. **Human ghost** — tests whether the human prior stabilizes poor value estimates and improves calibrated value estimates.

### Demo Overview

#### Demo 1 — contextual bandit

Synthetic waste stream + material probabilities + pick success + human ghost + learned value model.
Policies compared: random, imitation of the (noisy) human ghost, greedy without
exploration, greedy with epsilon-exploration, Thompson sampling on a Bayesian
linear-regression posterior, and the KL human-ghost policy.

Plots:

- recovered value per minute;
- cumulative reward;
- rolling reward;
- correct-bin rate;
- human interventions;
- epistemic value-model variance.

#### Demo 2 — LP assignment

Synthetic window with visible items, two robots, material bins, empty/full bin state, and a rolling LP planner.

Plots:

- selected value per window;
- correct-bin rate;
- contamination events in the CSV;
- robot utilization;
- LP solve time;
- number of visible items and legal actions.

#### Demo 3 — human ghost

Compare:

- bad greedy value model;
- pure human imitation;
- KL policy with degraded value estimate;
- KL policy with calibrated value estimate.

This diagnoses whether the human ghost helps when the value model is still unreliable.

---

## Outputs

After running the demos, expected output files include:

| Output | Created by |
|---|---|
| `outputs/contextual_bandit_results.html` | `demo_contextual_bandit.py` |
| `outputs/contextual_bandit_metrics.csv` | `demo_contextual_bandit.py` |
| `outputs/contextual_bandit_summary.json` | `demo_contextual_bandit.py` |
| `outputs/convex_assignment_results.html` | `demo_convex_assignment.py` |
| `outputs/convex_assignment_snapshot.html` | `demo_convex_assignment.py` |
| `outputs/convex_assignment_metrics.csv` | `demo_convex_assignment.py` |
| `outputs/human_ghost_results.html` | `demo_human_ghost.py` |
| `outputs/human_ghost_metrics.csv` | `demo_human_ghost.py` |
| `outputs/human_ghost_summary.json` | `demo_human_ghost.py` |

##  Staged roadmap

| Stage                              | Use when                                               | Main tools                            |
| ---------------------------------- | ------------------------------------------------------ | ------------------------------------- |
| Contextual bandit + KL human ghost | MVP, one-step pick selection                           | \(\widehat{Q}\), \(\mu_H\), KL policy |
| LP assignment                      | multiple feasible item/robot/bin actions in one window | linear programming + greedy rounding  |
| Full MDP/RL                        | timing and future state matter                         | PPO or other sequential RL            |
| Belief-MDP/POMDP                   | material and outcomes are latent                       | belief vectors in the state           |
| CMDP/shielding                     | human safety is in scope                               | hard action masks or constrained RL   |



---

## 21. Summary

The MVP is a rolling contextual bandit over actions:

$$
a_t=a(\tau,r,b)\quad\text{or}\quad a_t=\varnothing.
$$

Each action produces a trajectory instruction:

$$
\Gamma_t(a(\tau,r,b))
=
(\boldsymbol{\xi}_{\{\tau,r,b\}(t)},\mathrm{PICK}(\tau),\mathrm{PLACE}(b)).
$$

The expected one-action value is:

$$
\widehat{g}_{\{\tau,r,b\}(t)}
=
 p^{\mathrm{pick}}_{\{\tau,r\}(t)}
 p^{\mathrm{place}}_{\{\tau,r,b\}(t)}
 G_{\{\tau,b\}(t)}
-
C^{\mathrm{risk}}_{\{\tau,r,b\}(t)}
-
C^{\mathrm{motion}}_{\{\tau,r\}(t)}.
$$

The policy is regularized toward the human ghost:

$$
\pi^*(a\mid S_t)
\propto
\mu_H(a\mid S_t)
\exp(\widehat{Q}(S_t,a)/\beta).
$$

The LP planner assigns item/robot/bin triples subject to item uniqueness, robot capacity, reachability, and empty/full bin availability. Wrong-bin quality loss is handled through the expected gain formula using the no-credit indicator and \(C^{\mathrm{contamination}}_{M,b}\), rather than a separate bin-quality constraint.

---

## License

This repository is distributed under the MIT License unless otherwise stated.

---

Maintained by Elvis Borges @Triku Studio
© 2026 Triku Studio — All Rights Reserved.**
