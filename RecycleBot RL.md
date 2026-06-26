# RecycleBot RL 



### Objective

Continous improvement for trash -sorting system. **Focus on augmenting robot picking strategies:**

* maximize the overall value of the waste sorting process

* improve input based on mimicking human stream

* allow a human operator to (eventually) supervise multiple robot stations.

### Approach

#### System Components



1. **Value-based bin selection**

   * Use a “human ghost” concept to provide input for RL

   * The system converges to picking with selecting the highest-value trash

2. **Single waste stream**

   * Start with one waste stream while retaining the full object dimensionality

   *  e.g, position, 6D shape (position + orientation), pick-ability, attractiveness(value for picking), etc, more attributes?

3. **Perception streams**

   * T1: Object detection stream, provides continuous wast object stream

   * T2: Human hand-tracking stream to observe operator interventions/corrections/ parallel movement

4. **Learning**

   * Combine the perception streams.

   *  Converge towards an RKL bandit approach to adapt and improve pick selection based on both object observations and human feedback.



#### Optimization goal

* Optimize the steady state total value recovered from the waste stream,  (not value of individual picks)
* continuous optimization as stream improves

#### Outcome:

* higher overall sorting quality
* Continuously adaptable robot behavior based on human guidance 
* minimized amount of supervision required.



##### Further topics for discussion

Human-in-the-loop

* One operator monitors multiple robot stations
*  The operator only intervenes when necessary by correcting one robot at a time.
*  These interventions become learning signals that improve future autonomous decisions.



____

## Mathematical Setup

> Which trash item should which robot pick, and into which bin should it go?

#### current MVP uses three ideas:

1. a **contextual bandit** for one-step action choice;
2. a **human-ghost prior** to keep the robot close to experienced human behavior;
3. a **linear-programming assignment planner** when multiple robots can act in the same window.

---

### 1 - What the system sees

At time `t`, the system observes:

$$
S_t=(\mathcal{W}_t,\mathcal{R}_t,\mathcal{B}_t,H_t,E_t).
$$

This means:

| Part              | Meaning                                                      |
| ----------------- | ------------------------------------------------------------ |
| \(\mathcal{W}_t\) | visible tracked waste items                                  |
| \(\mathcal{R}_t\) | robot state: reachable area, busy/free state, gripper capability |
| \(\mathcal{B}_t\) | bin state: which bin, empty/full                             |
| \(H_t\)           | human operator state and corrections                         |
| \(E_t\)           | environment state: lighting and conveyor speed               |

One tracked item is called:

$$
\tau.
$$

The visible item set is:

$$
\mathcal{W}_t=\{\tau_{1(t)},\ldots,\tau_{n(t)}\}.
$$

For each item, the perception system builds a feature vector:

$$
\mathbf{F}_{\tau(t)}\in\mathbb{R}^p.
$$





### 2 - Current simplified object description

The material set is:

$$
\mathcal{M}=\{\mathrm{PET},\mathrm{HDPE},\mathrm{cardboard},\mathrm{metal},\mathrm{glass},\mathrm{organic},\mathrm{hazard}\}.
$$

The shape set is:

$$
\mathcal{F}=\{\mathrm{bottle},\mathrm{tray},\mathrm{can},\mathrm{box}\}.
$$

The item-condition set is:

$$
\mathcal{K}=\{\mathrm{clean},\mathrm{crumpled},\mathrm{occluded}\}.
$$

The environment is only:

$$
E_t=(U_{1,t},v_t^{\mathrm{belt}}),
$$

where \(U_{1,t}\) is lighting and \(v_t^{\mathrm{belt}}\) is conveyor speed.

The bin state is only empty/full for now. There is no separate bin-quality state in this simplified MVP.

____



### 3 -  an action

A single elementary action is:

$$
a_t=a(\tau,r,b).
$$

This means:

> pick item \(\tau\) with robot \(r\) and place it into bin \(b\).

The legal action space is:

$$
\mathbb{A}_t
=
\{a(\tau,r,b)\mid \tau\in\mathcal{W}_t,\ r\in\mathcal{R}_t,\ b\in\mathcal{B}_t\}
\cup
\{\varnothing\}.
$$

The null action \(\varnothing\) means: do not pick now.

A non-null action produces a robot instruction:

$$
\Gamma_t(a(\tau,r,b))
=
(\boldsymbol{\xi}_{\tau,r,b,t},\mathrm{PICK}(\tau),\mathrm{PLACE}(b)).
$$

Here \(\boldsymbol{\xi}_{\tau,r,b,t}\) is the trajectory or motion plan.

____



###  4 - Long term Robot picking strategy (robot should not simply pick the highest-value item)

A high-value item can still be a bad pick if it is:

- hard to grasp;
- almost outside the reachable workspace;
- likely to go into the wrong bin;
- hazardous;
- too slow to pick compared with several easier items.

So the system estimates the expected net value of one action:

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

This combines:

| Term                    | Meaning                                                      |
| ----------------------- | ------------------------------------------------------------ |
| \(p^{\mathrm{pick}}\)   | probability the robot successfully picks the item            |
| \(p^{\mathrm{place}}\)  | probability the robot successfully places it into the target bin |
| \(G_{\{\tau,b\}(t)}\)   | expected material-bin gain                                   |
| \(C^{\mathrm{risk}}\)   | risk/safety/hazard cost                                      |
| \(C^{\mathrm{motion}}\) | motion, cycle-time, wear, or energy cost                     |

____



### 5. Expected material-bin gain

The perception system does not know the true material with certainty. It has probabilities:

$$
P(M_\tau=M\mid z_{\tau,t}).
$$

The correct bin for material \(M\) is:

$$
B(M).
$$

The per-item value of correctly recovering material \(M\) is:

$$
V_M^{\mathrm{item}}.
$$

The expected gain of placing item \(\tau\) into bin \(b\) is:

$$
G_{\tau,b,t}
=
\sum_{M\in\mathcal{M}}
P(M_\tau=M\mid z_{\tau,t})
\left[
\mathbf{1}[b=B(M)]V_M^{\mathrm{item}}
-
C^{\mathrm{contamination}}_{M,b}
\right].
$$

The indicator \(\mathbf{1}[b=B(M)]\) means the system only gets item value if the item goes to the correct bin. The contamination cost is positive when material \(M\) damages the target stream \(b\).

---

 ### Algorithms

####  6 - Contextual bandit: the MVP learner

A contextual bandit is the simplest useful learning setup here.

At each step:

1. observe the current situation \(S_t\);
2. list all legal actions \(\mathbb{A}_t\);
3. estimate the value of every action with \(\widehat{Q}(S_t,a)\);
4. pick one action;
5. observe reward or delayed outcome;
6. update the value model.

The objective is:

$$
\max_\theta
\mathbb{E}_{S_t,a_t\sim\pi_\theta}
[\mathrm{reward}(S_t,a_t)].
$$

This is a good MVP because the first prototype can treat each pick as mostly independent.

Move to a full MDP/RL formulation when future effects become important: robot busy time, queueing, bin emptying, conveyor congestion, or operator workload.

#### 7 - Human ghost: using the operator as a behavioral prior

The human-ghost policy is:

$$
\mu_H(a\mid S_t).
$$

It means:

> given the current scene, how likely is a skilled human to choose action \(a\)?

It can be learned from hand tracking, corrections, manual picks, or operator choices.

The human ghost is not ground truth. It is a prior. Humans are useful but can be tired, inconsistent, or attention-limited.

#### 8 - KL human-ghost policy

The robot combines the learned value model and the human prior:

$$
\pi^*(a\mid S)
=
\frac{
\mu_H(a\mid S)\exp(\widehat{Q}(S,a)/\beta)
}{
\sum_{a'}\mu_H(a'\mid S)\exp(\widehat{Q}(S,a')/\beta)
}.
$$

Interpretation:

| Parameter       | Effect                                   |
| --------------- | ---------------------------------------- |
| large \(\beta\) | behave more like the human               |
| small \(\beta\) | behave more like the learned value model |
| \(\mu_H\)       | prevents early unstable robot behavior   |
| \(\widehat{Q}\) | lets the robot improve beyond imitation  |

The null action must be included in \(\mu_H\). Otherwise the robot is accidentally forced to pick something every time.

#### 9 - LP assignment planner

When multiple robots can act in one short window, the system chooses a compatible set of actions.

Define:

$$
y_{\{\tau,r,b\}(t)}\in[0,1].
$$

If \(y_{\tau,r,b,t}=1\), the planner dispatches action \(a(\tau,r,b)\).

The LP maximizes:

$$
\max_y
\sum_{\tau,r,b}
y_{\{\tau,r,b\}(t)}\widehat{g}_{\{\tau,r,b\}(t)}.
$$

**Subject to:**

##### One assignment per item

$$
\sum_{r,b}y_{\{\tau,r,b\}(t)}\le 1.
$$

##### Robot capacity

$$
\sum_{\tau,b}c^{\mathrm{cycle}}_{\{\tau,r\}(t)}y_{\{\tau,r,b\}(t)}\le C_{r(t)}.
$$

##### Reachability and full bins

Reachability and full-bin filtering are handled by removing illegal actions before the LP is solved.

There is no separate bin-quality constraint in the current MVP. We only keep empty/full bin state, while contamination costs remain inside the objective.

#### 10 - Human query rule

The robot should not ask the human constantly. Ask only when both conditions hold:

$$
\mathrm{Var}_{\mathrm{epi}}(\widehat{Q}(S_{r(t)},\cdot))>\eta
$$

and

$$
\Delta V^{\mathrm{human}}_{r(t)}>C^{\mathrm{human}}.
$$



1. the model is uncertain in a way that more data can fix;
2. the likely value of asking is larger than the cost of interrupting the human.

____

### 11 - Core value formula

The expected bin gain is:

$$
G_{\{\tau,b\}(t)}
=
\sum_{M\in\mathcal{M}}
P(M_\tau=M\mid z_{\tau(t)})
\left[
\mathbf{1}[b=B(M)]V_M^{\mathrm{item}}
-
C^{\mathrm{contamination}}_{M,b}
\right].
$$

The expected one-action net value is:

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

