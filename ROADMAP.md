# KotlinRL Algorithm Roadmap (August 2025)

This document outlines the current status and next priorities for algorithm development in KotlinRL, covering classical RL, planning, policy gradient, and function approximation tracks.

---

## ✅ Reprioritized TODOs (2025-08)

### 🔝 Tier 1 — Core Infrastructure Enablers

> These are blockers for multiple high-value algorithms

#### 🟢 1. Feature Extractors: `TileCoder`, `StateAggregator`
- Enables generalization and approximators
- Required for:
  - Linear V/Q functions
  - Policy gradient
  - Actor-Critic
- Use simple `FeatureExtractor<State>` / `SAFeatureExtractor<State, Action>`

➡️ **Must complete before anything involving function approximation or policy gradient**

---

### 🔝 Tier 2 — Easy Wins with High Value

#### 🟢 2. Prioritized Sweeping
- Big performance boost over random Dyna-Q
- Small incremental addition: add `predecessors` and a `PriorityQueue`
- Reuses all existing DynaQ and TD logic

➡️ High ROI without needing FA or policy gradient

#### 🟢 3. Dyna-Q+
- Add optimistic exploration bonuses in planning via `τ(s, a)`
- Small addition to your model interface and planning loop
- Very effective in sparse reward or deceptive reward tasks

➡️ Easy to implement, high impact

---

### 🔝 Tier 3 — First Policy Gradient Algorithms

> After feature extractors + softmax policy are in

#### 🟡 4. REINFORCE with Baseline
- First policy gradient method (episodic)
- Uses your value function predictor as a baseline
- Simple: fit baseline → compute Gt → apply gradient ascent
- Cleanly separates actor and critic

➡️ Great educational and functional stepping stone

---

### 🔝 Tier 4 — Deepening Online Control Methods

#### 🟡 5. One-step Actor-Critic (episodic)
- Uses:
  - TD(0) prediction for the critic
  - Policy gradient for the actor
- Requires: `gradLogPi(s, a)`, linear baseline

➡️ Connects TD learning and policy gradient; foundation for λ-trace and continuing versions

---

### 🔝 Tier 5 — High-Completeness Additions

#### 🟡 6. Tree Backup (n-step Expected SARSA generalization)
- Completes Sutton & Barto Chapter 7
- Naturally plugs into your `NStepTDQError`
- Not high priority unless targeting completeness

#### 🔵 7. Actor-Critic with Eligibility Traces
- λ-trace version of Actor-Critic
- Builds on One-Step Actor-Critic
- Uses Dutch traces for actor, standard TD(λ) for critic
- Required for best performance in noisy, long-term credit tasks

---

### 🔝 Tier 6 — Function Approximation Core

> Once extractors are in, implement approximators

#### 🔵 8. LinearValueFunction and LinearQFunction
- Parameterized approximators
- Power everything from True Online TD(λ) to policy gradient

#### 🔵 9. True Online TD(λ)
- Exact online λ-return
- Use Multik / KotlinDL or dot products
- Needed for accurate prediction under approximation

---

## 🧠 Summary

| Tier | Task                            | Priority | Notes |
|------|----------------------------------|----------|-------|
| 1    | Feature Extractors               | 🟢 HIGH   | Enables everything else |
| 2    | Prioritized Sweeping             | 🟢 HIGH   | High reward, low cost |
| 2    | Dyna-Q+                          | 🟢 HIGH   | Easy and effective |
| 3    | REINFORCE with Baseline          | 🟡 MEDIUM | First PG method |
| 4    | One-Step Actor-Critic            | 🟡 MEDIUM | Online PG + TD |
| 5    | Tree Backup                      | 🟡 MEDIUM | Completeness |
| 5    | Actor-Critic with Eligibility    | 🔵 LOW    | After One-Step AC |
| 6    | Linear Value/Q Function          | 🔵 LOW    | Needed for generalization |
| 6    | True Online TD(λ)                | 🔵 LOW    | High accuracy in FA |
