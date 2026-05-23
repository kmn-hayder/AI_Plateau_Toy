# AI Under Constraint: A Two-Model Framework for Scaling and Optimal Enterprise Investment

> **Status:** Structural toy model. Parameter values are illustrative, not predictive. The arguments below survive parameter drift; the snapshot does not.

---

## 0. Motivation

The dominant narrative around artificial intelligence assumes indefinite scaling. Larger models, more data, more compute, recursive self-improvement, eventually some form of open-ended acceleration. The framing is exponential.

But AI is not a purely algorithmic object. It is a capital-embedded production technology, dependent on energy, semiconductor fabrication, data center infrastructure, cooling, grid stability, and skilled labor. Once we treat AI as a materially constrained input, two questions follow:

1. **System level:** What does aggregate AI capability look like under binding physical and economic constraints?
2. **Firm level:** How much should a given enterprise invest in AI before incremental gains turn net negative?

The two models below answer these questions in sequence. Model 1 produces a capability ceiling $A^*$. Model 2 (static) and Model 2D (dynamic) take $A^*$ as input and return optimal firm-level investment. The dynamic version is structurally analogous to Card's optimal-schooling model from labor economics, extended to capital-stock accumulation under depreciation and time-to-build.

---

## 1. Model 1 — The Capability Plateau

### 1.1 Setup

Let $C(t)$ denote raw compute available at time $t$ and $E(t)$ denote algorithmic efficiency (capability extracted per unit of raw compute). Cumulative effective compute is:

$$X(t) = \int_0^t C(\tau) \cdot E(\tau) \, d\tau$$

Aggregate AI capability $A(X)$ is assumed to follow logistic dynamics in cumulative effective compute:

$$\frac{dA}{dX} = \kappa \cdot A \cdot \left(1 - \frac{A}{A^*}\right)$$

with closed-form solution:

$$A(X) = \frac{A^*}{1 + e^{-\kappa(X - X_0)}}$$

### 1.2 Why Logistic and Not Exponential

Exponential growth implies no upper bound. Logistic growth captures three empirically robust features of general-purpose technology diffusion:

1. **Early phase:** $\frac{dA}{dX} \approx \kappa A$ — near-exponential, dominated by architectural breakthroughs and capital inflows.
2. **Inflection:** rate of improvement peaks when $A \approx A^*/2$.
3. **Saturation:** $\frac{dA}{dX} \to 0$ as $A \to A^*$.

Steam engines, semiconductor density (Moore's law has been logistic, not exponential, when measured against the physical floor), network bandwidth, and energy conversion efficiency all follow this shape. AI improvement is embedded in mathematics, computation theory, and physics — none of which it has rewritten — so the same shape is the default expectation.

### 1.3 The Ceiling $A^*$ — Liebig's Law Applied

Let $\mathcal{L}$ denote the set of binding physical and economic constraints on AI capability. Each constraint $L_i$ has a finite magnitude and a conversion coefficient $\psi_i$ that maps constraint units (e.g., terawatt-hours, transformer units, tokens) to capability units.

The ceiling is set by the minimum over the constraint set:

$$A^* = \min_{i \in \mathcal{L}} \{\psi_i \cdot L_i\}$$

This is Liebig's law of the minimum, originally formulated for plant nutrients: growth is bounded by the scarcest required input, not the average availability.

**Structural claim:** $A^* < \infty$ if and only if at least one $L_i$ is finite. Since all known constraints are finite (energy, fabrication, data, capital, thermodynamic floors), $A^*$ is finite. The plateau is mathematically guaranteed.

The empirical question — which $L_i$ binds at a given moment — is separate from the structural claim that *some* $L_i$ binds.

### 1.4 Efficiency Gains and the Ceiling

A common objection: efficiency improvements (distillation, quantization, mixture-of-experts, algorithmic refinement) reduce compute per unit of capability, so the ceiling rises over time.

The framework accommodates this. Efficiency $E(t)$ enters cumulative effective compute $X(t)$, accelerating movement along the logistic curve. But this does not raise $A^*$ unless efficiency directly relaxes a binding $L_i$.

- **Movement along the curve:** quantization, distillation, better training recipes.
- **Movement of the ceiling:** breakthroughs that touch a binding constraint — e.g., new cooling architectures that lift $L_{\text{thermal}}$, new fabrication geometries that lift $L_{\text{fab}}$, novel energy generation that lifts $L_{\text{energy}}$.

Most algorithmic efficiency gains are the first kind. They get the system to the plateau faster. They do not move the plateau.

---

## 2. Model 2 — Static Optimal Enterprise Investment

### 2.1 Setup — Card-Analogous Structure

David Card's optimal-schooling model has an individual maximize log income net of schooling cost, with concave returns to schooling and convex marginal cost. The optimum:

$$S^* = \frac{\beta - r}{k + \gamma}$$

where $\beta$ is the return to schooling, $r$ the opportunity cost, $k$ the concavity of returns, $\gamma$ the convexity of cost.

The firm's AI investment problem has the same structure. Let $I$ denote AI expenditure. Firm net value:

$$V(I) = \pi_0 + \beta \cdot A^* \cdot I - \frac{k}{2} I^2 - r \cdot I - \frac{\gamma}{2} I^2$$

The four terms represent: baseline profit, productivity gain from AI (linear in firm absorption $\beta$ and capability ceiling $A^*$, linear in spend), diminishing returns ($k$), capital cost ($r$), and convex integration friction ($\gamma$).

### 2.2 First-Order Condition

$$\frac{dV}{dI} = \beta A^* - kI - r - \gamma I = 0$$

Solving:

$$\boxed{I^* = \frac{\beta A^* - r}{k + \gamma}}$$

Direct structural analog of Card's $S^*$.

### 2.3 Two Thresholds

| Threshold | Value | Interpretation |
|---|---|---|
| Marginal optimum | $I^* = (\beta A^* - r)/(k + \gamma)$ | Beyond this, each additional dollar destroys value |
| Total break-even | $I_{\max} = 2 I^*$ | Cumulative gains equal cumulative cost; beyond, firm is worse off than not adopting |

Spending in $(I^*, 2I^*)$ is suboptimal but still net positive. Spending beyond $2I^*$ makes the firm net worse off than the no-AI counterfactual.

### 2.4 Adoption Condition

$I^* > 0$ requires:

$$\beta A^* > r$$

If the firm's absorption capacity multiplied by the system-level capability ceiling does not exceed the cost of the AI dollar, the firm should not adopt.

---

## 3. Model 2D — Dynamic Version

### 3.1 What the Static Model Misses

The static model treats AI investment as a single shot. Three features the static version cannot capture:

1. **Capital stock dynamics:** AI is not consumed in one period. It accumulates and depreciates.
2. **Obsolescence:** New architectures make existing AI capital partially obsolete on a timescale shorter than the investment horizon.
3. **Time-to-build:** Investment commitments today do not deliver productive capacity today. There is a delay $\tau$ between order and delivery.

### 3.2 Setup

The firm maximizes the present value of net profit flow over an infinite horizon:

$$V = \int_0^\infty e^{-\rho t} \left[ \pi(K(t), A^*(t)) - c(I(t)) \right] dt$$

subject to capital accumulation with time-to-build delay:

$$\frac{dK}{dt} = I(t - \tau) - \delta \cdot K(t)$$

where:
- $K(t)$ is the firm's effective AI capital stock at time $t$
- $I(t)$ is gross investment at time $t$
- $\tau$ is the time-to-build delay
- $\delta$ is the depreciation / obsolescence rate
- $\rho$ is the firm's discount rate (WACC)
- $A^*(t)$ is the system-level capability ceiling (potentially time-varying)

Functional forms:

$$\pi(K, A^*) = \beta A^* K - \frac{k}{2} K^2 \qquad c(I) = r I + \frac{\gamma}{2} I^2$$

### 3.3 Hamiltonian and First-Order Conditions

Ignoring $\tau$ initially for clarity, the current-value Hamiltonian:

$$H = \beta A^* K - \frac{k}{2} K^2 - r I - \frac{\gamma}{2} I^2 + \lambda \cdot (I - \delta K)$$

**FOC on investment:**

$$\frac{\partial H}{\partial I} = -r - \gamma I + \lambda = 0 \quad \Rightarrow \quad I(t) = \frac{\lambda(t) - r}{\gamma}$$

**Co-state equation:**

$$\dot{\lambda} = \rho \lambda - \frac{\partial H}{\partial K} = (\rho + \delta) \lambda - \beta A^* + k K$$

### 3.4 Steady State

Set $\dot{K} = 0$ and $\dot{\lambda} = 0$:

- $\dot{K} = 0 \Rightarrow I^* = \delta K^*$
- $\dot{\lambda} = 0 \Rightarrow (\rho + \delta) \lambda^* = \beta A^* - k K^*$
- $\lambda^* = r + \gamma I^* = r + \gamma \delta K^*$

Substituting:

$$(\rho + \delta)(r + \gamma \delta K^*) = \beta A^* - k K^*$$

Solving for $K^*$:

$$\boxed{K^* = \frac{\beta A^* - (\rho + \delta) r}{k + \gamma \delta (\rho + \delta)}}$$

And:

$$\boxed{I^* = \delta K^* = \frac{\delta [\beta A^* - (\rho + \delta) r]}{k + \gamma \delta (\rho + \delta)}}$$

### 3.5 The User Cost of Capital

The term $(\rho + \delta) r$ is **Jorgenson's user cost of capital** — the dynamic generalization of the static "$r$" in Card's model. It says the relevant cost is not the price of the input alone, but the price multiplied by the rate at which capital must be replaced (depreciation) plus the rate at which future returns are discounted.

This is the central structural difference between the static and dynamic versions. Static: cost is $r$. Dynamic: effective cost is $(\rho + \delta) r$.

### 3.6 Incorporating Time-to-Build

Investment committed at time $t$ delivers productive capacity at time $t + \tau$. The present value of that delayed capacity is discounted by $e^{-\rho \tau}$. The modified FOC:

$$\lambda(t) \cdot e^{-\rho \tau} = r + \gamma I(t)$$

So:

$$I(t) = \frac{\lambda(t) e^{-\rho \tau} - r}{\gamma}$$

**Interpretation:** longer lead times reduce optimal investment today, because committed capital is more heavily discounted. This explains why firms with long lead times rationally preorder — the time-to-build wedge means that *not* committing early forces them further down the discount curve.

### 3.7 Dynamic Adoption Condition

$I^* > 0$ requires:

$$\beta A^* > (\rho + \delta) r$$

Stricter than the static condition. Adoption that looked rational in the static framework may fail under realistic depreciation and discounting.

---

## 4. Variable Dictionary

### 4.1 Model 1 Variables

| Symbol | Meaning | Units (illustrative) |
|---|---|---|
| $A(X)$ | Aggregate AI capability as a function of cumulative effective compute | Capability units |
| $A^*$ | Capability ceiling (plateau value) | Capability units |
| $X(t)$ | Cumulative effective compute at time $t$ | FLOP-equivalent |
| $C(t)$ | Raw compute available at time $t$ | FLOPs/second |
| $E(t)$ | Algorithmic efficiency at time $t$ | Capability per FLOP |
| $\kappa$ | Logistic growth rate parameter | 1/FLOP-equivalent |
| $X_0$ | Inflection point of the logistic curve | FLOP-equivalent |
| $\mathcal{L}$ | Set of binding physical/economic constraints | — |
| $L_i$ | Magnitude of the $i$-th constraint | Varies by constraint |
| $\psi_i$ | Conversion coefficient from $L_i$ to capability units | Capability per constraint unit |

### 4.2 Model 2 / 2D Variables

| Symbol | Meaning | Units (illustrative) |
|---|---|---|
| $I$ | Gross firm investment in AI | USD/year |
| $I^*$ | Optimal investment level | USD/year |
| $I_{\max}$ | Total break-even investment ($2 I^*$ in static) | USD/year |
| $K(t)$ | Firm AI capital stock at time $t$ | USD-equivalent |
| $K^*$ | Steady-state capital stock (dynamic) | USD-equivalent |
| $V$ | Firm net present value | USD |
| $\pi_0$ | Baseline profit without AI | USD/year |
| $\pi(K, A^*)$ | Firm profit flow as function of capital and ceiling | USD/year |
| $c(I)$ | Investment cost function | USD/year |
| $\beta$ | Firm absorption coefficient (capability → productivity) | Productivity per capability unit |
| $r$ | Effective per-unit price of AI input | USD per unit |
| $k$ | Concavity of productivity returns | USD per (capital)² |
| $\gamma$ | Convexity of integration / adjustment cost | USD per (investment)² |
| $\rho$ | Firm discount rate (WACC) | 1/year |
| $\delta$ | Depreciation / obsolescence rate of AI capital | 1/year |
| $\tau$ | Time-to-build delay | Years |
| $\lambda$ | Co-state variable (shadow price of capital) | USD per unit capital |
| $A^*(t)$ | Time-varying capability ceiling (from Model 1) | Capability units |

### 4.3 Derived Quantities

| Expression | Meaning |
|---|---|
| $(\rho + \delta) r$ | Jorgenson user cost of capital |
| $e^{-\rho \tau}$ | Time-to-build discount wedge |
| $\beta A^* > r$ | Static adoption condition |
| $\beta A^* > (\rho + \delta) r$ | Dynamic adoption condition |
| $k + \gamma \delta (\rho + \delta)$ | Effective curvature denominator (dynamic) |

---

## 5. Illustrative 2026 Snapshot

> **Caveat:** These values are point-in-time observations meant for illustration. They are not load-bearing for the argument and should not be carried forward as if structural. The set $\mathcal{L}$ itself is empirically determined and rotates over time. Treat the snapshot as a configuration file, not as part of the model.

### 5.1 Plausible Binding Constraints, 2026

| Constraint $L_i$ | 2026 observation | Source class |
|---|---|---|
| $L_{\text{energy}}$ | Capacity auctions clearing at regulatory caps; grid operators reporting shortfalls against reliability requirements | Wholesale electricity markets |
| $L_{\text{grid\_eq}}$ | Power transformer lead times in the 2–3 year range; switchgear and HV breakers similar | Procurement data |
| $L_{\text{thermal}}$ | Per-rack densities at the air-cooling thermodynamic limit; thermal runaway windows in seconds | Manufacturer specs |
| $L_{\text{fab}}$ | Leading-edge fab capacity concentrated, with multi-year build times for new facilities | Semiconductor industry data |
| $L_{\text{data}}$ | High-quality public training data increasingly exhausted for largest models | Research literature |
| $L_{\text{capital}}$ | Hyperscaler capex consuming significant fractions of revenue; bond issuance rising | Hyperscaler 10-Ks and analyst notes |

Which of these binds first is empirical and changes over time. As of 2026, $L_{\text{grid\_eq}}$ and $L_{\text{energy}}$ appear to bind ahead of $L_{\text{fab}}$ for many deployments.

### 5.2 Plausible Firm-Level Parameter Ranges, 2026

| Parameter | Indicative range | Notes |
|---|---|---|
| $\tau$ | 1–4 years | Procurement of grid equipment plus integration |
| $\delta$ | 0.25–0.50 / year | Implied by hardware refresh cycles and architecture transitions |
| $\rho$ | 0.08–0.15 / year | WACC range for large tech and enterprise adopters |
| $r$ | Tracks colocation pricing, electricity costs, and amortized hardware | Firm-specific |
| $\beta$ | Highly heterogeneous | Depends on workflow fit, talent, data quality |
| $k, \gamma$ | Difficult to estimate directly | Inferred from observed investment paths |

### 5.3 What These Numbers Imply (Today, Not Generally)

Plugging illustrative midpoints — $\rho = 0.10$, $\delta = 0.35$, $\tau = 3$ — yields:

- User cost multiplier $(\rho + \delta) = 0.45$, meaning the dynamic adoption hurdle is roughly 4–5x the static one for a given $r$.
- Time-to-build discount $e^{-\rho \tau} \approx e^{-0.30} \approx 0.74$, meaning investment dollars committed today are worth only ~74% of their face value in deployed-capacity terms.

If $\delta$ falls (architecture stabilizes), $\tau$ falls (supply chain catches up), and $\rho$ falls (capital cheapens), all three terms loosen and $I^*$ rises. None of this changes the structural shape of the model.

---

## 6. What Survives Parameter Drift

The empirical snapshot in §5 will look dated within 2–3 years. The following structural claims do not depend on it:

1. **A plateau exists.** The minimum over a finite set of finite constraints is finite. The argument requires no commitment to which $L_i$ binds.

2. **Optimal firm investment is bounded.** $I^*$ is finite for finite $A^*$. Unbounded linear scaling is mathematically excluded at the firm level, not just the system level.

3. **Time-to-build creates a structural wedge.** Any $\tau > 0$ means headline capital commitments overstate productive capacity. The $e^{-\rho \tau}$ discount is structural.

4. **Efficiency gains raise $A^*$ but do not eliminate it.** A higher ceiling shifts $I^*$ up proportionally — consistent with Jevons-style total demand growth alongside per-unit cost declines. Not a paradox in the model.

5. **The user cost of capital, not the price of capital, governs adoption.** The dynamic adoption condition $\beta A^* > (\rho + \delta) r$ is the correct one, and the static condition $\beta A^* > r$ understates the hurdle.

6. **The interesting research question is where the plateau sits, not whether it exists.** That question depends on $\min(L_i)$, which is empirical and changes as constraints relax.

---

## 7. Limitations

This is a structural toy model. It is not a forecasting tool. Honest limitations:

- **Functional forms are stylized.** Quadratic returns and costs are tractable but specific; CES, log-log, or other forms preserve the qualitative result but change closed forms.
- **$\beta$ is treated as constant.** In reality it likely grows with organizational learning, making $I^*$ dynamic over time even in the static framework.
- **$A^*$ is treated as exogenous to any single firm.** True at the firm level. False at the aggregate level, where firm investment shapes which $L_i$ binds.
- **No general equilibrium.** Heterogeneous firms with different $\beta$ values and competing for the same constrained inputs would produce richer dynamics — and potentially misallocation through arms-race effects.
- **Constraints are treated as deterministic.** Stochastic shocks (regulatory, geopolitical, architectural) would generate jump processes on $A^*$ and $\delta$.
- **Learning-by-doing on $\beta$ is absent.** A fully realistic model would feed $\beta$ back through firm output.

A predictive model addressing these would be a dissertation. The purpose of this framework is to structure an argument, not to forecast point estimates.

---

## 8. Summary

The plateau is mathematically guaranteed. The optimal firm-level investment is bounded by the system-level ceiling. The dynamic user cost of capital, $(\rho + \delta) r$, is the right adoption hurdle — not the static price $r$. Time-to-build creates a structural wedge between committed and deployed capital that explains observed preordering behavior without invoking irrationality.

Where the plateau sits, how high $I^*$ rises, and which constraint binds first are empirical questions whose answers will change. The structural shape of the model will not.
