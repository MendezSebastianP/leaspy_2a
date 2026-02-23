# LogisticModel

**Module:** `leaspy.models.logistic`
**Inherits from:** [`RiemanianManifoldModel`](RiemanianManifoldModel.md), [`LogisticInitializationMixin`](LogisticInitializationMixin.md)

The `LogisticModel` is the concrete implementation that defines the **shape** of disease progression as a logistic sigmoid curve. Building on the geometric framework of [`RiemanianManifoldModel`](RiemanianManifoldModel.md), it specifies the actual equation that transforms the reparametrized time into biomarker values.

This is the most commonly used shape in Leaspy, suitable for biomarkers that follow an S-shaped trajectory from normality (0) to pathology (1).

## The Mathematical Shape

The core contribution of this class is defining the function $S(t)$ and the metric $g(p)$.

1.  **The Logistic Function**:
    
    The model assumes that after time reparametrization and spatial mixing, the value of the $k$-th feature follows a logistic sigmoid curve. In the code (`model_with_sources`), this is implemented using **`torch.sigmoid`**:

    $$
    y_k(t) = \text{sigmoid}(\text{logit}_k(t))
    $$

    The "logit" is calculated using the Riemannian metric $G_k$ and the parameter $g_k$:

    $$ \text{logit}_k(t) = G_k \cdot (v_{0,k} \cdot \tilde{t}_k + \delta_k) - \ln(g_k) $$

2.  **The Metric ($G$) vs Parameter ($g$)**:

    Be careful to distinguish between the parameter $g$ (which controls the position) and the Riemannian metric $G$ (which scales the speed). The model explicitly defines their relationship:

    $$ 
    G_k = \frac{(1+g_k)^2}{g_k} 
    $$

    In the code:
    *   `metric` corresponds to the **Riemannian metric** $G_k$.
    *   `g` corresponds to the **shape parameter** $g_k$, which determines the value of the logistic curve at $t=0$ (before time-shifts are applied).

## Responsibilities

*   **Equation Definition**: Implements `model_with_sources` using the sigmoid formula.
*   **Variable Specification**: Adds the specific parameter `g` (related to the inflection point valid range) to the model's variable dictionary.

## Key Attributes & Parameters

*   `g` / `log_g`: A population parameter specific to the logistic shape.
*   `v0` / `log_v0`: Inherited from `RiemanianManifoldModel`, representing the initial velocity.

## Key Methods

*   `model_with_sources(...)`: The engine room. It computes, for each patient $i$, timepoint $t$, and feature $k$:

    1.  **Logit (`w_model_logit`)**: The logit is the **inverse of the sigmoid** — an unconstrained real number that gets squashed to $(0,1)$ at the last step. Working in logit space is necessary because the model needs to *add* contributions linearly (population time, individual space shifts, inflection offset), which is only valid before the nonlinear sigmoid is applied. The full expression is:

        $$\text{logit}_{i,t,k} = \underbrace{G_k \cdot v_{0,k} \cdot \tilde{t}_{i,t}}_{\text{population time}} + \underbrace{G_k \cdot \delta_{i,k}}_{\text{patient space shift}} - \underbrace{\ln(g_k)}_{\text{inflection offset}}$$

        In code: `metric[pop_s] * (v0[pop_s] * rt + space_shifts[:, None, ...]) - log(g[pop_s])`

        > **Why $G_k$ appears here**: $G_k = (1+g_k)^2/g_k$ is a *population-level* geometric normalizer — identical for all patients. It guarantees that any individual time reparametrisation ($\tau_i$, $\xi_i$) or space shift ($\delta_{i,k}$) in logit space produces another valid logistic curve of the exact same shape. Without it, shifting the logit would distort the S-curve rather than purely translate it.

        The `w_` prefix in `w_model_logit` signals it is a `WeightedTensor` — it carries an observation mask alongside the values to handle missing data. The mask is extracted via `WeightedTensor.get_filled_value_and_weight` before passing to `torch.sigmoid` (which cannot process NaNs directly).

    2.  **Sigmoid Activation**: `torch.sigmoid(model_logit)` maps the logit to $(0, 1)$, producing the final biomarker estimate. The observation mask is then re-applied via `WeightedTensor(...).weighted_value` so that missing observations remain masked throughout the NLL computation.

## Initialization Logic (Mixin)

Because non-linear models are sensitive to starting values, this class inherits initialization logic from `LogisticInitializationMixin`.
This separation keeps the *model definition* clean from the *heuristic estimation* code.

> See [`LogisticInitializationMixin`](LogisticInitializationMixin.md) for details on how we estimate initial `g`, `v0`, and `tau` from raw data before the main algorithm runs.

## Next Steps

This concludes the logistic model definition hierarchy.
*   To understand how this model connects to noisy data, you might look at [Observation Models](ObservationModel.md).
*   To see how parameters are estimated, look at the **Algorithms** (e.g., `McmcSaem`).
