# LogisticModel

**Module:** `leaspy.models.logistic`
**Inherits from:** [`RiemanianManifoldModel`](RiemanianManifoldModel.md), [`LogisticInitializationMixin`](LogisticInitializationMixin.md)

The `LogisticModel` is the concrete implementation that defines the **shape** of disease progression as a logistic sigmoid curve. Building on the geometric framework of [`RiemanianManifoldModel`](RiemanianManifoldModel.md), it specifies the actual equation that transforms the reparametrized time into biomarker values.

This is the most commonly used shape in Leaspy, suitable for biomarkers that follow an S-shaped trajectory from normality (0) to pathology (1).

## The Mathematical Shape

The core contribution of this class is defining the function $S(t)$ and the metric $g(p)$.

1.  **The Logistic Function**:
    The model assumes that after time reparametrization and spatial mixing, the value of the $k$-th feature follows:
    $$ y_k(t) = \frac{1}{1 + g_k \exp(-v_{0,k} \cdot \tilde{t}_k)} $$
    where $\tilde{t}_k$ includes time shifts and spatial source effects.

2.  **The Metric**:
    To be consistent with this shape, the Riemannian metric is defined as:
    $$ G_k(p) = \frac{(1+p_k)^2}{p_k} $$
    This metric ensures that the logistic curves are indeed geodesics (shortest paths) on the manifold equipped with this geometry.

## Responsibilities

*   **Equation Definition**: Implements `model_with_sources` using the sigmoid formula.
*   **Variable Specification**: Adds the specific parameter `g` (related to the inflection point valid range) to the model's variable dictionary.

## Key Attributes & Parameters

*   `g` / `log_g`: A population parameter specific to the logistic shape.
*   `v0` / `log_v0`: Inherited from `RiemanianManifoldModel`, representing the initial velocity.

## Key Methods

*   `model_with_sources(...)`: The engine room. It combines:
    1.  **Time Reparametrization** (from `TimeReparametrizedModel`): Getting $t_{reparam}$.
    2.  **Spatial Mixing** (from `RiemanianManifoldModel`): Mixing sources with direction $v_0$.
    3.  **Logistic Transform** (This class): Applying the sigmoid function.

## Initialization Logic (Mixin)

Because non-linear models are sensitive to starting values, this class inherits initialization logic from `LogisticInitializationMixin`.
This separation keeps the *model definition* clean from the *heuristic estimation* code.

> See [`LogisticInitializationMixin`](LogisticInitializationMixin.md) for details on how we estimate initial `g`, `v0`, and `tau` from raw data before the main algorithm runs.

## Next Steps

This concludes the logistic model definition hierarchy.
*   To understand how this model connects to noisy data, you might look at [Observation Models](ObservationModel.md).
*   To see how parameters are estimated, look at the **Algorithms** (e.g., `McmcSaem`).
