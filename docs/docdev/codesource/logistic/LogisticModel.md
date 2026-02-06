# LogisticModel

**Module:** `leaspy.models.logistic`

The `LogisticModel` is a concrete implementation of the Riemannian Manifold model, specifically designed to model disease progression using logistic curves. It inherits from `RiemanianManifoldModel` and `LogisticInitializationMixin`.

## Responsibilities

*   **Equation Definition**: It defines the specific mathematical form of the progression (logistic sigmoid).
*   **Variable Specification**: It adds the specific parameters `g` (related to the inflection point) to the variable specifications.

## Key Methods

*   `model_with_sources(cls, rt, space_shifts, metric, v0, g)`: This is the core function computing the model values. It combines all previous components:
    *   `rt`: Reparametrized time (from `TimeReparametrizedModel`).
    *   `space_shifts`: Shifts due to sources (from `TimeReparametrizedModel` / `RiemanianManifoldModel`).
    *   `v0`: Initial velocity.
    *   `g`: Parameter controlling the position of the logistic curve.
    *   `metric`: Geometric metric.

    The formula corresponds to a multidimensional logistic function, transforming the reparametrized time and sources into values between 0 and 1.

*   `metric(g)`: Defines the Riemannian metric specific to the logistic model.
*   `get_variables_specs()`: Adds `log_g` (and derived `g`) to the model variables.
