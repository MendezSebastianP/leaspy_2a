# RiemanianManifoldModel

**Module:** `leaspy.models.riemanian_manifold`
**Inherits from:** [`TimeReparametrizedModel`](TimeReparametrizedModel.md)

The `RiemanianManifoldModel` adds a layer of geometric consistency to the time-reparametrized framework. While [`TimeReparametrizedModel`](TimeReparametrizedModel.md) handles *when* changes happen (timing), `RiemanianManifoldModel` handles *how* multiple variables evolve together (direction and shape).

It provides the mathematical "skeleton" that supports multivariate progression, ensuring that different biomarkers evolve in a coordinated way as a geodesic (shortest path) on a Riemannian manifold.

## Core Geometric Enforcements

This class is largely abstract. Its primary job is to enforce the rules of the manifold on its subclasses (like `LogisticModel` or `LinearModel`) and maintain geometric stability during estimation.

### 1. Forcing the Shape (`metric` & `model_with_sources`)
The class does not know what the final curve will look like. Instead, it uses abstract methods to force subclasses later in the workflow to provide the exact geometric equations:

*   **`metric` (Abstract)**: Subclasses *must* define the Riemannian metric tensor. The metric is the fundamental property that defines the intrinsic geometry of the manifold, which in turn dictates the shape of the curve (the geodesic).
*   **`model_with_sources` (Abstract)**: Subclasses *must* define the actual mathematical formula that computes the biomarker values given the time, the metric, the velocity, and the spatial shifts.
*   *Note*: `model_no_sources` is automatically derived by calling `model_with_sources` with zeroed-out spatial shifts.

### 2. Preserving Geometric Hypotheses (`_center_xi_realizations`)
In a mixed-effects model on a manifold, there is a risk of redundancy (non-identifiability). If the population velocity ($v_0$) changes, and the individual velocity variations ($\xi$) change in the exact opposite way, the resulting trajectories are identical.

To prevent the estimation from drifting and to **ensure the geometric hypotheses of the manifold**, the model intercepts the state during the MCMC-SAEM sufficient statistics computation:
1.  It calculates the mean of all individual $\xi$ realizations.
2.  It centers $\xi$ by subtracting this mean (forcing the average individual variation to be 0).
3.  It absorbs this mean into the population parameter `log_v0`.

This internal gauge-fixing stabilizes the inference and ensures that the orthonormal basis (which is built dynamically from $v_0$ to project spatial sources) remains geometrically valid and collinear to the true direction of progression.

## Key Variables Introduced

This class adds specific geometric variables to the model's computational graph (`get_variables_specs`):

*   **`v0` / `log_v0`**: The initial velocity vector (tangent vector at the reference time). `log_v0` is the unconstrained parameter estimated by the algorithm, while `v0 = exp(log_v0)` is the strictly positive physical velocity.
*   **`metric`**: The Riemannian metric at the current point, linked to the abstract `metric` method.
*   **`orthonormal_basis`**: (Only if `source_dimension > 0`). A basis constructed orthogonally to $v_0$. It is used to mix the independent spatial sources (inter-subject variability) correctly onto the manifold without altering the main direction of progression.

## Next Step

The [LogisticModel](LogisticModel.md) (or `LinearModel`) inherits from this class and implements the `metric` and `model_with_sources`, defining effectively *what* the manifold looks like (e.g., a sigmoid surface).
