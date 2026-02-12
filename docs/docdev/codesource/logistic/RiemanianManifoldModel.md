# RiemanianManifoldModel

**Module:** `leaspy.models.riemanian_manifold`
**Inherits from:** [`TimeReparametrizedModel`](TimeReparametrizedModel.md)

The `RiemanianManifoldModel` adds a layer of geometric consistency to the time-reparametrized framework. While [`TimeReparametrizedModel`](TimeReparametrizedModel.md) handles *when* changes happen (timing), `RiemanianManifoldModel` handles *how* multiple variables evolve together (direction and shape).

It provides the mathematical "skeleton" that supports multivariate progression, ensuring that different biomarkers evolve in a coordinated way rather than as independent curves.

## The Geometric Perspective

In Leaspy, we view disease progression as a geodesic (shortest path) on a Riemannian manifold. This class implements the machinery for this view:

1.  **Metric**: Defines the "shape" of the manifold.
2.  **Velocity ($v_0$)**: Defines the *direction* of the disease trajectory at the reference time ($t_0$).
3.  **Parallel Transport**: Combines the time reparametrization with the velocity to generate the trajectory.

### Spatial Variability (Sources)

If the model is initialized with `source_dimension > 0`, this class acts as the mixing layer. It introduces **independent sources** (statistical components) to model "spatial" variability—differences in *how* the disease manifests across features (e.g., memory declines faster than language for Subject A, but vice versa for Subject B).

## Key Responsibilities

*   **Identifiability**: It implements a specialized trick in `compute_sufficient_statistics()` that centers the realizations of $\xi$ (acceleration) and absorbs the mean into the population parameter `v0`. This ensures the model parameters remain identifiable during estimation.
*   **Orthonormal Basis Construction**: It computes a dynamic basis based on $v_0$ to project the spatial sources correctly onto the manifold.
*   **Abstract Shape Definition**: It declares `model_with_sources` and `model_no_sources` as abstract methods. The actual shape (Linear, Logistic, etc.) is left to the subclass.

## Key Attributes

*   `dimension`: Number of features (biomarkers).
*   `source_dimension`: Number of spatial sources (latent variables for inter-subject variability pattern).

## Key Parameters & Variables

This class adds specific variables to the model specifications map (`get_variables_specs`):

*   **`v0` / `log_v0`**: The initial velocity vector (tangent vector at $t_0$). `log_v0` is the actual estimation parameter (unconstrained), while `v0 = exp(log_v0)` is the physical velocity.
*   **`metric`**: The Riemannian metric at the current point.
*   **`orthonormal_basis`**: (If sources exist) Constructed from $v_0$ to mix the sources orthogonally to the main progression direction.

## Next Step

The [LogisticModel](LogisticModel.md) (or `LinearModel`) inherits from this class and provides the concrete mathematical formula (the "shape") for `model_with_sources`, defining effectively *what* the manifold looks like (e.g., a sigmoid surface).
