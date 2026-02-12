# TimeReparametrizedModel

**Module:** `leaspy.models.time_reparametrized`

This class introduces the concept of time reparametrization, which is central to Leaspy's ability to model disease progression. It handles patient-specific time shifts ($\tau$) and acceleration factors ($\xi$).

## Responsibilities

*   **Time Normalization**: It maps real individual time (age) to a normalized "pathological" time used by the model.
*   **Individual Variability**: It defines and manages the individual parameters responsible for temporal variability ($\tau$, $\xi$) and optionally spatial variability (`sources`).
    *   $\tau$ (Time shift): Accounts for disease onset timing (subjects starting earlier or later).
    *   $\xi$ (Log-acceleration): Accounts for disease progression speed (subjects progressing faster or slower).
    *   `sources` (Spatial components): Accounts for inter-subject variability in the *pattern* of progression (e.g. different symptoms worsening at different relative rates), if `source_dimension > 0`.

## Key Methods

*   `get_variables_specs()`: Defines the statistical nature of the parameters (e.g., priors, distributions). It introduces:
    *   `xi`: Individual log-acceleration (Normal distribution).
    *   `tau`: Individual time shift (Normal distribution).
*   `time_reparametrization(t, alpha, tau)`: The static method that implements the core math equation for time mapping: $t_{reparam} = \alpha_i \cdot (t_{ij} - \tau_i)$.
    *   `sources`: (If applicable) Independent component sources for spatial variability.

## Key Attributes

*   `dimension` (Inherited): The number of features (biomarkers) in the model. This determines the shape of the mixing matrix if sources are used.
*   `source_dimension`: Number of independent sources modeling the spatial variability (optional).

## Next Step

The [RiemanianManifoldModel](RiemanianManifoldModel.md) builds upon this time reparametrization by adding a geometric structure (Riemannian metric) to the space of observations, essential for consistent multivariate modeling.
