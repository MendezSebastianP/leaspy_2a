# TimeReparametrizedModel

**Module:** `leaspy.models.time_reparametrized`

This class introduces the concept of time reparametrization, which is central to Leaspy's ability to model disease progression. It handles patient-specific time shifts ($\tau$) and acceleration factors ($\xi$).

## Responsibilities

*   **Time Normalization**: It maps real individual time (age) to a normalized "pathological" time used by the model.
*   **Individual Variability**: It defines and manages the individual parameters responsible for temporal variability:
    *   $\tau$ (Time shift): Accounts for disease onset timing (subjects starting earlier or later).
    *   $\xi$ (Log-acceleration): Accounts for disease progression speed (subjects progressing faster or slower).

## Key Methods

*   `get_variables_specs()`: Defines the statistical nature of the parameters (e.g., priors, distributions). It introduces:
    *   `xi`: Individual log-acceleration (Normal distribution).
    *   `tau`: Individual time shift (Normal distribution).
    *   `sources`: (If applicable) Independent component sources for spatial variability.

## Key Attributes

*   `source_dimension`: Number of independent sources modeling the spatial variability (optional).
