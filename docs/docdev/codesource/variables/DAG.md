# Model Variables & DAG

**Module:** `leaspy.variables.dag`

The DAG (Directed Acyclic Graph) is a fundamental structure in Leaspy. It organizes all the variables involved in the model (parameters, latent variables, random variables) into a hierarchy.

## Responsibilities

*   **Categorization**: It groups variables by their type (e.g., `ModelParameter` for fixed population parameters, `IndividualLatentVariable` for subject-specific random effects).
*   **Dependency Management**: It ensures that variables are computed in the correct order (e.g., population parameters -> individual variables -> derived variables).
*   **Specification Storage**: It holds the metadata (`specs`) for each variable, such as its distribution family, priors, and shape.

## Key Concepts

*   **Categories**:
    *   `ModelParameter`: Fixed parameters of the model (e.g., `v0_mean`, `xi_std`).
    *   `IndividualLatentVariable`: Random variables specific to each subject (e.g., `xi`, `tau`).
    *   `PopulationLatentVariable`: Random variables for the population (e.g., `log_v0`).
*   **Specs**: Define general attributes for each category.
