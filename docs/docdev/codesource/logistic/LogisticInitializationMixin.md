# LogisticInitializationMixin

**Module:** `leaspy.models.logistic`

This mixin class handles the specific initialization logic for the Logistic model parameters. Initialization strategies vary by model, and this class encapsulates the logic for the logistic curve.

## Responsibilities

*   **Initialization**: It computes stable initial values for model parameters (`g`, `v0`, etc.) based on the input data, often using heuristics or simplified estimations to ensure the main optimization starts from a reasonable point.

## Key Methods

*   `_compute_initial_values_for_model_parameters(self, dataset)`: (Protected) Calculates initial values for the model parameters using the provided `dataset`. It estimates parameters like `log_g_mean`, `log_v0_mean`, and `tau_mean` based on patient slopes and values distribution.

