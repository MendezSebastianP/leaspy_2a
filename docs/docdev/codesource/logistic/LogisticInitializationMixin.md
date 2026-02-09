# LogisticInitializationMixin

**Module:** `leaspy.models.logistic`

This mixin class handles the specific initialization logic for the Logistic model parameters. Initialization strategies vary by model, and this class encapsulates the logic for the logistic curve.

## Responsibilities

*   **Initialization**: It computes stable initial values for model parameters based on the input data. This ensures the main optimization algorithm (SAEM) starts from a reasonable point rather than random noise.

It specifically handles:
*   `tau_mean` (Average Onset): Estimated from the mean time of observations.
*   `xi_std` (Progression Variavility): Initialized to a default value (e.g. 0.5) or estimated variance.
*   `g` & `v0` (Logistic Shape): Estimated by fitting simplistic regressions to patient slopes and values.

## Key Methods

*   `_compute_initial_values_for_model_parameters(self, dataset)`: (Protected) Calculates initial values for the model parameters using the provided `dataset`.

    **How it works:**
    1.  Computes simple statistics on the dataset (mean values, mean slopes per patient).
    2.  Estimates `log_v0_mean` based on the slopes.
    3.  Estimates `log_g_mean` based on value ranges.
    4.  Returns a dictionary of tensors ready to be loaded into the model state.

