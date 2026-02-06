# McmcSaemCompatibleModel

**Module:** `leaspy.models.mcmc_saem_compatible`

This class extends `StatefulModel` to make it compatible with the **SAEM** (Stochastic Approximation Expectation-Maximization) algorithm using **MCMC** (Markov Chain Monte Carlo) sampling. It provides the mechanisms to update model parameters based on simulated states.

## Responsibilities

*   **Parameter Estimation**: It defines how parameters evolve during the optimization loop.
*   **Statistical Computation**: It calculates sufficient statistics required for the maximization step of SAEM.

## Key Methods

*   `update_parameters(cls, state, sufficient_statistics, burn_in)`: Updates the parameters categorized as `ModelParameter` in the DAG. This corresponds to the Maximization (M) step of SAEM.
*   `compute_sufficient_statistics(cls, state)`: Computes the sufficient statistics from the current `state`. It runs *before* parameter updates.
*   `get_variables_specs()`: Defines the specifications of variables used in the model, including timepoints and observation models.
