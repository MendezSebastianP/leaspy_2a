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

## The Algorithm's Loop

This class is typically used with the **MCMC-SAEM** algorithm. The algorithm relies on the methods defined here to execute its optimization loop:

```python
for iteration in range(n_iter):
    # E-step: Simulate individual parameters
    individual_params = algorithm.simulate_individual_parameters(model, dataset)
    
    # M-step (Part 1): Compute Statistics
    sufficient_statistics = algorithm.compute_statistics(model, individual_params)
    
    # M-step (Part 2): Update Parameters
    model.update_parameters(sufficient_statistics)
    
    # Check stopping criteria
    if algorithm.has_converged():
        break
```

### Understanding "Statistics" in the M-Step

During the M-step, the algorithm computes **sufficient statistics** from the simulated individual parameters. These are summary values (like sums, means, variances) computed across all simulations.

For example, if the algorithm simulated 100 possible values for each patient's $\tau_i$ (reference time), `compute_sufficient_statistics` calculates the mean and variance of these simulations across the entire population. These statistics are then passed to `update_parameters` to adjust the population parameters (like `tau_mean` and `tau_std`).

