# McmcSaemCompatibleModel

**Module:** `leaspy.models.mcmc_saem_compatible`
**Inherits from:** [`StatefulModel`](StatefulModel.md)

`document: leaspy.models.mcmc_saem_compatible.McmcSaemCompatibleModel`

[`StatefulModel`](StatefulModel.md) gives a model its internal variable representation (the `State` and [DAG](DAG.md)). But that alone does not tell the algorithm **how to use** those variables during optimization.

`McmcSaemCompatibleModel` bridges this gap. It defines the **contract** that the MCMC-SAEM algorithm expects: how to compute statistics, how to update parameters, and how to inject data and individual parameters into the state.

## What does the MCMC-SAEM algorithm need from a model?

The MCMC-SAEM algorithm alternates between two steps at each iteration:

1. **E-step (Expectation via MCMC)**: Sample individual latent variables ($\tau_i$, $\xi_i$, …) from their posterior, given the current population parameters and the observed data.
2. **M-step (Maximization via Stochastic Approximation)**: Update population parameters using summary statistics accumulated from those samples.

The algorithm itself is **generic**—it does not know what the model's variables are or what the math looks like. It only calls the methods that `McmcSaemCompatibleModel` guarantees will exist.

## The Three Core Methods

### 1. `compute_sufficient_statistics(state) → SuffStatsRW`

Called during the **M-step** (part 1). The algorithm has just finished sampling and the `state` holds fresh values for all individual latent variables. This method computes **sufficient statistics**: aggregate summaries over the population that are needed to update each parameter.

What are sufficient statistics? They are the *minimal set of numbers* that capture everything the data can tell us about a parameter. For instance, to update `tau_mean`, we need $\sum_i \tau_i$ and $n$—we do not need every individual $\tau_i$ separately.

In practice, each `ModelParameter` in the DAG knows how to compute its own sufficient statistics via a `suff_stats(state)` method. `compute_sufficient_statistics` iterates over all `ModelParameter` nodes and collects them:

```python
@classmethod
def compute_sufficient_statistics(cls, state: State) -> SuffStatsRW:
    suff_stats = {}
    for mp_var in state.dag.sorted_variables_by_type[ModelParameter].values():
        suff_stats.update(mp_var.suff_stats(state))
    # Also collect convergence metrics (nll_attach, nll_regul, ...)
    suff_stats["nll_attach"] = state["nll_attach"]
    suff_stats["nll_regul_ind_sum"] = state["nll_regul_ind_sum"]
    suff_stats["nll_tot"] = suff_stats["nll_attach"] + suff_stats["nll_regul_ind_sum"]
    return suff_stats
```

### 2. `update_parameters(state, sufficient_statistics, burn_in) → None`

Called during the **M-step** (part 2). Given the sufficient statistics, update every `ModelParameter` in the state.

A key design detail: all new values are **computed first**, then **applied in batch**. This avoids order-dependent bugs where updating parameter A changes a derived quantity that parameter B's update rule depends on.

```python
@classmethod
def update_parameters(cls, state, sufficient_statistics, *, burn_in):
    # Compute all updates first (no writes yet)
    params_updates = {}
    for mp_name, mp_var in state.dag.sorted_variables_by_type[ModelParameter].items():
        params_updates[mp_name] = mp_var.compute_update(
            state=state, suff_stats=sufficient_statistics, burn_in=burn_in
        )
    # Then apply them all at once
    for mp, val in params_updates.items():
        state[mp] = val
```

The `burn_in` flag tells the update rule whether we are still in the warm-up phase (where running averages are reset more aggressively) or in the convergence phase.

### 3. `put_individual_parameters(state, dataset)` *(abstract)*

Called during the **E-step**. The algorithm needs to inject the individual latent variables into the state so that derived quantities (like `rt`, `model`, `nll_attach`) can be computed. Each concrete model implements this differently depending on which individual variables it has.

## Additional Responsibilities

Beyond the three core methods, `McmcSaemCompatibleModel` handles several concerns that `StatefulModel` does not:

### Observation Models

The constructor accepts one or more `ObservationModel` objects, which define:

- The **noise distribution** (e.g., Gaussian with learned `noise_std`).
- The **likelihood function** that produces `nll_attach` (negative log-likelihood of the data given the model prediction).

These observation models contribute their own variables to the DAG (via `obs_model.get_variables_specs()`), which is why observation-related variables like `y`, `nll_attach_ind`, and `nll_attach` appear in the [logistic model's DAG](DAG.md).

### Hyperparameter Loading

`McmcSaemCompatibleModel` defines the abstract method `_load_hyperparameters(hyperparameters)`, called during `__init__`. Each concrete model overrides this to validate and store hyperparameters like `source_dimension` or `noise_model`. This is a **private hook** (prefixed with `_`), not part of the public API.

### Base Variable Specs

`McmcSaemCompatibleModel.get_variables_specs()` starts the accumulation chain that subclasses extend. It contributes the **root-level** variables that every SAEM-compatible model needs:

- `t`: a `DataVariable` for timepoints.
- Variables from each observation model (data variable for observations, `nll_attach`, etc.).

Subclasses like `TimeReparametrizedModel` then call `super().get_variables_specs()` and add their own variables on top (see [DAG – How VariablesDAG is built](DAG.md)).

### Individual Trajectory Computation

`compute_individual_trajectory(timepoints, individual_parameters)` lets you predict a single patient's trajectory **after** the model is fitted. It works by:

1. Cloning the current state (to avoid mutating the model's internal state).
2. Injecting the patient's timepoints and individual parameters.
3. Reading `state["model"]`—the DAG automatically propagates everything.

This is the method that powers `model.personalize(...)` under the hood.

## The Algorithm Loop (Putting It All Together)

Here is a simplified but faithful picture of how the MCMC-SAEM algorithm interacts with `McmcSaemCompatibleModel` at each iteration:

```python
for iteration in range(n_iter):
    # ── E-step: sample individual latent variables ──
    # The algorithm proposes new values for tau_i, xi_i, sources_i, ...
    # using Gibbs sampling (via the State's fork mechanism).
    sampler.sample(state, dataset)

    # ── M-step part 1: compute sufficient statistics ──
    suff_stats = model.compute_sufficient_statistics(state)

    # ── M-step part 2: update population parameters ──
    model.update_parameters(state, suff_stats, burn_in=(iteration < burn_in_threshold))
```

The `sampler.sample(...)` step internally uses the State's **forking** capability (described in [StatefulModel](StatefulModel.md)): it proposes a change in a temporary fork, evaluates the acceptance ratio, and either keeps or discards the proposal.

## What This Class Does NOT Do

`McmcSaemCompatibleModel` provides the **algorithm-facing contract**. It does not:

- Define which specific latent variables exist (that is the job of concrete models like `TimeReparametrizedModel`).
- Implement the mathematical model function (that is the job of `RiemanianManifoldModel` / `LogisticModel`).
- Run the algorithm itself (that is the job of `leaspy.algo`).

It is the **bridge** between the generic algorithm and the specific model.

