# Logistic Model Walkthrough

This page traces exactly what happens when you run a Logistic Model in Leaspy.

## 1. Instantiation
```python
from leaspy import LogisticModel
model = LogisticModel(name="test")
```
When {py:class}`~leaspy.models.logistic.LogisticModel` is created, it calls `get_variables_specs()`. This is crucial: it effectively "registers" the model parameters in the system.
*   It tells Leaspy: "I have a parameter `g` (log-normal), `tau` (Gaussian), etc."
*   These specs are used later to tell the Sampler what to sample.

## 2. Calling `fit()`
```python
model.fit(dataset, "mcmc_saem", n_iter=100)
```
The {py:meth}`~leaspy.models.base.BaseModel.fit` method is inherited from {py:class}`~leaspy.models.base.BaseModel`. It doesn't contain the math! It is an **Orchestrator**:
1.  **Data Prep**: It calls `_get_dataset(data)` to ensure we have valid Tensors.
2.  **Algo Factory**: It calls `algorithm_factory` to create an instance of {py:class}`leaspy.algo.fit.mcmc_saem.MCMCSAEM`.
3.  **Delegation**: It calls `algo.run(model, dataset)`.

## 3. Inside the Algorithm (`MCMCSAEM.run`)
The control is now passed to the Algorithm. The Model is just a passive object that answers questions.

```{mermaid}
sequenceDiagram
    autonumber
    
    %% Actors and Styles
    actor User
    participant Model as LogisticModel
    participant Algo as MCMC-SAEM
    participant State as State Object
    
    %% Boxes for logical grouping
    box rgb(240, 248, 255) User Space
        participant User
        participant Model
    end
    
    box rgb(255, 248, 240) Engine Room
        participant Algo
        participant State
    end

    User->>Model: fit(dataset)
    Model->>Algo: run(model, dataset)
    
    Note over Algo, State: Initialization Phase
    Algo->>Model: put_individual_parameters(state, dataset)
    Model->>State: Initialize z_i (random or default)
    
    loop Every Iteration (k=1 to n_iter)
        rect rgb(230, 230, 250)
            Note over Algo, State: Step 1: Simulation (S-step)
            Algo->>State: Sample Latent Variables (Gibbs)
            State-->>Algo: Updated z_i values
        end

        rect rgb(255, 239, 213)
            Note over Algo, Model: Step 2: Maximization (M-step)
            Algo->>Model: compute_sufficient_statistics(state)
            Model-->>Algo: Returns stats (S_k)
            Algo->>Model: update_parameters(state, S_k)
            Model->>State: Updates global parameters theta
        end
    end
    
    Algo-->>Model: Returns final state
    Model-->>User: Control returned
```

### Step 3.1: Initialization
The algorithm asks the model to initialize the individual parameters ($z_i$) for the dataset.
*   **Where**: {py:meth}`~leaspy.models.mcmc_saem_compatible.McmcSaemCompatibleModel.put_individual_parameters`
*   **What**: Creates entries in the `State` for `tau_i`, `xi_i`, etc., usually initialized to 0 or random values.

### Step 3.2: The Loop (SAEM)
For `n_iter` times, the algorithm coordinates the dance:

1.  **Sampling (S-step)**:
    The algorithm looks at the `State`'s "Random Variables" and runs a Sampler (e.g., Gibbs/Metropolis-Hastings) to tweak them.
    *   *Note*: The Model code is *not* involved here! The sampler is generic.

2.  **Maximization (M-step)**:
    *   **Stat Computation**: `Algo` asks `Model`: *"Given these current individual parameters $z_i$, what are the sufficient statistics?"*
        *   The Logistic Model calculates these using its Riemannian metric logic.
    *   **Parameter Update**: `Algo` tells `Model`: *"Here are the new averaged stats. Update yourself."*
        *   The Model updates its `parameters` attribute (e.g., `model.parameters['g']` changes).

## 4. Result
When `fit` returns, the `model` object has been mutated in-place. Its parameters $\theta$ now reflect the converged values.
