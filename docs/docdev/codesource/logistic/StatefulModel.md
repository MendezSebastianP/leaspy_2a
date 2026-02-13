# StatefulModel

**Module:** `leaspy.models.stateful`
**Inherits from:** `BaseModel`

`document: leaspy.models.stateful.StatefulModel`

While `BaseModel` handles the high-level orchestration of algorithms (Fit, Personalize, Simulate), `StatefulModel` is responsible for the **internal representation of model variables** (parameters, latent variables, derived variables).

This representation is what makes models compatible with the **MCMC-SAEM** family of algorithms: the algorithm can *propose changes* to some latent variables, and the model can *consistently update* everything that depends on them.

## The "State" Concept

In simple machine learning models, parameters might just be a flat list of weights. In Leaspy's generative models, parameters have **types** (population vs individual), **constraints**, and **dependencies** (some quantities are derived from others).

To manage this, `StatefulModel` exposes a `self.state` property, which returns a `State` object (see `leaspy.variables.state.State`).

### Responsibilities of `self.state`

1.  **Variable Registry**: It keeps track of every variable in the model:
    - **Population Parameters** (Fixed Effects): Global values like $g$ or $v_0$.
    - **Individual Parameters** (Random Effects): Patient-specific values like $\tau_i$ or $\xi_i$.
2.  **Dependency Graph (DAG)**: It knows how variables depend on each other.
3.  **Sampling Container**: During MCMC-SAEM, it acts as the container where proposed samples are stored, accepted/rejected, and kept consistent.

### What do we mean by “DAG/graph”?

Here, a **graph** is just a programming structure that represents *dependencies*.

- Each **node** is a variable (a parameter, a latent variable, or a derived quantity).
- Each **edge** means “this variable needs that one to be computed”.

Leaspy uses a **DAG** (Directed Acyclic Graph):

- **Directed**: dependencies have a direction (inputs → outputs).
- **Acyclic**: there is no circular dependency like “A depends on B and B depends on A”.

Example (informal):

```text
t_ij  ─┐
    ├──>  t_shifted = t_ij - tau_i
tau_i ─┘
```

This is about letting the code answer questions like:

- “If I change `tau_i`, which quantities must be recomputed?”
- “Which variables are individual random effects vs population parameters?”

### Key Implementation Details

`StatefulModel` does **not** fully build the `State` in `__init__`. Instead:

1. `__init__` starts with an empty internal state (`self._state = None`) and an optional list of variables to track.
2. `initialize()` calls `_initialize_state()`, which builds the DAG from the model variable specifications.

This is the core idea (simplified):

```python
def __init__(self, name: str, **kwargs):
    super().__init__(name, **kwargs)
    self._state = None
    self.tracked_variables = set()

def initialize(self, dataset=None):
    super().initialize(dataset=dataset)
    self._initialize_state()

def _initialize_state(self):
    self.state = State(
        VariablesDAG.from_dict(self.get_variables_specs()),
        auto_fork_type=StateForkType.REF,
    )
    self.state.track_variables(self.tracked_variables)
```

```{note}
The `State` is built from the model's *variable specifications* (`get_variables_specs()`), **not** from constructor arguments like `name` or `features` directly. Noise handling is part of the observation model layer (see [McmcSaemCompatibleModel](McmcSaemCompatibleModel.md)), not something `StatefulModel` itself manages.
```

Once the state exists, `StatefulModel` exposes higher-level helpers to interact with it without leaking the internal details:

*   `load_parameters(parameters)`: Validates and loads parameter values into the `State`.
*   `hyperparameters_names`, `parameters_names`, `population_variables_names`, `individual_variables_names`: Provide *typed* lists of variable names, derived from the DAG.

Note: there is **no** public `load_hyperparameters` method on `StatefulModel`. In Leaspy, hyperparameters are typically loaded via an internal hook named `_load_hyperparameters(...)` implemented in classes like `McmcSaemCompatibleModel` and concrete models.

## Architecture Decision: Stateful vs. Stateless

Leaspy V2 supports two distinct families of models. Understanding the difference helps developers choose the right base class for new models.

### 1. Stateful Models (The Generative Core)
*   **Base Class**: `leaspy.models.stateful.StatefulModel`
*   **Storage**: `self.state` (Object Graph)
*   **Use Case**: Bayesian Generative Models that require MCMC sampling.
*   **Examples**: `LogisticModel`, `LinearModel`, `UnivariateModel`, `JointModel`.
*   **Why it exists (longer explanation)**:
    - In MCMC-SAEM, the algorithm repeatedly proposes updates for latent variables (often individual random effects like $\tau_i, \xi_i$).
    - After proposing a change, it must evaluate how the proposal affects the model (log-likelihood / posterior terms) **without breaking consistency**.
    - If the model stored everything as an unstructured dict, the algorithm would need model-specific code like “when `tau` changes, recompute `t_shifted`, and then recompute `g(t_shifted)`…”. That would make the algorithm *not generic*.
    - With a `State` + DAG, the algorithm can remain generic: it can say “update this variable”, and the state machinery ensures that all dependent (derived) variables are recomputed correctly.
    - The `State` also supports **forking** (temporary copies / views of the state) which is a natural fit for MCMC proposals: propose in a fork → compute acceptance → keep or discard.

### 2. Stateless Models (The Benchmarks)
*   **Base Class**: `leaspy.models.stateless.StatelessModel`
*   **Storage**: `self._parameters` (Python Dictionary)
*   **Use Case**: Simple analytic models or wrappers around external libraries. They do generally not support the full Leaspy fitting process (SAEM) but can be used for prediction or benchmarking.
*   **Examples**:
    *   `LMEModel`: Wraps `statsmodels` to provide a Linear Mixed Effects baseline.
    *   `ConstantModel`: A trivial baseline that predicts constant values.
*   **Why it exists**: Sometimes we need a model that adheres to the Leaspy API (can be "personalized" or used to "simulate") but doesn't need the overhead of a variable graph.

## Developer Note: When to use which?

Use `StatefulModel` (or more commonly `McmcSaemCompatibleModel`) when:

- The model has **latent variables** (especially individual random effects) that you expect the algorithm to **sample / update iteratively**.
- You want the model to participate in the **MCMC-SAEM** workflow.
- You are ready to define a full variable specification via `get_variables_specs()` so the DAG can be built.
- You need “proposal-time consistency”: updating one variable should automatically update derived quantities.

Use `StatelessModel` when:

- The model is a **baseline / benchmark**, analytic estimator, or a thin wrapper around a library (e.g. `statsmodels`).
- You do not need the SAEM machinery and do not want to maintain variable specs / DAG logic.
- The model’s “parameters” are naturally represented as a small dict, and prediction/personalization can be implemented directly.
---

## What comes next?

`StatefulModel` gives a model its internal structure: the `State`, the [DAG](DAG.md), and typed access to variables. But it says nothing about **how the algorithm should use** that structure during optimization.

That is the role of [`McmcSaemCompatibleModel`](McmcSaemCompatibleModel.md), which inherits from `StatefulModel` and adds the contract that the MCMC-SAEM algorithm expects: computing sufficient statistics, updating parameters, and injecting data into the state.

→ Continue reading: [McmcSaemCompatibleModel](McmcSaemCompatibleModel.md)
