# StatefulModel

**Module:** `leaspy.models.stateful`
**Inherits from:** `BaseModel`

`document: leaspy.models.stateful.StatefulModel`

While `BaseModel` handles the high-level orchestration of algorithms (Fit, Personalize, Simulate), `StatefulModel` is responsible for the **internal representation of model variables** (parameters, latent variables, derived variables).

This representation is what makes models compatible with the **MCMC-SAEM** family of algorithms: the algorithm can *propose changes* to some latent variables, and the model can *consistently update* everything that depends on them.

## The "State" Concept

In standard programming, you might store your variables in a simple Python dictionary: `params = {"g": 1.2, "tau": 75}`.
In a **Generative Model** like Leaspy, this is not enough because the variables are strictly interconnected. If you change a patient's time shift ($\tau_i$), you implicitly change their reparametrized time, which changes their model score, which changes the likelihood of their data.

To handle this chain reaction, `StatefulModel` uses a dedicated object: **`self.state`** (an instance of `Leaspy.variables.State`).

### What is the State?

Think of the State as a **Smart Dictionary** that imposes strict rules:

1.  **Active Memory**: It stores the *current numerical values* of all variables (Population parameters, Individual parameters, and Derived values).
2.  **Consistency Enforcer**: It knows the **Dependency Graph** (DAG). If you update a variable (e.g., set a new `xi`), the State automatically invalidates or recomputes all downstream variables (like `alpha`). You can never have a "stale" state where $\alpha \neq e^{\xi}$.
3.  **Time-Machine (Forking)**: This is its most critical feature for MCMC.
    - When the algorithm proposes a new parameter (e.g., "What if $\tau = 80$?"), the State allows you to **Auto-Fork**.
    - If you **Accept** (Keep): You simply do nothing. The current state is already updated with the new value.
    - If you **Reject**: You **Revert** the State instantly to its previous values. It works like an "Undo" button.

> The "time machine" part uses a Computer Sciences concept called `fork`, and it is implemented in `src/leaspy/variables/state.py` when a changement of state is detected.

### Responsibilities of `StatefulModel`

While `BaseModel` handles the outer workflow (files, algorithms), `StatefulModel` manages the **internal physics**:

1.  **Defining the Physics**: It implements `get_variables_specs()` to tell the State what variables exist and how they connect.
2.  **Initializing the Machine**: It builds the `State` object during `initialize()`, transforming those specifications into a live, reactive graph.
3.  **Typed Access**: It provides clean properties (e.g., `population_variables_names`, `parameters`) so you don't have to query the raw graph manually.

### What do we mean by “DAG/graph”?

Here, a **graph** is just a programming structure that represents *dependencies*.

- Each **node** is a variable (a parameter, a latent variable, or a derived quantity).
- Each **edge** means “this variable needs that one to be computed”.

Leaspy uses a **DAG** (Directed Acyclic Graph):

- **Directed**: dependencies have a direction (inputs → outputs).
- **Acyclic**: there is no circular dependency like “A depends on B and B depends on A”.

This is about letting the code answer questions like:

- “If I change `tau_i`, which quantities must be recomputed?”
- “Which variables are individual random effects vs population parameters?”

> For more information about DAGs, check the [dedicated page](DAG.md)

### Key Implementation Details

`StatefulModel` separates creation from initialization:

1.  **`__init__`**: Sets up the shell. The `state` is empty (`None`) because we don't know the data dimensions yet.
2.  **`initialize(dataset)`**: Once data is available, it calls `_initialize_state()`. This method reads your `get_variables_specs()` (the blueprint) and builds the actual `State` object (the machine).

```python
def initialize(self, dataset=None):
    super().initialize(dataset=dataset)
    # The State is born here, derived from the variable specs
    self.state = State(VariablesDAG.from_dict(self.get_variables_specs()))
```

Once initialized, `StatefulModel` provides helpers like `load_parameters()` to safely inject values into the state, ensuring they match the defined shapes.

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
