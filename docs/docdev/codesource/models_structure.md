# Models Structure

In Leaspy, a "Model" isn't just a mathematical equation; it's a Python class that adheres to a strict contract. This allows generic algorithms (like MCMC-SAEM) to optimize *any* model that follows the rules.

## The Inheritance Hierarchy

Leaspy models are built using a mix of **Interface inheritance** and **Mixins**.

```{mermaid}
classDiagram
    %% Styles
    classDef interface fill:#e1f5fe,stroke:#01579b,stroke-width:2px,stroke-dasharray: 5 5;
    classDef base fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef impl fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    
    %% Connections styling
    linkStyle default stroke-width:2px,fill:none,stroke:#546e7a;

    class ModelInterface["ModelInterface"]:::interface {
        +name: str
        +dimension: int
        +fit()
        +personalize()
    }
    class BaseModel["BaseModel"]:::base {
        +features
        +load()
        +save()
        +_get_dataset()
    }
    class McmcSaemCompatibleModel["McmcSaemCompatibleModel"]:::base {
        <<Key Interface for SAEM>>
        +compute_sufficient_statistics()
        +update_parameters()
        +put_individual_parameters()
    }
    class TimeReparametrizedModel["TimeReparametrizedModel"]:::base {
        +time_reparametrization()
    }
    class LogisticModel["LogisticModel"]:::impl {
        +get_variables_specs()
        +metric()
    }
    class LinearModel["LinearModel"]:::impl {
        +get_variables_specs()
    }

    ModelInterface <|-- BaseModel
    BaseModel <|-- McmcSaemCompatibleModel
    McmcSaemCompatibleModel <|-- TimeReparametrizedModel
    TimeReparametrizedModel <|-- LogisticModel
    TimeReparametrizedModel <|-- LinearModel
```

## Key Classes

### 1. `BaseModel`
*   **Location**: {py:mod}`leaspy.models.base`
*   **Role**: Handles the boring stuff. Loading/Saving to JSON, checking input data types, and providing the public `fit()` entry point which delegates to the Algorithm factory.

### 2. `McmcSaemCompatibleModel`
*   **Location**: {py:mod}`leaspy.models.mcmc_saem_compatible`
*   **Role**: The bridge to the MCMC-SAEM algorithm.
*   **The Contract**: To use the SAEM algorithm, a model MUST implement methods that the algorithm calls blindly:
    *   {py:meth}`~leaspy.models.mcmc_saem_compatible.McmcSaemCompatibleModel.compute_sufficient_statistics`: "Here is the current state of the world, give me the raw numbers needed for optimization."
    *   {py:meth}`~leaspy.models.mcmc_saem_compatible.McmcSaemCompatibleModel.update_parameters`: "Here are the averaged stats, update your internal $\theta$."

### 3. `TimeReparametrizedModel`
*   **Location**: {py:mod}`leaspy.models.time_reparametrized`
*   **Role**: Implements the specific "time-warp" mechanic common to Linear, Logistic, and other models.
    *   It defines how real time $t_{ij}$ is converted to subject time $\tau_i + \alpha_i t_{ij}$.

## Extending the logic
When you create a new model (e.g. `LogisticModel`), you mostly just need to define:
1.  **Variables**: What parameters do you have? (`get_variables_specs`)
2.  **Geometry**: How does the manifold look? (`metric`)

:::{seealso}
See how this comes together in practice in the [Logistic Model Walkthrough](model_walkthrough.md).
:::
