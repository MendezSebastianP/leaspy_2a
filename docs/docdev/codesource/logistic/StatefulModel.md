# StatefulModel

**Module:** `leaspy.models.stateful`

`StatefulModel` extends `BaseModel` by introducing state management capabilities. It is a crucial layer that allows the model to handle variables and parameters dynamically during execution (like MCMC sampling).

## Responsibilities

*   **State Management**: It maintains an internal `State` object (`self.state`).
*   **MCMC Readiness**: It prepares the model infrastructure (State and DAG) required to run MCMC (Markov Chain Monte Carlo) algorithms.
*   **Variable Tracking**: It allows tracking specific variables (`tracked_variables`).

## Key Attributes

*   `state`: The internal state of the model, holding current values of variables.
*   `dag`: (Graph structure) Defines the dependencies between variables.

## Key Concepts

*   **State**: Holds the current realization of random variables during sampling.
*   **DAG**: Directed Acyclic Graph representing the hierarchical structure of model parameters and random variables.

