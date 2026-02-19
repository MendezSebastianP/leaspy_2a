# McmcSaemCompatibleModel

**Module:** `leaspy.models.mcmc_saem_compatible`
**Inherits from:** [`StatefulModel`](StatefulModel.md)

`document: leaspy.models.mcmc_saem_compatible.McmcSaemCompatibleModel`

`McmcSaemCompatibleModel` is the **bridge** between the generic MCMC-SAEM algorithm and a specific mathematical model. While [`StatefulModel`](StatefulModel.md) provides the variables (the `State`), this class defines the **interface** and methods required to optimize them.

It guarantees that the algorithm can perform its two main iterative steps without knowing the underlying model details:
1.  **E-Step**: Sampling individual latent variables (like $\tau_i, \xi_i$).
2.  **M-Step**: Updating population parameters (like $g, v_0$).

## The Optimization Contract

The class enforces three core methods that drive the MCMC-SAEM loop:

1.  **`put_individual_parameters(state, dataset)`** (E-Step)
    Used to initialize or inject individual latent variables into the state. Concrete models (like `LogisticModel`) implement this to map the dataset inputs into the model's specific latent variables.

2.  **`compute_sufficient_statistics(state)`** (M-Step, Part 1)
    After sampling, the algorithm needs to summarize the current state. This method aggregates **sufficient statistics** from all `ModelParameter` nodes in the DAG. It computes the minimal set of numbers (sums, counts) needed to update parameters, rather than storing every sample.

3.  **`update_parameters(state, sufficient_statistics, burn_in)`** (M-Step, Part 2)
    Updates the population parameters using the computed statistics. This is done in a **batch** operation: all new values are computed first, then applied simultaneously to avoid order-dependent inconsistencies. The `burn_in` flag controls whether the update step uses a "memoryless" approach (faster adaptation) or standard recursive averaging.

## Other Responsibilities

Beyond the optimization loop, this class handles:
*   **Observation Models**: Wraps the noise model (e.g., Gaussian noise), providing the likelihood function (`nll_attach`) needed to accept or reject samples.
*   **Individual Trajectories**: The `compute_individual_trajectory` method allows predicting a patient's score **after** the model is trained, effectively powering the `personalize` feature.

## The Algorithm Loop

The following diagram illustrates how the MCMC-SAEM algorithm interacts with this interface at each iteration:

```mermaid
%%{init: {
  "themeVariables": {
    "clusterBkg": "rgba(227, 242, 253, 0.6)",
    "clusterBorder": "#90caf9",
    "clusterTextColor": "#0d47a1",
    "fontSize": "14px"
  },
  "flowchart": {"rankSpacing":15, "nodeSpacing":15}
}}%%
graph TD
    %% Node styles
    classDef process fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#0d47a1,rx:10,ry:10;
    classDef decision fill:#e1f5fe,stroke:#0277bd,stroke-width:2px,color:#01579b,rx:10,ry:10;
    classDef math fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#4a148c,rx:10,ry:10;
    classDef io fill:#fffde7,stroke:#fbc02d,color:#f57f17,rx:10,ry:10;

    Start((Start)) --> Init[Initialize State]:::process
    Init --> Loop{{Loop k = 1 to N}}:::decision
    
    %% --- E-Step ---
    subgraph Estep["<b>1. E-Step: Sampling</b>"]
        direction TB
        Loop --> SelectVar["Select ϑ ∈ {τ, ξ, …}"]:::process
        SelectVar --> Propose["Propose ϑ*"]:::math
        Propose --> Fork["Fork State"]:::process
        Fork --> Alpha["Compute α = min(1, exp(−ΔH / T))"]:::math
        Alpha --> Accept{"Accept?"}:::decision
        
        Accept -- Yes --> Keep["Update State S ← S*"]:::process
        Accept -- No --> Revert["Revert State"]:::process
        
        Keep --> DoneVars{All vars?}:::decision
        Revert --> DoneVars
        DoneVars -- No --> SelectVar
    end
    style Estep fill:#e3f2fd,stroke:#90caf9,stroke-width:2px,color:#0d47a1;

    %% --- M-Step ---
    subgraph Mstep["<b>2. M-Step: Maximization</b>"]
        direction TB
        DoneVars -- Yes --> SuffStats["Sufficient Stats S<sub>k</sub>"]:::math
        SuffStats --> SAEM["Update Memory<br/>S<sub>k</sub> ← (1−γ<sub>k</sub>)S<sub>k−1</sub> + γ<sub>k</sub>S(v)"]:::math
        SAEM --> Update["Maximization<br/>θ<sub>k+1</sub> = arg max P(θ | S<sub>k</sub>)"]:::math
        Update --> Anneal["Decrease T"]:::process
    end
    style Mstep fill:#e3f2fd,stroke:#90caf9,stroke-width:2px,color:#0d47a1;

    Anneal --> Converged{Converged?}:::decision
    Converged -- No --> Loop
    Converged -- Yes --> End((End))
    
    class Start,End process
```

