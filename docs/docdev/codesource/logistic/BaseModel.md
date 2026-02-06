# BaseModel

**Module:** `leaspy.models.base`

`BaseModel` is the root class for Leaspy models. It acts as the orchestration layer between models, algorithms, and data. Every model inherits from BaseModel to gain the infrastructure needed to run optimization algorithms like MCMC-SAEM.

## The Bridge Between Model, Algorithm, and Data

When you write `model.fit(data)`, three components need to work together: the **model** (which defines the mathematical equations), the **algorithm** (which optimizes parameters), and the **data** (observations from patients). BaseModel acts as the bridge, providing a standard interface so any algorithm can work with any model type.

Consider this analogy: BaseModel is like a power outlet. The outlet provides a standard interface so any appliance (algorithm) can plug in and work with any power source (model) without knowing the internal wiring details.

## Anatomy of fit(): The Three-Step Orchestration

When you call:

```python
model = LogisticModel(name="test-model", source_dimension=2)
model.fit(data, algorithm="mcmc_saem", n_iter=1000, seed=0)
```

BaseModel's `fit()` method executes three critical steps:

### 1. Data Standardization

The first step normalizes your input into a consistent format:

```python
dataset = BaseModel._get_dataset(data)
```

You might pass a pandas DataFrame, a Leaspy `Data` object, or a `Dataset` directly. The algorithm doesn't care about these differences — it always receives a standardized `Dataset` object. This abstraction allows algorithms to focus on optimization logic rather than data format handling.

### 2. Model Initialization (First-Time Setup)

```python
if not self.is_initialized:
    self.initialize(dataset)
```

On the first call to `fit()`, the model must configure itself based on the data:

- **Dimension validation**: Verify that the data's feature count matches `source_dimension`
- **Feature extraction**: If you didn't specify feature names at construction, extract them from the data's column names
- **Parameter initialization**: Compute reasonable starting values for population parameters (tau_mean, xi_std, g, v0, etc.)

The `is_initialized` flag ensures this setup happens only once. Subsequent `fit()` calls skip directly to algorithm execution, allowing you to resume training or try different algorithm settings without reinitializing.

### 3. Algorithm Factory and Execution

```python
algorithm = BaseModel._get_algorithm(algorithm, algorithm_settings, **kwargs)
algorithm.run(self, dataset)
```

The factory pattern converts your algorithm specification into an executable object. When you pass `algorithm="mcmc_saem"`, the factory instantiates an `McmcSaemAlgorithm` class configured with your settings (`n_iter=1000`, `seed=0`, etc.).

The algorithm object has a `run()` method that accepts two arguments: the model instance (`self`) and the dataset. This design enables polymorphism — inside `algorithm.run()`, the code calls model methods like `compute_individual_trajectory()` or `compute_log_likelihood()` without knowing whether it's operating on a LogisticModel, LinearModel, or any future model type.

BaseModel defines these methods as abstract interfaces, guaranteeing that any concrete model implementation will provide the operations needed by the algorithm.

## Dimension vs Features: Two Paths to Model Configuration

Models can be configured through two equivalent approaches:

```python
# Approach 1: Explicit dimension (features determined later from data)
model = LogisticModel(name="test-model", source_dimension=2)

# Approach 2: Named features (dimension inferred as len(features))
model = LogisticModel(name="test-model", features=["memory", "language"])
```

Both approaches tell the model "predict 2 values at each timepoint," but differ in when feature names are resolved. With Approach 1, feature names are extracted from your data's columns during `initialize()`. With Approach 2, you specify them upfront and the model validates that your data matches.

This distinction matters for two reasons:

**Parameter allocation**: If your model has dimension 2, parameters like `g` (noise levels) will have 2 elements, one per feature. The algorithm needs to know this size to allocate memory and perform computations correctly.

**DThe Algorithm's Loop: Infrastructure vs Interface

To understand BaseModel's role, consider what happens inside `algorithm.run()` for MCMC-SAEM:

```python
for iteration in range(n_iter):
    # E-step: Simulate individual parameters
    individual_params = algorithm.simulate_individual_parameters(model, dataset)
    
    # M-step: Update population parameters
    sufficient_stats = algorithm.compute_statistics(model, individual_params)
    model.update_parameters(sufficient_stats)
    
    # Check stopping criteria
    if algorithm.has_converged():
        break
```

BaseModel divides responsibilities between **infrastructure** (orchestration) and **interface** (contract):

**Infrastructure** — What BaseModel implements directly:
- `fit()` orchestration: coordinates data preparation, initialization, and algorithm execution
- `initialize()`: configures model structure before the optimization loop
- `is_initialized`: guards against running algorithms on unprepared models

**Interface** — What BaseModel declares as abstract methods that subclasses must implement:
- `compute_individual_trajectory()`: predict values for a patient given their parameters
- `update_parameters()`: accept new population parameter values from the algorithm
- `compute_log_likelihood()`: evaluate how well current parameters explain the data

This separation means algorithms can be written generically. The MCMC-SAEM code doesn't need to know if it's fitting a LogisticModel or a LinearModel — it just calls the abstract methods and each model responds with its specific implementation.

### Understanding "Statistics" in the M-Step

During the M-step, the algorithm computes **sufficient statistics** from simulated individual parameters. These are summary values like means and variances computed across all simulated parameter sets. For example, if the algorithm simulated 100 possible values for each patient's τᵢ (reference time), it computes the mean and variance of these simulations across the population. These statistics then inform the update to tau_mean and tau_std.

The algorithm handles all statistics computation. The model simply receives the new parameter values and stores them.

### Convergence Detection

The algorithm monitors whether optimization should stop by checking:
- **Parameter stability**: Have tau_mean, xi_std, and other parameters stopped changing significantly?
- **Iteration limit**: Have we reached the maximum iterations (`n_iter=1000`)?
- **Custom criteria**: Any user-defined early stopping conditions

BaseModel doesn't implement convergence logic — it only provides the entry point that launches the algorithm. Once `algorithm.run()` starts, the algorithm controls the iteration loop and stopping decision

BasWhy fit() Appears Twice in the Code

If you examine `base.py`, you'll notice `fit()` is declared in two places:

**First declaration** (~line 167): This is in the abstract `ModelInterface` class. It defines the method signature and raises `NotImplementedError`. This is a **contract** that says "every Leaspy model must provide a `fit()` method with these parameters."

**Second declaration** (~line 713): This is in the concrete `BaseModel` class. It contains the **actual implementation** — the three-step orchestration described above.

This pattern (interface + implementation) ensures consistent APIs across all models while allowing BaseModel to provide shared orchestration logic that subclasses inherit automatically.

## From Abstract to Concrete: The Inheritance Chain

BaseModel is abstract — you cannot instantiate it directly. Concrete models like LogisticModel inherit from BaseModel through a chain of intermediate classes, each adding capabilities:

- **BaseModel**: Provides `fit()` orchestration and abstract method contracts
- **StatefulModel**: Adds parameter storage and state management
- **McmcSaemCompatibleModel**: Implements methods needed specifically for MCMC-SAEM
- **LogisticModel**: Implements the logistic sigmoid equation and parameter initialization

Each layer fulfills part of the contract BaseModel established. By the time you reach LogisticModel, all abstract methods have concrete implementations. This allows the algorithm to call methods like `compute_individual_trajectory()` and receive actual predictions based on the logistic curve formula.

See the [inheritance diagram](../architecture.md#simplified-workflow-structure) to visualize the complete

See the [inheritance diagram](../architecture.md#simplified-workflow-structure) to understand the full chain from BaseModel to LogisticModel.
