# BaseModel

**Module:** `leaspy.models.base`
**Inherits from:** [`ModelInterface`](ModelInterface.md)

While [`ModelInterface`](ModelInterface.md) defines the strict contract (the *what*), `BaseModel` provides the concrete implementation of the orchestration layer (the *how*). Every model inherits from BaseModel to gain the built-in infrastructure needed to run optimization algorithms like MCMC-SAEM without rewriting the boilerplate code.

## The Bridge Between Model, Algorithm, and Data

When you write `model.fit(data)`, three components need to work together: the **model** (which defines the mathematical equations), the **algorithm** (which optimizes parameters), and the **data** (observations from patients). BaseModel acts as the bridge, so any algorithm can work with any model type.

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

You might pass a pandas DataFrame, a Leaspy `Data` object, or a `Dataset` directly. The algorithm doesn't care about these differences — it always receives a standardized `Dataset` object. This abstraction allows algorithms to focus on optimization logic rather than data format handling. However models like `JointModels` need some specifications, so we advice to always give a `Data` object to your `fit()`.

### 2. Model Initialization (First-Time Setup)

```python
if not self.is_initialized:
    self.initialize(dataset)
```

On the first call to `fit()`, `BaseModel` triggers the initialization process. While `BaseModel` handles the state flag (`is_initialized`), the actual calculation of starting parameters is **delegated** to the specific model class (e.g., [`LogisticInitializationMixin`](LogisticInitializationMixin.md)).

This step ensures:
- **Dimension validation**: Verifying data matches model structure.
- **Paramater Initialization**: Computing heuristics for starting values (implemented by subclasses).

The `is_initialized` flag ensures this setup happens only once.

### 3. Algorithm Factory and Execution

```python
algorithm = BaseModel._get_algorithm(algorithm, algorithm_settings, **kwargs)
algorithm.run(self, dataset)
```

Finally, `BaseModel` instantiates the requested algorithm (e.g., MCMC-SAEM) and hands over control.

**Crucial Point**: Once `algorithm.run()` is called, **BaseModel's job is done**. The algorithm takes over the driver's seat. It will call back into the model to perform specific mathematical operations (like updating parameters or computing likelihoods), but the *loop itself* belongs to the algorithm. BaseModel defines these methods as abstract interfaces, guaranteeing that any concrete model implementation will provide the operations needed by the algorithm.

See [`McmcSaemCompatibleModel`](McmcSaemCompatibleModel.md) to understand how the algorithm interacts with the model during the optimization loop.

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

<br>

<div style="background-color: #f0f7ff; padding: 15px; border-left: 5px solid #0078d4; border-radius: 5px;">
<strong>Want to see the Algorithm Loop?</strong><br>
The complex "Simulate → Compute Statistics → Update Parameters" loop is not in BaseModel. It lives in the Algorithm classes and relies on methods defined in <a href="McmcSaemCompatibleModel.md#the-algorithms-loop-infrastructure-vs-interface">McmcSaemCompatibleModel</a>.
</div>

## Why fit() Appears Twice in the Code

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

See the [inheritance diagram](../architecture.md) to visualize the complete

See the [inheritance diagram](../architecture.md) to understand the full chain from BaseModel to LogisticModel.
