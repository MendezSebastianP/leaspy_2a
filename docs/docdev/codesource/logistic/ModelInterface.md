# ModelInterface

**Module:** `leaspy.models.base`

At the very top of the Leaspy model hierarchy sits `ModelInterface`. This is not a model you can use directly; it is a **contract**. It strictly defines what a Leaspy model *is* and what it *must do*.

## What is an Interface?

In software design, an interface acts like a blueprint. It declares the methods and properties that a class must implement, without providing the method code itself. This ensures consistency: any object that claims to be a "Leaspy Model" is guaranteed to have the standard methods (`fit`, `estimate`, `personalize`) that the algorithms expect.

> Use an Abstract Base Class (ABC) when you want to provide a common interface for a set of subclasses.
> — [Python `abc` module documentation](https://docs.python.org/3/library/abc.html)

## The Contract: Mandatory Capabilities

Any valid Leaspy model must provide implementations for these capabilities:

### Core Properties
*   `name` & `dimension`: Identity and structure of the model.
*   `features`: The specific variable names (e.g., "memory", "attention") corresponding to the dimensions.
*   `is_initialized`: A flag indicating if the model has been set up with data.
*   `parameters` & `hyperparameters`: Access to the underlying mathematical values (fixed effects).

### Core Methods
*   **`fit(data, algorithm)`**: Estimate the model parameters from population data.
*   **`personalize(data)`**: Estimate individual parameters for specific subjects.
*   **`estimate(timepoints, individual_parameters)`**: Predict values for subjects at specific times.
*   **`simulate(individual_parameters, ...)`**: Generate synthetic data based on model parameters.

### I/O
*   **`save(path)`** & **`load(path)`**: Serialize the model to/from JSON files for storage.

## Interface vs. Abstract Class: What's the Difference?

You might hear both terms used. In Python, the distinction is often subtle because both use the `abc` (Abstract Base Class) module, but the *intent* is different:

*   **Interface (`ModelInterface`)**: Pure contract. It contains **no logic**, only method signatures marked with `@abstractmethod`. It says "You *must* implement these methods," but doesn't help you do it.
*   **Abstract Class (`BaseModel`)**: Partial implementation. It obeys the interface but also provides shared code (like the `fit` orchestration). It cannot be used directly because it's still missing pieces (like the specific equation formulas), but it does a lot of the heavy lifting for its subclasses.

In Leaspy architecture:
1.  `ModelInterface` defines the strict rules.
2.  `BaseModel` implements the common rules.
3.  `LogisticModel` implements the specific rules.

## Why Do We Need This?

The algorithms in Leaspy (like MCMC-SAEM) are designed to be generic. They don't want to know if they are optimizing a `LogisticModel` or a custom `MyNewModel`. They only want to know that the object passed to them has a `.fit()` method and a `.parameters` property. As long as your new class inherits from `ModelInterface` and implements these methods, it will work with all existing Leaspy algorithms.

## Moving to Implementation: BaseModel

`ModelInterface` tells us *what* to do, but it doesn't do anything itself. Implementing all these methods from scratch for every new model would be tedious and error-prone.

To solve this, we have **[`BaseModel`](BaseModel.md)**. It takes this contract and provides the standard "plumbing" — the shared code that orchestrates how these methods work together.
