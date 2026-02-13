# Observation Models

**Modules:** `leaspy.models.obs_models._base`, `leaspy.models.obs_models._gaussian`

While the [`LogisticModel`](LogisticModel.md) defines the **hidden physiological process** (the smooth, noise-free trajectory of the disease), it doesn't describe the **data** directly. Real-world data is noisy and discrete.

The **Observation Model** is the bridge between the clean theoretical curve and the messy reality. It defines the probability distribution of the observed data given the model's prediction.

$$ P(y_{observed} | y_{model}) $$

## The Abstract Base: `ObservationModel`

**Module:** `leaspy.models.obs_models._base`

This class defines the interface for how data attaches to the model. In a probabilistic framework, this is often formalized as the **Negative Log-Likelihood (NLL)**.

### Responsibilities
*   **Data Retrieval**: Knows how to extract specific information from the raw `Dataset` (via a `getter`).
*   **Likelihood Computation**: Defines the symbolic distribution (e.g., Normal, Poisson) used to compute how "likely" the data is given the model's parameters.
*   **Variable Connection**: Generates the `nll_attach` variables that the optimization algorithms minimize.

### Key Attributes
*   `name`: The name of the observed variable (usually "y").
*   `dist`: A `SymbolicDistribution` object representing the statistical assumption (e.g., `Normal(loc="model", scale="noise_std")`).

---

## The Concrete Implementation: `GaussianObservationModel`

**Module:** `leaspy.models.obs_models._gaussian`
**Inherits from:** `ObservationModel`

This is the standard observation model used in Leaspy. It assumes that the observed data is just the model prediction plus some Gaussian noise.

$$ y_{observed} = y_{model}(t) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2) $$

### Responsibilities
*   **Noise Management**: It manages the `noise_std` parameter ($\sigma$).
*   **Residual Calculation**: It computes the squared differences between the model and the data (L2 norm) which drives the fitting process.

### `FullGaussianObservationModel`
A specialized subclass that handles the most common case:
*   **Scalar Noise**: One global $\sigma$ for all features (homoscedasticity).
*   **Diagonal Noise**: One distinct $\sigma_k$ per feature (heteroscedasticity across features).

It includes the **update rules** for estimating this noise variance during the MCMC-SAEM algorithm (M-step).

### Key Methods
*   `noise_std_specs(dimension)`: Defines whether the noise is scalar (dim=1) or diagonal (dim>1).
*   `scalar_noise_std_update(...)`: Closed-form formula to update the global noise variance based on the residuals.
*   `diagonal_noise_std_update(...)`: Closed-form formula to update feature-specific noise variances.

## Next Steps

Now that the Model (signal) and Observation (noise) are defined, the [McmcSaemCompatibleModel](McmcSaemCompatibleModel.md) ties them together to enable the actual fitting process.
