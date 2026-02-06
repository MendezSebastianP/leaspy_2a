# Data & Dataset

**Module:** `leaspy.io.data`

Leaspy uses specific internal representations for data to handle the complex structure of longitudinal datasets (multiple subjects, multiple timepoints, multiple features).

## Data

The `Data` class is the high-level container that user interacts with. It wraps the raw information (often a pandas DataFrame) and prepares it for the model.

## Dataset

The `Dataset` class is the internal representation used by the model during fitting. It converts the data into PyTorch tensors and handles:
*   **Timepoints**: Organization of visit times per subject.
*   **Values**: The observed measurements.
*   **Masking**: Handling of missing data (not all features are measured at all visits).
