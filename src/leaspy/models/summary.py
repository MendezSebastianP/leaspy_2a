"""Model inspection objects for programmatic access to model metadata.

This module provides :class:`Summary` and :class:`Info`, returned by
``model.summary()`` and ``model.info()`` respectively, along with the
:class:`~typing.TypedDict` schemas for training and dataset metadata.

Both classes auto-print when their return value is discarded (e.g.
``model.summary()``) and stay silent when stored in a variable
(e.g. ``s = model.summary()``).  See :class:`AutoPrintMixin` for details.
"""

import textwrap
from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict

import torch

__all__ = [
    "AutoPrintMixin",
    "DatasetInfo",
    "Info",
    "Summary",
    "TrainingInfo",
    "VisitsPerSubject",
    "get_axis_labels",
]


# ---------------------------------------------------------------------------
# TypedDict schemas for metadata
# ---------------------------------------------------------------------------


class VisitsPerSubject(TypedDict, total=False):
    """Per-subject visit distribution statistics."""

    median: float
    min: int
    max: int
    iqr: float


class DatasetInfo(TypedDict, total=False):
    """Statistics of the training dataset, computed during ``fit()``."""

    n_subjects: int
    n_scores: int
    n_visits: int
    n_observations: int
    visits_per_subject: VisitsPerSubject
    n_missing: int
    pct_missing: float
    n_events: int


class TrainingInfo(TypedDict, total=False):
    """Metadata about the training process, captured during ``fit()``."""

    algorithm: str
    seed: int
    n_iter: int
    converged: bool
    duration: str


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

# ANSI formatting constants
_WIDTH = 80
_BOLD = "\033[1m"
_RESET = "\033[0m"


def get_axis_labels(
    axis_name: Optional[str],
    size: int,
    feature_names: Optional[list[str]] = None,
) -> Optional[list[str]]:
    """Resolve human-readable labels for a parameter axis.

    Parameters
    ----------
    axis_name : str or None
        Semantic axis name (``"feature"``, ``"source"``, ``"cluster"``,
        ``"basis"``).
    size : int
        Number of elements along the axis.
    feature_names : list[str], optional
        Feature names used when *axis_name* is ``"feature"``.

    Returns
    -------
    list[str] or None
        Labels for the axis, or ``None`` if no meaningful labels are available.
    """
    if axis_name is None:
        return None

    if axis_name == "feature":
        if feature_names is not None:
            feats = feature_names[:size]
            return [f[:8] if len(f) <= 8 else f[:7] + "." for f in feats]
        return [f"f{i}" for i in range(size)]
    elif axis_name == "source":
        return [f"s{i}" for i in range(size)]
    elif axis_name == "cluster":
        return [f"c{i}" for i in range(size)]
    elif axis_name == "event":
        return None
    elif axis_name == "basis":
        return [f"b{i}" for i in range(size)]
    else:
        return None


def _wrap_text(label: str, text: str, indent: int = 0) -> list[str]:
    """Wrap *text* with a bold *label* prefix to fit within ``_WIDTH``."""
    prefix = f"{label}: " if label else ""
    initial_indent = " " * indent + prefix
    subsequent_indent = " " * (indent + 4)
    wrapper = textwrap.TextWrapper(
        width=_WIDTH,
        initial_indent=initial_indent,
        subsequent_indent=subsequent_indent,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapper.wrap(text)


# ---------------------------------------------------------------------------
# Auto-print mixin
# ---------------------------------------------------------------------------


class AutoPrintMixin:
    """Mixin that auto-prints when the object is discarded.

    Relies on CPython reference counting: when the return value of e.g.
    ``model.summary()`` is not assigned, the object is immediately
    garbage-collected, triggering ``__del__`` which prints it.

    When stored (``s = model.summary()``), any public attribute access
    sets ``_printed = True``, suppressing the ``__del__`` output.

    Subclasses must define a ``_printed: bool`` field (via dataclass)
    and a ``__str__`` method.
    """

    def __del__(self):
        if not object.__getattribute__(self, "_printed"):
            print(str(self))

    def __repr__(self) -> str:
        object.__setattr__(self, "_printed", True)
        return str(self)

    def __getattribute__(self, name: str):
        value = object.__getattribute__(self, name)
        # Suppress auto-print once any public attribute is accessed
        if not name.startswith("_") and name != "help":
            object.__setattr__(self, "_printed", True)
        return value


# ---------------------------------------------------------------------------
# Info
# ---------------------------------------------------------------------------


@dataclass(repr=False)
class Info(AutoPrintMixin):
    """Model configuration and training context (no parameter values).

    Returned by ``model.info()``.  Auto-prints when discarded; provides
    programmatic access when stored in a variable.

    Examples
    --------
    >>> model.info()              # prints info
    >>> i = model.info()          # store for programmatic access
    >>> i.n_subjects              # 150
    >>> i.pct_missing             # 2.5
    >>> i.help()                  # list available attributes
    """

    name: str
    model_type: str
    dimension: Optional[int] = None
    features: Optional[list[str]] = None
    source_dimension: Optional[int] = None
    n_clusters: Optional[int] = None
    obs_models: Optional[list[str]] = None
    n_total_params: Optional[int] = None
    training_info: TrainingInfo = field(default_factory=dict)
    dataset_info: DatasetInfo = field(default_factory=dict)
    leaspy_version: Optional[str] = None
    _printed: bool = field(default=False, repr=False)

    # -- Convenience properties: training ------------------------------------

    @property
    def algorithm(self) -> Optional[str]:
        """Algorithm name used for training."""
        return self.training_info.get("algorithm")

    @property
    def seed(self) -> Optional[int]:
        """Random seed used for training."""
        return self.training_info.get("seed")

    @property
    def n_iter(self) -> Optional[int]:
        """Number of iterations."""
        return self.training_info.get("n_iter")

    @property
    def converged(self) -> Optional[bool]:
        """Whether training converged."""
        return self.training_info.get("converged")

    @property
    def duration(self) -> Optional[str]:
        """Training duration."""
        return self.training_info.get("duration")

    # -- Convenience properties: dataset -------------------------------------

    @property
    def n_subjects(self) -> Optional[int]:
        """Number of subjects in the training dataset."""
        return self.dataset_info.get("n_subjects")

    @property
    def n_visits(self) -> Optional[int]:
        """Total number of visits."""
        return self.dataset_info.get("n_visits")

    @property
    def n_scores(self) -> Optional[int]:
        """Number of scored features."""
        return self.dataset_info.get("n_scores")

    @property
    def n_observations(self) -> Optional[int]:
        """Total number of observed data points."""
        return self.dataset_info.get("n_observations")

    @property
    def pct_missing(self) -> Optional[float]:
        """Percentage of missing observations."""
        return self.dataset_info.get("pct_missing")

    @property
    def n_missing(self) -> Optional[int]:
        """Number of missing observations."""
        return self.dataset_info.get("n_missing")

    @property
    def visits_per_subject(self) -> Optional[VisitsPerSubject]:
        """Per-subject visit distribution statistics."""
        return self.dataset_info.get("visits_per_subject")

    @property
    def n_events(self) -> Optional[int]:
        """Number of observed events (joint models only)."""
        return self.dataset_info.get("n_events")

    # -- Display -------------------------------------------------------------

    def __str__(self) -> str:
        lines = []
        sep = "=" * _WIDTH

        lines.append(sep)
        lines.append(f"{_BOLD}{'Model Information':^{_WIDTH}}{_RESET}")
        lines.append(sep)

        # Statistical Model
        lines.append(f"{_BOLD}Statistical Model{_RESET}")
        lines.append("-" * _WIDTH)
        lines.append(f"Type: {self.model_type}")
        lines.append(f"Name: {self.name}")
        lines.append(f"Dimension: {self.dimension}")
        if self.source_dimension is not None:
            lines.append(f"Source Dimension: {self.source_dimension}")
        if self.obs_models:
            lines.append(f"Observation Models: {', '.join(self.obs_models)}")
        if self.n_total_params is not None:
            lines.append(f"Total Parameters: {self.n_total_params}")
        if self.n_clusters is not None:
            lines.append(f"Clusters: {self.n_clusters}")

        # Training Dataset
        if self.dataset_info:
            lines.append("")
            lines.append(f"{_BOLD}Training Dataset{_RESET}")
            lines.append("-" * _WIDTH)
            di = self.dataset_info
            lines.append(f"Subjects: {di.get('n_subjects', 'N/A')}")
            lines.append(f"Visits: {di.get('n_visits', 'N/A')}")
            lines.append(f"Scores (Features): {di.get('n_scores', 'N/A')}")
            lines.append(f"Total Observations: {di.get('n_observations', 'N/A')}")
            if "visits_per_subject" in di:
                vps = di["visits_per_subject"]
                lines.append(
                    f"Visits per Subject: Median {vps['median']:.1f} "
                    f"[Min {vps['min']}, Max {vps['max']}, IQR {vps['iqr']:.1f}]"
                )
            if "n_missing" in di:
                lines.append(
                    f"Missing Data: {di['n_missing']} "
                    f"({di.get('pct_missing', 0):.2f}%)"
                )
            if "n_events" in di:
                lines.append(f"Events Observed: {di['n_events']}")

        # Training Details
        if self.training_info:
            lines.append("")
            lines.append(f"{_BOLD}Training Details{_RESET}")
            lines.append("-" * _WIDTH)
            ti = self.training_info
            lines.append(f"Algorithm: {ti.get('algorithm', 'N/A')}")
            if "seed" in ti:
                lines.append(f"Seed: {ti['seed']}")
            lines.append(f"Iterations: {ti.get('n_iter', 'N/A')}")
            if ti.get("converged") is not None:
                lines.append(f"Converged: {ti['converged']}")
            if "duration" in ti:
                lines.append(f"Duration: {ti['duration']}")

        # Leaspy Version
        if self.leaspy_version:
            lines.append("")
            lines.append(f"Leaspy Version: {self.leaspy_version}")

        lines.append(sep)
        return "\n".join(lines)

    def help(self) -> None:
        """Print available attributes and their meanings."""
        help_text = f"""
{_BOLD}Info Help{_RESET}
{'=' * 60}

The Info object provides access to model configuration and training context.

{_BOLD}Usage:{_RESET}
    model.info()          # Print model information
    i = model.info()      # Store to access individual attributes

{_BOLD}Available Attributes:{_RESET}

  {_BOLD}Model:{_RESET}
    name              Model name (str)
    model_type        Model class name (str)
    dimension         Number of features (int)
    features          Feature names (list[str])
    source_dimension  Number of sources (int or None)
    n_clusters        Number of clusters (int or None)
    obs_models        Observation model names (list[str] or None)
    n_total_params    Total number of scalar parameters (int)

  {_BOLD}Training:{_RESET}
    algorithm         Algorithm name (str)
    seed              Random seed (int)
    n_iter            Number of iterations (int)
    converged         Whether training converged (bool or None)
    duration          Training duration (str)

  {_BOLD}Dataset:{_RESET}
    n_subjects        Number of subjects (int)
    n_visits          Total visits (int)
    n_scores          Number of scored features (int)
    n_observations    Total observations (int)
    pct_missing       Percent missing data (float)
    n_missing         Count of missing observations (int)
    visits_per_subject  Visit distribution stats (dict)
    n_events          Observed events, joint models only (int or None)

  {_BOLD}Other:{_RESET}
    training_info     Full training metadata (TrainingInfo)
    dataset_info      Full dataset statistics (DatasetInfo)
    leaspy_version    Leaspy version (str)

{_BOLD}Examples:{_RESET}
    >>> i = model.info()
    >>> i.algorithm              # 'mcmc_saem'
    >>> i.n_subjects             # 150
    >>> i.pct_missing            # 2.5
"""
        print(help_text)
        object.__setattr__(self, "_printed", True)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


@dataclass(repr=False)
class Summary(AutoPrintMixin):
    """Structured summary of a Leaspy model including parameter values.

    Returned by ``model.summary()``.  Auto-prints when discarded; provides
    programmatic access when stored in a variable.

    Examples
    --------
    >>> model.summary()           # prints the formatted summary
    >>> s = model.summary()       # store for programmatic access
    >>> s.algorithm               # 'mcmc_saem'
    >>> s.get_param('tau_std')    # tensor([10.5])
    >>> s.help()                  # list available attributes
    """

    name: str
    model_type: str
    dimension: Optional[int] = None
    features: Optional[list[str]] = None
    source_dimension: Optional[int] = None
    n_clusters: Optional[int] = None
    obs_models: Optional[list[str]] = None
    nll: Optional[float] = None
    training_info: TrainingInfo = field(default_factory=dict)
    dataset_info: DatasetInfo = field(default_factory=dict)
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    leaspy_version: Optional[str] = None
    _param_axes: dict = field(default_factory=dict, repr=False)
    _feature_names: Optional[list[str]] = field(default=None, repr=False)
    _printed: bool = field(default=False, repr=False)

    # -- Convenience properties ----------------------------------------------

    @property
    def sources(self) -> Optional[list[str]]:
        """Source names (e.g. ``['s0', 's1']``) or ``None``."""
        if self.source_dimension is None:
            return None
        return [f"s{i}" for i in range(self.source_dimension)]

    @property
    def clusters(self) -> Optional[list[str]]:
        """Cluster names (e.g. ``['c0', 'c1']``) or ``None``."""
        if self.n_clusters is None:
            return None
        return [f"c{i}" for i in range(self.n_clusters)]

    @property
    def algorithm(self) -> Optional[str]:
        """Algorithm name used for training."""
        return self.training_info.get("algorithm")

    @property
    def seed(self) -> Optional[int]:
        """Random seed used for training."""
        return self.training_info.get("seed")

    @property
    def n_iter(self) -> Optional[int]:
        """Number of iterations."""
        return self.training_info.get("n_iter")

    @property
    def converged(self) -> Optional[bool]:
        """Whether training converged."""
        return self.training_info.get("converged")

    @property
    def n_subjects(self) -> Optional[int]:
        """Number of subjects in the training dataset."""
        return self.dataset_info.get("n_subjects")

    @property
    def n_visits(self) -> Optional[int]:
        """Total number of visits."""
        return self.dataset_info.get("n_visits")

    @property
    def n_observations(self) -> Optional[int]:
        """Total number of observations."""
        return self.dataset_info.get("n_observations")

    def get_param(self, name: str) -> Optional[Any]:
        """Get a parameter value by name, searching across all categories.

        Parameters
        ----------
        name : str
            Parameter name (e.g. ``'betas_mean'``, ``'tau_std'``).

        Returns
        -------
        value
            The parameter value (typically a ``torch.Tensor``), or ``None``.
        """
        for category_params in self.parameters.values():
            if name in category_params:
                return category_params[name]
        return None

    # -- Display -------------------------------------------------------------

    def __str__(self) -> str:
        lines = []
        sep = "=" * _WIDTH

        # Header
        lines.append(sep)
        lines.append(f"{_BOLD}{'Model Summary':^{_WIDTH}}{_RESET}")
        lines.append(sep)
        lines.append(f"{_BOLD}Model Name:{_RESET} {self.name}")
        lines.append(f"{_BOLD}Model Type:{_RESET} {self.model_type}")

        if self.features is not None:
            feat_str = ", ".join(self.features)
            lines.extend(
                _wrap_text(
                    f"{_BOLD}Features ({self.dimension}){_RESET}", feat_str
                )
            )

        if self.source_dimension is not None:
            sources = [f"Source {i} (s{i})" for i in range(self.source_dimension)]
            lines.extend(
                _wrap_text(
                    f"{_BOLD}Sources ({self.source_dimension}){_RESET}",
                    ", ".join(sources),
                )
            )

        if self.n_clusters is not None:
            clusters = [f"Cluster {i} (c{i})" for i in range(self.n_clusters)]
            lines.extend(
                _wrap_text(
                    f"{_BOLD}Clusters ({self.n_clusters}){_RESET}",
                    ", ".join(clusters),
                )
            )

        if self.obs_models:
            lines.extend(
                _wrap_text(
                    f"{_BOLD}Observation Models{_RESET}",
                    ", ".join(self.obs_models),
                )
            )

        if self.nll is not None:
            lines.append(f"{_BOLD}Neg. Log-Likelihood:{_RESET} {self.nll:.4f}")

        # Training Metadata
        if self.training_info:
            lines.append("")
            lines.append(f"{_BOLD}Training Metadata{_RESET}")
            lines.append("-" * _WIDTH)
            ti = self.training_info
            lines.append(f"Algorithm: {ti.get('algorithm', 'N/A')}")
            if "seed" in ti:
                lines.append(f"Seed: {ti['seed']}")
            lines.append(f"Iterations: {ti.get('n_iter', 'N/A')}")
            if ti.get("converged") is not None:
                lines.append(f"Converged: {ti['converged']}")
            if "duration" in ti:
                lines.append(f"Duration: {ti['duration']}")

        # Data Context
        if self.dataset_info:
            lines.append("")
            lines.append(f"{_BOLD}Data Context{_RESET}")
            lines.append("-" * _WIDTH)
            di = self.dataset_info
            lines.append(f"Subjects: {di.get('n_subjects', 'N/A')}")
            lines.append(f"Visits: {di.get('n_visits', 'N/A')}")
            lines.append(f"Total Observations: {di.get('n_observations', 'N/A')}")

        # Leaspy Version
        if self.leaspy_version:
            lines.append("")
            lines.append(f"Leaspy Version: {self.leaspy_version}")

        lines.append(sep)

        # Parameters by category
        for category, params in self.parameters.items():
            if params:
                lines.append("")
                lines.append(f"{_BOLD}{category}{_RESET}")
                lines.append("-" * _WIDTH)
                lines.extend(self._format_parameter_group(params))

        lines.append(sep)
        return "\n".join(lines)

    def help(self) -> None:
        """Print available attributes and their meanings."""
        help_text = f"""
{_BOLD}Summary Help{_RESET}
{'=' * 60}

The Summary object provides access to model metadata and parameters.

{_BOLD}Usage:{_RESET}
    model.summary()       # Print the formatted summary
    s = model.summary()   # Store to access individual attributes

{_BOLD}Available Attributes:{_RESET}

  {_BOLD}Model Information:{_RESET}
    name              Model name (str)
    model_type        Model class name, e.g., 'LogisticModel' (str)
    dimension         Number of features (int)
    features          List of feature names (list[str])
    sources           Source names, e.g., ['s0', 's1'] (list[str] or None)
    clusters          Cluster names, e.g., ['c0', 'c1'] (list[str] or None)
    source_dimension  Number of sources (int or None)
    n_clusters        Number of clusters (int or None)
    obs_models        Observation model names (list[str] or None)

  {_BOLD}Training:{_RESET}
    algorithm         Algorithm name, e.g., 'mcmc_saem' (str)
    seed              Random seed used (int)
    n_iter            Number of iterations (int)
    converged         Whether training converged (bool or None)
    nll               Negative log-likelihood (float or None)

  {_BOLD}Dataset:{_RESET}
    n_subjects        Number of subjects in training data (int)
    n_visits          Total number of visits (int)
    n_observations    Total number of observations (int)

  {_BOLD}Parameters:{_RESET}
    parameters        All parameters grouped by category (dict)
    get_param(name)   Get a specific parameter by name

  {_BOLD}Other:{_RESET}
    training_info     Full training metadata (TrainingInfo)
    dataset_info      Full dataset statistics (DatasetInfo)
    leaspy_version    Leaspy version used (str)

{_BOLD}Examples:{_RESET}
    >>> s = model.summary()
    >>> s.algorithm              # 'mcmc_saem'
    >>> s.seed                   # 42
    >>> s.n_subjects             # 150
    >>> s.get_param('tau_std')   # tensor([10.5])
"""
        print(help_text)
        object.__setattr__(self, "_printed", True)

    # -- Private formatting helpers ------------------------------------------

    def _format_parameter_group(self, params: dict[str, Any]) -> list[str]:
        """Format a group of parameters for display."""
        lines = []
        for name, value in params.items():
            if isinstance(value, torch.Tensor):
                lines.append(self._format_tensor(name, value))
            else:
                lines.append(f"  {name:<18} {value}")
        return lines

    def _format_tensor(self, name: str, value: torch.Tensor) -> str:
        """Format a tensor parameter with axis labels."""
        param_axes = object.__getattribute__(self, "_param_axes")
        feature_names = object.__getattribute__(self, "_feature_names")
        axes = param_axes.get(name, ())

        if value.ndim == 0:
            return f"  {name:<18} {value.item():.4f}"

        elif value.ndim == 1:
            n = len(value)
            if n > 10:
                return f"  {name:<18} Tensor of shape ({n},)"

            axis_name = axes[0] if len(axes) >= 1 else None
            col_labels = get_axis_labels(axis_name, n, feature_names)

            if col_labels:
                header = " " * 20 + "  ".join(f"{lbl:>8}" for lbl in col_labels)
                values = f"  {name:<18}" + "  ".join(
                    f"{v.item():>8.4f}" for v in value
                )
                return header + "\n" + values
            else:
                val_str = "[" + ", ".join(f"{v.item():.4f}" for v in value) + "]"
                return f"  {name:<18} {val_str}"

        elif value.ndim == 2:
            rows, cols = value.shape
            if rows > 8 or cols > 8:
                return f"  {name:<18} Tensor of shape {tuple(value.shape)}"

            row_axis = axes[0] if len(axes) >= 1 else None
            col_axis = axes[1] if len(axes) >= 2 else None
            row_labels = get_axis_labels(row_axis, rows, feature_names)
            col_labels = get_axis_labels(col_axis, cols, feature_names)

            result = [f"  {name}:"]
            if col_labels:
                header = " " * 20 + "  ".join(f"{lbl:>8}" for lbl in col_labels)
                result.append(header)

            for i, row in enumerate(value):
                row_lbl = row_labels[i] if row_labels else f"[{i}]"
                row_str = (
                    f"            {row_lbl:<8}"
                    + "  ".join(f"{v.item():>8.4f}" for v in row)
                )
                result.append(row_str)

            return "\n".join(result)

        else:
            return f"  {name:<18} Tensor of shape {tuple(value.shape)}"
