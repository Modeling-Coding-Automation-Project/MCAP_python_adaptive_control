"""Shared result types and validation for zero-state, discrete-time tuning."""
from dataclasses import dataclass
from enum import Enum
from numbers import Real

import numpy as np


class Status(str, Enum):
    SUCCESS = "success"
    ITERATION_LIMIT = "iteration_limit"
    RANK_DEFICIENT = "rank_deficient"
    INVALID_ARGUMENT = "invalid_argument"
    NON_FINITE = "non_finite"
    SINGULAR_INVERSE = "singular_inverse"


@dataclass
class SolverOptions:
    ridge: float = 0.0
    tolerance: float = 1e-8
    max_iterations: int = 500
    rank_tolerance: float = 1e-10
    discard_samples: int = 0

    def validate(self, sample_count):
        for name in ("ridge", "tolerance", "rank_tolerance"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
        if self.ridge < 0 or self.tolerance <= 0 or self.rank_tolerance <= 0:
            raise ValueError("ridge must be nonnegative; tolerances must be positive")
        if (isinstance(self.max_iterations, (bool, np.bool_))
                or not isinstance(self.max_iterations, (int, np.integer)) or self.max_iterations < 1):
            raise ValueError("max_iterations must be a positive integer")
        if (isinstance(self.discard_samples, (bool, np.bool_))
                or not isinstance(self.discard_samples, (int, np.integer))
                or not 0 <= self.discard_samples < sample_count):
            raise ValueError("discard_samples must leave at least one sample")


@dataclass
class FitResult:
    parameters: np.ndarray
    status: Status
    cost: float
    iterations: int = 0
    rank: int = 0

    @property
    def success(self):
        return self.status == Status.SUCCESS


def vector(value, name, size=None):
    out = np.asarray(value, dtype=float)
    if out.ndim != 1 or out.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if size is not None and out.size != size:
        raise ValueError(f"{name} must have length {size}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    return out


def dataset(u, y, options):
    u = vector(u, "u")
    y = vector(y, "y", u.size)
    options = SolverOptions() if options is None else options
    if not isinstance(options, SolverOptions):
        raise ValueError("options must be SolverOptions")
    options.validate(u.size)
    return u, y, options


def failure(parameter_count, status, iterations=0, rank=0):
    return FitResult(np.zeros(parameter_count), status, float("nan"), iterations, rank)
