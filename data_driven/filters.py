"""Causal rational filters with ascending powers of z^-1.

Every batch call starts from zero state. Coefficients may be padded with trailing
zeros to match the fixed polynomial capacity of the C++ implementation.
"""
import numpy as np

from .common import vector


class RationalFilter:
    def __init__(self, numerator, denominator):
        numerator = vector(numerator, "numerator")
        denominator = vector(denominator, "denominator")
        if denominator[0] == 0:
            raise ValueError("denominator[0] must be nonzero for a causal filter")
        size = max(numerator.size, denominator.size)
        with np.errstate(over="ignore", invalid="ignore"):
            self.numerator = np.pad(numerator, (0, size - numerator.size)) / denominator[0]
            self.denominator = np.pad(denominator, (0, size - denominator.size)) / denominator[0]
        if not (np.all(np.isfinite(self.numerator)) and np.all(np.isfinite(self.denominator))):
            raise ValueError("normalizing filter coefficients overflowed")

    def filter(self, signal):
        signal = vector(signal, "signal")
        state = self.state()
        out = np.empty(signal.size)
        with np.errstate(over="ignore", invalid="ignore"):
            for index, value in enumerate(signal):
                out[index] = state.step(value)
        if not np.all(np.isfinite(out)):
            raise FloatingPointError("filter response is not finite")
        return out

    def state(self):
        """Return an independent, resettable scalar realization."""
        return FilterState(self)

    def inverse(self):
        if self.numerator[0] == 0:
            raise ValueError("a causal inverse requires numerator[0] != 0")
        return RationalFilter(self.denominator, self.numerator)


class FilterState:
    """Direct form II transposed state; storage is O(filter order)."""
    def __init__(self, model):
        self.model = model
        self.delay = np.zeros(model.numerator.size - 1)

    def reset(self):
        self.delay.fill(0.0)

    def step(self, value):
        if not np.isfinite(value):
            raise ValueError("filter input must be finite")
        b, a = self.model.numerator, self.model.denominator
        if self.delay.size == 0:
            return float(b[0] * value)
        output = b[0] * value + self.delay[0]
        for index in range(self.delay.size - 1):
            self.delay[index] = self.delay[index + 1] + b[index + 1] * value - a[index + 1] * output
        self.delay[-1] = b[-1] * value - a[-1] * output
        return float(output)


class LinearControllerBasis:
    """C(rho)=sum_i rho_i B_i(z^-1)/A(z^-1).

For ``frit_least_squares`` this object instead represents C(rho)^-1.
Do not interpret those fitted inverse coefficients as forward PID gains.
"""
    def __init__(self, numerators, denominator):
        numerators = np.asarray(numerators, dtype=float)
        denominator = vector(denominator, "denominator")
        if numerators.ndim != 2 or 0 in numerators.shape or not np.all(np.isfinite(numerators)):
            raise ValueError("numerators must be a nonempty finite matrix")
        if denominator[0] == 0:
            raise ValueError("denominator[0] must be nonzero")
        size = max(numerators.shape[1], denominator.size)
        with np.errstate(over="ignore", invalid="ignore"):
            self.numerators = np.pad(numerators, ((0, 0), (0, size - numerators.shape[1]))) / denominator[0]
            self.denominator = np.pad(denominator, (0, size - denominator.size)) / denominator[0]
        if not (np.all(np.isfinite(self.numerators)) and np.all(np.isfinite(self.denominator))):
            raise ValueError("normalizing basis coefficients overflowed")

    @property
    def parameter_count(self):
        return self.numerators.shape[0]

    def controller(self, parameters):
        parameters = vector(parameters, "parameters", self.parameter_count)
        return RationalFilter(parameters @ self.numerators, self.denominator)

    def filter(self, signal):
        return np.column_stack([RationalFilter(row, self.denominator).filter(signal)
                                for row in self.numerators])

    @classmethod
    def pi(cls, sample_time):
        if not np.isfinite(sample_time) or sample_time <= 0:
            raise ValueError("sample_time must be finite and positive")
        return cls([[1.0, -1.0], [sample_time, 0.0]], [1.0, -1.0])

    @classmethod
    def pid(cls, sample_time):
        if not np.isfinite(sample_time) or sample_time <= 0:
            raise ValueError("sample_time must be finite and positive")
        return cls([[1.0, -1.0, 0.0], [sample_time, 0.0, 0.0],
                    [1.0 / sample_time, -2.0 / sample_time, 1.0 / sample_time]],
                   [1.0, -1.0, 0.0])


TransferFunction = RationalFilter
ControllerBasis = LinearControllerBasis
