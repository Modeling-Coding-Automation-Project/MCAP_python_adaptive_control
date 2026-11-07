"""Fictitious reference iterative tuning and its exact inverse-linear LS form.

The iterative form fits forward controller coefficients. The direct LS form
fits coefficients of C^-1 instead; it is exact for a fixed controller numerator
and variable denominator. It is not an approximate linearization of PID gains.
"""
import numpy as np

from .common import Status, dataset, failure, vector
from .filters import RationalFilter
from .least_squares import solve_least_squares
from .optimizer import SingularInverseError, optimize


def frit_regression(u, y, model, inverse_basis, options=None):
    u, y, options = dataset(u, y, options)
    target = y - model.filter(y)
    columns = inverse_basis.filter(u)
    matrix = np.column_stack([model.filter(columns[:, index])
                              for index in range(inverse_basis.parameter_count)])
    return matrix[options.discard_samples:], target[options.discard_samples:]


def frit_least_squares(u, y, model, inverse_basis, options=None):
    u, y, options = dataset(u, y, options)
    try:
        matrix, target = frit_regression(u, y, model, inverse_basis, options)
    except FloatingPointError:
        return failure(inverse_basis.parameter_count, Status.NON_FINITE)
    return solve_least_squares(matrix, target, options)


def _frit_objective(u, target, model, basis, options):
    def cost_gradient(parameters):
        numerator = parameters @ basis.numerators
        if numerator[0] == 0:
            raise SingularInverseError("C^-1 requires a nonzero controller direct coefficient")
        # B(rho)e=A u, e=C(rho)^-1 u.
        try:
            inverse = RationalFilter(basis.denominator, numerator)
        except ValueError as error:
            raise FloatingPointError("inverse filter coefficients are not representable") from error
        inverse_output = inverse.filter(u)
        residual = target - model.filter(inverse_output)
        residual = residual[options.discard_samples:]
        gradient = np.empty(basis.parameter_count)
        for index, row in enumerate(basis.numerators):
            # d e / d rho_i = -(B_i/B)e, hence d residual = M(B_i/B)e.
            try:
                derivative_filter = RationalFilter(row, numerator)
            except ValueError as error:
                raise FloatingPointError("sensitivity filter coefficients are not representable") from error
            derivative = model.filter(derivative_filter.filter(inverse_output))
            gradient[index] = 2 * residual @ derivative[options.discard_samples:] / residual.size
        cost = residual @ residual / residual.size + options.ridge * (parameters @ parameters)
        gradient += 2 * options.ridge * parameters
        return float(cost), gradient
    return cost_gradient


def frit_cost_gradient(parameters, u, y, model, basis, options=None):
    u, y, options = dataset(u, y, options)
    parameters = vector(parameters, "parameters", basis.parameter_count)
    return _frit_objective(u, y - model.filter(y), model, basis, options)(parameters)


def frit_optimize(u, y, model, basis, initial, lower, upper, options=None):
    u, y, options = dataset(u, y, options)
    try:
        target = y - model.filter(y)
    except FloatingPointError:
        return failure(basis.parameter_count, Status.NON_FINITE)
    return optimize(_frit_objective(u, target, model, basis, options),
                    basis.parameter_count, initial, lower, upper, options)


class FRIT:
    def __init__(self, reference_model, basis):
        self.reference_model = reference_model
        self.basis = basis

    def cost_gradient(self, parameters, u, y, options=None):
        return frit_cost_gradient(parameters, u, y, self.reference_model, self.basis, options)

    def fit_iterative(self, u, y, initial, lower, upper, options=None):
        return frit_optimize(u, y, self.reference_model, self.basis, initial, lower, upper, options)


class FRITLeastSquares:
    def __init__(self, reference_model, inverse_basis):
        self.reference_model = reference_model
        self.inverse_basis = inverse_basis

    def regression(self, u, y, options=None):
        return frit_regression(u, y, self.reference_model, self.inverse_basis, options)

    def fit_least_squares(self, u, y, options=None):
        return frit_least_squares(u, y, self.reference_model, self.inverse_basis, options)
