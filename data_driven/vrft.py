"""Virtual reference feedback tuning with the causal prefilter L=M.

For C(rho)=sum_i rho_i beta_i, minimize
mean((M u - sum_i rho_i beta_i (I-M)y)**2) + ridge*||rho||**2.
Using L=M avoids computing the often noncausal inverse of the desired model.
This prefilter is a practical choice, not a universal model-matching optimum.
"""
import numpy as np

from .common import Status, dataset, failure
from .least_squares import solve_least_squares
from .optimizer import optimize


def vrft_regression(u, y, model, basis, options=None, prefilter=None):
    u, y, options = dataset(u, y, options)
    # All filters run on the complete record before any startup samples are cut.
    target = model.filter(u)
    matrix = basis.filter(y - model.filter(y))
    if prefilter is not None:
        target = prefilter.filter(target)
        matrix = np.column_stack([prefilter.filter(column)
                                 for column in matrix.T])
    return matrix[options.discard_samples:], target[options.discard_samples:]


def vrft_least_squares(u, y, model, basis, options=None, prefilter=None):
    u, y, options = dataset(u, y, options)
    try:
        matrix, target = vrft_regression(
            u, y, model, basis, options, prefilter)
    except FloatingPointError:
        return failure(basis.parameter_count, Status.NON_FINITE)
    return solve_least_squares(matrix, target, options)


def vrft_optimize(u, y, model, basis, initial, lower, upper, options=None, prefilter=None):
    u, y, options = dataset(u, y, options)
    try:
        matrix, target = vrft_regression(
            u, y, model, basis, options, prefilter)
    except FloatingPointError:
        return failure(basis.parameter_count, Status.NON_FINITE)
    # Precompute the small quadratic: each PANOC iteration is O(P^2),
    # independent of the number of experimental samples.
    gram = matrix.T @ matrix / target.size + \
        options.ridge * np.eye(basis.parameter_count)
    linear = matrix.T @ target / target.size
    constant = float(target @ target / target.size)

    def cost_gradient(parameters):
        value = parameters @ gram @ parameters - 2 * linear @ parameters + constant
        return float(value), 2 * (gram @ parameters - linear)

    result = optimize(cost_gradient, basis.parameter_count,
                      initial, lower, upper, options)
    if result.status in (Status.SUCCESS, Status.ITERATION_LIMIT):
        residual = matrix @ result.parameters - target
        result.cost = float(residual @ residual / target.size
                            + options.ridge * (result.parameters @ result.parameters))
    return result


class VRFT:
    def __init__(self, reference_model, basis, prefilter=None):
        self.reference_model = reference_model
        self.basis = basis
        self.prefilter = prefilter

    def regression(self, u, y, options=None):
        return vrft_regression(u, y, self.reference_model, self.basis, options, self.prefilter)

    def fit_least_squares(self, u, y, options=None):
        return vrft_least_squares(u, y, self.reference_model, self.basis, options, self.prefilter)

    def fit_iterative(self, u, y, initial, lower, upper, options=None):
        return vrft_optimize(u, y, self.reference_model, self.basis, initial, lower, upper, options, self.prefilter)
