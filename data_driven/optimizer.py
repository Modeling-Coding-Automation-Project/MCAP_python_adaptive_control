"""Small adapter to the upstream MCAP Python PANOC implementation."""
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys

import numpy as np

from .common import FitResult, Status, failure, vector


class SingularInverseError(ArithmeticError):
    pass


@lru_cache(maxsize=1)
def _panoc_module():
    # Import the exact pinned dependency without changing global sys.path or
    # accidentally selecting a different checkout named python_optimization.
    path = (Path(__file__).resolve().parents[1] / "external_libraries" /
            "MCAP_python_optimization" / "python_optimization" / "panoc.py")
    if not path.is_file():
        raise ImportError("PANOC submodule is missing; run git submodule update --init --recursive")
    name = "_mcap_adaptive_control_panoc"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def optimize(cost_gradient, parameter_count, initial, lower, upper, options):
    initial = vector(initial, "initial", parameter_count)
    lower = vector(lower, "lower", parameter_count)
    upper = vector(upper, "upper", parameter_count)
    if np.any(lower > upper):
        raise ValueError("lower bounds must not exceed upper bounds")
    if np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("initial parameters must satisfy bounds")
    panoc = _panoc_module()
    last_parameters, last_value = None, None

    def evaluate(parameters):
        nonlocal last_parameters, last_value
        flat = parameters.reshape(-1)
        if not np.all(np.isfinite(flat)):
            raise FloatingPointError("trial parameters are not finite")
        if last_parameters is None or not np.array_equal(last_parameters, flat):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                value, gradient = cost_gradient(flat)
            if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                raise FloatingPointError("objective or gradient is not finite")
            last_parameters = flat.copy()
            last_value = (float(value), np.asarray(gradient).reshape(-1, 1))
        return last_value

    parameters = initial.copy().reshape(-1, 1)
    cache = panoc.PANOC_Cache(parameter_count, options.tolerance, 5)
    solver = panoc.PANOC_Optimizer(lambda p: evaluate(p)[0], lambda p: evaluate(p)[1],
                                   cache, lower.reshape(-1, 1), upper.reshape(-1, 1),
                                   max_iteration=options.max_iterations)
    try:
        status = solver.solve(parameters)
        # Upstream status.cost_value can refer to the pre-projection point.
        # Always evaluate the actual returned feasible parameters again.
        cost, gradient = evaluate(parameters)
    except SingularInverseError:
        return failure(parameter_count, Status.SINGULAR_INVERSE, cache.iteration)
    except (FloatingPointError, OverflowError):
        return failure(parameter_count, Status.NON_FINITE, cache.iteration)
    parameters = parameters.reshape(-1).copy()
    if status.exit_status == panoc.ExitStatus.NOT_FINITE_COMPUTATION:
        return failure(parameter_count, Status.NON_FINITE, status.number_of_iteration)
    if np.any(parameters < lower) or np.any(parameters > upper):
        return failure(parameter_count, Status.NON_FINITE, status.number_of_iteration)
    final_status = Status.SUCCESS if status.has_converged() else Status.ITERATION_LIMIT
    projected = np.clip(parameters - cache.gamma * gradient.reshape(-1), lower, upper)
    if np.linalg.norm(parameters - projected) > 10 * options.tolerance:
        final_status = Status.ITERATION_LIMIT
    return FitResult(parameters, final_status, cost, status.number_of_iteration)
