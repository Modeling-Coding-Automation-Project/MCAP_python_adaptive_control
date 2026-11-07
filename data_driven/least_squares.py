"""Direct, noniterative least squares (no nonlinear optimizer)."""
import numpy as np

from .common import FitResult, Status, failure


def solve_least_squares(matrix, target, options):
    count, size = matrix.shape
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(target)):
        return failure(size, Status.NON_FINITE)
    scaled_matrix = matrix / np.sqrt(count)
    scaled_target = target / np.sqrt(count)
    if options.ridge > 0:
        scaled_matrix = np.vstack(
            (scaled_matrix, np.sqrt(options.ridge) * np.eye(size)))
        scaled_target = np.concatenate((scaled_target, np.zeros(size)))
    try:
        parameters, _, rank, _ = np.linalg.lstsq(scaled_matrix, scaled_target,
                                                 rcond=options.rank_tolerance)
    except np.linalg.LinAlgError:
        return failure(size, Status.NON_FINITE)
    if rank < size:
        return failure(size, Status.RANK_DEFICIENT, rank=int(rank))
    residual = matrix @ parameters - target
    cost = float(residual @ residual / count +
                 options.ridge * (parameters @ parameters))
    if not np.isfinite(cost) or not np.all(np.isfinite(parameters)):
        return failure(size, Status.NON_FINITE)
    return FitResult(parameters, Status.SUCCESS, cost, rank=int(rank))
