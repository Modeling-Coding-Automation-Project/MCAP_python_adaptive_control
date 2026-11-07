import numpy as np
import pytest

from data_driven import (
    FRIT, FRITLeastSquares, VRFT, LinearControllerBasis, RationalFilter,
    SolverOptions, Status, frit_cost_gradient, frit_least_squares,
    frit_optimize, frit_regression, vrft_least_squares, vrft_optimize,
    vrft_regression,
)


@pytest.fixture
def experiment():
    # A known plant is used only by tests to verify tuning independently.
    # G=.2 z^-1/(1-.8 z^-1), M=.3 z^-1/(1-.7 z^-1)
    # implies C=M/[G(1-M)]=(1.5-1.2z^-1)/(1-z^-1).
    u = np.random.default_rng(123).uniform(-1, 1, 256)
    y = np.zeros(u.size)
    for k in range(1, u.size):
        y[k] = .8 * y[k - 1] + .2 * u[k - 1]
    model = RationalFilter([0, .3], [1, -.7])
    basis = LinearControllerBasis.pi(.1)
    inverse_basis = LinearControllerBasis([[1, 0], [0, 1]], [1.5, -1.2])
    return u, y, model, basis, inverse_basis


def test_filter_impulse_scaling_reset_and_inverse():
    model = RationalFilter([0, 2], [4, -2])
    impulse = np.r_[1., np.zeros(8)]
    expected = np.r_[0., .5 ** np.arange(1, 9)]
    np.testing.assert_allclose(model.filter(impulse), expected)
    state = model.state()
    np.testing.assert_allclose([state.step(x) for x in impulse], expected)
    state.reset()
    np.testing.assert_allclose([state.step(x) for x in impulse], expected)
    invertible = RationalFilter([2, -.8], [1, -.6])
    np.testing.assert_allclose(invertible.inverse().filter(invertible.filter(impulse)), impulse,
                               atol=1e-15)
    with pytest.raises(ValueError, match="causal inverse"):
        model.inverse()


def test_pi_pid_realization():
    error = np.array([1., 2., -.5, .25])
    dt = .1
    expected = 2 * error + 3 * dt * \
        np.cumsum(error) + .2 * np.diff(np.r_[0., error]) / dt
    basis = LinearControllerBasis.pid(dt)
    np.testing.assert_allclose(basis.controller(
        [2, 3, .2]).filter(error), expected)


def test_vrft_exact_pi_recovery(experiment):
    u, y, model, basis, _ = experiment
    for discard in (0, 10):
        result = vrft_least_squares(
            u, y, model, basis, SolverOptions(discard_samples=discard))
        assert result.success and result.rank == 2 and result.iterations == 0
        np.testing.assert_allclose(result.parameters, [1.2, 3.], atol=2e-13)
        assert result.cost < 1e-25


def test_frit_exact_inverse_linear_recovery(experiment):
    u, y, model, _, inverse_basis = experiment
    result = frit_least_squares(u, y, model, inverse_basis)
    assert result.success
    np.testing.assert_allclose(result.parameters, [1, -1], atol=1e-13)
    # Returned coefficients describe C^-1, not the forward PI basis.
    controller = inverse_basis.controller(result.parameters).inverse()
    np.testing.assert_allclose(controller.numerator, [1.5, -1.2], atol=1e-13)
    np.testing.assert_allclose(controller.denominator, [1, -1], atol=1e-13)


@pytest.mark.parametrize("solver", [vrft_optimize, frit_optimize])
def test_iterative_exact_pi_recovery(solver, experiment):
    u, y, model, basis, _ = experiment
    # Coefficient recovery needs tighter accuracy than the deployment default;
    # PANOC tolerance bounds a scaled residual, not parameter error.
    result = solver(u, y, model, basis, [.9, 1.4], [.1, .1], [3, 6],
                    SolverOptions(tolerance=1e-8))
    assert result.success
    assert 0 < result.iterations < 100
    np.testing.assert_allclose(result.parameters, [1.2, 3], atol=3e-6)
    assert result.cost < 1e-13


@pytest.mark.parametrize("solver", [vrft_optimize, frit_optimize])
def test_default_tolerance_converges_and_reduces_objective(solver, experiment):
    u, y, model, basis, _ = experiment
    initial = np.array([.9, 1.4])
    if solver is vrft_optimize:
        matrix, target = vrft_regression(u, y, model, basis)
        initial_cost = np.mean((matrix @ initial - target)**2)
    else:
        initial_cost, _ = frit_cost_gradient(initial, u, y, model, basis)
    result = solver(u, y, model, basis, initial, [.1, .1], [3, 6])
    assert result.success and np.all(np.isfinite(result.parameters))
    assert result.cost < initial_cost * 1e-4
    assert result.iterations < SolverOptions().max_iterations


def test_analytic_frit_gradient_matches_central_difference(experiment):
    u, y, model, basis, _ = experiment
    parameters = np.array([.9, 1.4])
    options = SolverOptions(ridge=.013, discard_samples=11)
    _, gradient = frit_cost_gradient(parameters, u, y, model, basis, options)
    finite_difference = np.zeros(2)
    h = 1e-5
    for index in range(2):
        step = np.zeros(2)
        step[index] = h
        plus = frit_cost_gradient(
            parameters + step, u, y, model, basis, options)[0]
        minus = frit_cost_gradient(
            parameters - step, u, y, model, basis, options)[0]
        finite_difference[index] = (plus - minus) / (2 * h)
    np.testing.assert_allclose(
        gradient, finite_difference, atol=1e-10, rtol=2e-8)


def test_bounds_are_respected_and_cost_is_returned_point(experiment):
    u, y, model, basis, _ = experiment
    lower, upper = np.array([.1, .1]), np.array([1., 6.])
    result = vrft_optimize(u, y, model, basis, [.8, 1.4], lower, upper)
    assert result.success
    assert np.all(result.parameters >= lower) and np.all(
        result.parameters <= upper)
    assert result.parameters[0] == pytest.approx(1.)
    matrix, target = vrft_regression(u, y, model, basis)
    residual = matrix @ result.parameters - target
    assert result.cost == pytest.approx(np.mean(residual**2), abs=1e-16)


def test_ridge_regularization_matches_independent_normal_equations(experiment):
    u, y, model, basis, _ = experiment
    y = y + np.random.default_rng(42).normal(0, .01, y.size)
    options = SolverOptions(ridge=.002, discard_samples=7, tolerance=1e-8)
    matrix, target = vrft_regression(u, y, model, basis, options)
    expected = np.linalg.solve(matrix.T @ matrix / target.size + options.ridge * np.eye(2),
                               matrix.T @ target / target.size)
    result = vrft_least_squares(u, y, model, basis, options)
    iterative = vrft_optimize(
        u, y, model, basis, [.9, 1.4], [-10, -10], [10, 10], options)
    np.testing.assert_allclose(result.parameters, expected, atol=1e-13)
    np.testing.assert_allclose(iterative.parameters, expected, atol=1e-6)


def test_rank_deficient_data_and_ridge(experiment):
    u, y, model, basis, _ = experiment
    result = vrft_least_squares(np.zeros_like(
        u), np.zeros_like(y), model, basis)
    assert result.status == Status.RANK_DEFICIENT and result.rank == 0
    assert np.isnan(result.cost)
    regularized = vrft_least_squares(np.zeros_like(u), np.zeros_like(y), model, basis,
                                     SolverOptions(ridge=.001))
    assert regularized.success
    np.testing.assert_array_equal(regularized.parameters, [0, 0])


def test_discard_happens_after_filtering(experiment):
    u, y, model, basis, inverse_basis = experiment
    for regression, family in ((vrft_regression, basis), (frit_regression, inverse_basis)):
        matrix, target = regression(u, y, model, family)
        trimmed, trimmed_target = regression(
            u, y, model, family, SolverOptions(discard_samples=17))
        np.testing.assert_array_equal(trimmed, matrix[17:])
        np.testing.assert_array_equal(trimmed_target, target[17:])


def test_extra_vrft_prefilter(experiment):
    u, y, model, basis, _ = experiment
    prefilter = RationalFilter([.2], [1, -.8])
    matrix, target = vrft_regression(u, y, model, basis)
    filtered, filtered_target = vrft_regression(
        u, y, model, basis, prefilter=prefilter)
    np.testing.assert_allclose(filtered_target, prefilter.filter(target))
    np.testing.assert_allclose(filtered, np.column_stack(
        [prefilter.filter(c) for c in matrix.T]))
    result = vrft_least_squares(u, y, model, basis, prefilter=prefilter)
    np.testing.assert_allclose(result.parameters, [1.2, 3], atol=1e-12)


def test_iteration_limit_and_singular_inverse(experiment):
    u, y, model, basis, _ = experiment
    options = SolverOptions(max_iterations=1)
    result = frit_optimize(
        u, y, model, basis, [.9, 1.4], [.1, .1], [3, 6], options)
    assert result.status == Status.ITERATION_LIMIT and result.iterations == 1
    result = frit_optimize(u, y, model, basis, [0, 0], [0, 0], [3, 6])
    assert result.status == Status.SINGULAR_INVERSE


def test_failure_on_unstable_overflow(experiment):
    u, y, _, basis, _ = experiment
    unstable_model = RationalFilter([1], [1, -1e8])
    result = vrft_least_squares(u, y, unstable_model, basis)
    assert result.status == Status.NON_FINITE


@pytest.mark.parametrize("bad_options", [
    SolverOptions(
        ridge=-1), SolverOptions(tolerance=0), SolverOptions(max_iterations=0),
    SolverOptions(rank_tolerance=np.nan), SolverOptions(discard_samples=256),
    SolverOptions(discard_samples=1.5),
    SolverOptions(discard_samples=False), SolverOptions(max_iterations=True),
    SolverOptions(max_iterations=2.5), SolverOptions(tolerance="invalid"),
    SolverOptions(ridge=1j), SolverOptions(ridge=True),
])
def test_invalid_options(bad_options, experiment):
    u, y, model, basis, _ = experiment
    with pytest.raises(ValueError):
        vrft_least_squares(u, y, model, basis, bad_options)


def test_invalid_data_and_bounds(experiment):
    u, y, model, basis, _ = experiment
    for bad_u, bad_y in (([], []), (u[:-1], y), (u[:, None], y), (u, y * np.nan)):
        with pytest.raises(ValueError):
            vrft_least_squares(bad_u, bad_y, model, basis)
    with pytest.raises(ValueError):
        vrft_optimize(u, y, model, basis, [.9, 1.4], [2, 2], [1, 1])
    with pytest.raises(ValueError):
        vrft_optimize(u, y, model, basis, [.9, 1.4], [1, 2], [3, 6])


def test_object_api(experiment):
    u, y, model, basis, inverse_basis = experiment
    assert VRFT(model, basis).fit_least_squares(u, y).success
    assert FRITLeastSquares(
        model, inverse_basis).fit_least_squares(u, y).success
    assert FRIT(model, basis).fit_iterative(
        u, y, [.9, 1.4], [.1, .1], [3, 6]).success


def test_mutated_options_are_revalidated(experiment):
    u, y, model, basis, _ = experiment
    options = SolverOptions()
    assert vrft_least_squares(u, y, model, basis, options).success
    options.tolerance = np.inf
    with pytest.raises(ValueError, match="finite real"):
        vrft_least_squares(u, y, model, basis, options)


def test_finite_but_unrepresentable_inverse_reports_numerical_failure():
    model = RationalFilter([1.], [1.])
    basis = LinearControllerBasis([[1.]], [1.])
    result = frit_optimize([1., 2.], [.2, .4], model, basis,
                           [1e-320], [1e-320], [1.])
    assert result.status == Status.NON_FINITE
