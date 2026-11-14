import numpy as np
import pytest

from sample.data_driven.vrft_demo import run_demo as run_vrft
from sample.data_driven.frit_demo import run_demo as run_frit


@pytest.mark.parametrize("run_demo", [run_vrft, run_frit], ids=["vrft", "frit"])
def test_tuning_data_is_recorded_from_initial_closed_loop(run_demo):
    result = run_demo()
    design = result["design"]
    reference, u, y = result["experiment_reference"], result["u"], result["y"]
    np.testing.assert_array_equal(reference, np.ones(design["sample_count"]))

    # Check the PI law independently, including its zero-state integral term.
    error = reference - y
    kp, ki = design["initial"]
    expected_u = kp * error + ki * design["sample_time"] * np.cumsum(error)
    np.testing.assert_allclose(u, expected_u, atol=1e-14)
    assert y[0] == 0.0
    np.testing.assert_allclose(
        y[1:], design["plant_a"] * y[:-1] + design["plant_b"] * u[:-1])

    # Validation restarts at zero, reproducing the initial experiment's prefix.
    count = min(design["sample_count"], design["simulation_count"])
    np.testing.assert_allclose(result["trajectories"]["initial PI"][:count], y[:count])
    np.testing.assert_allclose(result["controls"]["initial PI"][:count], u[:count])


@pytest.mark.parametrize("run_demo", [run_vrft, run_frit], ids=["vrft", "frit"])
def test_tuned_closed_loop_matches_independent_reference_model(run_demo):
    result = run_demo()
    assert len(result["fits"]) == 2
    initial_error = np.linalg.norm(
        result["trajectories"]["initial PI"] - result["desired"])
    assert initial_error > .1
    for name, fit in result["fits"].items():
        assert fit.success
        np.testing.assert_allclose(
            result["trajectories"][name], result["desired"], atol=1e-4)
