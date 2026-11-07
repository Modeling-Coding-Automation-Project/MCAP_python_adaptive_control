import numpy as np

from sample.data_driven.pi_tuning import run_demo


def test_tuned_closed_loop_matches_independent_reference_model():
    result = run_demo()
    initial_error = np.linalg.norm(result["trajectories"]["initial PI"] - result["desired"])
    assert initial_error > .1
    for name in result["fits"]:
        np.testing.assert_allclose(result["trajectories"][name], result["desired"], atol=2e-6)
