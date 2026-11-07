import numpy as np
import pytest

from sample.data_driven.vrft_demo import run_demo as run_vrft
from sample.data_driven.frit_demo import run_demo as run_frit


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
