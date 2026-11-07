"""Tune a PI controller from one experiment, then verify its step response.

Run: python sample/data_driven/pi_tuning.py [--plot]
The plant is known only to this demonstration's data acquisition and validation;
none of the four tuning functions receives a plant model.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_driven import (  # noqa: E402
    LinearControllerBasis, RationalFilter, SolverOptions,
    frit_least_squares, frit_optimize, vrft_least_squares, vrft_optimize,
)


def closed_loop(controller, reference):
    """Plant G=.2 z^-1/(1-.8 z^-1), with output sampled before input."""
    state = controller.state()
    response = np.zeros(len(reference))
    output = 0.0
    for index, command in enumerate(reference):
        response[index] = output
        control = state.step(command - output)
        output = .8 * output + .2 * control
    return response


def run_demo():
    sample_time = .1
    # 1. Record one persistently exciting input/output experiment.
    u = np.random.default_rng(2026).uniform(-1., 1., 512)
    plant = RationalFilter([0., .2], [1., -.8])
    y = plant.filter(u)

    # 2. Select desired response M and PI gains [Kp, Ki].
    model = RationalFilter([0., .3], [1., -.7])
    basis = LinearControllerBasis.pi(sample_time)
    initial = np.array([.7, 1.0])
    lower, upper = np.array([.1, .1]), np.array([3., 6.])
    options = SolverOptions(tolerance=1e-9)

    # 3. Tune with direct LS or the existing MCAP PANOC optimizer.
    fits = {
        "VRFT least squares": vrft_least_squares(u, y, model, basis, options),
        "VRFT PANOC": vrft_optimize(u, y, model, basis, initial, lower, upper, options),
        "FRIT PANOC": frit_optimize(u, y, model, basis, initial, lower, upper, options),
    }
    # Exact noniterative FRIT uses a different controller family: the inverse
    # controller is linear in rho. Here C^-1=(rho0+rho1*z^-1)/(1.5-1.2*z^-1).
    # Those coefficients are not [Kp, Ki]; the expected result is [1, -1].
    inverse_basis = LinearControllerBasis([[1., 0.], [0., 1.]], [1.5, -1.2])
    fits["FRIT inverse least squares"] = frit_least_squares(u, y, model, inverse_basis, options)
    if not all(result.success for result in fits.values()):
        raise RuntimeError("At least one tuning method failed; inspect FitResult.status")

    # 4. Independently simulate the tuned controller in feedback.
    reference = np.ones(100)
    desired = model.filter(reference)
    trajectories = {"initial PI": closed_loop(basis.controller(initial), reference)}
    for name, result in fits.items():
        controller = (inverse_basis.controller(result.parameters).inverse()
                      if name == "FRIT inverse least squares" else basis.controller(result.parameters))
        trajectories[name] = closed_loop(controller, reference)
    return {"fits": fits, "time": sample_time*np.arange(reference.size),
            "reference": reference, "desired": desired, "trajectories": trajectories}


def plot_demo(result):
    # Optional UI dependency; no plotting imports are needed for tuning or CI.
    from sample.simulation_manager.visualize.simulation_plotter_matplotlib import SimulationPlotterMatplotlib
    import matplotlib.pyplot as plt
    plotter = SimulationPlotterMatplotlib(activate_dump=False)
    signals = {"desired": result["desired"], **result["trajectories"]}
    for name, signal in signals.items():
        plotter.append_sequence_name(signal.reshape(-1, 1), name)
        plotter.assign(name, position=(0, 0), row=0, column=0,
                       x_sequence=result["time"].reshape(-1, 1), label=name)
    plotter.plot("Data-driven PI tuning")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="plot with simulation_manager")
    args = parser.parse_args()
    result = run_demo()
    for name, fit in result["fits"].items():
        rmse = np.sqrt(np.mean((result["trajectories"][name] - result["desired"])**2))
        print(f"{name}: rho={fit.parameters}, status={fit.status.value}, "
              f"cost={fit.cost:.3e}, iterations={fit.iterations}, step RMSE={rmse:.3e}")
    initial_rmse = np.sqrt(np.mean((result["trajectories"]["initial PI"] - result["desired"])**2))
    print(f"Initial PI step RMSE: {initial_rmse:.3e}")
    if args.plot:
        plot_demo(result)


if __name__ == "__main__":
    main()
