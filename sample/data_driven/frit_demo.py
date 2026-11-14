"""FRIT controller design and closed-loop response comparison.

Record a unit-step experiment in closed loop with the initial PI controller,
tune offline using that fixed dataset, then compare initial and tuned responses
with the reference model. Use --no-plot to run without plotting.
"""
import argparse
from pathlib import Path
import sys

import numpy as np


# Allow direct execution to import modules from this repository.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_driven import (
    FRIT,
    FRITLeastSquares,
    LinearControllerBasis,
    RationalFilter,
    SolverOptions,
)
from sample.simulation_manager.visualize.simulation_plotter_dash import (
    SimulationPlotterDash,
)


def create_design():
    """Define the experiment, reference model, controller basis, and solver settings."""
    sample_time = 0.1
    # Reference model: M(z) = 0.3*z^-1 / (1 - 0.7*z^-1).
    model = RationalFilter([0.0, 0.3], [1.0, -0.7])
    # PI controller parameters: [proportional gain Kp, integral gain Ki].
    basis = LinearControllerBasis.pi(sample_time)
    # Least squares parameterizes the inverse controller C^-1; its coefficients
    # are not PI gains.
    inverse_basis = LinearControllerBasis(
        [[1.0, 0.0], [0.0, 1.0]], [1.5, -1.2])
    return {
        "sample_time": sample_time,
        "sample_count": 128,
        "simulation_count": 100,
        # Simulated plant: x(k+1) = 0.8*x(k) + 0.2*u(k), y(k) = x(k).
        # Plant coefficients are used only for data collection and response evaluation,
        # not for controller tuning.
        "plant_a": 0.8,
        "plant_b": 0.2,
        "model": model,
        "basis": basis,
        "inverse_basis": inverse_basis,
        "initial": np.array([0.7, 1.0]),
        "lower": np.array([0.1, 0.1]),
        "upper": np.array([3.0, 6.0]),
        "options": SolverOptions(
            tolerance=2e-5, max_iterations=1000),
    }


def run_demo(design=None):
    """Tune controllers from input/output data and return their unit-step responses."""
    design = create_design() if design is None else design
    model, basis, options = design["model"], design["basis"], design["options"]

    # 1. Run a closed-loop unit-step experiment with the fixed initial PI.
    # For a physical system, replace this section with recorded r_ini, u_ini, y_ini.
    # The controller is not tuned during this experiment.
    experiment_reference = np.ones(design["sample_count"])
    u = np.zeros(design["sample_count"])
    y = np.zeros(design["sample_count"])
    initial_controller = basis.controller(design["initial"]).state()
    plant_state = 0.0
    for index, command in enumerate(experiment_reference):
        y[index] = plant_state
        u[index] = initial_controller.step(command - y[index])
        plant_state = (design["plant_a"] * plant_state
                       + design["plant_b"] * u[index])

    # 2. Fit parameters using input/output data, the reference model, and the basis.
    # Reuse the same recorded data in every optimization step; no plant is run.
    # Compare least squares with iterative PANOC optimization under parameter bounds.
    inverse_basis = design["inverse_basis"]
    direct_tuner = FRITLeastSquares(model, inverse_basis)
    tuner = FRIT(model, basis)
    direct = direct_tuner.fit_least_squares(u, y, options)
    iterative = tuner.fit_iterative(
        u, y, design["initial"], design["lower"], design["upper"], options)
    fits = {"FRIT inverse least squares": direct, "FRIT PANOC": iterative}
    if not all(fit.success for fit in fits.values()):
        raise RuntimeError("Tuning failed: " + ", ".join(
            f"{name}={fit.status.value}" for name, fit in fits.items()))

    # 3. Build the initial and tuned controllers.
    # Invert the least-squares inverse controller C^-1 to obtain controller C.
    direct_controller = inverse_basis.controller(direct.parameters).inverse()
    controllers = {
        "initial PI": basis.controller(design["initial"]),
        "FRIT inverse least squares": direct_controller,
        "FRIT PANOC": basis.controller(iterative.parameters),
    }

    # 4. Simulate each closed loop independently from zero state.
    # At each step, measure y(k), compute u(k) from the error, and update x(k+1).
    reference = np.ones(design["simulation_count"])
    desired = np.zeros(reference.size)
    trajectories = {name: np.zeros(reference.size) for name in controllers}
    controls = {name: np.zeros(reference.size) for name in controllers}
    controller_states = {name: controller.state()
                         for name, controller in controllers.items()}
    plant_states = {name: 0.0 for name in controllers}
    model_state = model.state()
    for index, command in enumerate(reference):
        desired[index] = model_state.step(command)
        for name, controller_state in controller_states.items():
            measured_output = plant_states[name]
            control = controller_state.step(command - measured_output)
            trajectories[name][index] = measured_output
            controls[name][index] = control
            plant_states[name] = (design["plant_a"] * measured_output
                                  + design["plant_b"] * control)

    experiment_time = design["sample_time"] * np.arange(u.size)
    return {"design": design, "fits": fits, "controllers": controllers,
            "u": u, "y": y,
            "experiment_time": experiment_time,
            "experiment_reference": experiment_reference,
            "time": design["sample_time"] * np.arange(reference.size),
            "reference": reference, "desired": desired,
            "trajectories": trajectories, "controls": controls}


def plot_demo(result):
    """Plot the recorded input/output data, tracking responses, and control inputs."""

    plotter = SimulationPlotterDash(activate_dump=False)

    def add_signals(signals, time, row):
        for name, signal in signals.items():
            plotter.append_sequence_name(signal.reshape(-1, 1), name)
            plotter.assign(name, position=(row, 0), x_sequence=time.reshape(-1, 1),
                           x_sequence_name="time [s]", label=name)

    # Separate pages make the acquisition, offline fitting, and validation explicit.
    experiment_time = result["experiment_time"]
    add_signals({"experiment reference r_ini": result["experiment_reference"],
                 "recorded output y_ini (initial PI)": result["y"]}, experiment_time, 0)
    add_signals(
        {"recorded input u_ini (initial PI)": result["u"]}, experiment_time, 1)
    plotter.pre_plot("1. Acquire data - closed loop with fixed initial PI")

    design = result["design"]
    model = design["model"]
    controllers = result["controllers"]
    # FRIT evaluates y_ini - M(C^-1 u_ini + y_ini) on the recorded data.
    offline_signals = {"recorded output y_ini": result["y"]}
    for name, controller in controllers.items():
        fictitious_reference = controller.inverse().filter(
            result["u"]) + result["y"]
        offline_signals[f"M r_tilde: {name}"] = model.filter(
            fictitious_reference)
    add_signals(offline_signals, experiment_time, 0)
    plotter.pre_plot("2. Offline FRIT tuning - reuse recorded u_ini, y_ini")

    time = result["time"]
    add_signals({"validation reference r": result["reference"],
                 "desired output M r": result["desired"],
                 **result["trajectories"]}, time, 0)
    add_signals({f"control u: {name}": signal
                 for name, signal in result["controls"].items()}, time, 1)
    plotter.pre_plot(
        "3. Validate - initial vs tuned closed-loop step responses")
    plotter.plot("FRIT - closed-loop experiment, offline tuning, validation")
    return plotter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true",
                        help="skip HTML response plotting")
    args = parser.parse_args()

    design = create_design()
    result = run_demo(design)
    print("FRIT: closed-loop experiment -> offline tuning -> validation")
    print("\n[1] Collect data with the fixed initial PI controller")
    print(f"  Kp={design['initial'][0]:.6g}, Ki={design['initial'][1]:.6g}; "
          f"unit-step reference, {design['sample_count']} samples, "
          f"Ts={design['sample_time']} s; zero initial states")
    print("  Recorded u_ini and y_ini from u_ini = C_ini(r_ini - y_ini).")
    print("\n[2] Tune offline using only the same recorded u_ini, y_ini")
    print("  r_tilde(rho) = C(rho)^-1 u_ini + y_ini; minimize mean((y_ini - M r_tilde)^2).")
    for name, fit in result["fits"].items():
        parameter_label = ("inverse-controller coefficients" if "inverse" in name
                           else "PI gains [Kp, Ki]")
        print(f"  {name}: {parameter_label}={fit.parameters}, "
              f"status={fit.status.value}, cost={fit.cost:.10g}, "
              f"iterations={fit.iterations}")
    print("\n[3] Validate each controller in a fresh closed-loop unit-step simulation")
    print(f"  {design['simulation_count']} samples; zero initial states; "
          "tracking RMSE relative to M r:")
    for name, output in result["trajectories"].items():
        rmse = np.sqrt(np.mean((output - result["desired"]) ** 2))
        print(f"  {name}: RMSE={rmse:.6g}")
    if not args.no_plot:
        plot_demo(result)


if __name__ == "__main__":
    main()
