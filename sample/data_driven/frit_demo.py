"""FRIT controller design and closed-loop response comparison.

Tune controllers from simulated plant input/output data using least squares
and PANOC, then compare their responses with the initial controller and the
reference model. Use --no-plot to run without plotting.
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
        "excitation_seed": 0x12345678,
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

    # 1. Collect input/output data from the plant starting at zero state.
    # For a physical system, replace this section with measured input u and output y.
    u = np.zeros(design["sample_count"])
    y = np.zeros(design["sample_count"])
    random_state = design["excitation_seed"]
    plant_state = 0.0
    for index in range(u.size):
        # Generate reproducible excitation using a linear congruential generator.
        random_state = (1664525 * random_state + 1013904223) & 0xFFFFFFFF
        u[index] = (random_state >> 16) / 32768.0 - 1.0
        y[index] = plant_state
        plant_state = (design["plant_a"] * plant_state
                       + design["plant_b"] * u[index])

    # 2. Fit parameters using input/output data, the reference model, and the basis.
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

    return {"design": design, "fits": fits, "u": u, "y": y,
            "time": design["sample_time"] * np.arange(reference.size),
            "reference": reference, "desired": desired,
            "trajectories": trajectories, "controls": controls}


def plot_demo(result):
    """Plot the recorded input/output data, tracking responses, and control inputs."""
    from sample.simulation_manager.visualize.simulation_plotter_dash import (
        SimulationPlotterDash,
    )

    plotter = SimulationPlotterDash(activate_dump=False)
    time = result["time"].reshape(-1, 1)
    outputs = {"reference": result["reference"], "desired": result["desired"],
               **result["trajectories"]}
    for name, signal in outputs.items():
        plotter.append_sequence_name(signal.reshape(-1, 1), name)
        plotter.assign(name, position=(0, 0), x_sequence=time,
                       x_sequence_name="time [s]", label=name)
    for name, signal in result["controls"].items():
        signal_name = "control: " + name
        plotter.append_sequence_name(signal.reshape(-1, 1), signal_name)
        plotter.assign(signal_name, position=(1, 0), x_sequence=time,
                       x_sequence_name="time [s]", label=signal_name)
    experiment_time = (result["design"]["sample_time"]
                       * np.arange(result["u"].size)).reshape(-1, 1)
    for row, (name, signal) in enumerate((
            ("recorded input", result["u"]), ("recorded output", result["y"]))):
        plotter.append_sequence_name(signal.reshape(-1, 1), name)
        plotter.assign(name, position=(row, 1), x_sequence=experiment_time,
                       x_sequence_name="experiment time [s]", label=name)
    plotter.plot("FRIT controller design")
    return plotter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true",
                        help="skip HTML response plotting")
    args = parser.parse_args()

    design = create_design()
    result = run_demo(design)
    print(f"FRIT experiment: {design['sample_count']} input/output samples, "
          f"sample time={design['sample_time']} s")
    for name, fit in result["fits"].items():
        print(f"{name}: parameters={fit.parameters}, status={fit.status.value}, "
              f"cost={fit.cost:.10g}, iterations={fit.iterations}")
    if not args.no_plot:
        plot_demo(result)


if __name__ == "__main__":
    main()
