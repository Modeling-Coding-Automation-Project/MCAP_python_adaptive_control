"""VRFTによる制御器設計と閉ループ応答の比較。

模擬プラントの入出力データから最小二乗法とPANOCで制御器を調整し、
初期制御器の応答および規範モデルの応答と比較する。
描画せずに実行する場合は --no-plot を指定する。
"""
import argparse
from pathlib import Path
import sys

import numpy as np


# このファイルを直接実行する場合も、リポジトリ内のモジュールを参照する。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_driven import (
    VRFT,
    LinearControllerBasis,
    RationalFilter,
    SolverOptions,
)


def create_design():
    """実験条件、規範モデル、制御器の構造と最適化条件を設定する。"""
    sample_time = 0.1
    # 規範モデル M(z) = 0.3*z^-1 / (1 - 0.7*z^-1)。
    model = RationalFilter([0.0, 0.3], [1.0, -0.7])
    # PI制御器のパラメータは [比例ゲイン Kp, 積分ゲイン Ki]。
    basis = LinearControllerBasis.pi(sample_time)

    return {
        "sample_time": sample_time,
        "sample_count": 128,
        "simulation_count": 100,
        "excitation_seed": 0x12345678,
        # 模擬プラント: x(k+1) = 0.8*x(k) + 0.2*u(k), y(k) = x(k)。
        # プラント係数はデータ収集と応答評価にのみ使い、制御器の調整には渡さない。
        "plant_a": 0.8,
        "plant_b": 0.2,
        "model": model,
        "basis": basis,
        "initial": np.array([0.7, 1.0]),
        "lower": np.array([0.1, 0.1]),
        "upper": np.array([3.0, 6.0]),
        "options": SolverOptions(
            tolerance=2e-5, max_iterations=1000),
    }


def run_demo(design=None):
    """入出力データから制御器を調整し、単位ステップ応答を返す。"""
    design = create_design() if design is None else design
    model, basis, options = design["model"], design["basis"], design["options"]

    # 1. 初期状態ゼロのプラントから入出力データを収集する。
    # 実機で設計する場合は、この部分を計測済みの入力 u と出力 y に置き換える。
    u = np.zeros(design["sample_count"])
    y = np.zeros(design["sample_count"])
    random_state = design["excitation_seed"]
    plant_state = 0.0
    for index in range(u.size):
        # 線形合同法で再現可能な励振入力を生成する。
        random_state = (1664525 * random_state + 1013904223) & 0xFFFFFFFF
        u[index] = (random_state >> 16) / 32768.0 - 1.0
        y[index] = plant_state
        plant_state = (design["plant_a"] * plant_state
                       + design["plant_b"] * u[index])

    # 2. 入出力データ、規範モデル、制御器の基底からパラメータを求める。
    # 最小二乗法と、上下限制約付きのPANOCによる反復最適化を比較する。
    tuner = VRFT(model, basis)
    direct = tuner.fit_least_squares(u, y, options)
    iterative = tuner.fit_iterative(
        u, y, design["initial"], design["lower"], design["upper"], options)
    fits = {"VRFT least squares": direct, "VRFT PANOC": iterative}
    if not all(fit.success for fit in fits.values()):
        raise RuntimeError("Tuning failed: " + ", ".join(
            f"{name}={fit.status.value}" for name, fit in fits.items()))

    # 3. 初期制御器と、調整後の制御器を組み立てる。
    controllers = {
        "initial PI": basis.controller(design["initial"]),
        "VRFT least squares": basis.controller(direct.parameters),
        "VRFT PANOC": basis.controller(iterative.parameters),
    }

    # 4. 各制御器の閉ループを初期状態ゼロから独立にシミュレーションする。
    # 各時刻で y(k) を測定し、偏差から u(k) を計算して x(k+1) を更新する。
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
    """収集した入出力データ、目標追従応答、制御入力を描画する。"""
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
    plotter.plot("VRFT controller design")
    return plotter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true",
                        help="応答のHTML描画を省略する")
    args = parser.parse_args()

    design = create_design()
    result = run_demo(design)
    print(f"VRFT experiment: {design['sample_count']} input/output samples, "
          f"sample time={design['sample_time']} s")
    for name, fit in result["fits"].items():
        print(f"{name}: parameters={fit.parameters}, status={fit.status.value}, "
              f"cost={fit.cost:.10g}, iterations={fit.iterations}")
    if not args.no_plot:
        plot_demo(result)


if __name__ == "__main__":
    main()
