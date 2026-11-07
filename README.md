# Python データ駆動制御ライブラリ

NumPyで設計したVRFT・FRITです。対応するC++11実装は
[python_adaptive_control_to_cpp](https://github.com/Modeling-Coding-Automation-Project/python_adaptive_control_to_cpp)
の `data_driven/` にあります。

| 手法 | 解法 | 調整する制御器 |
|---|---|---|
| VRFT | NumPy直接最小二乗法 | `C(rho) = sum rho_i beta_i` |
| VRFT | MCAP PANOC反復最適化、上下限制約付き | 同上 |
| FRIT | MCAP PANOC反復最適化、解析勾配 | `C(rho) = sum rho_i beta_i` |
| FRIT | NumPy直接最小二乗法 | `C(rho)^-1 = sum rho_i beta_i` |

**FRIT最小二乗法のパラメータは逆制御器の係数です。通常のPI/PIDゲインとは異なります。**
逆制御器がパラメータに線形な場合を厳密に解きます。

## 準備と実行

Python 3.10以降、NumPyを使用します。コア実装はSciPyに依存しません。
このリポジトリのルートをカレントディレクトリまたは `PYTHONPATH` に指定してください。

```bash
git submodule update --init --recursive
python -m pip install -r requirements-dev.txt
python -m pytest -q
python sample/data_driven/pi_tuning.py
```

表示も行う場合は `requirements-plot.txt` をインストールし、サンプルに `--plot` を付けます。
描画には `sample/simulation_manager` を使用します。

```python
from data_driven import RationalFilter, LinearControllerBasis, vrft_least_squares

model = RationalFilter([0.0, 0.3], [1.0, -0.7])
basis = LinearControllerBasis.pi(sample_time=0.1)
result = vrft_least_squares(measured_input, measured_output, model, basis)
if result.success:
    kp, ki = result.parameters
```

係数は `z^-1` の昇べき順です。データは1次元配列で、同じ時刻の入力と出力を並べて渡します。
詳細な数式・制約・失敗時の扱いは [利用ガイド](document/data_driven.md) を参照してください。

## 構成

- `data_driven/`: フィルタ、VRFT、FRIT、既存PANOCへのアダプタ
- `tests/`: 既知解、解析勾配、異常系、閉ループ検証
- `sample/data_driven/`: 四方式を使ったPI調整例
- `external_libraries/MCAP_python_optimization/`: PANOC実装（サブモジュール）
- `.github/`: 単体テスト・サンプル実行、Dependabot

ライセンス: [MIT](LICENSE.txt)
