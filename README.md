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
python -m pip install -r requirements-dev.txt -r requirements-plot.txt
python -m pytest -q
python sample/data_driven/vrft_demo.py
python sample/data_driven/frit_demo.py
```

通常実行では `SimulationPlotterDash` で応答をHTML表示します。先に `requirements-plot.txt` をインストールしてください。
自動テストなどで表示が不要な場合だけ `--no-plot` を指定します。
描画には、このリポジトリ内の `sample/simulation_manager` を使用します。
ブラウザが起動できない環境では、表示されたHTMLファイルのパスを開いてください。

GitHub Actionsで非公開サブモジュールを取得する場合は、このリポジトリのActions Secretに
`MCAP_SUBMODULE_TOKEN` を設定してください。再帰的に参照する各リポジトリへの読み取り権限を持つ
fine-grained PATまたはGitHub Appのトークンを指定します。未設定の場合は `github.token` を使いますが、
別の非公開リポジトリへのアクセスはできません。トークンをコードや設定ファイルへ直接記載しないでください。

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
- `sample/data_driven/`: VRFT・FRITを別々に実行するPI調整例
- `external_libraries/MCAP_python_optimization/`: PANOC実装（サブモジュール）
- `.github/`: 単体テスト・サンプル実行、Dependabot

ライセンス: [MIT](LICENSE.txt)
