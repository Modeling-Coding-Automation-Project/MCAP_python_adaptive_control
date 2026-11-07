# VRFT・FRIT 利用ガイド

## 対象と前提

対象は離散時間SISOの線形時不変系です。1回の実験の入力 `u[k]` と出力 `y[k]`、
目標閉ループモデル `M`、制御器の基底を使用します。推定関数にはプラントモデルを渡しません。
全フィルタはゼロ初期状態から計算します。データ取得時の初期状態・オフセットがある場合は、
事前処理と十分な助走を行い、`discard_samples` で初期区間を評価から外してください。
`discard_samples` は全区間をフィルタ処理した後に適用されます。
任意の非ゼロ初期状態を厳密に補償する機能ではありません。

`RationalFilter(b, a)` は `sum b[i] z^-i / sum a[i] z^-i` です。
`a[0] != 0` が必要です。`filter()` は毎回ゼロ状態、`state().step()` は状態を保持して1点ずつ計算します。
`LinearControllerBasis` は共通分母と複数の分子を保持します。
`pi(dt)` は `Kp + Ki*dt/(1-z^-1)`、`pid(dt)` はこれに `Kd*(1-z^-1)/dt` を加えます。
微分項のノイズ対策が必要な場合は、フィルタを含めた共通分母の基底を明示的に指定してください。

## 四つの計算法

### VRFT（直接最小二乗法・反復最適化）

前向き制御器を `C(rho)=sum rho_i beta_i` とします。
通常の仮想偏差は `(M^-1-I)y` ですが、デフォルトのプレフィルタを `L=M` とし、
`d=M u`、`X_i=beta_i (I-M)y` を直接構成します。
これにより遅れを持つ目標モデルでも `M^-1` を計算せずに使えます。
`prefilter=H` を追加指定すると実際のプレフィルタは `L=HM` になり、両辺に `H` を作用させます。
任意の目標・プラント・ノイズ条件について最適なプレフィルタであるとは限りません。

`vrft_least_squares` はNumPyのSVD最小二乗解を一度求めます。
`vrft_optimize` は同じ評価関数をPANOCで反復的に最小化し、係数の上下限制約を扱います。
反復前に `X.T X` と `X.T d` を計算するため、各反復はデータ長に依存しない `O(P^2)` です。

### FRIT（反復最適化）

前向き制御器 `C(rho)=B(rho)/A` に対し、擬似参照 `r_tilde=C(rho)^-1 u+y` を構成し、
`e=y-M r_tilde` の二乗平均を最小化します。
解析勾配には `v=C^-1u` と `dv/drho_i=-(B_i/B)v` を使用します。
有限差分で勾配を近似する必要はありません。

逆フィルタを計算するため、`B(rho)[0] != 0` が必要です。
上下限は因果的で安定した逆フィルタを持つ探索範囲に選び、ゼロゲインから開始しないでください。
一般の基底に対して制約範囲全体の逆フィルタ安定性を自動証明する機能はありません。
有限データで計算できても、推定制御器の閉ループ安定性は別途検証が必要です。
PANOCは非凸FRITに対して大域的最適解を保証しません。

### FRIT（直接最小二乗法）

逆制御器を `C(rho)^-1=sum rho_i beta_i` と線形パラメータ化します。
`d=(I-M)y`、`X_i=M beta_i u` に対する最小二乗問題です。
制御器の分子を固定し、分母の係数を調整する場合を厳密に解きます。
これは通常のPI/PIDゲインに対するFRIT目的関数を線形化したものではありません。
推定後は `inverse_basis.controller(result.parameters).inverse()` により前向き制御器を構成します。
この操作にも先頭係数が非ゼロであることが必要です。

## APIと結果

```python
options = SolverOptions(ridge=0.0, tolerance=2e-5, max_iterations=500,
                        rank_tolerance=1e-10, discard_samples=0)

vrft_least_squares(u, y, model, basis, options, prefilter=None)
vrft_optimize(u, y, model, basis, initial, lower, upper, options, prefilter=None)
frit_least_squares(u, y, model, inverse_basis, options)
frit_optimize(u, y, model, basis, initial, lower, upper, options)
```

同じ操作を `VRFT`、`FRIT`、`FRITLeastSquares` クラスからも呼べます。
反復計算では有限な上下限と、その範囲内の初期値を指定します。
`FitResult` には `parameters`、`status`、`cost`、`iterations`、`rank` があります。
評価値は、使用するデータ数を `n` として `||residual||^2/n + ridge*||rho||^2` です。
正則化の強さはデータ数に依存しません。直接法は `X/sqrt(n)` と `sqrt(ridge) I` を
積み重ねた行列をSVDで解くため、正規方程式の明示的な逆行列を作りません。

`status` は `success`、`iteration_limit`、`rank_deficient`、`non_finite`、
`singular_inverse` です。不正な引数や係数は `ValueError` になります。
`invalid_argument` は対応するC++ APIとの共通定義として存在します。
階数不足はゼロのパラメータとNaNの評価値を返します。正則化を指定した場合の `rank` は
正則化行を含む行列の階数です。反復法の `rank` は未計算を示す0です。
`iteration_limit` の候補は返しますが、収束済みとして扱わないでください。
上流PANOCの終了評価値は返却点と異なる場合があるため、実際の返却係数で評価値と射影残差を再確認します。

## 数値計算と検証

Python版はNumPy float64による基準実装です。C++版は固定長配列と
`python_numpy_to_cpp` の既存QRを用います。反復ソルバーの既定許容値は `2e-5` です。
これは射影残差の停止条件で、係数の絶対誤差ではありません。厳密な係数回復が
必要な場合は、明示的に小さい許容値を指定し、実際の応答も確認してください。
通常の全階数問題で解は一致しますが、特異値とQR対角成分の違いにより、閾値付近の階数判定は一致を保証しません。
データの励起不足、過大なスケール差、測定ノイズ、目標モデルの実現可能性を確認してください。
入力・出力の同時刻対応やサンプリング周期を誤ると、両言語が同じ結果でも正しい制御器にはなりません。

`tests/` は独立に既知解を導出できるプラント、解析勾配と中心差分、上下限制約、
正則化、階数不足、遅れ、初期区間除去、オーバーフロー、閉ループ応答を検証します。
C++とのSILは親リポジトリの `test_sil/` で行います。

方式の対応: ローカル参考資料『データ駆動制御入門』第3章（FRIT）、4.3節（VRFT最小二乗法）、
4.5.1節（逆制御器を線形パラメータ化したFRIT）。参考資料のPDFや本文は本リポジトリに含めません。
