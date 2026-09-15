# Krusell=Smith 家計ブロックの Sequence Space Jacobian

講義ノート `sequence_space_jacobian_lecture_notes_ch4_11.pdf` 第7章 「直接法：ヤコビアンを一列ずつ計算する」の宿題

> $T = 5$ で直接法を使って家計ブロックのヤコビアンを求める（章末確認問題1）

に対する実装。

## ファイル構成

|                                       | 内容                                                                                                   |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `01_homework_direct_method.ipynb`     | **宿題そのもの。** 定常状態 → ブロック写像 → $J^{K,r}_{:,3}$ を一列 → 4本のヤコビアン → 章末問題の答え |
| `02_experiments.ipynb`                | 周辺の検証と実験。キャリブレーション、差分幅、計算量、ライブラリ突き合わせ、失業保険の感応度           |
| `docs/WALKTHROUGH.md`                 | `ks` パッケージが何をしているかを、実行順に言葉と実コードを交互に並べて解説                            |
| `docs/note.md`                        | 読みながら出た疑問と、その回答（実装へのリンク付き）                                                   |
| `docs/summary.md`                     | 自分の言葉でのまとめ                                                                                   |
| `marimo/`                             | 01・02 の marimo 版（`tools/marimo_convert.py` で .ipynb から生成）                                     |
| `ks/`                                 | 実装本体                                                                                               |
| `8_2_krusell_and_smith.py` / `.ipynb` | 既存の KS(1998) 元祖解法（参照用。このリポジトリでは使わない）                                         |

`ks/` の中身:

| モジュール          | 役割                                                                 |
| ------------------- | -------------------------------------------------------------------- |
| `ks/types.py`       | 配列の型エイリアスと状態ラベル。**まずここを読むと配列の形が分かる** |
| `ks/calibration.py` | `KSCalibration`（pydantic）、`KSModel`、KS マルコフ連鎖の積分        |
| `ks/household.py`   | EGM の後ろ向きステップ、前向きステップ、定常状態、移行経路           |
| `ks/jacobian.py`    | 直接法（第7章）とフェイクニュース法（第8章）                         |
| `ks/ssj.py`         | sequence-jacobian ライブラリ版（このモジュールだけ外部依存あり）     |

## セットアップ

`uv` + `pyproject.toml` で管理している。依存は `uv.lock` に固定済み。Python は 3.14。

```bash
uv sync                      # .venv を作って依存を入れる（ks も editable で入る）
uv run python -c "import ks" # 実行は uv run 経由が確実
uv run jupyter lab           # ノートブック
```

### marimo で開く

正本は `.ipynb` だが、同じ内容を marimo でも開ける。`marimo/*.py` は変換済みのものを置いてあり、.ipynb を変えたら 1 コマンドで作り直す:

```bash
uv run tools/marimo_convert.py                                    # 01, 02 → marimo/*.py
uv run --env-file .env marimo edit marimo/01_homework_direct_method.py
```

`marimo convert` がセルをそのまま移し、複数セルで再定義される名前（`fig, ax` など）はセル内変数 `_fig, _ax` に直す。
IPython の `display(...)` だけは変換スクリプトが `mo.output.append(...)` に置き換える。
marimo 側で編集した内容を戻すなら `uv run marimo export ipynb marimo/xxx.py -o xxx.ipynb`。

### JIT

CPython 3.14 の実験的 JIT は `PYTHON_JIT=1` を**インタプリタ起動前に**環境変数で渡すと有効になる （`-X` オプションや `os.environ` では有効にならない）。`.env` にその1行を置いてあるので、

```bash
uv run --env-file .env python ...      # スクリプト
uv run --env-file .env jupyter lab     # ノートブック（カーネルにも引き継がれる）
uv run --env-file .env python -c "import sys; print(sys._jit.is_enabled())"   # 確認
```

JIT が組み込まれているのは uv 管理の CPython（`python-preference = "only-managed"` で固定）。 Homebrew の python@3.14 は JIT なしでビルドされている。 なお CPython の JIT は Python のバイトコードを速くするもので、このコードの時間は numpy の呼び出しと実行時型検査（C 側）に消えているので効果は出ない（実測で差なし）。

### 速度

配列が 2 x 200 と小さいので、numpy の計算そのものではなく「小さな演算を何万回も呼ぶオーバーヘッド」が支配的。
内側ループ（`interpolate_y`・`forward_endogenous`）は numba の kernel にしてある（`ks/_jit.py`）。
初回だけコンパイルに 1〜2 秒かかり、以後は `__pycache__` のキャッシュが使われる。

| | 実行時型検査あり（既定） | `KS_TYPECHECK=0` |
|---|---|---|
| 定常状態 `solve_general_equilibrium` | 1.1 s | 0.25 s |
| `jacobian_direct(T=5)` | 6 ms | 3 ms |
| `backward_egm` 1 回 | 31 µs | 6 µs |

型検査（beartype）は 1 回 10 µs 前後かかるので、既定では時間の大半がそれ。重い実験では `KS_TYPECHECK=0` を付ける。

### Lint / 型検査 / テスト

```bash
uv format                    # ruff format（ノートブックは対象外）
uv run ruff check .          # ノートブックも対象。8_2_* の元祖コードは除外
uv run mypy                  # ks/ と tests/ を strict + α で検査（設定は pyproject.toml）
uv run pytest                # tests/。各テストの docstring がその関数の仕様の説明になっている
```

`ks` はパッケージとしてインストールされるので、`ks/*.py` を編集すれば 再インストールなしで反映される。どのディレクトリからでも `import ks` できる。

`ks.ssj` 以外は numpy / numba / pydantic / jaxtyping / beartype だけで動く。

> **シェルで pyenv などの仮想環境が有効だと** `VIRTUAL_ENV ... does not match` の警告が出る。 `uv` 側は `.venv` を使うので実害はないが、`python` を直接叩くと別の環境に行く。 `uv run` を通すか、`pyenv deactivate` してから作業する。 VS Code でノートブックを開くときはインタプリタに `.venv/bin/python` を選ぶ。

## 使い方

```python
import ks

model = ks.KSModel.build()                   # パラメータと導出量
ss = ks.solve_general_equilibrium(model)     # 定常状態
J = ks.jacobian_direct(model, ss, T=5)       # 直接法でヤコビアン
J.K_r                                        # dK_t / dr_s の 5x5 行列
```

## 型について

配列は「何次元で、各軸が何を表し、モデルのどの対象なのか」が型から読めるようにしてある。

```python
def backward_egm(
    expected_Va: MarginalValue,   # Float[np.ndarray, "n_e n_a"]
    a_grid:      AssetGrid,       # Float[np.ndarray, "n_a"]
    y:           IncomeByState,   # Float[np.ndarray, "n_e"]
    r: float, beta: float, eis: float,
) -> BackwardStep:                # NamedTuple(Va, a, c) いずれも "n_e n_a"
```

軸の名前の約束:

|       | 意味               | KS での値                                            |
| ----- | ------------------ | ---------------------------------------------------- |
| `n_e` | 個別の雇用状態の数 | 2（失業 / 就業）                                     |
| `n_a` | 資産グリッドの点数 | 200                                                  |
| `T`   | 系列空間の切断期間 | 5（宿題）                                            |
| `n_A` | 集計状態の数       | 2（不況 / 好況）。定常状態を作る過程で積分して消える |

- パラメータは pydantic で検証される（`beta` が $(0,1)$ の外なら `ValidationError`）
- 配列の形は jaxtyping + beartype で実行時に検査される。同じ軸名は長さの一致まで見る
- 重い計算で外したい場合は環境変数 `KS_TYPECHECK=0`（約2倍速くなる）
- 添字は `Employment.UNEMPLOYED` / `AggregateState.BAD` のような `IntEnum` で書ける

## 主な数値

```
K = 11.60936401    r = 0.00999853    w = 2.37449987
1/beta - 1 = 0.01010101       beta * (1 + r) = 0.99989854
```

均衡が非忍耐性の境界に極端に近いため、分布の前向き反復が実用的に収束しない。 定常分布は反復ではなく線形方程式 $D = \Lambda' D,\ \mathbf{1}'D = 1$ で直接解いている。

検証（$T = 5$）:

| 比較                                                          | 最大差  |
| ------------------------------------------------------------- | ------- |
| 自作 直接法（中央）vs ライブラリ fake news（中央）            | 1.7e-07 |
| 自作 fake news vs ライブラリ fake news                        | 1.6e-07 |
| 自作 直接法（片側）vs ライブラリ 直接法（片側）、同一定常状態 | 0.0     |

## 注意点

- `sequence_jacobian` の `.jacobian()` も有限差分。既定は片側差分なので、直接法と 突き合わせるときは `twosided=True` にする（既定のままだと 2.6e-4 ずれる）
- `impulse_nonlinear` は「$Y_{ss}$ からの偏差」を返す。直接法に使うならゼロショックの 走行を1本流して基準走行 $Y^0$ を差し引く（講義ノート 7.7）
- 既存コードの失業保険は課税で賄われていない純粋な移転なので、財市場が $-u \times 0.07$ だけずれる。部分均衡のヤコビアンには影響しないが GE では要注意
- 集計ショックに伴う失業率の変動（好況 4% / 不況 10%）は 2×2 の `Pi` に潰した時点で 落ちている。復活させるなら SSJ 側で `Pi` 自体を入力として動かす
