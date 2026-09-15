"""配列の「形」と「経済学的な意味」を型で表すための定義。

このモジュールの狙い
--------------------
`np.ndarray` とだけ書いてあると、その配列が

  - 何次元なのか
  - 各軸が何を表しているのか（雇用状態？資産？時点？）
  - モデルのどの対象なのか（政策関数？分布？ヤコビアン？）

がコードから読み取れない。ここではその3つを型エイリアスに畳み込む。
`jaxtyping` の記法 `Float[np.ndarray, "n_e n_a"]` は「float の2次元配列で、
軸0 の長さが n_e、軸1 の長さが n_a」を意味する。

軸の名前の約束
--------------
    n_e : 個別の雇用状態の数。KS では 2（失業 / 就業）
    n_a : 資産グリッドの点数
    T   : 系列空間の切断期間（宿題では 5）
    n_A : 集計状態の数。KS では 2（不況 / 好況）
          定常状態を作る過程でのみ登場し、SSJ の家計ブロックには残らない

同じ名前の軸は、一つの関数呼び出しの中で長さが一致していることが実行時に検査される
（`typed` デコレータを付けた関数のみ）。
"""

from __future__ import annotations

import os
from collections.abc import Callable
from enum import IntEnum

import numpy as np
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped

__all__ = [
    "AggregatePath",
    "AggregateState",
    "AggregateTransition",
    "AssetGrid",
    "Distribution",
    "DistributionPath",
    "Employment",
    "EmploymentTransition",
    "FloatArray",
    "IncomeByState",
    "IntArray",
    "JacobianMatrix",
    "JointDistribution",
    "JointTransition",
    "LotteryIndex",
    "LotteryWeight",
    "MarginalValue",
    "PolicyFunction",
    "PolicyPath",
    "PricePath",
    "TransitionMatrix",
    "typed",
]


# =====================================================================
# 状態のラベル（添字を意味のある名前で書けるようにする）
# =====================================================================


class Employment(IntEnum):
    """個別の雇用状態。配列の軸0 の添字として使う。

    既存コード 8_2_krusell_and_smith.py の `il` / `lgrid` に対応する。
    """

    UNEMPLOYED = 0
    EMPLOYED = 1


class AggregateState(IntEnum):
    """集計状態。既存コードの `iA` / `Agrid` に対応する。

    SSJ の定常状態では積分されて消えるので、キャリブレーションの中でしか現れない。
    """

    BAD = 0
    GOOD = 1


# =====================================================================
# 基底の配列型
# =====================================================================

type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
"""float64 の ndarray。形は jaxtyping の軸名で別途指定する。

裸の `np.ndarray` は `ndarray[tuple[Any, ...], dtype[Any]]` に展開されて Any を含むので、
静的型検査を厳しくするために dtype まで固定しておく。
"""

type IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]
"""int64 の ndarray。添字用。"""


# =====================================================================
# 個別状態の上の対象
# =====================================================================

type AssetGrid = Float[FloatArray, "n_a"]
"""資産グリッド a。昇順。既存コードの `agrid`。"""

type IncomeByState = Float[FloatArray, "n_e"]
"""雇用状態ごとの労働所得 y(e)。並びは [失業, 就業]。"""

type PolicyFunction = Float[FloatArray, "n_e n_a"]
"""個別状態 (e_t, a_{t-1}) 上の政策関数。期末資産 a_t または消費 c_t。

軸0 = 今期の雇用状態 e_t、軸1 = 期首資産 a_{t-1}。
"""

type MarginalValue = Float[FloatArray, "n_e n_a"]
"""価値関数の資産微分 V_a。EGM の後ろ向き変数。形は政策関数と同じ。"""

type Distribution = Float[FloatArray, "n_e n_a"]
"""時点 t 冒頭の個別状態 (e_t, a_{t-1}) 上の分布 D_t。総和は 1。

講義ノート ii ページの D_t そのもの。
"""


# =====================================================================
# 遷移
# =====================================================================

type EmploymentTransition = Float[FloatArray, "n_e n_e"]
"""雇用の遷移行列 Pi(e' | e)。行 = 今の雇用状態、列 = 来期の雇用状態。各行の和は 1。

SSJ の家計ブロックが受け取る外生過程はこれ（2x2）。
"""

type AggregateTransition = Float[FloatArray, "n_A n_A"]
"""集計状態の遷移行列 P(A' | A)。行 = 今、列 = 来期。"""

type JointTransition = Float[FloatArray, "n_An_e n_An_e"]
"""(A, l) 同時過程の遷移行列（KS では 4x4）。行 = n_e * A + l、列 = n_e * A' + l'。

既存コード `markov_KS` の返す `p_lA_lA` / `p_l_lAA` と同じ並び。
"""

type JointDistribution = Float[FloatArray, "n_An_e"]
"""(A, l) 同時分布（KS では長さ 4）。並びは [不況&失業, 不況&就業, 好況&失業, 好況&就業]。"""

type LotteryIndex = Int[IntArray, "n_e n_a"]
"""政策 a_t をグリッドに落とすときの、下側の格子点の添字 i。"""

type LotteryWeight = Float[FloatArray, "n_e n_a"]
"""同上の下側の格子点にかかる重み pi。a_t = pi * grid[i] + (1 - pi) * grid[i+1]。"""

type TransitionMatrix = Float[FloatArray, "n_states n_states"]
"""個別状態空間 (e, a) 上の遷移行列 Lambda を平らに並べたもの。n_states = n_e * n_a。

D_{t+1} = Lambda_t' D_t（講義ノート式 7.19）。
"""


# =====================================================================
# 系列空間の対象
# =====================================================================

type PricePath = Float[FloatArray, "T"]
"""価格の経路。r_0, ..., r_{T-1} を積み上げたもの（講義ノートの太字 r, w）。"""

type AggregatePath = Float[FloatArray, "T"]
"""集計量の経路。K_0, ..., K_{T-1} または C_0, ..., C_{T-1}。"""

type PolicyPath = Float[FloatArray, "T n_e n_a"]
"""政策関数の経路。軸0 が時点、軸1・2 が個別状態。"""

type DistributionPath = Float[FloatArray, "T n_e n_a"]
"""分布の経路 D_0, ..., D_{T-1}。形は政策経路と同じだが意味が違う。"""

type JacobianMatrix = Float[FloatArray, "T T"]
"""Sequence Space Jacobian。(t, s) 要素が dY_t / dX_s（講義ノートの J^{Y,X}）。

行 t = 応答の時点、列 s = ショックの時点。異質的家計ブロックでは密行列になる。
"""


# =====================================================================
# 実行時の型・形状チェック
# =====================================================================

_RUNTIME_TYPECHECK = os.environ.get("KS_TYPECHECK", "1").lower() not in ("0", "false", "no")


def typed[F: Callable[..., object]](fn: F) -> F:
    """引数と返り値の dtype・形状を実行時に検査するデコレータ。

    同じ軸名（例えば "n_a"）が複数の引数に現れる場合、その長さが一致していることも
    検査される。形の取り違えは異質的主体モデルで最も起きやすいバグなので、
    既定では有効にしてある。

    重い計算で外したい場合は環境変数 `KS_TYPECHECK=0` を設定する。
    """
    if not _RUNTIME_TYPECHECK:
        return fn
    return jaxtyped(typechecker=beartype)(fn)
