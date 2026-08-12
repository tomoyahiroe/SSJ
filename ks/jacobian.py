# -*- coding: utf-8 -*-
"""家計ブロックのヤコビアン。

第7章の直接法（`jacobian_direct`）と、第8章のフェイクニュース法（`jacobian_fake_news`）。
宿題は前者。後者は前者を検証するために置いてある。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

import numpy as np
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, Field

from .calibration import KSModel
from .household import (
    Lottery,
    SteadyState,
    asset_lottery,
    backward_egm,
    forward_step,
    simulate_transition,
)
from .types import Distribution, JacobianMatrix, PolicyFunction, typed

__all__ = [
    "DifferenceScheme",
    "BlockInput",
    "BlockOutput",
    "HouseholdJacobians",
    "jacobian_direct",
    "jacobian_fake_news",
]


class DifferenceScheme(StrEnum):
    """有限差分の取り方（講義ノート 7.6）。"""

    ONE_SIDED = "one_sided"
    """片側差分。誤差 O(eps)。摂動走行が1本で済む。"""

    CENTRAL = "central"
    """中央差分。誤差 O(eps^2)。計算量は約2倍。"""


BlockInput = Literal["r", "w"]
"""家計ブロックの入力。価格の2本だけ。"""

BlockOutput = Literal["K", "C"]
"""家計ブロックの出力。集計資本と集計消費。"""


class HouseholdJacobians(BaseModel):
    """家計ブロックの4本のヤコビアン（講義ノート式 3）。

        (dK, dC)' = [[J^{K,r}, J^{K,w}], [J^{C,r}, J^{C,w}]] (dr, dw)'

    どの行列も (t, s) 要素が dY_t / dX_s。行 t = 応答の時点、列 s = ショックの時点。
    dict の文字列キーではなく属性でアクセスできるようにしてある。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    K_r: JacobianMatrix = Field(description="dK_t / dr_s")
    K_w: JacobianMatrix = Field(description="dK_t / dw_s")
    C_r: JacobianMatrix = Field(description="dC_t / dr_s")
    C_w: JacobianMatrix = Field(description="dC_t / dw_s")

    eps: float | None = Field(default=None, description="使った差分幅")
    scheme: DifferenceScheme | None = Field(default=None, description="使った差分の取り方")

    @property
    def T(self) -> int:
        return self.K_r.shape[0]

    @typed
    def get(self, output: BlockOutput, input_: BlockInput) -> JacobianMatrix:
        """出力名と入力名で取り出す（ループを回すとき用）。"""
        return getattr(self, f"{output}_{input_}")

    @typed
    def max_difference(self, other: "HouseholdJacobians") -> float:
        """4本すべてを比べたときの最大の絶対差。実装同士の突き合わせに使う。"""
        return max(
            float(np.max(np.abs(self.get(o, i) - other.get(o, i))))
            for o in ("K", "C")
            for i in ("r", "w")
        )


# =====================================================================
# 第7章：直接法
# =====================================================================


@typed
def jacobian_direct(
    model: KSModel,
    ss: SteadyState,
    T: int = 5,
    eps: float = 1e-4,
    scheme: DifferenceScheme = DifferenceScheme.ONE_SIDED,
    verbose: bool = False,
) -> HouseholdJacobians:
    """直接法でヤコビアンを求める（講義ノート 定義 7.1）。

    入力経路の各座標を一つずつ摂動し、そのたびに非線形の家計問題と分布動学を解いて
    出力経路の差分を求め、ヤコビアンの列を順番に構成する。

    高価な後ろ向き・前向き計算の回数は、入力1本あたり T 列 x 長さ T = O(T^2)。

    Parameters
    ----------
    eps
        差分幅（ノート 7.6.3 のトレードオフ）。
    scheme
        片側差分か中央差分か。
    """
    r_ss = np.full(T, ss.prices.r)
    w_ss = np.full(T, ss.prices.w)

    # ノート 7.7「基準走行を差し引く理由」:
    # 定常値 Y_ss * 1 ではなく、同じ有限期間・同じ終端条件・同じ数値手順で走らせた
    # Y^0 を差し引く。摂動と無関係な基準誤差が 1/eps 倍されるのを避けるため。
    baseline = simulate_transition(model, ss, r_ss, w_ss)

    columns: dict[str, JacobianMatrix] = {
        f"{o}_{i}": np.empty((T, T)) for o in ("K", "C") for i in ("r", "w")
    }

    for input_ in ("r", "w"):
        base = r_ss if input_ == "r" else w_ss
        for s in range(T):
            # 式 7.4:  X^{s,+}_t = X_ss + eps * 1{t = s}
            up = base.copy()
            up[s] += eps
            path_up = simulate_transition(
                model, ss, up if input_ == "r" else r_ss, w_ss if input_ == "r" else up
            )

            if scheme is DifferenceScheme.CENTRAL:
                down = base.copy()
                down[s] -= eps
                path_down = simulate_transition(
                    model, ss, down if input_ == "r" else r_ss, w_ss if input_ == "r" else down
                )
                denominator = 2.0 * eps
                reference_K, reference_C = path_down.K, path_down.C
            else:
                denominator = eps
                reference_K, reference_C = baseline.K, baseline.C

            columns[f"K_{input_}"][:, s] = (path_up.K - reference_K) / denominator
            columns[f"C_{input_}"][:, s] = (path_up.C - reference_C) / denominator

            if verbose:
                print(f"  入力 {input_} の第 {s} 列 完了")

    return HouseholdJacobians(**columns, eps=eps, scheme=scheme)


# =====================================================================
# 第8章：フェイクニュース・アルゴリズム（直接法の検証用）
# =====================================================================


class _FakeNewsIngredients(BaseModel):
    """フェイクニュース法の第1段階の出力。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    curly_Y: dict[str, Float[np.ndarray, "T"]] = Field(
        description="s 期先のニュースに対する、時点0 の集計反応"
    )
    curly_D: Float[np.ndarray, "T n_e n_a"] = Field(
        description="s 期先のニュースが時点1 の分布に残すずれ"
    )


@typed
def jacobian_fake_news(
    model: KSModel,
    ss: SteadyState,
    T: int = 5,
    eps: float = 1e-4,
    scheme: DifferenceScheme = DifferenceScheme.CENTRAL,
) -> HouseholdJacobians:
    """フェイクニュース・アルゴリズム（講義ノート第8章）。

    高価な計算は後ろ向き T 回 + 前向き T 回の O(T) で済む。

    重要: この方法も有限差分である。ただし微分するのは「経路全体の写像」ではなく
    「1期分の後ろ向きステップ」で、その誤差が対角方向の累積を通じて J に伝わる。
    そのため既定は中央差分にしてある（片側だと直接法と O(eps) ずれる）。
    sequence-jacobian の `.jacobian()` は既定が片側差分なので、突き合わせるときは
    `twosided=True` を指定すること。
    """
    c = model.calibration
    lottery_ss = asset_lottery(model.a_grid, ss.a)
    two_sided = scheme is DifferenceScheme.CENTRAL
    denominator = 2.0 * eps if two_sided else eps

    def ingredients(input_: BlockInput) -> _FakeNewsIngredients:
        curly_Y = {"K": np.empty(T), "C": np.empty(T)}
        curly_D = np.empty((T, model.n_e, model.n_a))

        def shocked(sign: float) -> tuple[float, float]:
            return (
                ss.prices.r + sign * eps if input_ == "r" else ss.prices.r,
                ss.prices.w + sign * eps if input_ == "w" else ss.prices.w,
            )

        def steady_step(Va: Float[np.ndarray, "n_e n_a"]):
            return backward_egm(
                model.Pi @ Va, model.a_grid, model.income(ss.prices.w), ss.prices.r, c.beta, c.eis
            )

        up = down = None
        Va_up = Va_down = ss.Va
        for s in range(T):
            # s 期先のニュースに対する時点0 の政策 = ショック1回 + 定常ステップ s 回
            if s == 0:
                r_u, w_u = shocked(+1.0)
                up = backward_egm(
                    model.Pi @ ss.Va, model.a_grid, model.income(w_u), r_u, c.beta, c.eis
                )
                if two_sided:
                    r_d, w_d = shocked(-1.0)
                    down = backward_egm(
                        model.Pi @ ss.Va, model.a_grid, model.income(w_d), r_d, c.beta, c.eis
                    )
            else:
                up = steady_step(Va_up)
                if two_sided:
                    down = steady_step(Va_down)
            Va_up = up.Va
            a_down: PolicyFunction = down.a if two_sided else ss.a
            c_down: PolicyFunction = down.c if two_sided else ss.c
            if two_sided:
                Va_down = down.Va

            curly_Y["K"][s] = np.vdot(ss.D, up.a - a_down) / denominator
            curly_Y["C"][s] = np.vdot(ss.D, up.c - c_down) / denominator

            D_up = forward_step(ss.D, model.Pi, asset_lottery(model.a_grid, up.a))
            D_down = (
                forward_step(ss.D, model.Pi, asset_lottery(model.a_grid, a_down))
                if two_sided
                else forward_step(ss.D, model.Pi, lottery_ss)
            )
            curly_D[s] = (D_up - D_down) / denominator

        return _FakeNewsIngredients(curly_Y=curly_Y, curly_D=curly_D)

    def expectation_vectors(outcome: PolicyFunction) -> Float[np.ndarray, "T_minus_1 n_e n_a"]:
        """curly_E[t]: 分布の痕跡を将来の集計量に変換するベクトル（ノート 8.5）。"""
        rows = np.arange(model.n_e)[:, None]
        E = np.empty((max(T - 1, 1), model.n_e, model.n_a))
        E[0] = outcome
        for t in range(1, T - 1):
            tmp = model.Pi @ E[t - 1]
            E[t] = (
                lottery_ss.weight * tmp[rows, lottery_ss.index]
                + (1.0 - lottery_ss.weight) * tmp[rows, lottery_ss.index + 1]
            )
        return E

    E = {"K": expectation_vectors(ss.a), "C": expectation_vectors(ss.c)}

    result: dict[str, JacobianMatrix] = {}
    for input_ in ("r", "w"):
        ing = ingredients(input_)
        for output in ("K", "C"):
            F = np.empty((T, T))
            F[0, :] = ing.curly_Y[output]  # ノート 8.6
            for t in range(1, T):
                F[t, :] = np.tensordot(E[output][t - 1], ing.curly_D, axes=([0, 1], [1, 2]))
            J = F.copy()
            for t in range(1, T):  # ノート 8.7: J[t, s] = J[t-1, s-1] + F[t, s]
                J[t, 1:] += J[t - 1, :-1]
            result[f"{output}_{input_}"] = J

    return HouseholdJacobians(**result, eps=eps, scheme=scheme)
