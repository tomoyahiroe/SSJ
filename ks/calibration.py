# -*- coding: utf-8 -*-
"""パラメータと、集計ショックを積分して消す作業。

既存コード 8_2_krusell_and_smith.py の `Setting`（replicate_model = 0、
すなわち Krusell and Smith 1998）を、検証つきの pydantic モデルに置き換えたもの。

SSJ の定常状態には集計ショック A が存在しないので、KS の (A, l) 4状態チェーンを
l だけの2状態チェーンに潰す必要がある。その手順もここに置く。
"""

from __future__ import annotations

from functools import cached_property

import numpy as np
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, Field

from .types import (
    AggregateTransition,
    AssetGrid,
    EmploymentTransition,
    IncomeByState,
    JointDistribution,
    JointTransition,
    typed,
)

__all__ = [
    "Prices",
    "KSCalibration",
    "KSMarkovChains",
    "KSModel",
    "stationary_distribution",
    "exponential_grid",
]


# =====================================================================
# 価格
# =====================================================================


class Prices(BaseModel):
    """家計ブロックが外から受け取る価格。SSJ の家計ブロックの入力はこの2つだけ。"""

    model_config = ConfigDict(frozen=True)

    r: float = Field(description="純資本収益率 r_t = alpha Z (K_{t-1}/N)^(alpha-1) - delta")
    w: float = Field(description="賃金率 w_t = (1-alpha) Z (K_{t-1}/N)^alpha")


# =====================================================================
# 汎用ユーティリティ
# =====================================================================


@typed
def stationary_distribution(
    P: Float[np.ndarray, "n n"], tol: float = 1e-14, max_iter: int = 1_000_000
) -> Float[np.ndarray, "n"]:
    """行確率行列 P の定常分布 x（x P = x, 1'x = 1）を反復で求める。

    既存コードの `stationary_by_iteration` と同じ。状態数が小さいチェーン専用。
    """
    x = np.full(P.shape[0], 1.0 / P.shape[0])
    for _ in range(max_iter):
        y = x @ P
        if np.max(np.abs(y - x)) < tol:
            return y
        x = y
    raise RuntimeError("stationary_distribution: 収束しませんでした")


@typed
def exponential_grid(a_min: float, a_max: float, n_a: int) -> AssetGrid:
    """対数空間で等間隔な資産グリッド。既存コードの `exponential_grid` と同一。"""
    return np.expm1(np.linspace(np.log1p(a_min), np.log1p(a_max), n_a))


# =====================================================================
# パラメータ
# =====================================================================


class KSCalibration(BaseModel):
    """Krusell=Smith のプリミティブ・パラメータ（スカラーのみ）。

    既定値は既存コード `Setting` の `replicate_model = 0`（KS 1998）に一致する。
    配列は一切持たない。導出される配列は `KSModel` 側に置く。
    """

    model_config = ConfigDict(frozen=True)

    # --- 技術 ---
    alpha: float = Field(0.36, gt=0.0, lt=1.0, description="資本分配率")
    delta: float = Field(0.025, gt=0.0, lt=1.0, description="四半期の資本減耗率")

    # --- 選好 ---
    beta: float = Field(0.99, gt=0.0, lt=1.0, description="四半期の割引因子")
    eis: float = Field(1.0, gt=0.0, description="異時点間代替の弾力性。1 なら log 効用")

    # --- 労働と失業保険 ---
    lbar: float = Field(0.3271, gt=0.0, description="就業時の労働時間（既存コードの lbar）")
    unemployment_insurance: float = Field(
        0.07,
        gt=0.0,
        description="失業給付。既存コードの replicate_model=0 では w に比例しない「水準」",
    )

    # --- 集計ショックのマルコフ連鎖（定常状態を作る過程で積分して消す） ---
    u_bad: float = Field(0.10, gt=0.0, lt=1.0, description="不況時の失業率")
    u_good: float = Field(0.04, gt=0.0, lt=1.0, description="好況時の失業率")
    z_bad: float = Field(0.99, gt=0.0, description="不況時の TFP")
    z_good: float = Field(1.01, gt=0.0, description="好況時の TFP")
    dur_unemployed_good: float = Field(1.5, gt=1.0, description="好況時の平均失業継続期間（四半期）")
    dur_unemployed_bad: float = Field(2.5, gt=1.0, description="不況時の平均失業継続期間（四半期）")
    dur_good: float = Field(8.0, gt=1.0, description="好況の平均継続期間（四半期）")
    dur_bad: float = Field(8.0, gt=1.0, description="不況の平均継続期間（四半期）")
    add_assumption_1: float = Field(1.25, gt=0.0, description="好況→不況の失業継続への調整（KS の追加仮定）")
    add_assumption_2: float = Field(0.75, gt=0.0, description="不況→好況の失業継続への調整（KS の追加仮定）")

    # --- 資産グリッド ---
    n_a: int = Field(200, ge=10, description="資産グリッドの点数。政策と分布で共用する")
    a_min: float = Field(0.0, ge=0.0, description="借入制約 a_t >= a_min")
    a_max: float = Field(300.0, gt=0.0, description="資産グリッドの上端")

    @property
    def r_impatience_bound(self) -> float:
        """r がこれ以上だと資産需要が発散する上限 1/beta - 1。"""
        return 1.0 / self.beta - 1.0


# =====================================================================
# 集計ショックのマルコフ連鎖（既存コード markov_KS の移植）
# =====================================================================


class KSMarkovChains(BaseModel):
    """KS(1998) の (A, l) 同時マルコフ連鎖と、そこから導いた2状態雇用チェーン。

    SSJ の家計ブロックが実際に使うのは `employment` （2x2）だけ。
    残りは「その 2x2 がどこから来たか」を追えるようにするために持っている。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    aggregate: AggregateTransition = Field(description="P(A' | A)。行 = [不況, 好況]")
    joint: JointTransition = Field(description="P(A', l' | A, l)。行 = 2*A + l")
    employment_given_A: JointTransition = Field(description="P(l' | A, A', l)。行 = 2*A + l")

    aggregate_stationary: Float[np.ndarray, "n_A"] = Field(description="A の定常分布（既存コードの A_ss）")
    joint_stationary: JointDistribution = Field(description="(A, l) の同時定常分布")

    employment: EmploymentTransition = Field(description="A を積分した2状態チェーン Pi(l' | l)")
    u_ss: float = Field(description="Pi の定常失業率。同時定常分布の周辺失業率と一致する")

    @property
    def employment_marginal(self) -> Float[np.ndarray, "n_e"]:
        """同時定常分布から得た雇用の周辺分布 [失業, 就業]。"""
        j = self.joint_stationary
        return np.array([j[0] + j[2], j[1] + j[3]])

    @property
    def mean_unemployment_duration(self) -> float:
        """積分後のチェーンが含意する平均失業継続期間（四半期）。"""
        return 1.0 / (1.0 - self.employment[0, 0])

    @typed
    def conditional_employment_matrix(
        self, a_now: int, a_next: int
    ) -> EmploymentTransition:
        """(A, A') の組を一つ指定して、その 2x2 雇用遷移行列を取り出す。

        Pi はこの4枚を重み付き平均したものになっている。
        """
        return self.employment_given_A[2 * a_now : 2 * a_now + 2, 2 * a_next : 2 * a_next + 2]

    @typed
    def mixing_weights(self, employment_state: int) -> Float[np.ndarray, "n_An_e"]:
        """Pi の第 `employment_state` 行を作るときの、4枚の行列にかかる重み（合計 1）。

        重み = pi(A | l) * P(A' | A)。並びは [(不況,不況), (不況,好況), (好況,不況), (好況,好況)]。
        重みが行ごとに異なる点が重要で、共通の pi(A) で潰すと定常失業率がずれる。
        """
        j, marg = self.joint_stationary, self.employment_marginal
        return np.array(
            [
                j[2 * i_a + employment_state] / marg[employment_state] * self.aggregate[i_a, j_a]
                for i_a in range(2)
                for j_a in range(2)
            ]
        )


def _build_markov_chains(c: KSCalibration) -> KSMarkovChains:
    """既存コード `markov_KS`（Fortran の trans_prob）をそのまま移植し、A を積分する。"""
    # --- 集計状態の遷移。q = (D - 1) / D は「平均継続期間 D」と等価 ---
    p_gg = (c.dur_good - 1.0) / c.dur_good
    p_bb = (c.dur_bad - 1.0) / c.dur_bad
    aggregate = np.array([[p_bb, 1.0 - p_bb], [1.0 - p_gg, p_gg]])  # 行 = [不況, 好況]

    # --- (A, A') の組ごとの雇用遷移 ---
    pgg00 = (c.dur_unemployed_good - 1.0) / c.dur_unemployed_good
    pgg01 = (c.u_good - c.u_good * pgg00) / (1.0 - c.u_good)
    pbb00 = (c.dur_unemployed_bad - 1.0) / c.dur_unemployed_bad
    pbb01 = (c.u_bad - c.u_bad * pbb00) / (1.0 - c.u_bad)
    pbg00 = c.add_assumption_1 * pbb00
    pbg01 = (c.u_bad - c.u_good * pbg00) / (1.0 - c.u_good)
    pgb00 = c.add_assumption_2 * pgg00
    pgb01 = (c.u_good - c.u_bad * pgb00) / (1.0 - c.u_bad)

    def m(p00: float, p01: float) -> EmploymentTransition:
        return np.array([[p00, 1.0 - p00], [p01, 1.0 - p01]])

    by_a = [
        [m(pbb00, pbb01), m(pgb00, pgb01)],  # 不況 -> 不況, 好況
        [m(pbg00, pbg01), m(pgg00, pgg01)],  # 好況 -> 不況, 好況
    ]

    joint = np.zeros((4, 4))
    employment_given_a = np.zeros((4, 4))
    for i_a in range(2):
        for i_l in range(2):
            row = 2 * i_a + i_l
            for j_a in range(2):
                for j_l in range(2):
                    col = 2 * j_a + j_l
                    employment_given_a[row, col] = by_a[i_a][j_a][i_l, j_l]
                    joint[row, col] = aggregate[i_a, j_a] * by_a[i_a][j_a][i_l, j_l]

    # --- A を同時定常分布で積分して 2x2 にする ---
    joint_stationary = stationary_distribution(joint)
    marg_l = np.array(
        [joint_stationary[0] + joint_stationary[2], joint_stationary[1] + joint_stationary[3]]
    )
    employment = np.zeros((2, 2))
    for i_a in range(2):
        for i_l in range(2):
            row = 2 * i_a + i_l
            for j_a in range(2):
                for j_l in range(2):
                    employment[i_l, j_l] += (
                        joint_stationary[row] * aggregate[i_a, j_a] * employment_given_a[row, 2 * j_a + j_l]
                    )
    employment /= marg_l[:, None]

    return KSMarkovChains(
        aggregate=aggregate,
        joint=joint,
        employment_given_A=employment_given_a,
        aggregate_stationary=stationary_distribution(aggregate),
        joint_stationary=joint_stationary,
        employment=employment,
        u_ss=float(stationary_distribution(employment)[0]),
    )


# =====================================================================
# モデル（パラメータ + 導出された配列）
# =====================================================================


class KSModel(BaseModel):
    """スカラーのパラメータから導いた配列をまとめて持つ。

    `KSCalibration` が「入力」、こちらが「そこから決まるもの」という分担。
    家計問題を解く関数はすべてこのオブジェクトを受け取る。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    calibration: KSCalibration
    chains: KSMarkovChains
    a_grid: AssetGrid = Field(description="資産グリッド。政策と分布で共用する")

    @classmethod
    def build(cls, calibration: KSCalibration | None = None) -> "KSModel":
        c = calibration or KSCalibration()
        return cls(
            calibration=c,
            chains=_build_markov_chains(c),
            a_grid=exponential_grid(c.a_min, c.a_max, c.n_a),
        )

    # --- よく使うものへの近道 ---

    @property
    def Pi(self) -> EmploymentTransition:
        """雇用の遷移行列 Pi(e'|e)。SSJ の家計ブロックの外生過程。"""
        return self.chains.employment

    @property
    def u_ss(self) -> float:
        """定常失業率。"""
        return self.chains.u_ss

    @property
    def n_e(self) -> int:
        return self.Pi.shape[0]

    @property
    def n_a(self) -> int:
        return self.calibration.n_a

    @cached_property
    def N(self) -> float:
        """集計労働 N = lbar * (1 - u)。既存コードの L_agg に対応する。"""
        return self.calibration.lbar * (1.0 - self.u_ss)

    @cached_property
    def Z(self) -> float:
        """定常状態の TFP。集計状態の定常分布で加重平均したもの。"""
        z = np.array([self.calibration.z_bad, self.calibration.z_good])
        return float(self.chains.aggregate_stationary @ z)

    @cached_property
    def K_impatience_bound(self) -> float:
        """r(K) = 1/beta - 1 となる K。均衡 K はこれより大きくないと定常分布が存在しない。"""
        c = self.calibration
        return self.N * ((c.alpha * self.Z) / (c.r_impatience_bound + c.delta)) ** (
            1.0 / (1.0 - c.alpha)
        )

    # --- 企業側の関係式（既存コード vfi_numba の中の式と同一） ---

    @typed
    def prices(self, K: float) -> Prices:
        """期首資本 K = K_{t-1} から (r_t, w_t) を作る。"""
        c = self.calibration
        k_n = K / self.N
        return Prices(
            r=c.alpha * self.Z * k_n ** (c.alpha - 1.0) - c.delta,
            w=(1.0 - c.alpha) * self.Z * k_n**c.alpha,
        )

    @typed
    def output(self, K: float) -> float:
        """産出 Y = Z K^alpha N^(1-alpha)。"""
        c = self.calibration
        return self.Z * K**c.alpha * self.N ** (1.0 - c.alpha)

    @typed
    def income(self, w: float) -> IncomeByState:
        """雇用状態ごとの労働所得 y(e)。並びは [失業, 就業]。

        既存コードの replicate_model = 0 と同じで、失業給付は w に比例しない水準。
        """
        c = self.calibration
        return np.array([c.unemployment_insurance, w * c.lbar])
