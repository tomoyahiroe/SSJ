"""家計問題：後ろ向きの政策、前向きの分布、定常状態、移行経路。

講義ノート第4章の3段階をそのまま関数にしたもの。

    後ろ向き   v_t     = v(v_{t+1}, X_t)          （式 7.18）
    前向き     D_{t+1} = Lambda(v_{t+1}, X_t)' D_t  （式 7.19）
    集計       Y_t     = y(v_{t+1}, X_t)' D_t      （式 7.20）

補間は sequence_jacobian.utilities.interpolate と数値的に同一の結果を返すように
書いてある（ライブラリ版と突き合わせるときに補間の差が混ざらないようにするため）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, Field

from .calibration import KSModel, Prices
from .types import (
    AggregatePath,
    AssetGrid,
    Distribution,
    DistributionPath,
    EmploymentTransition,
    FloatArray,
    IncomeByState,
    LotteryIndex,
    LotteryWeight,
    MarginalValue,
    PolicyFunction,
    PolicyPath,
    PricePath,
    TransitionMatrix,
    typed,
)

__all__ = [
    "BackwardStep",
    "Lottery",
    "StationaryDistribution",
    "SteadyState",
    "TransitionPath",
    "asset_lottery",
    "backward_egm",
    "forward_endogenous",
    "forward_step",
    "initial_marginal_value",
    "interpolate_y",
    "simulate_transition",
    "solve_general_equilibrium",
    "solve_household",
    "stationary_from_policy",
    "transition_matrix",
]


# =====================================================================
# 補間
# =====================================================================


@typed
def interpolate_y(
    x: Float[FloatArray, "n_e n_a"],
    xq: Float[FloatArray, "n_e n_a"],
    y: Float[FloatArray, "n_a"],
) -> Float[FloatArray, "n_e n_a"]:
    """各行について、増加データ点 x に対して y を xq で線形補間する（範囲外は線形外挿）。

    EGM で「内生グリッド上の消費」を「外生グリッド上の手持ち資産」へ移すのに使う。
    """
    n_a = x.shape[-1]
    i = np.empty(xq.shape, dtype=np.int64)
    for row in range(x.shape[0]):
        i[row] = np.searchsorted(x[row], xq[row], side="left") - 1
    np.clip(i, 0, n_a - 2, out=i)

    x_lo = np.take_along_axis(x, i, axis=-1)
    x_hi = np.take_along_axis(x, i + 1, axis=-1)
    weight = (x_hi - xq) / (x_hi - x_lo)
    return weight * y[i] + (1.0 - weight) * y[i + 1]


@dataclass(frozen=True, slots=True)
class Lottery:
    """政策 a_t をグリッド上の2点に振り分ける「くじ」表現。

        a_t = weight * a_grid[index] + (1 - weight) * a_grid[index + 1]

    NamedTuple にしていないのは、フィールド名 `index` が `tuple.index` と衝突するため。
    """

    index: LotteryIndex
    weight: LotteryWeight


@typed
def asset_lottery(a_grid: AssetGrid, policy: PolicyFunction) -> Lottery:
    """政策をくじ表現に直す。SSJ の interpolate_coord_robust と同一の結果。"""
    n_a = a_grid.shape[0]
    index = np.searchsorted(a_grid, policy, side="left") - 1
    np.clip(index, 0, n_a - 2, out=index)
    weight = (a_grid[index + 1] - policy) / (a_grid[index + 1] - a_grid[index])
    return Lottery(index=index, weight=weight)


# =====================================================================
# 後ろ向きステップ
# =====================================================================


class BackwardStep(NamedTuple):
    """1期分の後ろ向きステップの出力。

    pydantic ではなく NamedTuple にしてあるのは、直接法の内側ループで何千回も
    作られるため。注釈で形は明示してあるので読み取りには困らない。
    """

    Va: MarginalValue
    """V_a,t。次のステップの入力になる。"""

    a: PolicyFunction
    """期末資産の政策 a_t(e_t, a_{t-1})。"""

    c: PolicyFunction
    """消費の政策 c_t(e_t, a_{t-1})。"""


@typed
def backward_egm(
    expected_Va: MarginalValue,
    a_grid: AssetGrid,
    y: IncomeByState,
    r: float,
    beta: float,
    eis: float,
) -> BackwardStep:
    """内生グリッド法による1期分の後ろ向きステップ。

    Parameters
    ----------
    expected_Va
        E_t[V_a,{t+1}] = Pi @ Va_{t+1}。軸1 は期末資産 a_t。
        期待を取る操作は呼び出し側の責任（sequence-jacobian では HetBlock が代行する）。
    y
        時点 t の労働所得ベクトル。
    r
        時点 t の純資本収益率。

    オイラー方程式 u'(c_t) = beta E_t[(1 + r_{t+1}) u'(c_{t+1})] を、
    V_a,{t+1} = (1 + r_{t+1}) u'(c_{t+1}) と定義して解いている。
    """
    uc_nextgrid = beta * expected_Va
    c_nextgrid = uc_nextgrid ** (-eis)
    coh = (1.0 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]  # 手持ち資産
    a = interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    np.maximum(a, a_grid[0], out=a)  # 借入制約 a_t >= a_min
    c = coh - a
    return BackwardStep(Va=(1.0 + r) * c ** (-1.0 / eis), a=a, c=c)


@typed
def initial_marginal_value(
    a_grid: AssetGrid, y: IncomeByState, r: float, eis: float
) -> MarginalValue:
    """後ろ向き反復の初期値（消費が手持ちの1割、という粗い当て推量）。"""
    coh = (1.0 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    return (1.0 + r) * (0.1 * coh) ** (-1.0 / eis)


# =====================================================================
# 前向きステップ
# =====================================================================


@typed
def forward_endogenous(D: Distribution, lottery: Lottery) -> Distribution:
    """資産の遷移だけを進める（雇用状態はまだ動かさない）。"""
    out = np.zeros_like(D)
    for e in range(D.shape[0]):
        np.add.at(out[e], lottery.index[e], lottery.weight[e] * D[e])
        np.add.at(out[e], lottery.index[e] + 1, (1.0 - lottery.weight[e]) * D[e])
    return out


@typed
def forward_step(D: Distribution, Pi: EmploymentTransition, lottery: Lottery) -> Distribution:
    """D_{t+1} = Lambda_t' D_t。まず資産、次に雇用状態の順（SSJ の規約と同じ）。"""
    return Pi.T @ forward_endogenous(D, lottery)


@typed
def transition_matrix(Pi: EmploymentTransition, lottery: Lottery) -> TransitionMatrix:
    """遷移行列 Lambda を (n_e*n_a, n_e*n_a) の密行列として明示的に作る。

    状態数は 2 * n_a しかないので密行列で持って問題ない。
    定常分布を反復ではなく線形方程式で解くために使う。
    """
    n_e, n_a = lottery.index.shape
    lam = np.zeros((n_e * n_a, n_e * n_a))
    for e in range(n_e):
        for i in range(n_a):
            row = e * n_a + i
            lo, w = lottery.index[e, i], lottery.weight[e, i]
            for e_next in range(n_e):
                lam[row, e_next * n_a + lo] += Pi[e, e_next] * w
                lam[row, e_next * n_a + lo + 1] += Pi[e, e_next] * (1.0 - w)
    return lam


class StationaryDistribution(NamedTuple):
    """定常分布と、その残差 max|Lambda' D - D|。"""

    D: Distribution
    residual: float


@typed
def stationary_from_policy(Pi: EmploymentTransition, lottery: Lottery) -> StationaryDistribution:
    """政策から定常分布を直接解く: D = Lambda' D, 1'D = 1。

    KS のキャリブレーションは beta*(1+r) が 1 に極端に近く、前向き反復では
    「見かけ上収束したのにまだ真の定常分布から遠い」状態になりやすい。
    そこを線形方程式で一発で解く。
    """
    n_e, n_a = lottery.index.shape
    n = n_e * n_a
    lam = transition_matrix(Pi, lottery)
    lhs = lam.T - np.eye(n)
    lhs[-1, :] = 1.0  # 最終行を正規化条件 1'D = 1 で置き換える
    rhs = np.zeros(n)
    rhs[-1] = 1.0
    d = np.linalg.solve(lhs, rhs)
    return StationaryDistribution(
        D=d.reshape(n_e, n_a), residual=float(np.max(np.abs(lam.T @ d - d)))
    )


# =====================================================================
# 定常状態
# =====================================================================


class SteadyState(BaseModel):
    """集計ショックのない定常状態。ヤコビアンはこの点のまわりで測る。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    prices: Prices
    K: float = Field(description="家計の期末資産の集計。資本市場が清算していれば期首資本にも等しい")
    C: float = Field(description="集計消費")

    y: IncomeByState = Field(description="雇用状態ごとの労働所得")
    Va: MarginalValue = Field(description="定常の限界価値。移行経路の終端条件になる")
    a: PolicyFunction = Field(description="定常の貯蓄政策")
    c: PolicyFunction = Field(description="定常の消費政策")
    D: Distribution = Field(description="定常分布。移行経路の初期条件になる")

    backward_iterations: int = Field(description="政策関数の収束までの反復回数")
    distribution_residual: float = Field(
        description="max|Lambda' D - D|。基準走行の平坦さを左右する"
    )
    top_grid_mass: float = Field(
        description="資産グリッド上端10点の質量。小さいほどグリッドが足りている"
    )

    @typed
    def excess_demand(self, K_supplied: float) -> float:
        """資産需要 - 供給。一般均衡ではこれをゼロにする。"""
        return self.K - K_supplied


@typed
def solve_household(
    model: KSModel,
    prices: Prices,
    Va_init: MarginalValue | None = None,
    *,
    backward_tol: float = 1e-11,
    backward_maxit: int = 50_000,
) -> SteadyState:
    """価格を所与に、家計問題の定常状態（政策と分布）を解く。部分均衡。

    `Va_init` を渡すと後ろ向き反復を温かいところから始められる（二分法の中で使う）。
    """
    c = model.calibration
    y = model.income(prices.w)

    Va = (
        Va_init if Va_init is not None else initial_marginal_value(model.a_grid, y, prices.r, c.eis)
    )
    a_previous: PolicyFunction = np.zeros((model.n_e, model.n_a))
    step = BackwardStep(Va=Va, a=a_previous, c=a_previous)
    iterations = 0
    for iterations in range(1, backward_maxit + 1):
        step = backward_egm(model.Pi @ step.Va, model.a_grid, y, prices.r, c.beta, c.eis)
        if iterations % 10 == 0:
            if np.max(np.abs(step.a - a_previous)) < backward_tol:
                break
            a_previous = step.a
    else:
        msg = "政策関数が収束しませんでした"
        raise RuntimeError(msg)

    stationary = stationary_from_policy(model.Pi, asset_lottery(model.a_grid, step.a))

    return SteadyState(
        prices=prices,
        K=float(np.vdot(stationary.D, step.a)),
        C=float(np.vdot(stationary.D, step.c)),
        y=y,
        Va=step.Va,
        a=step.a,
        c=step.c,
        D=stationary.D,
        backward_iterations=iterations,
        distribution_residual=stationary.residual,
        top_grid_mass=float(stationary.D[:, -10:].sum()),
    )


@typed
def solve_general_equilibrium(
    model: KSModel,
    K_low: float | None = None,
    K_high: float | None = None,
    *,
    tol: float = 1e-11,
    verbose: bool = False,
) -> SteadyState:
    """資本市場を清算する定常状態を二分法で解く: 家計の資産需要 = K。

    r(K) は K について減少、資産需要は r について増加なので、超過需要は K について
    減少する。下側の境界には `K_impatience_bound` を使う（そこを下回ると
    r >= 1/beta - 1 となり定常分布が存在しない）。
    """
    K_low = K_low if K_low is not None else model.K_impatience_bound * (1.0 + 1e-8)
    K_high = K_high if K_high is not None else 40.0 * model.N

    warm: MarginalValue | None = None

    def evaluate(K: float) -> SteadyState:
        nonlocal warm
        ss = solve_household(model, model.prices(K), Va_init=warm)
        warm = ss.Va
        return ss

    ss = evaluate(K_high)
    if ss.excess_demand(K_high) > 0.0:
        msg = f"上側の境界で超過需要 ({ss.excess_demand(K_high):.3e})。K_high を大きく。"
        raise RuntimeError(msg)

    while K_high - K_low > tol * max(1.0, K_low):
        K_mid = 0.5 * (K_low + K_high)
        candidate = evaluate(K_mid)
        excess = candidate.excess_demand(K_mid)
        if verbose:
            print(  # noqa: T201  verbose=True のときの進捗表示
                f"  K = {K_mid:10.6f}   r = {candidate.prices.r:9.6f}   超過需要 = {excess: .3e}"
            )
        if excess > 0.0:
            K_low = K_mid
        else:
            K_high = K_mid
            ss = candidate

    return evaluate(0.5 * (K_low + K_high))


# =====================================================================
# 移行経路：ブロック写像 H
# =====================================================================


class TransitionPath(BaseModel):
    """価格経路 1 本に対する集計経路。これがブロック写像 H の出力。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    K: AggregatePath = Field(description="集計資本（家計の期末資産）の経路")
    C: AggregatePath = Field(description="集計消費の経路")

    a: PolicyPath | None = Field(
        default=None, description="各時点の貯蓄政策（store_internals=True のとき）"
    )
    c: PolicyPath | None = Field(default=None, description="各時点の消費政策")
    D: DistributionPath | None = Field(default=None, description="各時点の分布")

    @property
    def T(self) -> int:
        """経路の長さ。"""
        return self.K.shape[0]


@typed
def simulate_transition(
    model: KSModel,
    ss: SteadyState,
    r_path: PricePath,
    w_path: PricePath,
    *,
    store_internals: bool = False,
) -> TransitionPath:
    """価格経路を受け取って集計経路を返す。ブロック写像 H そのもの。

    講義ノート 7.2 の3段階:

      第1段階  終端条件 V_T = V_ss から t = T-1, ..., 0 と後ろ向きに政策を求める
      第2段階  初期条件 D_0 = D_ss から t = 0, ..., T-1 と前向きに分布を進める
      第3段階  各時点で集計する

    ショックが時点 s にあっても、t < s の政策を求めるには将来の r_s が必要なので、
    「時点 s だけ解けばよい」のではなく終端から時点 0 までの後ろ向き計算が要る
    （ノート 7.2.1）。
    """
    T = r_path.shape[0]
    c = model.calibration

    # ---- 第1段階：後ろ向き（終端 = 定常状態） ----
    a_path = np.empty((T, model.n_e, model.n_a))
    c_path = np.empty((T, model.n_e, model.n_a))
    Va = ss.Va
    for t in reversed(range(T)):
        step = backward_egm(
            model.Pi @ Va, model.a_grid, model.income(w_path[t]), r_path[t], c.beta, c.eis
        )
        Va, a_path[t], c_path[t] = step.Va, step.a, step.c

    # ---- 第2・3段階：前向きに分布を進めつつ集計 ----
    K = np.empty(T)
    C = np.empty(T)
    D_path = np.empty((T, model.n_e, model.n_a))
    D = ss.D.copy()
    for t in range(T):
        D_path[t] = D
        K[t] = np.vdot(D, a_path[t])
        C[t] = np.vdot(D, c_path[t])
        D = forward_step(D, model.Pi, asset_lottery(model.a_grid, a_path[t]))

    if store_internals:
        return TransitionPath(K=K, C=C, a=a_path, c=c_path, D=D_path)
    return TransitionPath(K=K, C=C)
