"""sequence-jacobian ライブラリ版。

https://github.com/shade-econ/sequence-jacobian を使って、`ks` パッケージの自作実装と
「まったく同じモデル・同じグリッド・同じ定常状態」でヤコビアンを求める。

このモジュールだけ `sequence_jacobian` に依存する。

    uv pip install --python .venv/bin/python sequence-jacobian scipy numba
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sequence_jacobian as sj
from sequence_jacobian import SteadyStateDict, het, simple

from .calibration import KSModel, Prices
from .household import SteadyState
from .jacobian import DifferenceScheme, HouseholdJacobians
from .types import AssetGrid, FloatArray, IncomeByState, JacobianMatrix, MarginalValue, typed

__all__ = [
    "STRICT_TOLERANCES",
    "build_calibration",
    "direct_jacobians_via_library",
    "firm",
    "hh",
    "hh_extended",
    "library_jacobians",
    "market_clearing",
    "solve_general_equilibrium_with_library",
    "solve_steady_state",
    "steady_state_to_ks",
]

# sequence-jacobian は simple ブロックの引数に Displace 等の独自オブジェクトを渡すので、
# ライブラリが呼ぶコールバックの引数は Any にしてある。
# また、ライブラリは return 文の「変数名」を出力名として読む（行末コメントも不可）ので、
# `y = ...; return y` の形を崩せない（RET504 はこのファイルだけ無効）。


# =====================================================================
# HetBlock
#
# 自作の ks.household.backward_egm と一行ずつ対応する。違いは Pi @ Va_{t+1} の
# 期待計算だけで、それは HetBlock が backward_fun を呼ぶ前に済ませてくれる。
# =====================================================================


def hh_init(a_grid: AssetGrid, y: IncomeByState, r: float, eis: float) -> MarginalValue:
    """後ろ向き反復の初期値。ks.household.initial_marginal_value と同一。

    注意: sequence-jacobian は return 文に書かれた「変数名」を出力名として読み取るので、
    式をそのまま返さず必ず名前付きの変数を返すこと。
    """
    coh = (1 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    Va: MarginalValue = (1 + r) * (0.1 * coh) ** (-1 / eis)
    return Va


@het(exogenous="Pi", policy="a", backward="Va", backward_init=hh_init)
def hh(
    Va_p: MarginalValue, a_grid: AssetGrid, y: IncomeByState, r: float, beta: float, eis: float
) -> tuple[MarginalValue, FloatArray, FloatArray]:
    """内生グリッド法による1期分の後ろ向きステップ。Va_p = E_t[Va_{t+1}]。"""
    uc_nextgrid = beta * Va_p
    c_nextgrid = uc_nextgrid ** (-eis)
    coh = (1 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    a = sj.interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    sj.misc.setmin(a, a_grid[0])
    c = coh - a
    Va = (1 + r) * c ** (-1 / eis)
    return Va, a, c


def income(w: float, lbar: float, unemployment_insurance: float) -> IncomeByState:
    """雇用状態ごとの労働所得。ks.calibration.KSModel.income と同一。"""
    y = np.array([unemployment_insurance, w * lbar])
    return y


hh_extended = hh.add_hetinputs([income])


@simple
def firm(K: Any, N: Any, Z: Any, alpha: Any, delta: Any) -> tuple[Any, Any, Any]:  # noqa: ANN401
    """時点 t の生産は期首資本 K(-1) = K_{t-1} を使う（ノート ii ページのタイミング規約）。"""
    r = alpha * Z * (K(-1) / N) ** (alpha - 1) - delta
    w = (1 - alpha) * Z * (K(-1) / N) ** alpha
    Y = Z * K(-1) ** alpha * N ** (1 - alpha)
    return r, w, Y


@simple
def market_clearing(K: Any, A: Any) -> Any:  # noqa: ANN401
    """家計の期末資産集計 A が、企業に貸し出される資本 K に一致する。"""
    asset_mkt = A - K
    return asset_mkt


# =====================================================================
# 許容誤差
# =====================================================================

STRICT_TOLERANCES: dict[str, float] = {
    "backward_tol": 1e-12,
    "backward_maxit": 100_000,
    "forward_tol": 1e-13,
    "forward_maxit": 3_000_000,
}
"""既定より厳しい許容誤差。

このキャリブレーションでは beta * (1 + r_ss) = 0.99990 と 1 に極端に近く、遷移行列の
第2固有値も 1 に近い。そのため分布の前向き反復が「見かけ上は収束したのにまだ真の
定常分布から遠い」状態になりやすい（実測: max|D_new - D| < 1e-10 に 33,000 回かかり、
そのときの D はまだ真の値から 1.2e-7 ずれていて K が 1.1e-4 ずれる）。

既定の forward_tol = 1e-10 のままだと、この 1e-4 の誤差が定常状態に残る。
自作版は定常分布を反復ではなく線形方程式 D = Lambda' D で直接解いているのでこの問題がない。
"""


# =====================================================================
# 自作実装との橋渡し
# =====================================================================


@typed
def build_calibration(model: KSModel, prices: Prices) -> dict[str, float | FloatArray]:
    """`KSModel` から sequence-jacobian の calibration 辞書を作る。

    グリッドも遷移行列も自作版と同じオブジェクトを渡すので、両者の差は
    アルゴリズムの差だけになる。
    """
    c = model.calibration
    return {
        "Pi": model.Pi,
        "a_grid": model.a_grid,
        "beta": c.beta,
        "eis": c.eis,
        "lbar": c.lbar,
        "unemployment_insurance": c.unemployment_insurance,
        "r": prices.r,
        "w": prices.w,
    }


@typed
def solve_steady_state(model: KSModel, prices: Prices, *, strict: bool = True) -> SteadyStateDict:
    """価格を所与に、ライブラリで家計ブロックの定常状態を解く。"""
    options = STRICT_TOLERANCES if strict else {}
    return hh_extended.steady_state(build_calibration(model, prices), **options)


@typed
def steady_state_to_ks(ss_library: SteadyStateDict) -> SteadyState:
    """ライブラリの SteadyStateDict を自作の `SteadyState` に詰め替える。

    同じ定常状態の上で両方の直接法を走らせて、実装が一致することを確かめるために使う。
    """
    internals = ss_library.internals["hh"]
    return SteadyState(
        prices=Prices(r=float(ss_library["r"]), w=float(ss_library["w"])),
        K=float(ss_library["A"]),
        C=float(ss_library["C"]),
        y=internals["y"],
        Va=internals["Va"],
        a=internals["a"],
        c=internals["c"],
        D=internals["D"],
        backward_iterations=-1,
        distribution_residual=float("nan"),
        top_grid_mass=float(internals["D"][:, -10:].sum()),
    )


@typed
def library_jacobians(
    ss_library: SteadyStateDict, T: int = 5, eps: float = 1e-4, *, twosided: bool = True
) -> HouseholdJacobians:
    """ライブラリの `.jacobian()`（フェイクニュース法）を自作の型に詰め替える。

    `twosided` の既定を True にしてあるのは、ライブラリの既定（片側差分 h=1e-4）だと
    1期分の後ろ向きステップの微分誤差が O(1e-4) 残り、直接法と突き合わせたときに
    ずれて見えるため。
    """
    J = hh_extended.jacobian(
        ss_library, inputs=["r", "w"], outputs=["A", "C"], T=T, h=eps, twosided=twosided
    )
    return HouseholdJacobians(
        K_r=np.asarray(J["A"]["r"]),
        K_w=np.asarray(J["A"]["w"]),
        C_r=np.asarray(J["C"]["r"]),
        C_w=np.asarray(J["C"]["w"]),
        eps=eps,
        scheme=DifferenceScheme.CENTRAL if twosided else DifferenceScheme.ONE_SIDED,
    )


@typed
def direct_jacobians_via_library(
    ss_library: SteadyStateDict, T: int = 5, eps: float = 1e-4, *, subtract_baseline: bool = True
) -> HouseholdJacobians:
    """ライブラリの非線形移行経路を使って直接法でヤコビアンを構成する。

    `impulse_nonlinear` は自作の `simulate_transition` とまったく同じ計算をするが、
    返すのは「定常値 Y_ss からの偏差」であって「基準走行 Y^0 からの偏差」ではない。
    `subtract_baseline=True` でゼロショックを1本流して Y^0 を作り、それを差し引く
    （講義ノート 7.7）。差は実測で 3.5e-6 あり、そのままヤコビアンの誤差になる。
    """
    zero = {"K": np.zeros(T), "C": np.zeros(T)}
    if subtract_baseline:
        base = hh_extended.impulse_nonlinear(ss_library, {"r": np.zeros(T)}, outputs=["A", "C"])
        zero = {"K": np.asarray(base["A"]), "C": np.asarray(base["C"])}

    columns: dict[str, JacobianMatrix] = {
        f"{o}_{i}": np.empty((T, T)) for o in ("K", "C") for i in ("r", "w")
    }
    for input_ in ("r", "w"):
        for s in range(T):
            shock = np.zeros(T)
            shock[s] = eps
            path = hh_extended.impulse_nonlinear(ss_library, {input_: shock}, outputs=["A", "C"])
            columns[f"K_{input_}"][:, s] = (np.asarray(path["A"]) - zero["K"]) / eps
            columns[f"C_{input_}"][:, s] = (np.asarray(path["C"]) - zero["C"]) / eps

    return HouseholdJacobians.from_columns(columns, eps=eps, scheme=DifferenceScheme.ONE_SIDED)


@typed
def solve_general_equilibrium_with_library(model: KSModel, K_low: float = 11.58) -> SteadyStateDict:
    """一般均衡の定常状態をライブラリに解かせる（第10・11章の入口）。

    資産需要は r = 1/beta - 1 の近くで爆発するので、下側の境界は発散点
    `model.K_impatience_bound` より少し上に取る必要がある。
    """
    c = model.calibration
    ks_model = sj.create_model([hh_extended, firm, market_clearing], name="Krusell-Smith")
    calibration = {
        "Pi": model.Pi,
        "a_grid": model.a_grid,
        "beta": c.beta,
        "eis": c.eis,
        "lbar": c.lbar,
        "unemployment_insurance": c.unemployment_insurance,
        "alpha": c.alpha,
        "delta": c.delta,
        "Z": model.Z,
        "N": model.N,
    }
    return ks_model.solve_steady_state(
        calibration,
        unknowns={"K": (K_low, 40.0 * model.N)},
        targets={"asset_mkt": 0.0},
        solver="brentq",
        options={"hh": STRICT_TOLERANCES},
    )
