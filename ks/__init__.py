"""Krusell=Smith 型家計ブロックの Sequence Space Jacobian。

講義ノート `sequence_space_jacobian_lecture_notes_ch4_11.pdf` 第7章
「直接法：ヤコビアンを一列ずつ計算する」の実装。

使い方
------
    import ks

    model = ks.KSModel.build()                        # パラメータと導出量
    ss = ks.solve_general_equilibrium(model)          # 定常状態
    J = ks.jacobian_direct(model, ss, T=5)            # 直接法でヤコビアン
    J.K_r                                             # dK_t / dr_s の 5x5 行列

依存
----
`ks.ssj` だけは sequence-jacobian ライブラリを必要とする。そのほかは numpy のみ。

型について
----------
配列の型エイリアスは `ks.types` にまとめてある。`Float[FloatArray, "n_e n_a"]` の
ように軸の名前まで書いてあるので、関数シグネチャを見れば形が分かる。
既定では実行時にも検査される（環境変数 `KS_TYPECHECK=0` で無効化）。
"""

from .calibration import (
    KSCalibration,
    KSMarkovChains,
    KSModel,
    Prices,
    exponential_grid,
    stationary_distribution,
)
from .household import (
    BackwardStep,
    Lottery,
    SteadyState,
    TransitionPath,
    asset_lottery,
    backward_egm,
    forward_step,
    initial_marginal_value,
    interpolate_y,
    simulate_transition,
    solve_general_equilibrium,
    solve_household,
    stationary_from_policy,
    transition_matrix,
)
from .jacobian import (
    DifferenceScheme,
    HouseholdJacobians,
    jacobian_direct,
    jacobian_fake_news,
)
from .types import (
    AggregatePath,
    AggregateState,
    AssetGrid,
    Distribution,
    Employment,
    EmploymentTransition,
    IncomeByState,
    JacobianMatrix,
    MarginalValue,
    PolicyFunction,
    PricePath,
)

__all__ = [
    # パラメータ
    "KSCalibration",
    "KSMarkovChains",
    "KSModel",
    "Prices",
    "exponential_grid",
    "stationary_distribution",
    # 家計問題
    "BackwardStep",
    "Lottery",
    "SteadyState",
    "TransitionPath",
    "asset_lottery",
    "backward_egm",
    "forward_step",
    "initial_marginal_value",
    "interpolate_y",
    "simulate_transition",
    "solve_general_equilibrium",
    "solve_household",
    "stationary_from_policy",
    "transition_matrix",
    # ヤコビアン
    "DifferenceScheme",
    "HouseholdJacobians",
    "jacobian_direct",
    "jacobian_fake_news",
    # 型
    "AggregatePath",
    "AggregateState",
    "AssetGrid",
    "Distribution",
    "Employment",
    "EmploymentTransition",
    "IncomeByState",
    "JacobianMatrix",
    "MarginalValue",
    "PolicyFunction",
    "PricePath",
]
