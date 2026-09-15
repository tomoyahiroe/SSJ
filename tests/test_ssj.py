"""`ks.ssj` — sequence-jacobian ライブラリ版と自作実装の突き合わせの仕様。

「同じモデル・同じグリッド・同じ定常状態」でライブラリと自作の結果が一致することが、
自作実装の検証になっている。
"""

import numpy as np
import pytest
from sequence_jacobian import SteadyStateDict

import ks
from ks import ssj as lib
from ks.jacobian import DifferenceScheme

T = 5


@pytest.fixture(scope="module")
def ss_lib(model: ks.KSModel, ss: ks.SteadyState) -> SteadyStateDict:
    """自作の均衡価格をライブラリに渡して解いた、家計ブロックの定常状態（締めた許容誤差）。"""
    return lib.solve_steady_state(model, ss.prices, strict=True)


@pytest.fixture(scope="module")
def j_direct_one(model: ks.KSModel, ss: ks.SteadyState) -> ks.HouseholdJacobians:
    """自作の直接法・片側差分。"""
    return ks.jacobian_direct(model, ss, T=T, scheme=DifferenceScheme.ONE_SIDED)


@pytest.fixture(scope="module")
def j_direct_central(model: ks.KSModel, ss: ks.SteadyState) -> ks.HouseholdJacobians:
    """自作の直接法・中央差分。"""
    return ks.jacobian_direct(model, ss, T=T, scheme=DifferenceScheme.CENTRAL)


# =====================================================================
# ブロックの部品は自作版と同一
# =====================================================================


def test_hh_init_equals_initial_marginal_value(model: ks.KSModel) -> None:
    """hh_init は ks.household.initial_marginal_value と同じ初期値を返す。"""
    y = model.income(2.0)
    np.testing.assert_array_equal(
        lib.hh_init(model.a_grid, y, 0.01, 1.0),
        ks.initial_marginal_value(model.a_grid, y, 0.01, 1.0),
    )


def test_income_equals_model_income(model: ks.KSModel) -> None:
    """income は ks.calibration.KSModel.income と同じ所得ベクトルを返す。"""
    c = model.calibration
    np.testing.assert_array_equal(
        lib.income(2.0, c.lbar, c.unemployment_insurance), model.income(2.0)
    )


def test_build_calibration_passes_same_objects(model: ks.KSModel, ss: ks.SteadyState) -> None:
    """build_calibration は自作版と同じ Pi・a_grid をそのまま渡す（差はアルゴリズムだけになる）。"""
    cal = lib.build_calibration(model, ss.prices)
    assert set(cal) == {"Pi", "a_grid", "beta", "eis", "lbar", "unemployment_insurance", "r", "w"}
    assert cal["Pi"] is model.Pi
    assert cal["a_grid"] is model.a_grid
    assert cal["r"] == ss.prices.r


# =====================================================================
# 定常状態
# =====================================================================


def test_library_steady_state_matches_ours(ss: ks.SteadyState, ss_lib: SteadyStateDict) -> None:
    """締めた許容誤差なら、ライブラリの定常状態は自作版と K, C が 1e-6 未満で一致。"""
    converted = lib.steady_state_to_ks(ss_lib)
    assert converted.K == pytest.approx(ss.K, abs=1e-6)
    assert converted.C == pytest.approx(ss.C, abs=1e-6)
    assert converted.prices == ss.prices
    assert converted.D.shape == ss.D.shape
    assert converted.D.sum() == pytest.approx(1.0)


def test_strict_tolerances_matter(model: ks.KSModel, ss: ks.SteadyState) -> None:
    """ライブラリ既定の許容誤差（strict=False）だと分布の反復が早く止まり、K が自作版からずれる。

    beta (1 + r) が 1 に極端に近いこのキャリブレーション特有の問題。
    STRICT_TOLERANCES はそれを締める。
    """
    loose = lib.steady_state_to_ks(lib.solve_steady_state(model, ss.prices, strict=False))
    strict = lib.steady_state_to_ks(lib.solve_steady_state(model, ss.prices, strict=True))
    assert abs(loose.K - ss.K) > abs(strict.K - ss.K)
    assert {"backward_tol", "backward_maxit", "forward_tol", "forward_maxit"} == set(
        lib.STRICT_TOLERANCES
    )


def test_library_general_equilibrium_matches_ours(model: ks.KSModel, ss: ks.SteadyState) -> None:
    """ライブラリに一般均衡を解かせても、均衡 K は自作の二分法と 1e-6 未満で一致する。"""
    ss_ge = lib.solve_general_equilibrium_with_library(model)
    assert float(ss_ge["K"]) == pytest.approx(ss.K, abs=1e-6)
    assert float(ss_ge["asset_mkt"]) == pytest.approx(0.0, abs=1e-8)


# =====================================================================
# ヤコビアン
# =====================================================================


def test_library_fake_news_matches_direct(
    ss_lib: SteadyStateDict, j_direct_central: ks.HouseholdJacobians
) -> None:
    """ライブラリの .jacobian() は twosided=True なら自作の直接法（中央差分）と一致する。"""
    j_lib = lib.library_jacobians(ss_lib, T=T, twosided=True)
    assert j_lib.T == T
    assert j_lib.scheme is DifferenceScheme.CENTRAL
    assert j_lib.max_difference(j_direct_central) < 1e-6


def test_library_one_sided_is_marked(ss_lib: SteadyStateDict) -> None:
    """twosided=False で作ったものは scheme = ONE_SIDED と記録される。"""
    j_lib = lib.library_jacobians(ss_lib, T=T, twosided=False)
    assert j_lib.scheme is DifferenceScheme.ONE_SIDED


def test_direct_via_library_needs_baseline_subtraction(
    ss_lib: SteadyStateDict, j_direct_one: ks.HouseholdJacobians
) -> None:
    """ライブラリの非線形移行経路で直接法を組むと、基準走行 Y^0 を差し引けば自作版と一致する。

    impulse_nonlinear は「定常値からの偏差」を返すので、差し引かない（subtract_baseline=False）と
    基準走行の誤差が 1/eps 倍されてヤコビアンに残る（講義ノート 7.7）。
    """
    with_base = lib.direct_jacobians_via_library(ss_lib, T=T, subtract_baseline=True)
    without = lib.direct_jacobians_via_library(ss_lib, T=T, subtract_baseline=False)
    assert with_base.max_difference(j_direct_one) < 1e-6
    assert without.max_difference(j_direct_one) > with_base.max_difference(j_direct_one)
    assert with_base.scheme is DifferenceScheme.ONE_SIDED
