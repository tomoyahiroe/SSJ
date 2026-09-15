"""`ks.jacobian` — 直接法（第7章）とフェイクニュース法（第8章）の仕様。"""

import numpy as np
import pytest

import ks
from ks.jacobian import BlockInput, BlockOutput, DifferenceScheme

T = 5
OUTPUTS: tuple[BlockOutput, ...] = ("K", "C")
INPUTS: tuple[BlockInput, ...] = ("r", "w")


@pytest.fixture(scope="module")
def j_one(model: ks.KSModel, ss: ks.SteadyState) -> ks.HouseholdJacobians:
    """直接法・片側差分（宿題の設定）。"""
    return ks.jacobian_direct(model, ss, T=T, scheme=DifferenceScheme.ONE_SIDED)


@pytest.fixture(scope="module")
def j_central(model: ks.KSModel, ss: ks.SteadyState) -> ks.HouseholdJacobians:
    """直接法・中央差分。"""
    return ks.jacobian_direct(model, ss, T=T, scheme=DifferenceScheme.CENTRAL)


@pytest.fixture(scope="module")
def j_fake_news(model: ks.KSModel, ss: ks.SteadyState) -> ks.HouseholdJacobians:
    """フェイクニュース法・中央差分（既定）。"""
    return ks.jacobian_fake_news(model, ss, T=T)


# =====================================================================
# 入れ物
# =====================================================================


def test_difference_scheme_values() -> None:
    """DifferenceScheme は文字列 enum。片側 "one_sided" と中央 "central"。"""
    assert DifferenceScheme.ONE_SIDED.value == "one_sided"
    assert DifferenceScheme.CENTRAL.value == "central"


def test_jacobians_container(j_one: ks.HouseholdJacobians) -> None:
    """HouseholdJacobians は 4 本の T x T 行列を持ち、get(出力, 入力) で属性と同じものを返す。"""
    assert j_one.T == T
    for output in OUTPUTS:
        for input_ in INPUTS:
            J = j_one.get(output, input_)
            assert J.shape == (T, T)
            assert J is getattr(j_one, f"{output}_{input_}")
    assert j_one.eps == 1e-4
    assert j_one.scheme is DifferenceScheme.ONE_SIDED


def test_from_columns_round_trip(j_one: ks.HouseholdJacobians) -> None:
    """from_columns は {"K_r": ..., "C_w": ...} の辞書から同じものを組み立てる。"""
    cols = {f"{o}_{i}": j_one.get(o, i) for o in OUTPUTS for i in INPUTS}
    rebuilt = ks.HouseholdJacobians.from_columns(cols, eps=j_one.eps, scheme=j_one.scheme)
    assert rebuilt.max_difference(j_one) == 0.0
    assert rebuilt.scheme is j_one.scheme


def test_max_difference_is_symmetric(
    j_one: ks.HouseholdJacobians, j_central: ks.HouseholdJacobians
) -> None:
    """max_difference は 4 本すべてにわたる最大絶対差。自分自身とは 0、対称。"""
    assert j_one.max_difference(j_one) == 0.0
    assert j_one.max_difference(j_central) == j_central.max_difference(j_one)
    assert j_one.max_difference(j_central) > 0.0


# =====================================================================
# 直接法
# =====================================================================


def test_direct_budget_identity_at_time_zero(
    model: ks.KSModel, ss: ks.SteadyState, j_one: ks.HouseholdJacobians
) -> None:
    """時点 0 の予算制約 K_0 + C_0 = (1 + r_0) K_ss + Y_0 をヤコビアンが満たす。

    dK_0/dr_s + dC_0/dr_s は s = 0 で K_ss、s > 0 で 0。
    dK_0/dw_s + dC_0/dw_s は s = 0 で N = lbar (1 - u)、s > 0 で 0。
    """
    r_total = j_one.K_r[0] + j_one.C_r[0]
    w_total = j_one.K_w[0] + j_one.C_w[0]
    expected_r = np.zeros(T)
    expected_r[0] = ss.K
    expected_w = np.zeros(T)
    expected_w[0] = model.N
    np.testing.assert_allclose(r_total, expected_r, atol=1e-8)
    np.testing.assert_allclose(w_total, expected_w, atol=1e-8)


def test_direct_response_signs(j_one: ks.HouseholdJacobians) -> None:
    """r の上昇は同時点の貯蓄を増やし（対角 > 0）、それを見越して以前の時点でも貯蓄を増やす。"""
    assert (np.diag(j_one.K_r) > 0).all()
    assert (j_one.K_r[:3, 3] > 0).all()


def test_direct_central_vs_one_sided(
    j_one: ks.HouseholdJacobians, j_central: ks.HouseholdJacobians
) -> None:
    """片側差分と中央差分の差は O(eps)。同じ列を測っているので桁違いにはならない。"""
    diff = j_one.max_difference(j_central)
    assert 1e-8 < diff < 1e-2
    assert j_central.scheme is DifferenceScheme.CENTRAL


def test_direct_verbose(
    model: ks.KSModel, ss: ks.SteadyState, capsys: pytest.CaptureFixture[str]
) -> None:
    """verbose=True で列ごとの進捗を表示する（入力 2 本 x T 列）。"""
    ks.jacobian_direct(model, ss, T=2, verbose=True)
    out = capsys.readouterr().out
    assert out.count("完了") == 2 * 2


# =====================================================================
# フェイクニュース法（直接法の検証用）
# =====================================================================


def test_fake_news_matches_direct_central(
    j_central: ks.HouseholdJacobians, j_fake_news: ks.HouseholdJacobians
) -> None:
    """中央差分どうしなら、フェイクニュース法は直接法と 1e-6 未満で一致する。"""
    assert j_fake_news.max_difference(j_central) < 1e-6
    assert j_fake_news.scheme is DifferenceScheme.CENTRAL


def test_fake_news_one_sided_deviates_by_eps(
    model: ks.KSModel, ss: ks.SteadyState, j_one: ks.HouseholdJacobians
) -> None:
    """フェイクニュース法を片側差分にすると、直接法（片側）から O(eps) ずれる。

    微分しているのが「1期分の後ろ向きステップ」で、その誤差が対角方向に累積するため。
    """
    j_fn_one = ks.jacobian_fake_news(model, ss, T=T, scheme=DifferenceScheme.ONE_SIDED)
    diff = j_fn_one.max_difference(j_one)
    assert 1e-6 < diff < 1e-2
