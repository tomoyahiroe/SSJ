"""`ks.calibration` — パラメータ、マルコフ連鎖の積分、価格と所得の仕様。"""

import numpy as np
import pytest
from pydantic import ValidationError

import ks

# =====================================================================
# 汎用ユーティリティ
# =====================================================================


def test_stationary_distribution_is_fixed_point_of_chain() -> None:
    """stationary_distribution は x P = x, Σx = 1, x >= 0 を満たすベクトルを返す。"""
    P = np.array([[0.9, 0.1], [0.5, 0.5]])
    x = ks.stationary_distribution(P)
    np.testing.assert_allclose(x @ P, x, atol=1e-13)
    assert x.sum() == pytest.approx(1.0)
    assert (x >= 0).all()
    # 解析解は 5/6 と 1/6
    np.testing.assert_allclose(x, [5 / 6, 1 / 6], atol=1e-12)


def test_stationary_distribution_raises_when_not_converged() -> None:
    """max_iter 回の反復で収束しなければ RuntimeError を投げる（黙って途中の値を返さない）。"""
    P = np.array([[0.9, 0.1], [0.5, 0.5]])
    with pytest.raises(RuntimeError, match="収束しませんでした"):
        ks.stationary_distribution(P, max_iter=0)


def test_exponential_grid_endpoints_and_spacing() -> None:
    """exponential_grid は [a_min, a_max] を n_a 点で、下側ほど密に、狭義単調増加で切る。"""
    grid = ks.exponential_grid(0.0, 300.0, 50)
    assert grid.shape == (50,)
    assert grid[0] == pytest.approx(0.0)
    assert grid[-1] == pytest.approx(300.0)
    spacing = np.diff(grid)
    assert (spacing > 0).all()
    assert (np.diff(spacing) > 0).all(), "格子の間隔は上に行くほど広がる"


# =====================================================================
# KSCalibration
# =====================================================================


def test_calibration_defaults_match_ks1998() -> None:
    """既定値は Krusell and Smith (1998) の四半期キャリブレーション。"""
    c = ks.KSCalibration()
    assert c.alpha == 0.36
    assert c.delta == 0.025
    assert c.beta == 0.99
    assert c.eis == 1.0
    assert (c.u_bad, c.u_good) == (0.10, 0.04)
    assert (c.z_bad, c.z_good) == (0.99, 1.01)
    assert (c.n_a, c.a_min, c.a_max) == (200, 0.0, 300.0)


def test_calibration_is_frozen() -> None:
    """KSCalibration は不変。属性への代入は ValidationError。"""
    c = ks.KSCalibration()
    with pytest.raises(ValidationError):
        c.beta = 0.95  # type: ignore[misc]


@pytest.mark.parametrize(
    "bad",
    [
        {"beta": 1.0},  # 割引因子は 1 未満
        {"alpha": 0.0},  # 資本分配率は正
        {"n_a": 5},  # グリッドは 10 点以上
        {"a_min": -1.0},  # 借入は不可
        {"dur_good": 1.0},  # 平均継続期間は 1 より大
    ],
)
def test_calibration_rejects_out_of_range_values(bad: dict[str, float]) -> None:
    """範囲外のパラメータは構築時に ValidationError で弾く。"""
    with pytest.raises(ValidationError):
        ks.KSCalibration.model_validate(bad)


def test_r_impatience_bound() -> None:
    """r_impatience_bound = 1/beta - 1。r がこれ以上だと資産需要が発散する。"""
    c = ks.KSCalibration(beta=0.95)
    assert c.r_impatience_bound == pytest.approx(1 / 0.95 - 1)


# =====================================================================
# KSMarkovChains（集計ショックの積分）
# =====================================================================


def test_transition_matrices_are_row_stochastic(model: ks.KSModel) -> None:
    """aggregate (2x2)・joint (4x4)・employment (2x2) はいずれも行確率行列。"""
    ch = model.chains
    for name, P in [
        ("aggregate", ch.aggregate),
        ("joint", ch.joint),
        ("employment", ch.employment),
    ]:
        np.testing.assert_allclose(P.sum(axis=1), 1.0, err_msg=name)
        assert (P >= 0).all(), name
    assert ch.aggregate.shape == (2, 2)
    assert ch.joint.shape == (4, 4)
    assert ch.employment.shape == (2, 2)


def test_aggregate_chain_has_expected_duration(model: ks.KSModel) -> None:
    """集計状態の継続確率は (D-1)/D。既定では好況・不況とも平均 8 四半期続く。"""
    ch = model.chains
    assert ch.aggregate[0, 0] == pytest.approx(7 / 8)  # 不況 → 不況
    assert ch.aggregate[1, 1] == pytest.approx(7 / 8)  # 好況 → 好況
    np.testing.assert_allclose(ch.aggregate_stationary, [0.5, 0.5])


def test_stationary_unemployment_is_consistent(model: ks.KSModel) -> None:
    """u_ss は 2状態チェーン Pi の定常失業率で、(A, l) 同時定常分布の周辺失業率と一致する。"""
    ch = model.chains
    assert ch.u_ss == pytest.approx(ks.stationary_distribution(ch.employment)[0])
    assert ch.u_ss == pytest.approx(ch.employment_marginal[0])
    assert ch.employment_marginal.sum() == pytest.approx(1.0)
    # 好況の失業率 4% と不況の 10% の間に入る
    assert model.calibration.u_good < ch.u_ss < model.calibration.u_bad


def test_employment_chain_is_mixture_of_conditional_matrices(model: ks.KSModel) -> None:
    """Pi の第 e 行は、(A, A') ごとの 2x2 行列の第 e 行を mixing_weights(e) で加重平均したもの。

    重み = pi(A | l) P(A' | A) は行ごとに異なる（共通の pi(A) で潰すと定常失業率がずれる）。
    """
    ch = model.chains
    for e in range(2):
        w = ch.mixing_weights(e)
        assert w.sum() == pytest.approx(1.0)
        mixed = sum(
            w[2 * a + a_next] * ch.conditional_employment_matrix(a, a_next)[e]
            for a in range(2)
            for a_next in range(2)
        )
        np.testing.assert_allclose(mixed, ch.employment[e], atol=1e-14)


def test_mean_unemployment_duration(model: ks.KSModel) -> None:
    """積分後のチェーンが含意する平均失業継続期間は 1 / (1 - Pi[失業, 失業])。"""
    ch = model.chains
    assert ch.mean_unemployment_duration == pytest.approx(1 / (1 - ch.employment[0, 0]))
    c = model.calibration
    assert c.dur_unemployed_good < ch.mean_unemployment_duration < c.dur_unemployed_bad


# =====================================================================
# KSModel（導出量と企業側の関係式）
# =====================================================================


def test_model_shapes(model: ks.KSModel) -> None:
    """n_e = 2、n_a はキャリブレーションどおり、a_grid は n_a 点。"""
    assert model.n_e == 2
    assert model.n_a == model.calibration.n_a
    assert model.a_grid.shape == (model.n_a,)
    assert model.Pi.shape == (2, 2)


def test_build_uses_given_calibration() -> None:
    """build(calibration) は渡したパラメータで組み立てる。省略時は既定値。"""
    custom = ks.KSCalibration(n_a=50, a_max=100.0)
    m = ks.KSModel.build(custom)
    assert m.calibration is custom
    assert m.a_grid.shape == (50,)
    assert m.a_grid[-1] == pytest.approx(100.0)
    assert ks.KSModel.build().calibration == ks.KSCalibration()


def test_aggregate_labor_and_tfp(model: ks.KSModel) -> None:
    """N = lbar (1 - u_ss)。Z は集計状態の定常分布で加重した TFP で、z_bad と z_good の間。"""
    c = model.calibration
    assert model.N == pytest.approx(c.lbar * (1 - model.u_ss))
    assert model.Z == pytest.approx(0.5 * (c.z_bad + c.z_good))
    assert c.z_bad < model.Z < c.z_good


def test_prices_from_capital(model: ks.KSModel) -> None:
    """prices(K) は r = alpha Z (K/N)^(alpha-1) - delta, w = (1-alpha) Z (K/N)^alpha。

    r は K について減少、w は K について増加。
    """
    c = model.calibration
    K = 12.0
    p = model.prices(K)
    k_n = K / model.N
    assert p.r == pytest.approx(c.alpha * model.Z * k_n ** (c.alpha - 1) - c.delta)
    assert p.w == pytest.approx((1 - c.alpha) * model.Z * k_n**c.alpha)
    p_more = model.prices(K + 1.0)
    assert p_more.r < p.r
    assert p_more.w > p.w


def test_impatience_bound_capital(model: ks.KSModel) -> None:
    """K_impatience_bound は r(K) = 1/beta - 1 となる K。"""
    r_at_bound = model.prices(model.K_impatience_bound).r
    assert r_at_bound == pytest.approx(model.calibration.r_impatience_bound)


def test_output_and_income(model: ks.KSModel) -> None:
    """output(K) = Z K^alpha N^(1-alpha)。income(w) = [失業給付, w lbar]。

    失業給付は w に比例しない「水準」。
    """
    c = model.calibration
    assert model.output(12.0) == pytest.approx(model.Z * 12.0**c.alpha * model.N ** (1 - c.alpha))
    y = model.income(2.0)
    assert y.shape == (2,)
    assert y[ks.Employment.UNEMPLOYED] == pytest.approx(c.unemployment_insurance)
    assert y[ks.Employment.EMPLOYED] == pytest.approx(2.0 * c.lbar)
    assert model.income(3.0)[0] == y[0], "失業給付は賃金に依存しない"


def test_prices_are_frozen() -> None:
    """Prices は不変の値オブジェクト。"""
    p = ks.Prices(r=0.01, w=2.0)
    with pytest.raises(ValidationError):
        p.r = 0.02  # type: ignore[misc]
