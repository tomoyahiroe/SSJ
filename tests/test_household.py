"""`ks.household` — 補間、くじ表現、EGM、分布の前向き、定常状態、移行経路の仕様。"""

import numpy as np
import pytest

import ks

# =====================================================================
# 補間とくじ表現
# =====================================================================


def test_interpolate_y_matches_np_interp_inside_range() -> None:
    """interpolate_y は各行で線形補間する。データ点の範囲内では np.interp と一致。"""
    x = np.array([[0.0, 1.0, 2.0, 4.0], [1.0, 2.0, 3.0, 5.0]])
    y = np.array([0.0, 10.0, 20.0, 40.0])
    xq = np.array([[0.5, 1.5, 3.0, 2.0], [1.5, 2.5, 4.0, 3.0]])
    out = ks.interpolate_y(x, xq, y)
    for row in range(2):
        np.testing.assert_allclose(out[row], np.interp(xq[row], x[row], y))


def test_interpolate_y_extrapolates_linearly() -> None:
    """範囲外では端の2点の直線で外挿する（np.interp のように端の値で頭打ちにしない）。

    x と xq は同じ形 (n_e, n_a) でなければならない（`typed` が検査する）。
    """
    x = np.array([[0.0, 1.0, 2.0]])
    y = np.array([0.0, 10.0, 20.0])
    out = ks.interpolate_y(x, np.array([[-1.0, 3.0, 1.0]]), y)
    np.testing.assert_allclose(out, [[-10.0, 30.0, 10.0]])


def test_asset_lottery_reconstructs_policy(model: ks.KSModel, ss: ks.SteadyState) -> None:
    """くじ表現は政策を再現する: weight * grid[index] + (1 - weight) * grid[index + 1] = a。

    index は [0, n_a - 2]。グリッド内の政策なら weight は [0, 1]。
    上端を超える政策は最後の区間で線形外挿になる（weight < 0）が、再現の等式は保たれる。
    """
    lot = ks.asset_lottery(model.a_grid, ss.a)
    assert lot.index.shape == ss.a.shape
    assert lot.index.min() >= 0
    assert lot.index.max() <= model.n_a - 2
    inside = ss.a <= model.a_grid[-1]
    assert (lot.weight[inside] >= 0).all()
    assert (lot.weight[inside] <= 1).all()
    rebuilt = lot.weight * model.a_grid[lot.index] + (1 - lot.weight) * model.a_grid[lot.index + 1]
    np.testing.assert_allclose(rebuilt, ss.a, atol=1e-12)


def test_asset_lottery_at_grid_points() -> None:
    """政策が格子点 grid[k] そのものなら index = k - 1, weight = 0（上側の点に重み 1）。

    下端 grid[0] だけは index = 0, weight = 1。いずれも SSJ の interpolate_coord_robust と同じ規約。
    """
    grid = np.array([0.0, 1.0, 3.0, 6.0])
    lot = ks.asset_lottery(grid, np.array([[0.0, 1.0, 3.0, 6.0]]))
    np.testing.assert_array_equal(lot.index, [[0, 0, 1, 2]])
    np.testing.assert_allclose(lot.weight, [[1.0, 0.0, 0.0, 0.0]])


# =====================================================================
# 後ろ向きステップ（EGM）
# =====================================================================


def test_backward_egm_step_properties(model: ks.KSModel, ss: ks.SteadyState) -> None:
    """backward_egm は1期分の後ろ向きステップ。

    - 予算制約 c + a = (1 + r) a_{-1} + y を等式で満たす
    - 借入制約 a >= a_min を満たす
    - Va = (1 + r) c^(-1/eis)（限界価値の定義）
    - 期首資産が多いほど c も a も減らない（単調）
    """
    c = model.calibration
    step = ks.backward_egm(model.Pi @ ss.Va, model.a_grid, ss.y, ss.prices.r, c.beta, c.eis)
    coh = (1 + ss.prices.r) * model.a_grid[None, :] + ss.y[:, None]
    np.testing.assert_allclose(step.c + step.a, coh, rtol=1e-12)
    assert (step.a >= model.a_grid[0]).all()
    np.testing.assert_allclose(step.Va, (1 + ss.prices.r) * step.c ** (-1 / c.eis), rtol=1e-12)
    assert (np.diff(step.a, axis=1) >= 0).all()
    assert (np.diff(step.c, axis=1) > 0).all()


def test_backward_egm_binds_at_low_assets(model: ks.KSModel, ss: ks.SteadyState) -> None:
    """失業・資産ゼロの家計は借入制約に張り付く（a = a_min）。"""
    c = model.calibration
    step = ks.backward_egm(model.Pi @ ss.Va, model.a_grid, ss.y, ss.prices.r, c.beta, c.eis)
    assert step.a[ks.Employment.UNEMPLOYED, 0] == model.a_grid[0]


def test_initial_marginal_value_formula(model: ks.KSModel) -> None:
    """initial_marginal_value は「消費 = 手持ちの 1 割」から作った (1 + r) c^(-1/eis)。"""
    y = model.income(2.0)
    Va = ks.initial_marginal_value(model.a_grid, y, 0.01, 1.0)
    coh = 1.01 * model.a_grid[None, :] + y[:, None]
    np.testing.assert_allclose(Va, 1.01 * (0.1 * coh) ** (-1.0))
    assert Va.shape == (2, model.n_a)


# =====================================================================
# 前向きステップと定常分布
# =====================================================================


def test_forward_step_preserves_mass(model: ks.KSModel, ss: ks.SteadyState) -> None:
    """forward_step は確率質量を保存し、非負性を保つ。"""
    lot = ks.asset_lottery(model.a_grid, ss.a)
    D_next = ks.forward_step(ss.D, model.Pi, lot)
    assert D_next.sum() == pytest.approx(1.0)
    assert (D_next >= 0).all()


def test_forward_step_equals_transition_matrix(model: ks.KSModel, ss: ks.SteadyState) -> None:
    """forward_step(D) は D' = Lambda' D と同じ（式 7.19）。

    Lambda は transition_matrix が作る (n_e n_a) x (n_e n_a) の行確率行列。
    """
    lot = ks.asset_lottery(model.a_grid, ss.a)
    lam = ks.transition_matrix(model.Pi, lot)
    n = model.n_e * model.n_a
    assert lam.shape == (n, n)
    np.testing.assert_allclose(lam.sum(axis=1), 1.0)
    D0 = ss.D
    D0 = D0 / D0.sum()
    via_matrix = (lam.T @ D0.ravel()).reshape(model.n_e, model.n_a)
    np.testing.assert_allclose(ks.forward_step(D0, model.Pi, lot), via_matrix, atol=1e-15)


def test_stationary_from_policy_is_fixed_point(model: ks.KSModel, ss: ks.SteadyState) -> None:
    """stationary_from_policy は D = Lambda' D, Σ D = 1 を線形方程式で解き、残差も返す。"""
    lot = ks.asset_lottery(model.a_grid, ss.a)
    st = ks.stationary_from_policy(model.Pi, lot)
    assert st.D.shape == (model.n_e, model.n_a)
    assert st.D.sum() == pytest.approx(1.0)
    assert st.residual < 1e-12
    np.testing.assert_allclose(ks.forward_step(st.D, model.Pi, lot), st.D, atol=1e-12)


# =====================================================================
# 定常状態
# =====================================================================


def test_solve_household_aggregates(model: ks.KSModel, ss: ks.SteadyState) -> None:
    """solve_household の K, C は定常分布で政策を集計したもの。政策は後ろ向き反復の不動点。"""
    part = ks.solve_household(model, ss.prices)
    assert part.K == pytest.approx(float(np.vdot(part.D, part.a)))
    assert part.C == pytest.approx(float(np.vdot(part.D, part.c)))
    c = model.calibration
    again = ks.backward_egm(model.Pi @ part.Va, model.a_grid, part.y, part.prices.r, c.beta, c.eis)
    assert np.max(np.abs(again.a - part.a)) < 1e-9
    assert part.distribution_residual < 1e-12


def test_solve_household_warm_start(model: ks.KSModel, ss: ks.SteadyState) -> None:
    """Va_init に定常の Va を渡すと、少ない反復で同じ定常状態に着く。"""
    warm = ks.solve_household(model, ss.prices, Va_init=ss.Va)
    assert warm.K == pytest.approx(ss.K, abs=1e-8)
    assert warm.backward_iterations <= 20


def test_solve_household_raises_if_not_converged(model: ks.KSModel, ss: ks.SteadyState) -> None:
    """backward_maxit 回で政策が収束しなければ RuntimeError。"""
    with pytest.raises(RuntimeError, match="収束しませんでした"):
        ks.solve_household(model, ss.prices, backward_maxit=10)


def test_general_equilibrium_clears_capital_market(model: ks.KSModel, ss: ks.SteadyState) -> None:
    """solve_general_equilibrium は「家計の資産需要 = K」かつ「価格 = prices(K)」を満たす。

    均衡の r は 1/beta - 1 より小さく、K は K_impatience_bound より大きい。
    """
    assert ss.excess_demand(ss.K) == pytest.approx(0.0, abs=1e-8)
    p = model.prices(ss.K)
    assert ss.prices.r == pytest.approx(p.r, abs=1e-8)
    assert ss.prices.w == pytest.approx(p.w, abs=1e-8)
    assert ss.prices.r < model.calibration.r_impatience_bound
    assert ss.K > model.K_impatience_bound
    assert ss.D.sum() == pytest.approx(1.0)


def test_general_equilibrium_raises_if_upper_bound_too_low(model: ks.KSModel) -> None:
    """K_high での超過需要が正なら（区間に均衡がない）RuntimeError。

    K_impatience_bound（約 11.56）のすぐ上では資産需要が急増するので、K_high = 11.6 で超過需要は正。
    """
    with pytest.raises(RuntimeError, match="超過需要"):
        ks.solve_general_equilibrium(model, K_low=11.59, K_high=11.6)


def test_general_equilibrium_verbose(
    model: ks.KSModel, ss: ks.SteadyState, capsys: pytest.CaptureFixture[str]
) -> None:
    """verbose=True で二分法の各ステップ（K, r, 超過需要）を表示する。"""
    ks.solve_general_equilibrium(model, K_low=ss.K - 1e-6, K_high=ss.K + 1e-6, verbose=True)
    out = capsys.readouterr().out
    assert "超過需要" in out


# =====================================================================
# 移行経路（ブロック写像 H）
# =====================================================================


def test_simulate_transition_baseline_is_flat(model: ks.KSModel, ss: ks.SteadyState) -> None:
    """定常価格を T 期流すと、集計経路は定常値のまま（基準走行 Y^0 ≈ Y_ss）。"""
    T = 5
    r_ss = np.full(T, ss.prices.r)
    w_ss = np.full(T, ss.prices.w)
    path = ks.simulate_transition(model, ss, r_ss, w_ss)
    assert path.T == T
    np.testing.assert_allclose(path.K, ss.K, atol=1e-9)
    np.testing.assert_allclose(path.C, ss.C, atol=1e-9)
    assert path.a is None
    assert path.D is None


def test_simulate_transition_store_internals(model: ks.KSModel, ss: ks.SteadyState) -> None:
    """store_internals=True なら各時点の政策と分布も返す。D_0 は定常分布（初期条件）。"""
    T = 4
    path = ks.simulate_transition(
        model, ss, np.full(T, ss.prices.r), np.full(T, ss.prices.w), store_internals=True
    )
    assert path.a is not None
    assert path.c is not None
    assert path.D is not None
    assert path.a.shape == path.c.shape == path.D.shape == (T, model.n_e, model.n_a)
    np.testing.assert_array_equal(path.D[0], ss.D)


def test_simulate_transition_anticipates_future_shock(
    model: ks.KSModel, ss: ks.SteadyState
) -> None:
    """時点 s の r の上昇は、時点 s の貯蓄を増やし、それ以前の時点にも（ニュースとして）効く。"""
    T, s = 5, 3
    r_ss = np.full(T, ss.prices.r)
    w_ss = np.full(T, ss.prices.w)
    r_pert = r_ss.copy()
    r_pert[s] += 1e-4
    base = ks.simulate_transition(model, ss, r_ss, w_ss)
    pert = ks.simulate_transition(model, ss, r_pert, w_ss)
    assert pert.K[s] > base.K[s]
    assert (pert.K[:s] > base.K[:s]).all(), "t < s でも将来の r を見て貯蓄が動く"
