# -*- coding: utf-8 -*-
"""
Krusell=Smith 型家計ブロックのヤコビアン ―― sequence-jacobian ライブラリ版
=========================================================================

https://github.com/shade-econ/sequence-jacobian （PyPI: sequence-jacobian）を使って、
ks_direct_scratch.py と「まったく同じモデル・同じグリッド・同じ定常状態」で
家計ブロックのヤコビアンを求め、自作の直接法と数値を突き合わせる。

実行:
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python sequence-jacobian scipy numba matplotlib
    .venv/bin/python ks_direct_ssj.py

このファイルが示すこと
--------------------
  A. HetBlock の書き方が、自作版の backward_egm とほぼ一行ずつ対応していること
  B. ライブラリの .jacobian() は第8章のフェイクニュース法で計算されていること
  C. ライブラリの .impulse_nonlinear() を T 回呼べば、それがそのまま第7章の直接法であること
  D. 直接法（自作・ライブラリ）とフェイクニュース法（ライブラリ）が一致すること
"""

from __future__ import annotations

import numpy as np
import sequence_jacobian as sj
from sequence_jacobian import het, simple

import ks_direct_scratch as scratch


# =====================================================================
# Part 1  HetBlock の定義
#         自作版 backward_egm との対応:
#
#           uc_nextgrid = beta * Va_p          <-  beta * EVa_p
#           c_nextgrid  = uc_nextgrid ** -eis  <-  同じ
#           coh         = (1+r) a + y          <-  同じ
#           a           = interpolate_y(...)   <-  同じ
#           Va          = (1+r) c ** (-1/eis)  <-  同じ
#
#         違いは Pi @ Va_{t+1} の期待計算だけ。ライブラリでは HetBlock が
#         backward_fun を呼ぶ前に Va_p = Pi @ Va を作って渡してくれる。
# =====================================================================


def hh_init(a_grid, y, r, eis):
    """後ろ向き反復の初期値。scratch.backward_egm_init と同一。

    注意: sequence-jacobian は return 文に書かれた「変数名」を出力名として読み取るので、
    式をそのまま返さず必ず名前付きの変数を返すこと（ここでは Va）。
    """
    coh = (1 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    Va = (1 + r) * (0.1 * coh) ** (-1 / eis)
    return Va


@het(exogenous="Pi", policy="a", backward="Va", backward_init=hh_init)
def hh(Va_p, a_grid, y, r, beta, eis):
    """内生グリッド法による1期後ろ向きステップ。Va_p = E_t[Va_{t+1}] は HetBlock が用意する。"""
    uc_nextgrid = beta * Va_p
    c_nextgrid = uc_nextgrid ** (-eis)
    coh = (1 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    a = sj.interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    sj.misc.setmin(a, a_grid[0])
    c = coh - a
    Va = (1 + r) * c ** (-1 / eis)
    return Va, a, c


def income(w, lbar, unemployment_insurance):
    """雇用状態ごとの労働所得。並びは [失業, 就業]。

    元の 8_2_krusell_and_smith.py（replicate_model = 0）と同じで、
    失業保険は w に比例しない水準で与えられる。
    """
    y = np.array([unemployment_insurance, w * lbar])
    return y


hh_ext = hh.add_hetinputs([income])


# =====================================================================
# Part 2  企業ブロックと市場清算（第10・11章の入口）
# =====================================================================


@simple
def firm(K, N, Z, alpha, delta):
    """時点 t の生産は期首資本 K(-1) = K_{t-1} を使う。ノート ii ページのタイミング規約。"""
    r = alpha * Z * (K(-1) / N) ** (alpha - 1) - delta
    w = (1 - alpha) * Z * (K(-1) / N) ** alpha
    Y = Z * K(-1) ** alpha * N ** (1 - alpha)
    return r, w, Y


@simple
def mkt_clearing(K, A):
    """家計の期末資産集計 A が、企業に貸し出される資本 K に一致する。"""
    asset_mkt = A - K
    return asset_mkt


# =====================================================================
# Part 3  自作版とまったく同じキャリブレーションを組み立てる
# =====================================================================


SS_OPTIONS = dict(
    backward_tol=1e-12,
    backward_maxit=100_000,
    forward_tol=1e-13,
    forward_maxit=3_000_000,
)
"""既定より厳しい許容誤差。

このキャリブレーションでは beta * (1 + r_ss) = 0.99989... と 1 に極端に近いため、
遷移行列 Λ_ss の第2固有値も 1 に非常に近く、分布の前向き反復が「見かけ上は収束したのに
まだ真の定常分布から遠い」という状態になりやすい。実測では

    max|D_new - D| < 1e-10 に到達  … 33,000 回
    しかしそのときの D は真の定常分布から max 1.2e-7 ずれており、K は 1.1e-4 ずれる

既定の forward_tol = 1e-10 のままだと、この 1e-4 の誤差が定常状態に残る。
ノート 7.7 が言うとおり、直接法では同じ基準走行 Y^0 を差し引くのでこの誤差の大半は
相殺されるが、自作版と数値を突き合わせるときは邪魔になるので締めておく。
"""


def build_calibration(p: scratch.KSParams, r: float, w: float) -> dict:
    return dict(
        Pi=p.Pi,                    # 雇用の遷移行列（ノートブックの定常分布から作ったもの）
        a_grid=p.a_grid,            # 資産グリッド（元コードの exponential_grid と同一）
        beta=p.beta,
        eis=p.eis,
        lbar=p.lbar,
        unemployment_insurance=p.unemployment_insurance,
        r=r,
        w=w,
    )


# =====================================================================
# Part 4  ライブラリで「直接法」を回す
#         ノート 7.2 の3段階は impulse_nonlinear がそのまま実行してくれる
# =====================================================================


def jacobian_direct_via_library(hh_block, ss, T: int = 5, h: float = 1e-4,
                                twosided: bool = False, subtract_baseline: bool = True) -> dict:
    """ライブラリの非線形移行経路を使って直接法でヤコビアンを構成する。

    impulse_nonlinear(ss, {'r': dr}) は
        「終端 V_T = V_ss から後ろ向き → 初期 D_0 = D_ss から前向き → 集計」
    を行う。つまり自作版の td_household とまったく同じもの。

    ただし返ってくるのは「定常値 Y_ss からの偏差」であって、「基準走行 Y^0 からの偏差」
    ではない。有限期間・終端条件・数値誤差のせいで Y^0 は完全には平坦にならないので、
    ノート 7.7 が言うとおり Y^0 を差し引いた方がよい。
    subtract_baseline=True だとゼロショックを1本流して Y^0 を作り、それを差し引く。
    """
    J = {f"{o},{i}": np.empty((T, T)) for o in ("A", "C") for i in ("r", "w")}
    zero = {o: np.zeros(T) for o in ("A", "C")}
    if subtract_baseline:
        base = hh_block.impulse_nonlinear(ss, {"r": np.zeros(T)}, outputs=["A", "C"])
        zero = {o: np.asarray(base[o]) for o in ("A", "C")}

    for i in ("r", "w"):
        for s in range(T):
            e_s = np.zeros(T)
            e_s[s] = h
            td = hh_block.impulse_nonlinear(ss, {i: e_s}, outputs=["A", "C"])
            if twosided:
                td_m = hh_block.impulse_nonlinear(ss, {i: -e_s}, outputs=["A", "C"])
                for o in ("A", "C"):
                    J[f"{o},{i}"][:, s] = (td[o] - td_m[o]) / (2 * h)
            else:
                for o in ("A", "C"):
                    J[f"{o},{i}"][:, s] = (td[o] - zero[o]) / h
    return J


# =====================================================================
# Part 5  実行
# =====================================================================


def scratch_ss_from_ssj(p: scratch.KSParams, ss) -> scratch.SteadyState:
    """SSJ の SteadyStateDict を、自作版の SteadyState に詰め替える。

    同じ定常状態の上で両方の直接法を走らせて、実装が一致することを確かめるために使う。
    """
    d = ss.internals["hh"]
    return scratch.SteadyState(
        K=float(ss["A"]), r=float(ss["r"]), w=float(ss["w"]), y=d["y"],
        Va=d["Va"], a=d["a"], c=d["c"], D=d["D"], C=float(ss["C"]),
        a_max_share=float(d["D"][:, -10:].sum()), backward_its=-1, D_resid=np.nan,
    )


def _fmt(M, width=12, prec=6):
    return "\n".join("  " + "".join(f"{v:{width}.{prec}f}" for v in row) for row in M)


def _maxdiff(J1, J2, keymap):
    return max(np.max(np.abs(J1[k1] - J2[k2])) for k1, k2 in keymap)


def main() -> None:
    np.set_printoptions(precision=6, suppress=True, linewidth=140)
    T = 5

    # ---- 1. 自作版で定常状態を解く（両者で完全に同じ点を使うため） ----
    print("=" * 78)
    print("1. 定常状態（ks_direct_scratch.py と共通）")
    print("=" * 78)
    p = scratch.KSParams()
    ss_scratch = scratch.solve_ss(p)
    print(f"  自作版:  K = {ss_scratch.K:.8f}   r = {ss_scratch.r:.8f}   w = {ss_scratch.w:.8f}")

    calib = build_calibration(p, ss_scratch.r, ss_scratch.w)

    ss_loose = hh_ext.steady_state(calib)                      # ライブラリの既定 tol
    ss = hh_ext.steady_state(calib, **SS_OPTIONS)              # 締めた tol
    print(f"  SSJ版(既定 tol):  A = {ss_loose['A']:.10f}   誤差 = {ss_loose['A'] - ss_scratch.K:+.3e}")
    print(f"  SSJ版(締めた tol): A = {ss['A']:.10f}   誤差 = {ss['A'] - ss_scratch.K:+.3e}")
    print(f"  自作版         :  K = {ss_scratch.K:.10f}   C = {ss_scratch.C:.10f}")
    print(f"                    C = {ss['C']:.10f}")
    print(f"  分布の差 max|D_ssj - D_scratch| = "
          f"{np.max(np.abs(ss.internals['hh']['D'] - ss_scratch.D)):.3e}")
    print(f"  政策の差 max|a_ssj - a_scratch| = "
          f"{np.max(np.abs(ss.internals['hh']['a'] - ss_scratch.a)):.3e}")
    print()
    print("  既定 tol だと 1e-4 ほどずれるのは、beta*(1+r) = "
          f"{p.beta * (1 + ss_scratch.r):.8f} が 1 に極端に近く、")
    print("  分布の前向き反復が『見かけ上収束したのにまだ真の定常分布から遠い』状態に")
    print("  なりやすいため。自作版は定常分布を反復ではなく線形方程式 D = Λ'D で直接解いている。")

    # ---- 2. 一般均衡の定常状態もライブラリに解かせてみる（第10・11章の予習） ----
    print()
    print("=" * 78)
    print("2. 一般均衡の定常状態をライブラリに解かせる（おまけ）")
    print("=" * 78)
    ks = sj.create_model([hh_ext, firm, mkt_clearing], name="Krusell-Smith")
    calib_ge = dict(Pi=p.Pi, a_grid=p.a_grid, beta=p.beta, eis=p.eis, lbar=p.lbar,
                    unemployment_insurance=p.unemployment_insurance,
                    alpha=p.alpha, delta=p.delta, Z=p.Z, N=p.N)
    # 資産需要は r = 1/beta - 1 の近くで爆発的に増えるので、下側の境界は
    # 発散点 K_min_for_impatience = 11.5564 より少し上に取る必要がある。
    # （K = 11.57 だと定常分布が存在せず、ライブラリの前向き反復が発散する）
    ss_ge = ks.solve_steady_state(
        calib_ge,
        unknowns={"K": (11.58, 40.0 * p.N)},
        targets={"asset_mkt": 0.0},
        solver="brentq",
        options={"hh": SS_OPTIONS},
    )
    print(f"  K = {ss_ge['K']:.8f}   r = {ss_ge['r']:.8f}   w = {ss_ge['w']:.8f}")
    print(f"  自作版の二分法との差 |dK| = {abs(ss_ge['K'] - ss_scratch.K):.3e}")
    print(f"  資産需要曲線はこの辺りでほぼ垂直: K = 11.58 で A = 20.4, K = 11.65 で A = 8.5。")
    print(f"  これは beta = 0.99 の KS(1998) 設定が非忍耐性の境界 r = 1/beta - 1 = "
          f"{p.r_max:.6f} に張り付いているため。")

    # ---- 3. ヤコビアン3種類を突き合わせる ----
    print()
    print("=" * 78)
    print(f"3. ヤコビアン（T = {T}）")
    print("=" * 78)

    def flat(Jd):
        return {f"{o},{i}": Jd[o][i] for o in ("A", "C") for i in ("r", "w")}

    # ライブラリの .jacobian() は「1期分の後ろ向きステップ」を有限差分で微分している。
    # 既定は片側差分 h=1e-4。twosided=True にすると中央差分になる。
    J_lib = flat(hh_ext.jacobian(ss, inputs=["r", "w"], outputs=["A", "C"], T=T))
    J_lib_two = flat(hh_ext.jacobian(ss, inputs=["r", "w"], outputs=["A", "C"], T=T,
                                     h=1e-4, twosided=True))
    J_lib_direct = jacobian_direct_via_library(hh_ext, ss, T=T, h=1e-4)
    J_lib_direct_nobase = jacobian_direct_via_library(hh_ext, ss, T=T, h=1e-4,
                                                      subtract_baseline=False)
    base_run = hh_ext.impulse_nonlinear(ss, {"r": np.zeros(T)}, outputs=["A", "C"])

    J_scratch = scratch.jacobian_direct(p, ss_scratch, T=T, h=1e-4)
    J_scratch_two = scratch.jacobian_direct(p, ss_scratch, T=T, h=1e-4, twosided=True)
    J_scratch_fn = scratch.jacobian_fake_news(p, ss_scratch, T=T, h=1e-4, twosided=True)

    # 自作の直接法を「ライブラリの定常状態の上で」走らせる。
    # これで定常状態の差が消えるので、二つの実装が同じ計算をしていることを機械精度で確認できる。
    ss_lib_as_scratch = scratch_ss_from_ssj(p, ss)
    J_scratch_at_lib = scratch.jacobian_direct(p, ss_lib_as_scratch, T=T, h=1e-4)

    print("\n  【ライブラリ .jacobian(twosided=True) = フェイクニュース法】  J^{A,r}")
    print(_fmt(J_lib_two["A,r"]))

    print("\n  【自作 直接法（中央差分, eps = 1e-4）】  J^{K,r}")
    print(_fmt(J_scratch_two["K,r"]))

    print("\n  差分の表（4本すべての max|・| ）")
    same = [(k, k) for k in ("A,r", "A,w", "C,r", "C,w")]
    cross = [("K,r", "A,r"), ("K,w", "A,w"), ("C,r", "C,r"), ("C,w", "C,w")]
    print(f"    自作 直接法(中央)    vs ライブラリ fake news(中央) : "
          f"{_maxdiff(J_scratch_two, J_lib_two, cross):.3e}   <- 一致。両者は同じ導関数")
    print(f"    自作 fake news(中央) vs ライブラリ fake news(中央) : "
          f"{_maxdiff(J_scratch_fn, J_lib_two, cross):.3e}")
    print(f"    自作 直接法(片側)    vs ライブラリ 直接法(片側)     : "
          f"{_maxdiff(J_scratch_at_lib, J_lib_direct, cross):.3e}   <- 同じ定常状態なら機械精度で一致")
    print(f"    同上、基準走行 Y^0 を差し引かない場合               : "
          f"{_maxdiff(J_scratch_at_lib, J_lib_direct_nobase, cross):.3e}   <- ノート 7.7")
    print()
    print(f"    ライブラリ fake news(片側) vs (中央)               : "
          f"{_maxdiff(J_lib, J_lib_two, same):.3e}")
    print()
    print("  基準走行について（ノート 7.7）:")
    print("    ゼロショックを流したときの A_t - A_ss = "
          + "  ".join(f"{v:.3e}" for v in np.asarray(base_run["A"])))
    print("    定常状態を解いた数値誤差のせいで完全には平坦にならない。これを eps = 1e-4 で")
    print("    割ると 3e-6 程度になり、そのままヤコビアンの誤差になる。")
    print("    ライブラリの impulse_nonlinear は『Y_ss からの偏差』を返すので、直接法に使うなら")
    print("    ゼロショックの走行を1本流して差し引くのが安全。")
    print()
    print("  読み方: ライブラリの .jacobian() も有限差分である。ただし微分する対象が")
    print("          『経路全体の写像』ではなく『1期分の後ろ向きステップ』で、その誤差が")
    print("          対角方向の累積を通じて J 全体に伝わる。既定の片側差分 h=1e-4 では")
    print("          その誤差が O(1e-4) 残るので、直接法と突き合わせるときは")
    print("          twosided=True にするか h を小さくすること。")
    print("          フェイクニュース法が速いのは重複計算を除くからであって、")
    print("          有限差分をやめるからではない。")

    # ---- 4. 予算制約からくる恒等式のチェック ----
    print()
    print("=" * 78)
    print("4. 予算制約からくる恒等式（ヤコビアンの健全性チェック）")
    print("=" * 78)
    K = ss_scratch.K
    N = p.N
    print("  各家計の予算制約  c_t + k_t = (1 + r_t) k_{t-1} + y_t  を集計して微分すると")
    print("    dC_0/dr_0 + dK_0/dr_0 = K_ss        （r_0 は期首資本 K_ss にかかる）")
    print("    dC_0/dr_s + dK_0/dr_s = 0     (s > 0)")
    print("    dC_0/dw_0 + dK_0/dw_0 = N     （w_0 は集計労働 N にかかる）")
    print()
    row0_r = J_lib_two["A,r"][0, :] + J_lib_two["C,r"][0, :]
    row0_w = J_lib_two["A,w"][0, :] + J_lib_two["C,w"][0, :]
    print(f"    実際: dC_0/dr_s + dK_0/dr_s = {np.array2string(row0_r, precision=8)}")
    print(f"          K_ss = {K:.8f}    誤差 = {abs(row0_r[0] - K):.3e},  "
          f"s>0 の最大 = {np.max(np.abs(row0_r[1:])):.3e}")
    print(f"    実際: dC_0/dw_s + dK_0/dw_s = {np.array2string(row0_w, precision=8)}")
    print(f"          N = {N:.8f}    誤差 = {abs(row0_w[0] - N):.3e},  "
          f"s>0 の最大 = {np.max(np.abs(row0_w[1:])):.3e}")

    # ---- 5. 直接法の計算量（ノート 7.4） ----
    print()
    print("=" * 78)
    print("5. 直接法の計算量が O(T^2) になること（ノート 7.4）")
    print("=" * 78)
    import time
    print("    入力1本あたりの高価な後ろ向きステップ数 = 列数 T x 1列の長さ T = T^2")
    print(f"    {'T':>4}  {'直接法(秒)':>12}  {'fake news(秒)':>14}  {'比':>8}")
    for T_ in (5, 10, 20, 40):
        t0 = time.perf_counter()
        scratch.jacobian_direct(p, ss_scratch, T=T_, h=1e-4)
        t_dir = time.perf_counter() - t0
        t0 = time.perf_counter()
        hh_ext.jacobian(ss, inputs=["r", "w"], outputs=["A", "C"], T=T_)
        t_fn = time.perf_counter() - t0
        print(f"    {T_:>4}  {t_dir:>12.4f}  {t_fn:>14.4f}  {t_dir / t_fn:>8.1f}x")
    print("\n    T を2倍にすると直接法はおよそ4倍になる（T^2）。")
    print("    fake news 法はおよそ2倍（T）で済む。これが第8章の動機。")


if __name__ == "__main__":
    main()
