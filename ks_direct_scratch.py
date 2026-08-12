# -*- coding: utf-8 -*-
"""
Krusell=Smith 型家計ブロックの Sequence Space Jacobian を「直接法」で求める（ライブラリ不使用）
=================================================================================

講義ノート `sequence_space_jacobian_lecture_notes_ch4_11.pdf` 第7章
「直接法：ヤコビアンを一列ずつ計算する」の実装。章末確認問題1
「T = 5 で J^{K,r}_{:,3} を片側差分で求める計算順序を記述せよ」に対応する。

このファイルは numpy だけで動く（numba も sequence_jacobian も不要）。
ライブラリ版は ks_direct_ssj.py を参照。


既存の 8_2_krusell_and_smith.py との関係
--------------------------------------
元のコードは Krusell and Smith (1998) の「元祖」解法だった:

    集計状態 (K, A) を状態変数に持つ  →  予測式 log K' = a0 + a1 log K で近似
    →  VFI  →  6000期シミュレーション  →  OLS で予測係数を更新（有界合理性）

SSJ は同じモデルを別の切り口で解く:

    集計ショックのない決定論的定常状態を解く
    →  価格経路 {r_t, w_t} を所与とした有限期間 T の写像として家計を扱う
    →  その写像を定常状態のまわりで一次微分したものがヤコビアン

したがって K グリッド・予測式・シミュレーションは全部消える。残るのは
「後ろ向きに政策 → 前向きに分布 → 集計」の3段階（ノート 7.2）だけである。
パラメータ (alpha, delta, beta, lbar, 失業保険, 所得の作り方) は元のコードの
replicate_model = 0（KS 1998）をそのまま引き継ぐ。


タイミング規約（ノート ii ページ「通読のための記号とタイミング」に一致）
--------------------------------------------------------------------
    時点 t の冒頭に家計は資本 k_{t-1} と個別状態 e_t を持つ。
    価格 (r_t, w_t) を所与に消費 c_t と期末資本 k_t を選ぶ。

        c_t + k_t = (1 + r_t) k_{t-1} + y_t(e_t),     k_t >= 0

    D_t は時点 t 冒頭の (e_t, k_{t-1}) 上の分布。
    Λ_t は政策と個別状態遷移から作られる遷移行列で  D_{t+1} = Λ_t' D_t。
    集計は  K_t = k_t' D_t,  C_t = c_t' D_t。

    企業は時点 t に K_{t-1} と N_t で生産するので、r_t と w_t は K_{t-1} の関数。
    （このため企業ブロックのヤコビアンには1期の遅れが現れる。ノート 5.4）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

import numpy as np

# =====================================================================
# Part 0  補間・グリッドのユーティリティ
#         sequence_jacobian.utilities.interpolate と数値的に同一の結果を返す
#         （ライブラリ版と突き合わせるときに補間の差が混ざらないようにするため）
# =====================================================================


def exponential_grid(a_min: float, a_max: float, n: int) -> np.ndarray:
    """対数空間で等間隔なグリッド。8_2_krusell_and_smith.py の exponential_grid と同一。"""
    return np.expm1(np.linspace(np.log1p(a_min), np.log1p(a_max), n))


def interpolate_y(x: np.ndarray, xq: np.ndarray, y: np.ndarray) -> np.ndarray:
    """各行について、増加データ点 x に対して y を xq で線形補間する（範囲外は線形外挿）。

    x : (ne, na) 各行が増加      … EGM の内生グリッド c_nextgrid + a_grid
    xq: (ne, na) 各行が増加      … 手持ち資産 coh
    y : (na,)                    … a_grid
    """
    na = x.shape[-1]
    # searchsorted(side='left') - 1 は SSJ の単調スイープと同じブラケットを返す
    i = np.empty(xq.shape, dtype=np.int64)
    for row in range(x.shape[0]):
        i[row] = np.searchsorted(x[row], xq[row], side="left") - 1
    np.clip(i, 0, na - 2, out=i)

    x_lo = np.take_along_axis(x, i, axis=-1)
    x_hi = np.take_along_axis(x, i + 1, axis=-1)
    pi = (x_hi - xq) / (x_hi - x_lo)
    return pi * y[i] + (1.0 - pi) * y[i + 1]


def interpolate_coord(grid: np.ndarray, xq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """xq を grid 上の「くじ」表現に直す:  xq = pi * grid[i] + (1 - pi) * grid[i+1]

    SSJ の interpolate_coord_robust と同一（xq の単調性を仮定しない、範囲外は外挿）。
    """
    n = grid.shape[0]
    i = np.searchsorted(grid, xq, side="left") - 1
    np.clip(i, 0, n - 2, out=i)
    pi = (grid[i + 1] - xq) / (grid[i + 1] - grid[i])
    return i, pi


def stationary_dist(P: np.ndarray, tol: float = 1e-14, maxit: int = 1_000_000) -> np.ndarray:
    """行確率行列 P の定常分布を反復で求める。元コードの stationary_by_iteration と同じ。"""
    P = np.asarray(P, dtype=float)
    x = np.full(P.shape[0], 1.0 / P.shape[0])
    for _ in range(maxit):
        y = x @ P
        if np.max(np.abs(y - x)) < tol:
            return y
        x = y
    raise RuntimeError("stationary_dist: 収束しませんでした")


# =====================================================================
# Part 1  キャリブレーション
#         「ノートブックで実際に得られる定常分布」から2状態雇用チェーンを作る
# =====================================================================


def markov_KS(
    ugrid: np.ndarray = np.array([0.10, 0.04]),
    durug: float = 1.5,
    durub: float = 2.5,
    durgd: float = 8.0,
    durbd: float = 8.0,
    add_assumption_num1: float = 1.25,
    add_assumption_num2: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """8_2_krusell_and_smith.py の markov_KS をそのまま移植（nl = nA = 2 専用）。

    状態の並び:  A = 0 が bad, A = 1 が good /  l = 0 が失業, l = 1 が就業
    Returns
    -------
    p_A_A   : (2, 2)  P(A' | A)
    p_lA_lA : (4, 4)  P(A', l' | A, l)   行 = 2*A + l, 列 = 2*A' + l'
    p_l_lAA : (4, 4)  P(l' | A, A', l)
    """
    ugrid = np.asarray(ugrid, dtype=float)
    unempb, unempg = float(ugrid[0]), float(ugrid[1])

    # --- 集計状態の遷移 ---
    pgg = (durgd - 1.0) / durgd
    pbg = 1.0 - pgg
    pbb = (durbd - 1.0) / durbd
    pgb = 1.0 - pbb
    p_A_A = np.array([[pbb, pgb], [pbg, pgg]])  # 行 = [bad, good]

    # --- 集計状態の組ごとの雇用遷移 ---
    pgg00 = (durug - 1.0) / durug
    pgg01 = (unempg - unempg * pgg00) / (1.0 - unempg)
    pbb00 = (durub - 1.0) / durub
    pbb01 = (unempb - unempb * pbb00) / (1.0 - unempb)
    pbg00 = add_assumption_num1 * pbb00
    pbg01 = (unempb - unempg * pbg00) / (1.0 - unempg)
    pgb00 = add_assumption_num2 * pgg00
    pgb01 = (unempg - unempb * pgb00) / (1.0 - unempb)

    def M(p00: float, p01: float) -> np.ndarray:
        return np.array([[p00, 1.0 - p00], [p01, 1.0 - p01]])

    M_by_A = [[M(pbb00, pbb01), M(pgb00, pgb01)],   # bad  -> bad, good
              [M(pbg00, pbg01), M(pgg00, pgg01)]]   # good -> bad, good

    p_l_lAA = np.zeros((4, 4))
    p_lA_lA = np.zeros((4, 4))
    for iA in range(2):
        for il in range(2):
            row = 2 * iA + il
            for jA in range(2):
                for jl in range(2):
                    col = 2 * jA + jl
                    p_l_lAA[row, col] = M_by_A[iA][jA][il, jl]
                    p_lA_lA[row, col] = p_A_A[iA, jA] * M_by_A[iA][jA][il, jl]
    return p_A_A, p_lA_lA, p_l_lAA


def employment_chain_from_ks(**kwargs) -> Tuple[np.ndarray, float, Dict[str, np.ndarray]]:
    """KS(1998) の同時チェーン (A, l) を、その「実際の定常分布」で A について積分し、
    集計ショックのない2状態雇用チェーン Pi(l' | l) を作る。

        Pi(l'|l) = Σ_{A, A'} π(A, l) P(A'|A) P(l'|A, A', l) / π(l)

    π は p_lA_lA（4状態の同時チェーン）の定常分布。
    式の意味は「実際に起きている l -> l' のフロー ÷ 実際に l にいるストック」で、
    条件付き確率の定義そのもの。A は観測しないので足し上げて消している。

    失業率が壊れないこと
    --------------------
    A を消して2状態に潰すと、普通は何かが壊れる。壊れうるものの一つが失業率の水準である。

        潰す前: 元の4状態経済の平均失業率           = 0.07
        潰した後: Pi を単独のマルコフ連鎖として回した失業率 = ?

    この2つは別々の数字だが、上の式で Pi を作ると必ず一致する（どちらも 0.07 になる）。
    0.07 に合うように何かを調整したのではなく、Pi の作り方から自動的にそうなる。

    理由はフローで見ると早い。上の式の分母を払うと

        π(l) * Pi(l'|l) = 元の経済で実際に起きている l -> l' のフロー

    で、元の経済は定常状態なので流入と流出が釣り合っている。つまり
    π(u)Pi(e|u) = π(e)Pi(u|e)。2状態マルコフ連鎖の定常分布はまさにこの式で決まるので、
    [π(u), π(e)] = [0.07, 0.93] が Pi の定常分布になる。

    重みを間違えると壊れる
    ----------------------
    重みは行ごとに違う π(A|l) であって、共通の π(A) ではない:

        l = 失業: π(bad|l) = 0.7143, π(good|l) = 0.2857   <- 失業者は不況に偏在
        l = 就業: π(bad|l) = 0.4839, π(good|l) = 0.5161

    これを共通の [0.5, 0.5] で潰して単純平均すると、定常失業率は 0.0634 になり 0.07 から外れる。
    （なお失業率の「水準」の方は π(A) 加重で正しく、u = 0.5*0.10 + 0.5*0.04 = 0.07。
      水準は単純加重でよいが、遷移行列は条件付き加重でないといけない、という非対称に注意。）

    近似になっている点
    ------------------
    元の KS 経済では l 単独は厳密にはマルコフではない（遷移が A に依存するため）。
    Pi は 1 期先の条件付き分布を定常分布のもとで平均的に再現するが、2 期先以降はずれる
    （実測で最大 6.9e-3）。「不況では失業リスクが上がる」という連動も落ちる。
    集計ショックを落とす以上この近似は避けられないが、必要なら SSJ 側で Pi 自体を
    入力として動かすことで復活させられる。

    ノートブックの数値: A_ss = [0.5, 0.5] → u = 0.07, Z = 0.5*0.99 + 0.5*1.01 = 1.0
    """
    p_A_A, p_lA_lA, p_l_lAA = markov_KS(**kwargs)
    pi_joint = stationary_dist(p_lA_lA)             # 行 = 2*A + l
    marg_l = np.array([pi_joint[0] + pi_joint[2], pi_joint[1] + pi_joint[3]])

    Pi = np.zeros((2, 2))
    for iA in range(2):
        for il in range(2):
            row = 2 * iA + il
            for jA in range(2):
                for jl in range(2):
                    Pi[il, jl] += pi_joint[row] * p_A_A[iA, jA] * p_l_lAA[row, 2 * jA + jl]
    Pi /= marg_l[:, None]

    u_ss = float(stationary_dist(Pi)[0])
    diag = {
        "p_A_A": p_A_A,
        "A_ss": stationary_dist(p_A_A),
        "pi_joint_Al": pi_joint,
        "marg_l": marg_l,
    }
    return Pi, u_ss, diag


@dataclass
class KSParams:
    """replicate_model = 0（Krusell and Smith 1998）のパラメータ。"""

    # 技術・選好（元コードの Setting と同一）
    alpha: float = 0.36
    delta: float = 0.025
    beta: float = 0.99
    lbar: float = 0.3271
    eis: float = 1.0            # log 効用（元コードの bellman_objective は np.log(consume)）
    Z: float = 1.0              # 集計ショックの定常分布での平均 = 0.5*0.99 + 0.5*1.01 = 1.0

    # 失業保険。元コードでは水準（w に比例しない）で与えられている:
    #   il == 0:  income = (1+r)a + unemployment_insurance
    #   il == 1:  income = (1+r)a + w * lbar
    unemployment_insurance: float = 0.07

    # 資産グリッド（元コードの agrid と同じ作り方。Na は分布計算と共用するので少し多め）
    n_a: int = 200
    a_min: float = 0.0
    a_max: float = 300.0

    # 以下は __post_init__ で作る
    Pi: np.ndarray = field(init=False)
    u_ss: float = field(init=False)
    a_grid: np.ndarray = field(init=False)
    markov_diag: Dict[str, np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        self.Pi, self.u_ss, self.markov_diag = employment_chain_from_ks()
        self.a_grid = exponential_grid(self.a_min, self.a_max, self.n_a)

    # --- 集計労働と価格 ---
    @property
    def N(self) -> float:
        """集計労働。元コードの L_agg = lbar * (1 - u) に対応（u は定常失業率）。"""
        return self.lbar * (1.0 - self.u_ss)

    def prices(self, K: float) -> Tuple[float, float]:
        """期首資本 K から (r, w) を作る。元コードの vfi_numba 内の式と同一。"""
        r = self.alpha * self.Z * K ** (self.alpha - 1.0) * self.N ** (1.0 - self.alpha) - self.delta
        w = (1.0 - self.alpha) * self.Z * K ** self.alpha * self.N ** (-self.alpha)
        return r, w

    def income(self, w: float) -> np.ndarray:
        """雇用状態ごとの労働所得ベクトル y(e)。並びは [失業, 就業]。"""
        return np.array([self.unemployment_insurance, w * self.lbar])

    @property
    def r_max(self) -> float:
        """r >= 1/beta - 1 だと資産需要が発散するので、その上限。"""
        return 1.0 / self.beta - 1.0

    @property
    def K_min_for_impatience(self) -> float:
        """r(K) = 1/beta - 1 となる K。均衡 K はこれより大きい必要がある。"""
        return self.N * ((self.alpha * self.Z) / (self.r_max + self.delta)) ** (1.0 / (1.0 - self.alpha))


# =====================================================================
# Part 2  家計問題：1期分の後ろ向きステップ
#         v_t = v(v_{t+1}, X_t)   （ノート式 7.18）
# =====================================================================


def backward_egm(
    EVa_p: np.ndarray, a_grid: np.ndarray, y: np.ndarray, r: float, beta: float, eis: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """内生グリッド法による1期後ろ向きステップ。

    Parameters
    ----------
    EVa_p : (ne, na)  E_t[ V_a,{t+1} ] = Pi @ Va_{t+1}。行は時点 t の個別状態、列は期末資産 a_t。
    y     : (ne,)     時点 t の労働所得
    r     : float     時点 t の純資本収益率

    Returns
    -------
    Va : (ne, na)  V_a,t  （時点 t の状態 (e_t, a_{t-1}) 上）
    a  : (ne, na)  期末資本 k_t
    c  : (ne, na)  消費 c_t

    オイラー方程式 u'(c_t) = beta * E_t[(1 + r_{t+1}) u'(c_{t+1})] を、
    Va_{t+1} = (1 + r_{t+1}) u'(c_{t+1}) と定義して使っている。
    sequence_jacobian の hetblocks.hh_sim.hh と同じ式。
    """
    uc_nextgrid = beta * EVa_p                       # = u'(c_t) が満たすべき値
    c_nextgrid = uc_nextgrid ** (-eis)               # 内生グリッド上の消費
    coh = (1.0 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]   # 手持ち資産（外生グリッド上）
    a = interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    np.maximum(a, a_grid[0], out=a)                  # 借入制約 k_t >= 0
    c = coh - a
    Va = (1.0 + r) * c ** (-1.0 / eis)
    return Va, a, c


def backward_egm_init(a_grid: np.ndarray, y: np.ndarray, r: float, eis: float) -> np.ndarray:
    """後ろ向き反復の初期値（消費が手持ちの1割、という粗い当て推量）。"""
    coh = (1.0 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    return (1.0 + r) * (0.1 * coh) ** (-1.0 / eis)


def backward_vfi(
    EV_p: np.ndarray,
    a_grid: np.ndarray,
    y: np.ndarray,
    r: float,
    beta: float,
    tol: float = 1e-12,
    maxit: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """黄金分割法による1期後ろ向きステップ（EGM の検証用）。

    元コード 8_2_krusell_and_smith.py の bellman_objective + golden_minimize と同じ考え方だが、
    全格子点について同時に黄金分割を回すよう numpy でベクトル化してある。
    効用は log（元コードと同じ）なので eis = 1 に対応する。

    EV_p : (ne, na)  E_t[V_{t+1}]。列は期末資産 a_t（a_grid 上）。
    """
    ne, na = EV_p.shape
    coh = (1.0 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]

    def objective(ap: np.ndarray) -> np.ndarray:
        """-(log c + beta * E V(a'))。実行不能なら大きな値。"""
        c = coh - ap
        cont = np.empty_like(ap)
        for e in range(ne):
            cont[e] = np.interp(ap[e], a_grid, EV_p[e])
        out = -(np.log(np.maximum(c, 1e-300)) + beta * cont)
        return np.where(c <= 0.0, 1e18, out)

    lo = np.full_like(coh, a_grid[0])
    hi = np.minimum(coh - 1e-10, a_grid[-1])
    hi = np.maximum(hi, lo)

    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    x1 = hi - invphi * (hi - lo)
    x2 = lo + invphi * (hi - lo)
    f1, f2 = objective(x1), objective(x2)
    for _ in range(maxit):
        if np.max(hi - lo) < tol:
            break
        left = f1 < f2
        hi = np.where(left, x2, hi)
        lo = np.where(left, lo, x1)
        x1_new = hi - invphi * (hi - lo)
        x2_new = lo + invphi * (hi - lo)
        # 片側は前回の内点を使い回せるが、ベクトル化の可読性を優先して両方評価する
        x1, x2 = x1_new, x2_new
        f1, f2 = objective(x1), objective(x2)

    a = 0.5 * (lo + hi)
    np.maximum(a, a_grid[0], out=a)
    np.minimum(a, coh - 1e-10, out=a)
    c = coh - a
    cont = np.empty_like(a)
    for e in range(ne):
        cont[e] = np.interp(a[e], a_grid, EV_p[e])
    V = np.log(c) + beta * cont
    return V, a, c


def backward_vfi_init(a_grid: np.ndarray, y: np.ndarray, r: float, beta: float) -> np.ndarray:
    coh = (1.0 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    return np.log(0.1 * coh) / (1.0 - beta)


# =====================================================================
# Part 3  前向きステップと定常状態
#         D_{t+1} = Λ_t' D_t   （ノート式 7.19）
# =====================================================================


def forward_endog(D: np.ndarray, a_i: np.ndarray, a_pi: np.ndarray) -> np.ndarray:
    """資産の遷移だけを進める（政策 a_t のくじ表現を使う）。"""
    Dnew = np.zeros_like(D)
    for e in range(D.shape[0]):
        np.add.at(Dnew[e], a_i[e], a_pi[e] * D[e])
        np.add.at(Dnew[e], a_i[e] + 1, (1.0 - a_pi[e]) * D[e])
    return Dnew


def forward_step(D: np.ndarray, Pi: np.ndarray, a_i: np.ndarray, a_pi: np.ndarray) -> np.ndarray:
    """D_{t+1} = Λ_t' D_t 。まず資産、次に個別状態の順（SSJ の規約と同じ）。"""
    return Pi.T @ forward_endog(D, a_i, a_pi)


def transition_matrix(Pi: np.ndarray, a_i: np.ndarray, a_pi: np.ndarray) -> np.ndarray:
    """遷移行列 Λ を (ne*na, ne*na) の密行列として明示的に作る。

    Λ[(e, i), (e', j)] = Pi[e, e'] * （政策 a(e, i) が格子点 j に落ちるくじの重み）

    状態数は ne*na = 2*n_a しかないので、密行列で持っても何の問題もない。
    定常分布を反復ではなく線形方程式で解くために使う（beta*(1+r) が 1 に非常に近い
    KS のキャリブレーションでは、反復は収束が極端に遅い）。
    """
    ne, na = a_i.shape
    n = ne * na
    Lam = np.zeros((n, n))
    for e in range(ne):
        for i in range(na):
            row = e * na + i
            lo, hi_w = a_i[e, i], a_pi[e, i]
            for ep in range(ne):
                Lam[row, ep * na + lo] += Pi[e, ep] * hi_w
                Lam[row, ep * na + lo + 1] += Pi[e, ep] * (1.0 - hi_w)
    return Lam


def stationary_from_policy(Pi: np.ndarray, a_i: np.ndarray, a_pi: np.ndarray) -> Tuple[np.ndarray, float]:
    """政策から定常分布 D を直接解く:  D = Λ' D,  1'D = 1 。

    Returns (D, 残差)。残差は max|Λ'D - D| で、機械精度なら 1e-15 程度になる。
    """
    ne, na = a_i.shape
    Lam = transition_matrix(Pi, a_i, a_pi)
    n = ne * na
    A = Lam.T - np.eye(n)
    A[-1, :] = 1.0                      # 最終行を正規化条件 1'D = 1 で置き換える
    b = np.zeros(n)
    b[-1] = 1.0
    d = np.linalg.solve(A, b)
    resid = float(np.max(np.abs(Lam.T @ d - d)))
    return d.reshape(ne, na), resid


@dataclass
class SteadyState:
    K: float
    r: float
    w: float
    y: np.ndarray
    Va: np.ndarray
    a: np.ndarray
    c: np.ndarray
    D: np.ndarray            # 時点 t 冒頭の (e_t, a_{t-1}) 上の分布
    C: float
    a_max_share: float       # 資産グリッド上端付近の質量（グリッドが足りているかの点検）
    backward_its: int
    D_resid: float           # max|Λ'D - D| 。基準走行が平坦かどうかを左右する


def solve_household_ss(
    p: KSParams,
    r: float,
    w: float,
    Va_init: np.ndarray | None = None,
    backward_tol: float = 1e-11,
    backward_maxit: int = 50_000,
) -> SteadyState:
    """価格 (r, w) を所与に、家計問題の定常状態（政策と分布）を解く。部分均衡。

    Va_init を渡すと後ろ向き反復を温かいところから始められる（二分法の中で使う）。
    """
    y = p.income(w)

    # --- 後ろ向き反復：政策関数の不動点 ---
    Va = backward_egm_init(p.a_grid, y, r, p.eis) if Va_init is None else Va_init.copy()
    a_old = np.zeros((2, p.n_a))
    its = 0
    for its in range(1, backward_maxit + 1):
        Va, a, c = backward_egm(p.Pi @ Va, p.a_grid, y, r, p.beta, p.eis)
        if its % 10 == 0:
            if np.max(np.abs(a - a_old)) < backward_tol:
                break
            a_old = a
    else:
        raise RuntimeError("政策関数が収束しませんでした")

    # --- 定常分布：反復ではなく線形方程式で直接解く ---
    # beta*(1+r) が 1 に極めて近い KS のキャリブレーションでは反復が非常に遅いうえ、
    # 中途半端に打ち切ると基準走行 K^0_t が平坦にならず、直接法の差分に誤差が乗る。
    a_i, a_pi = interpolate_coord(p.a_grid, a)
    D, resid = stationary_from_policy(p.Pi, a_i, a_pi)

    K = float(np.vdot(D, a))
    C = float(np.vdot(D, c))
    a_max_share = float(D[:, -10:].sum())
    return SteadyState(K=K, r=r, w=w, y=y, Va=Va, a=a, c=c, D=D, C=C,
                       a_max_share=a_max_share, backward_its=its, D_resid=resid)


def solve_ss(p: KSParams, K_lo: float | None = None, K_hi: float | None = None,
             tol: float = 1e-11, verbose: bool = False) -> SteadyState:
    """資本市場を清算する定常状態を解く:  家計の資産需要 = K 。

    r(K) は K について減少、資産需要は r について増加。よって
        f(K) = 資産需要(r(K), w(K)) - K
    は K について減少するので、二分法で解ける。
    K が p.K_min_for_impatience を下回ると r >= 1/beta - 1 となり資産需要が発散し
    定常分布が存在しないので、そこを下側の境界に使う。
    """
    if K_lo is None:
        K_lo = p.K_min_for_impatience * (1.0 + 1e-8)
    if K_hi is None:
        K_hi = 40.0 * p.N

    warm: np.ndarray | None = None

    def excess(K: float) -> Tuple[float, SteadyState]:
        nonlocal warm
        r, w = p.prices(K)
        ss = solve_household_ss(p, r, w, Va_init=warm)
        warm = ss.Va
        return ss.K - K, ss

    f_hi, ss = excess(K_hi)
    if f_hi > 0.0:
        raise RuntimeError(f"上側の境界で超過需要 (f={f_hi:.3e})。K_hi を大きくしてください。")

    while K_hi - K_lo > tol * max(1.0, K_lo):
        K_mid = 0.5 * (K_lo + K_hi)
        f_mid, ss_mid = excess(K_mid)
        if verbose:
            print(f"  K = {K_mid:10.6f}   r = {ss_mid.r:9.6f}   資産需要 - K = {f_mid: .3e}")
        if f_mid > 0.0:
            K_lo = K_mid
        else:
            K_hi = K_mid
            ss = ss_mid

    # 最終的な K で解き直して、返す定常状態と価格を厳密に整合させる
    _, ss = excess(0.5 * (K_lo + K_hi))
    return ss


# =====================================================================
# Part 4  移行経路：ブロック写像 H （ノート式 2, 7.21）
#         (K, C) = H(r, w)
# =====================================================================


def td_household(
    p: KSParams,
    ss: SteadyState,
    r_path: np.ndarray,
    w_path: np.ndarray,
    backward: str = "egm",
) -> Tuple[np.ndarray, np.ndarray]:
    """価格経路を1本受け取り、集計経路 (K_t, C_t) を返す。これがブロック写像 H そのもの。

    ノート 7.2 の3段階をそのまま実装している:
      第1段階  終端条件 V_T = V_ss から t = T-1, ..., 0 と後ろ向きに政策を求める
      第2段階  初期条件 D_0 = D_ss から t = 0, ..., T-1 と前向きに分布を求める
      第3段階  各時点で集計する

    重要（ノート 7.2.1 の注意）:
      ショックが時点 s にあっても、t < s の政策を求めるには将来の r_s が必要なので、
      「時点 s だけ解けばよい」のではなく終端から時点 0 までの後ろ向き計算が要る。
    """
    T = len(r_path)
    assert len(w_path) == T

    # ---- 第1段階：後ろ向き（終端 = 定常状態） ----
    a_path = np.empty((T,) + ss.a.shape)
    c_path = np.empty((T,) + ss.c.shape)
    if backward == "egm":
        V = ss.Va
        step = lambda EV, r, w: backward_egm(EV, p.a_grid, p.income(w), r, p.beta, p.eis)
    elif backward == "vfi":
        V = ss.Va  # solve_household_ss_vfi が入れた V
        step = lambda EV, r, w: backward_vfi(EV, p.a_grid, p.income(w), r, p.beta)
    else:
        raise ValueError("backward は 'egm' か 'vfi'")

    for t in reversed(range(T)):
        V, a_path[t], c_path[t] = step(p.Pi @ V, r_path[t], w_path[t])

    # ---- 第2段階 + 第3段階：前向きに分布を回しつつ集計 ----
    K = np.empty(T)
    C = np.empty(T)
    D = ss.D.copy()
    for t in range(T):
        K[t] = np.vdot(D, a_path[t])
        C[t] = np.vdot(D, c_path[t])
        a_i, a_pi = interpolate_coord(p.a_grid, a_path[t])
        D = forward_step(D, p.Pi, a_i, a_pi)
    return K, C


# =====================================================================
# Part 5  直接法（第7章）
# =====================================================================


def jacobian_direct(
    p: KSParams,
    ss: SteadyState,
    T: int = 5,
    h: float = 1e-4,
    twosided: bool = False,
    backward: str = "egm",
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """直接法で家計ブロックのヤコビアン4本を求める。

    ノート 定義 7.1:
      入力経路の各座標を一つずつ摂動し、そのたびに非線形の家計問題と分布動学を解いて
      出力経路の差分を求め、ヤコビアンの列を順番に構成する。

    返り値のキーは 'K,r', 'K,w', 'C,r', 'C,w'。(t, s) 要素が ∂Y_t / ∂X_s。

    Parameters
    ----------
    h        : 差分幅 ε（ノート 7.6.3）
    twosided : True なら中央差分（ノート 7.6.2）。誤差 O(ε^2) だが計算量は約2倍。
    """
    r0 = np.full(T, ss.r)
    w0 = np.full(T, ss.w)

    # ノート 7.7「基準走行を差し引く理由」:
    # 定常値 Yss * 1 ではなく、同じ有限期間・同じ終端条件・同じ数値手順で走らせた
    # Y^0 を差し引く。摂動と無関係な基準誤差が 1/ε 倍されるのを避けるため。
    K0, C0 = td_household(p, ss, r0, w0, backward=backward)

    J = {f"{o},{i}": np.empty((T, T)) for o in ("K", "C") for i in ("r", "w")}

    for i, base in (("r", r0), ("w", w0)):
        for s in range(T):
            # 式 7.4:  X^{s,+}_t = X_ss + ε * 1{t = s}
            up = base.copy()
            up[s] += h
            if i == "r":
                Kp, Cp = td_household(p, ss, up, w0, backward=backward)
            else:
                Kp, Cp = td_household(p, ss, r0, up, backward=backward)

            if twosided:
                dn = base.copy()
                dn[s] -= h
                if i == "r":
                    Km, Cm = td_household(p, ss, dn, w0, backward=backward)
                else:
                    Km, Cm = td_household(p, ss, r0, dn, backward=backward)
                J[f"K,{i}"][:, s] = (Kp - Km) / (2.0 * h)   # 式 7.14
                J[f"C,{i}"][:, s] = (Cp - Cm) / (2.0 * h)
            else:
                J[f"K,{i}"][:, s] = (Kp - K0) / h            # 式 7.3 / 7.9
                J[f"C,{i}"][:, s] = (Cp - C0) / h

            if verbose:
                print(f"  第{s}列 (入力 {i}) 完了")
    return J


# =====================================================================
# Part 6  おまけ：フェイクニュース・アルゴリズム（第8章の予習）
#         直接法の結果を検証するために使う
# =====================================================================


def jacobian_fake_news(
    p: KSParams, ss: SteadyState, T: int = 5, h: float = 1e-4, twosided: bool = False
) -> Dict[str, np.ndarray]:
    """フェイクニュース・アルゴリズムで同じヤコビアンを求める（ノート第8章）。

    直接法が O(T^2) 回の高価な後ろ向き・前向き計算を要するのに対し、こちらは
    後ろ向き T 回 + 前向き T 回で済む。ここでは「直接法が正しいことの確認」に使う。

    第1段階  curlyY[s], curlyD[s] : 時点 s のニュースに対する時点 0 の反応
    第2段階  curlyE[t]            : 分布の痕跡を将来の集計量に変換する期待ベクトル
    第3段階  F = フェイクニュース行列
    第4段階  J[t, s] = J[t-1, s-1] + F[t, s]

    重要な注意（sequence-jacobian の .jacobian() も同じ）:
      この方法も結局は有限差分である。ただし微分するのは「経路全体の写像」ではなく
      「1期分の後ろ向きステップ」であり、その微分誤差が対角方向の累積を通じて
      J に伝わる。したがって片側差分 (twosided=False) だと、直接法を同じ h で
      走らせたときより誤差が大きくなることがある。実測では

          直接法(中央差分, h=1e-4)      真値との差  約 3e-8
          fake news(片側差分, h=1e-4)   真値との差  約 3e-4      <- h に比例
          fake news(中央差分, h=1e-4)   真値との差  約 2e-7

      「fake news が速い」のは重複計算を除くからであって、有限差分をやめるからではない。
    """
    a_i_ss, a_pi_ss = interpolate_coord(p.a_grid, ss.a)

    def shocked_prices(input_name: str, sign: float) -> Tuple[float, float]:
        return (ss.r + sign * h if input_name == "r" else ss.r,
                ss.w + sign * h if input_name == "w" else ss.w)

    def curly_YD(input_name: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """s 期先のニュースに対する「時点0の集計反応」と「時点1に残る分布のずれ」。"""
        curlyY = {"K": np.empty(T), "C": np.empty(T)}
        curlyD = np.empty((T,) + ss.D.shape)

        step_ss = lambda V: backward_egm(p.Pi @ V, p.a_grid, p.income(ss.w), ss.r, p.beta, p.eis)
        denom = 2.0 * h if twosided else h

        V_up = V_dn = ss.Va
        for s in range(T):
            # s 期先にショックがあるときの時点 0 の政策 ＝ ショック1回 + 定常ステップ s 回
            if s == 0:
                r_u, w_u = shocked_prices(input_name, +1.0)
                V_up, a_up, c_up = backward_egm(p.Pi @ ss.Va, p.a_grid, p.income(w_u), r_u, p.beta, p.eis)
                if twosided:
                    r_d, w_d = shocked_prices(input_name, -1.0)
                    V_dn, a_dn, c_dn = backward_egm(p.Pi @ ss.Va, p.a_grid, p.income(w_d), r_d, p.beta, p.eis)
                else:
                    a_dn, c_dn = ss.a, ss.c
            else:
                V_up, a_up, c_up = step_ss(V_up)
                if twosided:
                    V_dn, a_dn, c_dn = step_ss(V_dn)
                else:
                    a_dn, c_dn = ss.a, ss.c

            curlyY["K"][s] = np.vdot(ss.D, a_up - a_dn) / denom
            curlyY["C"][s] = np.vdot(ss.D, c_up - c_dn) / denom

            # 政策のずれが分布に与える影響（くじの重みの微分）
            D_up = forward_step(ss.D, p.Pi, *interpolate_coord(p.a_grid, a_up))
            if twosided:
                D_dn = forward_step(ss.D, p.Pi, *interpolate_coord(p.a_grid, a_dn))
            else:
                D_dn = forward_step(ss.D, p.Pi, a_i_ss, a_pi_ss)
            curlyD[s] = (D_up - D_dn) / denom
        return curlyY, curlyD

    def curly_E(o_ss: np.ndarray) -> np.ndarray:
        """期待ベクトル curlyE[t] = Λ_ss^t を o_ss に作用させたもの（ノート 8.5）。"""
        E = np.empty((T - 1,) + o_ss.shape)
        E[0] = o_ss
        for t in range(1, T - 1):
            # 期待は Λ' の随伴、すなわち「まず個別状態、次に資産」の順で後ろ向きに作用させる
            tmp = p.Pi @ E[t - 1]
            E[t] = a_pi_ss * tmp[np.arange(tmp.shape[0])[:, None], a_i_ss] \
                 + (1.0 - a_pi_ss) * tmp[np.arange(tmp.shape[0])[:, None], a_i_ss + 1]
        return E

    Es = {"K": curly_E(ss.a), "C": curly_E(ss.c)}

    J = {}
    for i in ("r", "w"):
        curlyY, curlyD = curly_YD(i)
        for o in ("K", "C"):
            F = np.empty((T, T))
            F[0, :] = curlyY[o]
            for t in range(1, T):
                F[t, :] = np.tensordot(Es[o][t - 1], curlyD, axes=([0, 1], [1, 2]))
            Jmat = F.copy()
            for t in range(1, T):
                Jmat[t, 1:] += Jmat[t - 1, :-1]
            J[f"{o},{i}"] = Jmat
    return J


# =====================================================================
# Part 7  実行
# =====================================================================


def _fmt(M: np.ndarray, width: int = 12, prec: int = 6) -> str:
    return "\n".join("  " + "".join(f"{v:{width}.{prec}f}" for v in row) for row in M)


def main() -> None:
    np.set_printoptions(precision=6, suppress=True, linewidth=140)
    T = 5

    print("=" * 78)
    print("1. キャリブレーション：ノートブックの実際の定常分布から2状態雇用チェーンを作る")
    print("=" * 78)
    p = KSParams()
    d = p.markov_diag
    print(f"  P(A'|A) 行=[bad, good]:\n{_fmt(d['p_A_A'], 10, 4)}")
    print(f"  集計状態の定常分布 A_ss           = {d['A_ss']}")
    print(f"  (A, l) 同時定常分布               = {d['pi_joint_Al']}")
    print(f"  A を積分した Pi(l'|l) 行=[失業, 就業]:\n{_fmt(p.Pi, 12, 6)}")
    print(f"  定常失業率 u                      = {p.u_ss:.6f}")
    print(f"  平均失業継続期間 1/(1-Pi[0,0])    = {1.0 / (1.0 - p.Pi[0, 0]):.4f} 四半期")
    print(f"  集計労働 N = lbar * (1 - u)       = {p.N:.6f}")
    print(f"  資産需要が発散する下限 K          = {p.K_min_for_impatience:.6f}")

    print()
    print("=" * 78)
    print("2. 定常状態（資本市場清算）")
    print("=" * 78)
    ss = solve_ss(p, verbose=False)
    Y = p.Z * ss.K ** p.alpha * p.N ** (1.0 - p.alpha)
    print(f"  K   = {ss.K:.6f}      (KS(1998) の報告値は約 11.5)")
    print(f"  r   = {ss.r:.6f}      (1/beta - 1 = {p.r_max:.6f})")
    print(f"  w   = {ss.w:.6f}")
    print(f"  Y   = {Y:.6f}       K/Y = {ss.K / Y:.4f}")
    print(f"  C   = {ss.C:.6f}")
    print(f"  所得 y = [失業 {ss.y[0]:.4f}, 就業 {ss.y[1]:.4f}]  代替率 {ss.y[0] / ss.y[1]:.3f}")
    print(f"  後ろ向き反復 {ss.backward_its} 回")
    print(f"  定常分布の残差 max|Λ'D - D| = {ss.D_resid:.3e}")
    print(f"  資産グリッド上端10点の質量  = {ss.a_max_share:.3e}  (小さいほど良い)")
    walras = Y - ss.C - p.delta * ss.K
    print(f"\n  財市場: Y - C - delta*K = {walras:+.6f}")
    print(f"    これは失業保険の分 -u * ui = {-p.u_ss * p.unemployment_insurance:+.6f} に一致する。")
    print("    元の 8_2_krusell_and_smith.py では失業保険が課税で賄われていない（純粋な移転）ため、")
    print("    資源制約がその分だけずれる。家計ブロックのヤコビアン（部分均衡）には影響しないが、")
    print("    一般均衡に閉じるときは税で賄うか、Walras の法則の点検を緩める必要がある。")

    print()
    print("=" * 78)
    print(f"3. 直接法によるヤコビアン（T = {T}, 片側差分, eps = 1e-4）")
    print("=" * 78)
    J = jacobian_direct(p, ss, T=T, h=1e-4, twosided=False)
    for key in ("K,r", "K,w", "C,r", "C,w"):
        o, i = key.split(",")
        print(f"\n  J^{{{o},{i}}}   (t, s) 要素 = d{o}_t / d{i}_s")
        print(_fmt(J[key]))

    print()
    print("=" * 78)
    print("4. 精度の点検")
    print("=" * 78)
    # 真値の代理として、中央差分・十分小さい eps の直接法を使う
    truth = jacobian_direct(p, ss, T=T, h=1e-6, twosided=True)

    print("  (a) 直接法：差分幅 eps に対する収束（真値の代理 = 中央差分 eps=1e-6）")
    print(f"    {'eps':>10}  {'片側差分':>14}  {'中央差分':>14}")
    for h in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        e1 = max(np.max(np.abs(jacobian_direct(p, ss, T=T, h=h)[k] - truth[k])) for k in truth)
        e2 = max(np.max(np.abs(jacobian_direct(p, ss, T=T, h=h, twosided=True)[k] - truth[k]))
                 for k in truth)
        print(f"    {h:10.0e}  {e1:14.3e}  {e2:14.3e}")
    print("    片側差分の誤差は eps に比例（O(eps)、ノート 7.6.1）。")
    print("    中央差分は eps^2 で落ちるが、eps を小さくしすぎると丸め誤差が効き始める")
    print("    （eps=1e-7 で悪化するのがそれ。ノート 7.6.3 のトレードオフ）。")

    print("\n  (b) 直接法 vs フェイクニュース法（第8章）")
    print(f"    {'eps':>10}  {'fake news 片側':>16}  {'fake news 中央':>16}")
    for h in (1e-3, 1e-4, 1e-5, 1e-6):
        e1 = max(np.max(np.abs(jacobian_fake_news(p, ss, T=T, h=h)[k] - truth[k])) for k in truth)
        e2 = max(np.max(np.abs(jacobian_fake_news(p, ss, T=T, h=h, twosided=True)[k] - truth[k]))
                 for k in truth)
        print(f"    {h:10.0e}  {e1:16.3e}  {e2:16.3e}")
    print("    フェイクニュース法も有限差分である。ただし微分するのは『経路全体の写像』ではなく")
    print("    『1期分の後ろ向きステップ』で、その誤差が対角方向の累積を通じて J に伝わる。")
    print("    速いのは重複計算を除くからであって、有限差分をやめるからではない。")

    print("\n  (c) 直接法の第3列を eps ごとに並べる（J^{K,r}[:, 3], 片側差分）:")
    for h in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        Jh = jacobian_direct(p, ss, T=T, h=h)
        print(f"    eps = {h:8.1e}   {np.array2string(Jh['K,r'][:, 3], precision=6)}")

    print()
    print("=" * 78)
    print("5. 章末確認問題1：T = 5 で J^{K,r} の第3列を片側差分で求める")
    print("=" * 78)
    r_pert = np.full(T, ss.r)
    r_pert[3] += 1e-4
    K_base, _ = td_household(p, ss, np.full(T, ss.r), np.full(T, ss.w))
    K_pert, _ = td_household(p, ss, r_pert, np.full(T, ss.w))
    print("  手順:")
    print("    (1) r 経路 = [r_ss, r_ss, r_ss, r_ss + eps, r_ss] を作る")
    print("    (2) V_5 = V_ss から t = 4, 3, 2, 1, 0 と後ろ向きに政策 {a_t, c_t} を求める")
    print("    (3) D_0 = D_ss から t = 0, ..., 4 と前向きに分布を進める")
    print("    (4) K_t = a_t' D_t を集計し、基準走行との差を eps で割る")
    print()
    print(f"    基準走行  K^0_t   = {np.array2string(K_base, precision=8)}")
    print(f"    摂動走行  K^{{3,+}}_t = {np.array2string(K_pert, precision=8)}")
    print(f"    第3列     J[:, 3] = {np.array2string((K_pert - K_base) / 1e-4, precision=6)}")
    print()
    print("  読み方: t < 3 の成分がゼロでないのは、時点3の金利上昇を「予見」した家計が")
    print("          それ以前から貯蓄を変えるため（ニュース効果）。ノート 6.3。")


if __name__ == "__main__":
    main()
