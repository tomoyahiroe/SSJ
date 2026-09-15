# 分かっていないところ

## 1. KS モデルを SSJ で解くために、モデルの設定を変えている部分はどこか？

**1 か所に集約してある:** `ks/calibration.py` の `_build_markov_chains`。 そこから派生する `KSModel.Z` と `KSModel.N` もその結果を使う。 家計問題（`household.py`）・価格の式・所得の式は KS(1998) のまま。

| 場所                   | 変えていること                                                                                               |
| ---------------------- | ------------------------------------------------------------------------------------------------------------ |
| `_build_markov_chains` | 集計ショック $A$ を積分して消し、$(A, l)$ の 4 状態チェーンを $l$ だけの 2 状態チェーン `Pi` にする          |
| `KSModel.Z`            | TFP を $\{0.99, 1.01\}$ の確率変数から、その定常分布での平均 $Z = 1.0$ という定数にする                      |
| `KSModel.N`            | 集計労働を、状態ごとの $u_A$ ではなく潰したチェーンの定常失業率 $u_{ss}$ で $N = \bar l (1 - u_{ss})$ にする |
| `Prices`               | 家計ブロックの入力を $(r, w)$ の 2 本だけにする（$A$ や $K$ の予測式は入力に含まれない）                     |

なお「解き方」の違いはモデルの設定変更ではない。元コードは $K$ グリッド + 予測式（`Kgrid`, `vfi_numba`）で $A$ を状態変数に持ったまま解くが、こちらは「$A$ のない定常状態 + そのまわりのヤコビアン」で解く。この違いが、上の表の変更を**必要にしている**理由。

理由はモジュールの冒頭に書いてある:

```python
"""パラメータと、集計ショックを積分して消す作業。

既存コード 8_2_krusell_and_smith.py の `Setting`（replicate_model = 0、
すなわち Krusell and Smith 1998）を、検証つきの pydantic モデルに置き換えたもの。

SSJ の定常状態には集計ショック A が存在しないので、KS の (A, l) 4状態チェーンを
l だけの2状態チェーンに潰す必要がある。その手順もここに置く。
"""
```

## 2. 何をどう変えているか？

**変えているのは「雇用の遷移確率が景気に依存する」という部分で、景気を平均して消している。**

KS では雇用の遷移 $P(l' \mid l)$ が今期と来期の景気 $(A, A')$ に依存する（好況で $u = 0.04$、不況で $u = 0.10$）。 SSJ の定常状態には $A$ がないので、家計が見る遷移行列を 1 枚に決めなければならない。やっているのは次の 3 段階。

1. **4 状態の同時チェーンを作る** — 元コード `markov_KS` の移植。`joint[row, col]` $= P(A' \mid A)\, P(l' \mid A, A', l)$
2. **同時定常分布 $\pi(A, l)$ を求める** — `stationary_distribution(joint)` → `[0.05, 0.45, 0.02, 0.48]`（順に 不況&失業, 不況&就業, 好況&失業, 好況&就業）
3. **$A$ で加重平均する** —

$$
\Pi(l' \mid l) = \frac{\sum_{A, A'} \pi(A, l)\, P(A' \mid A)\, P(l' \mid A, A', l)}{\pi(l)}
$$

分子は「$l$ から $l'$ へのフロー」、分母 `marg_l` は「$l$ にいるストック」。重みが $l$ ごとに違う（$\pi(A \mid l)$）点が肝で、共通の $\pi(A) = 0.5$ で潰すと定常失業率が 0.07 からずれる。

これが実装（`ks/calibration.py:239-255`）:

```python
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
                        joint_stationary[row]
                        * aggregate[i_a, j_a]
                        * employment_given_a[row, 2 * j_a + j_l]
                    )
    employment /= marg_l[:, None]
```

結果は $\Pi = \begin{pmatrix} 0.5074 & 0.4926 \\ 0.0371 & 0.9629 \end{pmatrix}$（行 = 今、列 = 来期、\[失業, 就業\]）。 この 1 枚が `KSModel.Pi` として家計ブロックの外生過程になる。

同じ考え方で TFP も平均する（`KSModel.Z`）:

```python
    @cached_property
    def Z(self) -> float:
        """定常状態の TFP。集計状態の定常分布で加重平均したもの。"""
        z = np.array([self.calibration.z_bad, self.calibration.z_good])
        return float(self.chains.aggregate_stationary @ z)
```

`dur_good = dur_bad = 8` で $A$ の定常分布が $[0.5, 0.5]$ になるので $Z = 0.5 \times 0.99 + 0.5 \times 1.01 = 1.0$ ちょうど。

変えて**いない**もの: 選好（$\beta = 0.99$, log 効用）、技術（$\alpha, \delta$）、失業給付の水準 0.07、借入制約 $a \ge 0$、資産グリッド。 「Pi が 4 枚の条件付き行列の加重平均になっている」ことはテスト `test_employment_chain_is_mixture_of_conditional_matrices` で確かめている。

## 3. 定常失業率はどうやって計算できるのか？

**潰した 2 状態チェーン** `Pi` の定常分布の第 0 成分。 $x \Pi = x$, $\sum x = 1$ を解いた $x = [u_{ss}, 1 - u_{ss}]$。

2 状態なら閉じた式がある。$\Pi = \begin{pmatrix} 1-a & a \\ b & 1-b \end{pmatrix}$（$a$ = 失業→就業、$b$ = 就業→失業）のとき

$$
u_{ss} = \frac{b}{a + b} = \frac{0.037074}{0.49256 + 0.037074} = 0.0700
$$

コードでは閉じた式ではなく、汎用の反復（$x \leftarrow x\Pi$ を収束まで）で解いている（`ks/calibration.py:264`、`stationary_distribution`）:

```python
    return KSMarkovChains(
        ...
        employment=employment,
        u_ss=float(stationary_distribution(employment)[0]),
    )
```

```python
@typed
def stationary_distribution(
    P: Float[FloatArray, "n n"], tol: float = 1e-14, max_iter: int = 1_000_000
) -> Float[FloatArray, "n"]:
    """行確率行列 P の定常分布 x（x P = x, 1'x = 1）を反復で求める。"""
    x: FloatArray = np.full(P.shape[0], 1.0 / P.shape[0], dtype=np.float64)
    for _ in range(max_iter):
        y = x @ P
        if np.max(np.abs(y - x)) < tol:
            return y
        x = y
```

**検算が 2 通りある。**

- 同時定常分布の周辺: $\pi(u) = \pi(b, u) + \pi(g, u) = 0.05 + 0.02 = 0.07$（`employment_marginal`）。KS は「来期の景気 $A'$ での失業率がちょうど $u_{A'}$ になる」ように遷移確率を作っているので、$P(u \mid A = b) = 0.10$, $P(u \mid A = g) = 0.04$ が正確に成り立ち、$u = 0.5 \times 0.10 + 0.5 \times 0.04 = 0.07$。
- 2 つが一致することはテスト `test_stationary_unemployment_is_consistent` で確かめている。

注意: 失業**率** 0.07 と失業**給付** `unemployment_insurance = 0.07` は無関係で、数値の一致は偶然。

## 4. `p_gg, p_bb` とは何で、どうやって計算しているのか？なぜその計算で良いのか？

**景気の持続確率。** `p_gg` $= P(\text{来期 好況} \mid \text{今 好況})$、`p_bb` $= P(\text{来期 不況} \mid \text{今 不況})$。

計算は「平均継続期間 $D$」からの逆算（`ks/calibration.py:205-208`）:

```python
    # --- 集計状態の遷移。q = (D - 1) / D は「平均継続期間 D」と等価 ---
    p_gg = (c.dur_good - 1.0) / c.dur_good
    p_bb = (c.dur_bad - 1.0) / c.dur_bad
    aggregate = np.array([[p_bb, 1.0 - p_bb], [1.0 - p_gg, p_gg]])  # 行 = [不況, 好況]
```

**なぜそれで良いか**: 2 状態マルコフ連鎖では、好況に入ってから抜けるまでの長さが幾何分布に従う。毎期確率 $p$ で留まり、$1 - p$ で抜けるので

$$
P(\text{ちょうど } k \text{ 期続く}) = p^{k-1}(1 - p), \qquad
E[\text{継続期間}] = \sum_{k \ge 1} k\, p^{k-1}(1 - p) = \frac{1}{1 - p}
$$

これを $p$ について解くと $p = 1 - 1/D = (D - 1)/D$。KS(1998) は「好況・不況の平均長さは 8 四半期」と置いたので $p = 7/8 = 0.875$、抜ける確率は $0.125$。

つまり `dur_good` は「観測できる量（景気の平均長さ）」で、`p_gg` は「モデルの中の量（遷移確率）」。前者から後者を決めるのがキャリブレーション。 実際 `aggregate` の定常分布は $[0.5, 0.5]$（`dur` が対称なので）で、これは `test_aggregate_chain_has_expected_duration` が確かめている。

同じ逆算が雇用側にもある（`ks/calibration.py:211-218`）。`pgg00 = (1.5 - 1) / 1.5 = 1/3` は「好況が続くとき、失業者が来期も失業している確率」で、失業の平均長さ 1.5 四半期の逆算。そこから残りの確率は「来期の失業率がちょうど $u_{A'}$ になる」という条件で決まる:

$$
u_{A'} = u_A\, p_{00} + (1 - u_A)\, p_{01} \quad\Longrightarrow\quad p_{01} = \frac{u_{A'} - u_A\, p_{00}}{1 - u_A}
$$

（$p_{01}$ = 就業者が失業する確率。コードの添字は Fortran の慣習で `pbg` = 「好況から不況へ」のように**来期が先**。`by_a` の並びで確認できる。）

## 5. SSJ って定常状態が求まっていることを前提としている？

**はい。SSJ は「定常状態のまわりでのブロック写像 $H$ の微分」なので、定常状態は前提（入力）。** 使い方の順番がそのまま答えになっている:

```python
model = ks.KSModel.build()
ss = ks.solve_general_equilibrium(model)    # まず定常状態
J = ks.jacobian_direct(model, ss, T=5)      # それを渡してヤコビアン
```

Q: solve_general_equilibriumでやっていることは非線形にモデルの定常状態を求めるということ？普通に定常状態になるような金利を入れてみて、ループ回して調整する的なことをしている？つまり、SSJとかKSとかというよりも普通にモデルの設定のもとで、定常状態を解いている？

**A:** そのとおり。SSJ にも KS にも固有の話ではなく、Aiyagari 型モデルの定常状態を求める標準の手順そのもの。ただし 2 点だけ補足がある。

1. **ループで動かしているのは金利ではなく $K$。** 金利は $K$ から企業の一階条件で決まる（`model.prices(K)` → $r = \alpha Z (K/N)^{\alpha-1} - \delta$）ので、どちらで回しても同じ。$K$ にしてあるのは、下側の境界を「$r = 1/\beta - 1$ になる $K$」（`K_impatience_bound`）で切れて分かりやすいから。
2. **解いているのは「$A$ を消した後のモデル」の定常状態。** KS(1998) そのものには集計ショックがあるので厳密な定常状態はなく、元コードは予測式で「確率的定常状態」を扱う。ここで解いているのは質問 1・2 で潰したチェーン `Pi` と $Z = 1.0$ のもとでの、集計リスクのない経済の定常状態。SSJ が前提にするのはこちら。

手順は「$K$ を仮置き → 価格 → 家計問題を非線形に解く → 資産需要 − $K$ の符号で区間を半分に」の二分法（[`solve_general_equilibrium`](../ks/household.py#L356-L400)）:

```python
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
        ...
        if excess > 0.0:
            K_low = K_mid
        else:
            K_high = K_mid
            ss = candidate
```

「非線形に」の中身は `evaluate` が呼ぶ [`solve_household`](../ks/household.py#L307-L352)。価格を固定して EGM の後ろ向き反復を政策が収束するまで回し（[`backward_egm`](../ks/household.py#L155)）、その政策から定常分布を線形方程式で解く（[`stationary_from_policy`](../ks/household.py#L251-L271)）。線形化は一切していない。`warm`（前回の $V_a$ から再開）は速度のためだけで、答えは変わらない。二分法が使えるのは超過需要が $K$ について単調減少だから（$r(K)$ は減少、資産需要は $r$ について増加）。`verbose=True` で各ステップの $K, r$, 超過需要が見える。


Q: これsequence jacobian パッケージでは、裏で勝手に定常状態を求めた上で、各ブロックのヤコビアンを求めるべく、SSJに渡してくれているの？

**A:** 定常状態は「裏で勝手に」ではなく、**自分で明示的に解いて渡す**。自動なのはその後（各ブロックのヤコビアンを作って DAG に沿って合成するところ）。ライブラリの API はどれも定常状態を第 1 引数に取る:

```
block.steady_state(calibration)                     → SteadyStateDict   （ブロック単体・価格所与）
model.solve_steady_state(calibration, unknowns, targets, solver=...)  → SteadyStateDict （一般均衡）
block.jacobian(ss, inputs, outputs, T)              ← ss を渡す
model.solve_jacobian(ss, unknowns, targets, inputs, outputs, T)       ← ss を渡す
block.impulse_nonlinear(ss, shocks)                 ← ss を渡す
```

このリポジトリでの対応（[`ks/ssj.py`](../ks/ssj.py)）:

| 段階 | 自作版 | ライブラリ版 |
|---|---|---|
| 家計ブロックの定常状態（価格所与） | `solve_household` | [`solve_steady_state`](../ks/ssj.py#L150-L154) → `hh_extended.steady_state(...)` |
| 一般均衡（資本市場の清算） | `solve_general_equilibrium`（二分法） | [`solve_general_equilibrium_with_library`](../ks/ssj.py#L232-L259) → `ks_model.solve_steady_state(..., solver="brentq")` |
| ヤコビアン | `jacobian_direct(model, ss, T)` | [`library_jacobians`](../ks/ssj.py#L180-L199) → `hh_extended.jacobian(ss_library, ...)` |

一般均衡の呼び出しが一番分かりやすい。`unknowns` に「振る変数と探索区間」、`targets` に「ゼロにする式」を渡すと、ライブラリが `solve_general_equilibrium` と同じ「$K$ を振って `asset_mkt = A - K` の符号を見る」ループを brentq で回す（[`ks/ssj.py:253-259`](../ks/ssj.py#L253-L259)）:

```python
    return ks_model.solve_steady_state(
        calibration,
        unknowns={"K": (K_low, 40.0 * model.N)},
        targets={"asset_mkt": 0.0},
        solver="brentq",
        options={"hh": STRICT_TOLERANCES},
    )
```

そして得た `SteadyStateDict` を、ヤコビアンには**こちらから**渡す（[`ks/ssj.py:189-191`](../ks/ssj.py#L189-L191)）:

```python
    J = hh_extended.jacobian(
        ss_library, inputs=["r", "w"], outputs=["A", "C"], T=T, h=eps, twosided=twosided
    )
```

`HetBlock.steady_state` の中でやっていることは自作版と同じで、`backward_init=hh_init` の初期値から後ろ向き反復を `backward_tol` まで回し、分布を前向き反復で `forward_tol` まで回す。ただし分布を反復で解くので、このキャリブレーションでは既定の許容誤差だと $K$ が $10^{-4}$ ずれる（`STRICT_TOLERANCES` を渡している理由。`02_experiments` 参照）。

「勝手にやってくれる」のは `solve_jacobian` / `solve_impulse_linear` の側で、家計ブロックはフェイクニュース法、`@simple` ブロックは自動微分でそれぞれヤコビアンを作り、DAG の順に掛け合わせて一般均衡の応答まで出す。定常状態だけは、どの経路でも入力。


コードの中で定常状態が使われる場所は 3 つ。

| どこ                  | 何に使うか                                                                    |
| --------------------- | ----------------------------------------------------------------------------- |
| `jacobian_direct`     | 摂動する経路の土台 `r_ss = np.full(T, ss.prices.r)`。基準走行も定常価格で流す |
| `simulate_transition` | 後ろ向きの**終端条件** `Va = ss.Va`、前向きの**初期条件** `D = ss.D`          |
| `library_jacobians`   | ライブラリの `.jacobian()` も `SteadyStateDict` を最初の引数に取る            |

一番大事なのは 2 つ目。移行経路は「$T$ 期後には定常状態に戻っている」「時点 0 の分布は定常分布」という両端の条件がないと解けない（`ks/household.py:452-466`）:

```python
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
```

補足が 3 つ。

- **どの定常状態か**: 家計ブロックのヤコビアンだけなら、価格 $(r, w)$ を与えて解いた部分均衡の定常状態（`solve_household`）で足りる。資本市場を清算する一般均衡（`solve_general_equilibrium`）が要るのは「定常価格そのもの」を決めるため。`02_experiments` ではライブラリが解いた定常状態 `ss_from_lib` の上でも同じヤコビアンが出ることを確かめている。
- **精度が要る**: 定常状態が不正確だと基準走行 $Y^0$ が平らにならず、その誤差が $1/\varepsilon$ 倍されてヤコビアンに乗る。講義ノート 7.7 の「基準走行を差し引く」はその対策で、分布を反復ではなく線形方程式で解く（`stationary_from_policy`）のも同じ理由。`02_experiments` の「sloppy な定常状態」の実験がこれを見せている。
- **局所的な道具**: ヤコビアンは定常状態での一次微分なので、大きなショックや定常状態から遠い経路には使えない。それでも非線形の移行経路（MIT ショック）を解くときも、両端の条件として定常状態は必要。
