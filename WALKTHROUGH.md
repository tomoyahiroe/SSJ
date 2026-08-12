# `ks` パッケージの歩き方

`ks.jacobian_direct(model, ss, T=5)` を呼んだとき、中で何が起きているのかを
**実行される順番どおりに**、言葉と実際のコードを交互に並べて追う。

引用しているコードはすべて実物（コメントとドックストリングは一部省略）。

---

## 全体像

やりたいことは1本の式に集約される。家計ブロックは価格の経路を受け取って集計量の経路を返す
写像であり、それを定常状態のまわりで一次微分したものがヤコビアン。

$$
\begin{pmatrix} \mathcal{K} \\ \mathcal{C} \end{pmatrix} = H\!\left(\begin{pmatrix} r \\ w \end{pmatrix}\right)
\qquad\Longrightarrow\qquad
\begin{pmatrix} d\mathcal{K} \\ d\mathcal{C} \end{pmatrix}
= \begin{pmatrix} J^{K,r} & J^{K,w} \\ J^{C,r} & J^{C,w} \end{pmatrix}
  \begin{pmatrix} dr \\ dw \end{pmatrix}
$$

呼び出しの流れ:

```
ks.KSModel.build()                      calibration.py   パラメータ → 導出量
    └── _build_markov_chains()                           (A,l) 4状態 → l だけの 2状態
    └── exponential_grid()                               資産グリッド

ks.solve_general_equilibrium(model)     household.py     資本市場を清算する K を二分法で
    └── solve_household(model, prices)                   価格所与で政策と分布
            └── backward_egm()          ×2万回           EGM で1期分の後ろ向き
            └── stationary_from_policy()                 D = Λ'D を線形方程式で

ks.jacobian_direct(model, ss, T=5)      jacobian.py      直接法（第7章）
    └── simulate_transition()           ×(2入力 × 5列 + 基準走行1)
            └── backward_egm()          ×T              終端 V_ss から後ろ向き
            └── forward_step()          ×T              初期 D_ss から前向き
```

| モジュール | 役割 |
|---|---|
| `ks/types.py` | 語彙。配列の形と意味を型で表す |
| `ks/calibration.py` | パラメータ、集計ショックの消去、価格と所得 |
| `ks/household.py` | EGM・前向き・定常状態・移行経路 |
| `ks/jacobian.py` | 直接法（第7章）とフェイクニュース法（第8章） |
| `ks/ssj.py` | sequence-jacobian ライブラリ版 |

---

## 0. 語彙をそろえる — `ks/types.py`

異質的主体モデルで一番よく起きるバグは**配列の形の取り違え**である。
`np.ndarray` としか書いていないと、それが政策関数なのか分布なのかヤコビアンなのか、
軸がどちらが雇用状態でどちらが資産なのかが読み取れない。

そこで軸の名前まで型に書く。

```python
n_e : 個別の雇用状態の数。KS では 2（失業 / 就業）
n_a : 資産グリッドの点数
T   : 系列空間の切断期間（宿題では 5）
n_A : 集計状態の数。KS では 2（不況 / 好況）
      定常状態を作る過程でのみ登場し、SSJ の家計ブロックには残らない
```

```python
PolicyFunction: TypeAlias = Float[np.ndarray, "n_e n_a"]
"""個別状態 (e_t, a_{t-1}) 上の政策関数。期末資産 a_t または消費 c_t。"""

MarginalValue: TypeAlias = Float[np.ndarray, "n_e n_a"]
"""価値関数の資産微分 V_a。EGM の後ろ向き変数。形は政策関数と同じ。"""

Distribution: TypeAlias = Float[np.ndarray, "n_e n_a"]
"""時点 t 冒頭の個別状態 (e_t, a_{t-1}) 上の分布 D_t。総和は 1。"""

JacobianMatrix: TypeAlias = Float[np.ndarray, "T T"]
"""(t, s) 要素が dY_t / dX_s。行 t = 応答の時点、列 s = ショックの時点。"""
```

`PolicyFunction` と `MarginalValue` と `Distribution` は**形は同じ `"n_e n_a"` だが名前が違う**。
これは意図的で、シグネチャを見たときに「これは分布であって政策ではない」と分かるようにしている。

添字も名前で書けるようにする。

```python
class Employment(IntEnum):
    UNEMPLOYED = 0
    EMPLOYED = 1
```

これで `ss.a[0]` ではなく `ss.a[Employment.UNEMPLOYED]` と書ける。

型注釈は飾りではなく、実行時にも検査される。

```python
_RUNTIME_TYPECHECK = os.environ.get("KS_TYPECHECK", "1").lower() not in ("0", "false", "no")

def typed(fn: _F) -> _F:
    """引数と返り値の dtype・形状を実行時に検査するデコレータ。"""
    if not _RUNTIME_TYPECHECK:
        return fn
    return jaxtyped(typechecker=beartype)(fn)
```

同じ軸名（例えば `"n_a"`）が複数の引数に現れる場合、**その長さが一致していることまで**見る。
資産グリッドが 200 点なのに政策が 150 点、という取り違えはその場で落ちる。
重い計算で外したいときは `KS_TYPECHECK=0`（約2倍速くなる）。

---

## 1. パラメータを持つ — `KSCalibration`

既存コード `8_2_krusell_and_smith.py` の `Setting`（`replicate_model = 0`）を、
検証つきの pydantic モデルに置き換えたもの。**スカラーだけ**を持ち、配列は一切持たない。

```python
class KSCalibration(BaseModel):
    model_config = ConfigDict(frozen=True)

    # --- 技術 ---
    alpha: float = Field(0.36, gt=0.0, lt=1.0, description="資本分配率")
    delta: float = Field(0.025, gt=0.0, lt=1.0, description="四半期の資本減耗率")

    # --- 選好 ---
    beta: float = Field(0.99, gt=0.0, lt=1.0, description="四半期の割引因子")
    eis: float = Field(1.0, gt=0.0, description="異時点間代替の弾力性。1 なら log 効用")

    # --- 労働と失業保険 ---
    lbar: float = Field(0.3271, gt=0.0, description="就業時の労働時間（既存コードの lbar）")
    unemployment_insurance: float = Field(
        0.07,
        gt=0.0,
        description="失業給付。既存コードの replicate_model=0 では w に比例しない「水準」",
    )
    ...
```

`gt=0.0, lt=1.0` が効くので、`KSCalibration(beta=1.5)` は `ValidationError` になる。
タイプミスで非現実的なパラメータのまま数時間走る、という事故が防げる。

`frozen=True` なので後から書き換えられない。定常状態を解いたあとにパラメータが
書き換わって整合性が壊れる、ということが起きない。

派生的な量はプロパティで持つ。

```python
    @property
    def r_impatience_bound(self) -> float:
        """r がこれ以上だと資産需要が発散する上限 1/beta - 1。"""
        return 1.0 / self.beta - 1.0
```

---

## 2. 集計ショックを消す — `_build_markov_chains`

ここが KS を SSJ に持ち込むときの一番の関門。

**問題**: SSJ の定常状態には集計ショック $A$ が存在しない。だが KS の雇用遷移は $A$ に依存する
（好況 $u = 0.04$、不況 $u = 0.10$）。$(A, l)$ の4状態チェーンを $l$ だけの2状態に潰す必要がある。

まず既存コードの `markov_KS` をそのまま移植して、4状態の同時チェーンを作る。
`dur` パラメータから確率を作るところは「2状態チェーンで留まる確率 $q$ の期待継続期間は
$1/(1-q)$」という関係の逆算である。

```python
    # --- 集計状態の遷移。q = (D - 1) / D は「平均継続期間 D」と等価 ---
    p_gg = (c.dur_good - 1.0) / c.dur_good
    p_bb = (c.dur_bad - 1.0) / c.dur_bad
    aggregate = np.array([[p_bb, 1.0 - p_bb], [1.0 - p_gg, p_gg]])  # 行 = [不況, 好況]
```

`dur_good = dur_bad = 8` なので $q = 7/8 = 0.875$、行列は対称になる。
これが効いて $A$ の定常分布が $[0.5, 0.5]$ ちょうどになり、$u = 0.07$、$Z = 1.00$ が出る。

次に $(A, A')$ の組ごとの雇用遷移を作り、4×4 に組み上げる。

```python
    joint = np.zeros((4, 4))
    employment_given_a = np.zeros((4, 4))
    for i_a in range(2):
        for i_l in range(2):
            row = 2 * i_a + i_l
            for j_a in range(2):
                for j_l in range(2):
                    col = 2 * j_a + j_l
                    employment_given_a[row, col] = by_a[i_a][j_a][i_l, j_l]
                    joint[row, col] = aggregate[i_a, j_a] * by_a[i_a][j_a][i_l, j_l]
```

同時遷移が $P(A'\mid A) \cdot P(l' \mid A, A', l)$ と積に分解できるのは KS の作り方による。
$P(A'\mid A)$ が $l$ に依存しないのは個人が無限小だから。

**そして $A$ を積分して消す。** ここが要点。

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
                        joint_stationary[row] * aggregate[i_a, j_a]
                        * employment_given_a[row, 2 * j_a + j_l]
                    )
    employment /= marg_l[:, None]
```

これは数式で書くと

$$\Pi(l' \mid l) = \frac{\sum_{A, A'} \pi(A, l)\, P(A' \mid A)\, P(l' \mid A, A', l)}{\pi(l)}$$

で、やっているのは**条件付き確率の定義そのもの**である。分子は「実際に起きている
$l \to l'$ のフロー」、分母は「実際に $l$ にいるストック」。$A$ は観測しないので
可能な $(A, A')$ の組を全部足し上げて消している。

最後の `employment /= marg_l[:, None]` の `[:, None]` が重要で、**行ごとに違う数で割っている**。
失業者は不況に偏在している（失業者の 71.4% が不況にいる。不況の頻度は 50% しかないのに）ので、
失業者の行だけ不況側の遷移行列が重くなる。共通の $\pi(A) = [0.5, 0.5]$ で潰すと
定常失業率が 0.0657 になり 0.07 から外れる。

得られた `Pi` は `[[0.507440, 0.492560], [0.037074, 0.962926]]`、定常失業率はちょうど 0.07。
検証は `02_experiments.ipynb` 節1 にある。

> **落ちているもの**: 元の KS 経済では $l$ 単独は厳密にはマルコフではない。
> $\Pi$ は1期先の条件付き分布を平均的に再現するが、2期先以降はずれる（最大 6.9e-3）。
> 「不況では失業リスクが上がる」という連動も消える。

---

## 3. モデルを組み立てる — `KSModel`

`KSCalibration` が「入力」、`KSModel` が「そこから決まるもの」という分担。

```python
class KSModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    calibration: KSCalibration
    chains: KSMarkovChains
    a_grid: AssetGrid = Field(description="資産グリッド。政策と分布で共用する")

    @classmethod
    def build(cls, calibration: KSCalibration | None = None) -> "KSModel":
        c = calibration or KSCalibration()
        return cls(
            calibration=c,
            chains=_build_markov_chains(c),
            a_grid=exponential_grid(c.a_min, c.a_max, c.n_a),
        )
```

企業側の関係式もここに置く。既存コードの `vfi_numba` の中にあった式と同一。

```python
    @typed
    def prices(self, K: float) -> Prices:
        """期首資本 K = K_{t-1} から (r_t, w_t) を作る。"""
        c = self.calibration
        k_n = K / self.N
        return Prices(
            r=c.alpha * self.Z * k_n ** (c.alpha - 1.0) - c.delta,
            w=(1.0 - c.alpha) * self.Z * k_n**c.alpha,
        )

    @typed
    def income(self, w: float) -> IncomeByState:
        """雇用状態ごとの労働所得 y(e)。並びは [失業, 就業]。"""
        c = self.calibration
        return np.array([c.unemployment_insurance, w * c.lbar])
```

`income` の失業側が `w` を掛けていないのが `replicate_model = 0` の特徴。
失業給付 0.07 は水準であって賃金比例ではない。

均衡がどこにあり得るかの上限も持っておく。

```python
    @cached_property
    def K_impatience_bound(self) -> float:
        """r(K) = 1/beta - 1 となる K。均衡 K はこれより大きくないと定常分布が存在しない。"""
```

このモデルでは 11.5564。均衡 $K = 11.6094$ はそのすぐ上にあり、
$\beta(1+r) = 0.99990$ と1に極端に近い。この事実がこの先ずっと効いてくる。

---

## 4. 家計問題の1期分 — `backward_egm`

ここが計算の最小単位。講義ノート式 7.18 の $v_t = v(v_{t+1}, X_t)$ にあたる。

```python
@typed
def backward_egm(
    expected_Va: MarginalValue,
    a_grid: AssetGrid,
    y: IncomeByState,
    r: float,
    beta: float,
    eis: float,
) -> BackwardStep:
    uc_nextgrid = beta * expected_Va
    c_nextgrid = uc_nextgrid ** (-eis)
    coh = (1.0 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]  # 手持ち資産
    a = interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    np.maximum(a, a_grid[0], out=a)  # 借入制約 a_t >= a_min
    c = coh - a
    return BackwardStep(Va=(1.0 + r) * c ** (-1.0 / eis), a=a, c=c)
```

1行ずつ:

| 行 | 内容 |
|---|---|
| `uc_nextgrid = beta * expected_Va` | オイラー方程式の右辺。これが $u'(c_t)$ の満たすべき値 |
| `c_nextgrid = uc_nextgrid ** (-eis)` | 限界効用を逆にして消費に戻す。**内生グリッド上の**消費 |
| `coh = (1+r) * a_grid + y` | 手持ち資産 cash-on-hand。**外生グリッド上** |
| `interpolate_y(c_nextgrid + a_grid, coh, a_grid)` | 内生グリッド $c + a'$ を外生グリッド $coh$ に貼り替える。EGM の心臓部 |
| `np.maximum(a, a_grid[0])` | 借入制約。オイラー方程式が成り立たない領域をここで潰す |
| `Va = (1+r) * c ** (-1/eis)` | 次の後ろ向きステップに渡す量 |

引数の `expected_Va` は $E_t[V_{a,t+1}] = \Pi \cdot V_{a,t+1}$ で、
**期待を取る操作は呼び出し側の責任**にしてある。sequence-jacobian の HetBlock は
これを代行するので、ライブラリ版のコードには `Pi @` が現れない。この対応関係は節10で見る。

返り値はタプルではなく名前つき。

```python
class BackwardStep(NamedTuple):
    Va: MarginalValue
    a: PolicyFunction
    c: PolicyFunction
```

> pydantic ではなく `NamedTuple` にしてあるのは、直接法の内側ループで何千回も
> 作られるため。注釈で形は明示してあるので読み取りには困らない。

---

## 5. 分布の1期分 — `forward_step`

講義ノート式 7.19 の $D_{t+1} = \Lambda_t^{\top} D_t$。

政策 $a_t$ はグリッド上の点に落ちるとは限らないので、まず「くじ」に直す。

```python
class Lottery(NamedTuple):
    """政策 a_t をグリッド上の2点に振り分ける「くじ」表現。

        a_t = weight * a_grid[index] + (1 - weight) * a_grid[index + 1]
    """
    index: LotteryIndex
    weight: LotteryWeight
```

そのうえで、**資産 → 雇用状態の順**に進める。

```python
@typed
def forward_endogenous(D: Distribution, lottery: Lottery) -> Distribution:
    """資産の遷移だけを進める（雇用状態はまだ動かさない）。"""
    out = np.zeros_like(D)
    for e in range(D.shape[0]):
        np.add.at(out[e], lottery.index[e], lottery.weight[e] * D[e])
        np.add.at(out[e], lottery.index[e] + 1, (1.0 - lottery.weight[e]) * D[e])
    return out


@typed
def forward_step(D: Distribution, Pi: Float[np.ndarray, "n_e n_e"], lottery: Lottery) -> Distribution:
    """D_{t+1} = Lambda_t' D_t。まず資産、次に雇用状態の順（SSJ の規約と同じ）。"""
    return Pi.T @ forward_endogenous(D, lottery)
```

順番は規約の問題だが、**sequence-jacobian と合わせておかないと数値が一致しない**ので
そちらに揃えてある。`D_t` は「時点 $t$ 冒頭、$e_t$ が判明したあと、$a_{t-1}$ を持っている状態」の分布。

---

## 6. 定常状態 — `solve_household` と `solve_general_equilibrium`

### 6.1 価格所与で解く（部分均衡）

政策は不動点に達するまで後ろ向きに回す。

```python
    Va = Va_init if Va_init is not None else initial_marginal_value(model.a_grid, y, prices.r, c.eis)
    a_previous = np.zeros((model.n_e, model.n_a))
    step = BackwardStep(Va=Va, a=a_previous, c=a_previous)
    iterations = 0
    for iterations in range(1, backward_maxit + 1):
        step = backward_egm(model.Pi @ step.Va, model.a_grid, y, prices.r, c.beta, c.eis)
        if iterations % 10 == 0:
            if np.max(np.abs(step.a - a_previous)) < backward_tol:
                break
            a_previous = step.a
```

`model.Pi @ step.Va` が節4で言った「期待を取る操作」。10回に1回しか収束判定しないのは
判定のコストを削るためで、sequence-jacobian も同じことをしている。

分布の方は**反復しない**。ここがこの実装の特徴。

```python
@typed
def stationary_from_policy(
    Pi: Float[np.ndarray, "n_e n_e"], lottery: Lottery
) -> StationaryDistribution:
    """政策から定常分布を直接解く: D = Lambda' D, 1'D = 1。"""
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
```

**なぜ反復しないのか。** $\beta(1+r) = 0.99990$ と1に極端に近いため、遷移行列 $\Lambda_{ss}$ の
第2固有値も1に近く、前向き反復が「見かけ上収束したのにまだ真の定常分布から遠い」状態になる。
実測では `max|D_new - D| < 1e-10` に 33,000 回かかり、そのときの $D$ はまだ真の値から
1.2e-7 ずれていて $K$ が 1.1e-4 ずれる。

状態数は $2 \times 200 = 400$ しかないので、密行列を作って `np.linalg.solve` に投げれば一発。
残差は 4.7e-16 まで落ちる。これが後で効いてくる（節8の「基準走行」）。

### 6.2 資本市場を清算する（一般均衡）

$r(K)$ は $K$ について減少、資産需要は $r$ について増加なので、超過需要は $K$ について減少する。
よって二分法で解ける。

```python
    def evaluate(K: float) -> SteadyState:
        nonlocal warm
        ss = solve_household(model, model.prices(K), Va_init=warm)
        warm = ss.Va
        return ss
```

`warm` で前回の $V_a$ を使い回すので、二分法の各ステップの後ろ向き反復が 20 回程度で済む。

下側の境界は節3で計算した発散点を使う。

```python
    K_low = K_low if K_low is not None else model.K_impatience_bound * (1.0 + 1e-8)
```

結果は $K = 11.6093640145$、$r = 0.00999853$、$w = 2.37449987$。

---

## 7. ブロック写像 $H$ — `simulate_transition`

**ここが宿題の中心**。価格経路を1本受け取って集計経路を返す。これが $H$ そのもの。

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
    for t in range(T):
        D_path[t] = D
        K[t] = np.vdot(D, a_path[t])
        C[t] = np.vdot(D, c_path[t])
        D = forward_step(D, model.Pi, asset_lottery(model.a_grid, a_path[t]))
```

講義ノート 7.2 の3段階がそのまま3つのループになっている。

- **`Va = ss.Va` から始めて `reversed(range(T))`** — 終端条件 $V_T = V_{ss}$ から後ろ向き。
  ここが「時点 $s$ だけ解けばよいわけではない」の正体。$r$ を1点だけ動かしても、
  そのループを通じて $t < s$ の政策が全部変わる。
- **`D = ss.D.copy()` から `range(T)`** — 初期条件 $D_0 = D_{ss}$ から前向き。
- **`np.vdot(D, a_path[t])`** — 集計。$K_t = a_t' D_t$。

方向が逆の2つのループが必要で、しかも両方とも長さ $T$ 必要。
これが直接法の1列あたりのコストで、$O(T)$。

---

## 8. 直接法 — `jacobian_direct`

節7を全部の $s$ と全部の入力について繰り返せばヤコビアンになる。$T$ 列 × 長さ $T$ で $O(T^2)$。

まず基準走行を1本流す。

```python
    r_ss = np.full(T, ss.prices.r)
    w_ss = np.full(T, ss.prices.w)

    # ノート 7.7「基準走行を差し引く理由」:
    # 定常値 Y_ss * 1 ではなく、同じ有限期間・同じ終端条件・同じ数値手順で走らせた
    # Y^0 を差し引く。摂動と無関係な基準誤差が 1/eps 倍されるのを避けるため。
    baseline = simulate_transition(model, ss, r_ss, w_ss)
```

理想的には `baseline.K` は $K_{ss}$ で平坦なはずだが、有限期間・補間・収束誤差のせいで
完全には平坦にならない。$Y_{ss}\mathbf{1}$ を引くとその誤差まで $1/\varepsilon$ 倍される。
節6.1 で分布を線形方程式で解いておいたおかげで、ここでは 1e-12 まで平坦になっている。

そして列を1本ずつ作る。

```python
    for input_ in ("r", "w"):
        base = r_ss if input_ == "r" else w_ss
        for s in range(T):
            # 式 7.4:  X^{s,+}_t = X_ss + eps * 1{t = s}
            up = base.copy()
            up[s] += eps
            path_up = simulate_transition(
                model, ss, up if input_ == "r" else r_ss, w_ss if input_ == "r" else up
            )

            if scheme is DifferenceScheme.CENTRAL:
                ...
                denominator = 2.0 * eps
                reference_K, reference_C = path_down.K, path_down.C
            else:
                denominator = eps
                reference_K, reference_C = baseline.K, baseline.C

            columns[f"K_{input_}"][:, s] = (path_up.K - reference_K) / denominator
            columns[f"C_{input_}"][:, s] = (path_up.C - reference_C) / denominator
```

`up[s] += eps` の1行が「時点 $s$ だけを $\varepsilon$ 動かす」で、
`columns[...][:, s] = ...` が「その結果を第 $s$ 列に入れる」。
二重ループの外側が入力（$r$ と $w$）、内側が列。

返り値は4本まとめて名前つきで持つ。

```python
class HouseholdJacobians(BaseModel):
    K_r: JacobianMatrix = Field(description="dK_t / dr_s")
    K_w: JacobianMatrix = Field(description="dK_t / dw_s")
    C_r: JacobianMatrix = Field(description="dC_t / dr_s")
    C_w: JacobianMatrix = Field(description="dC_t / dw_s")

    eps: float | None = Field(default=None, description="使った差分幅")
    scheme: DifferenceScheme | None = Field(default=None, description="使った差分の取り方")
```

`J["K,r"]` のような文字列キーではなく `J.K_r` で取れる。
`eps` と `scheme` を一緒に持たせてあるので、後から「この行列はどの設定で作ったのか」が分かる。

実装同士の突き合わせ用のメソッドもある。

```python
    @typed
    def max_difference(self, other: "HouseholdJacobians") -> float:
        """4本すべてを比べたときの最大の絶対差。実装同士の突き合わせに使う。"""
```

---

## 9. フェイクニュース法 — `jacobian_fake_news`

第8章の内容で、宿題には要らないが**直接法の答え合わせに使う**。
高価な計算が $O(T^2)$ から $O(T)$ に落ちる。

3つの部品からなる。

**① `curly_Y[s]`, `curly_D[s]`** — 「$s$ 期先のニュース」に対する時点0の反応。
$s$ 期先のショックへの時点0の政策は「ショック1回 + 定常ステップ $s$ 回」で得られる。

```python
        for s in range(T):
            # s 期先のニュースに対する時点0 の政策 = ショック1回 + 定常ステップ s 回
            if s == 0:
                r_u, w_u = shocked(+1.0)
                up = backward_egm(
                    model.Pi @ ss.Va, model.a_grid, model.income(w_u), r_u, c.beta, c.eis
                )
                ...
            else:
                up = steady_step(Va_up)
```

**② `expectation_vectors`** — 分布の痕跡を将来の集計量に変換するベクトル。

```python
    def expectation_vectors(outcome: PolicyFunction) -> Float[np.ndarray, "T_minus_1 n_e n_a"]:
        """curly_E[t]: 分布の痕跡を将来の集計量に変換するベクトル（ノート 8.5）。"""
        rows = np.arange(model.n_e)[:, None]
        E = np.empty((max(T - 1, 1), model.n_e, model.n_a))
        E[0] = outcome
        for t in range(1, T - 1):
            tmp = model.Pi @ E[t - 1]
            E[t] = (
                lottery_ss.weight * tmp[rows, lottery_ss.index]
                + (1.0 - lottery_ss.weight) * tmp[rows, lottery_ss.index + 1]
            )
        return E
```

**③ 組み立て** — フェイクニュース行列 $F$ を作り、対角方向に累積して $J$ にする。

```python
            F = np.empty((T, T))
            F[0, :] = ing.curly_Y[output]  # ノート 8.6
            for t in range(1, T):
                F[t, :] = np.tensordot(E[output][t - 1], ing.curly_D, axes=([0, 1], [1, 2]))
            J = F.copy()
            for t in range(1, T):  # ノート 8.7: J[t, s] = J[t-1, s-1] + F[t, s]
                J[t, 1:] += J[t - 1, :-1]
```

最後の2行の `J[t, 1:] += J[t - 1, :-1]` が「対角方向に累積する」の実装。

> **この方法も有限差分である**という点は押さえておきたい。省いているのは
> 「列ごとに同じ後ろ向き・前向き計算を繰り返す重複」であって、有限差分そのものではない。
> ただし微分する対象が「経路全体の写像」ではなく「1期分の後ろ向きステップ」なので、
> その誤差が対角方向の累積を通じて $J$ 全体に伝わる。だから既定を中央差分にしてある。

---

## 10. ライブラリ版 — `ks/ssj.py`

sequence-jacobian で同じものを書くとこうなる。

```python
@het(exogenous="Pi", policy="a", backward="Va", backward_init=hh_init)
def hh(Va_p, a_grid, y, r, beta, eis):
    """内生グリッド法による1期分の後ろ向きステップ。Va_p = E_t[Va_{t+1}]。"""
    uc_nextgrid = beta * Va_p
    c_nextgrid = uc_nextgrid ** (-eis)
    coh = (1 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    a = sj.interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    sj.misc.setmin(a, a_grid[0])
    c = coh - a
    Va = (1 + r) * c ** (-1 / eis)
    return Va, a, c
```

節4の `backward_egm` と**ほぼ一行ずつ対応している**。違いは2つだけ。

1. 引数が `Va_p`（すでに期待を取ったもの）になっている。`@het(exogenous="Pi", ...)` と
   宣言すると、HetBlock が `backward_fun` を呼ぶ前に $\Pi \cdot V_a$ を計算してくれる。
2. 定常状態・移行経路・ヤコビアンのループはライブラリが持っている。

つまり自作版で書いた節5〜9は、ライブラリでは全部この1関数の外側に隠れている。
**逆に言えば、自作版を通したことで「隠れている部分に何があるか」が分かる。**

```python
def income(w, lbar, unemployment_insurance):
    y = np.array([unemployment_insurance, w * lbar])
    return y

hh_extended = hh.add_hetinputs([income])
```

> **落とし穴**: sequence-jacobian は `return` 文に書いた**変数名**を出力名として読み取る。
> 式をそのまま `return` してはいけない（必ず `return Va` のように名前付きの変数を返す）。

自作の型に詰め替えるヘルパも用意してある。同じ定常状態の上で両方の直接法を走らせると
差はゼロになる。

```python
@typed
def steady_state_to_ks(ss_library) -> SteadyState:
    """ライブラリの SteadyStateDict を自作の `SteadyState` に詰め替える。"""
```

許容誤差は締めてある。

```python
STRICT_TOLERANCES: dict[str, Any] = dict(
    backward_tol=1e-12,
    backward_maxit=100_000,
    forward_tol=1e-13,
    forward_maxit=3_000_000,
)
```

節6.1 で見たとおり、既定の `forward_tol = 1e-10` のままだと定常状態が 1e-4 ずれるため。

---

## 付録: 検証の全体像

このパッケージは「同じものを4通りで計算して一致を見る」構造になっている。

| 実装 | 微分の対象 | 場所 |
|---|---|---|
| 自作 直接法 | 経路全体の写像 $H$ | `ks.jacobian_direct` |
| 自作 fake news | 1期分の後ろ向きステップ | `ks.jacobian_fake_news` |
| ライブラリ 直接法 | 経路全体の写像 | `ks.ssj.direct_jacobians_via_library` |
| ライブラリ fake news | 1期分の後ろ向きステップ | `ks.ssj.library_jacobians` |

$T = 5$ での結果:

```
自作 直接法(中央)    vs ライブラリ fake news(中央) : 1.7e-07
自作 fake news(中央) vs ライブラリ fake news(中央) : 1.6e-07
自作 直接法(片側)    vs ライブラリ 直接法(片側)     : 0.0      ← 同一定常状態なら完全一致
同上、基準走行を差し引かない場合                    : 3.5e-06   ← ノート 7.7 の効果
ライブラリ fake news 片側 vs 中央                  : 2.6e-04   ← 差分幅の影響
```

さらに、予算制約から来る恒等式でも検算できる。

$$\frac{\partial C_0}{\partial r_0} + \frac{\partial K_0}{\partial r_0} = K_{ss},
\qquad
\frac{\partial C_0}{\partial w_0} + \frac{\partial K_0}{\partial w_0} = N$$

どちらも $s > 0$ ではゼロ。実測で 1e-11 のオーダーまで合う。

---

## どこから読むか

- **宿題だけ知りたい**: `01_homework_direct_method.ipynb` → このファイルの節7・8
- **実装を追いたい**: 節0（語彙）→ 節4（EGM）→ 節7（$H$）→ 節8（直接法）
- **キャリブレーションが気になる**: 節2 → `02_experiments.ipynb` 節1
- **ライブラリとの対応**: 節10 → `02_experiments.ipynb` 節5
