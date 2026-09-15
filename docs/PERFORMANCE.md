# 速度の仕組み

このリポジトリの計算は「小さい配列（2 × 200）に対する演算を、何万回も繰り返す」形をしている。
そのため速度を決めるのは numpy の計算量ではなく、**1 回の呼び出しにかかる固定費**と、
**呼び出しの回数**の 2 つ。速くする手段もこの 2 つに対応する。

| 手段 | 効くもの | どこにあるか |
|---|---|---|
| numba で内側ループを 1 本に融合 | 呼び出しの固定費 | `ks/_jit.py`, `ks/household.py` の `_*_kernel` |
| 実行時型検査のオン/オフ | 呼び出しの固定費 | `KS_TYPECHECK=0` |
| 定常分布を線形方程式で解く | 呼び出しの回数（と精度） | `stationary_from_policy` |
| 二分法の温かい再開 | 呼び出しの回数 | `solve_general_equilibrium` |
| くじ表現による疎な前向き | 1 回の計算量 | `asset_lottery`, `forward_step` |
| フェイクニュース法 O(T) | 呼び出しの回数 | `jacobian_fake_news` |

---

## 0. 時間がどこに消えているか

定常状態を 1 回解くと、後ろ向きステップ `backward_egm` が数千〜数万回呼ばれる。
1 回の `backward_egm` は 10 個ほどの numpy 演算からなり、配列が 400 要素しかないので
**各演算は計算 0.1 µs、呼び出しの固定費 1 µs** という比率になる。
つまり numpy の内側（C）ではなく、Python から C を呼ぶ境界で時間が消えている。

プロファイルすると上位はこう並ぶ（型検査オン）:

```
19%  beartype の isinstance      ← 実行時型検査
 7%  interpolate_y
 6%  jaxtyping の軸検査
 5%  backward_egm 本体
 3%  np.take_along_axis
```

CPython 3.14 の JIT（`PYTHON_JIT=1`、`.env` で有効化）が効かなかったのはこのため。
JIT が速くするのは Python バイトコードで、ここの時間は C 側の固定費と型検査に消えている。
実測でも差なし（4.2 s vs 4.1 s）。

---

## 1. numba kernel — 呼び出しの固定費を消す

numpy の演算 10 個を **1 本のコンパイル済みループ**に融合すれば、固定費は 1 回分になる。
これをやっているのが `ks/household.py` の 2 つの kernel。

### `_interpolate_y_kernel` — 補間

EGM の心臓部で、後ろ向きステップの時間の 7 割を占めていた。
numpy 版は `searchsorted` → `clip` → `take_along_axis` × 2 → 算術、の 5 呼び出し。

kernel は「直前の区間から走査する」方式:

```python
i = 0
for j in range(n_q):
    v = xq[row, j]
    while i < n_a - 2 and x[row, i + 1] < v:   # 上へ
        i += 1
    while i > 0 and x[row, i] >= v:            # 下へ
        i -= 1
```

- EGM では `xq`（手持ち資産）が昇順なので、行全体で O(n)。
- 昇順でなくても正しい（`searchsorted` と結果が bit 単位で一致。重複点も可）。
- numba の中で `np.searchsorted` を呼ぶ案は 11 µs、要素ごとの二分探索は 3.5 µs、この走査が 1.0 µs。
  「numba の中でも numpy の関数呼び出しは高い」ことの例。

### `_forward_endogenous_kernel` — 分布の前向き

numpy 版の `np.add.at` は汎用で遅い（6 µs）。明示的な 2 重ループにして 0.8 µs。
下側の格子点を全部足してから上側を足す **2 パス**にしてあるのは、`np.add.at` 版と
加算の順序を揃えて結果を bit 単位で一致させるため。

### 自作の `njit`（`ks/_jit.py`）

`numba.njit(cache=True)` を型付きで包んだもの（[TYPING_AND_LINT.md](TYPING_AND_LINT.md) §4）。

- `cache=True`: コンパイル結果を `__pycache__/` に保存。初回だけ 1〜2 秒、以後はゼロ。
- kernel は private 関数で、外側の公開関数（`interpolate_y`, `forward_endogenous`）に
  `@typed` を残す。呼び出し側から見た仕様は numpy 版と同じ。

### 結果（型検査オフ、1 回あたり）

| | numpy | numba |
|---|---|---|
| `interpolate_y` | 15.7 µs | 1.0 µs |
| `forward_endogenous` | 5.8 µs | 0.8 µs |
| `backward_egm` 全体 | 21.9 µs | 6.4 µs |
| 定常状態（一般均衡） | 0.75 s | 0.25 s |

`backward_egm` の残り 6 µs は、まだ numpy で書いてある算術演算 5〜6 個の固定費。
これも kernel に入れれば 2 µs 程度になるが、`backward_egm` は
[WALKTHROUGH](WALKTHROUGH.md) で 1 行ずつ解説する教材なので、読める形を優先して残している。

---

## 2. 実行時型検査 — 最大の固定費

`@typed`（jaxtyping + beartype）は 1 回 10 µs 前後。`backward_egm` は内側で `interpolate_y` を
呼ぶので、1 ステップにつき検査が 2 回入る。numba 化した後は**時間の 7 割以上がこの検査**:

| | 型検査あり（既定） | `KS_TYPECHECK=0` |
|---|---|---|
| 定常状態 | 1.07 s | 0.25 s |
| `backward_egm` 1 回 | 31 µs | 6 µs |

既定でオンにしてあるのは、形の取り違えがこの種のモデルで一番多いバグだから。
重い実験（`02_experiments` の T = 80 など）を回すときだけ `KS_TYPECHECK=0` を付ける。
検査は関数の**入口と出口**にしかないので、切っても計算結果は変わらない。

---

## 3. 定常分布を線形方程式で解く — 回数と精度を同時に

定常分布 $D = \Lambda' D$ は、前向き反復 $D \leftarrow \Lambda' D$ で解くのが普通。
だが KS のキャリブレーションは $\beta(1+r) = 0.9999$ と 1 に極端に近く、$\Lambda$ の第 2 固有値も
1 に近いので、反復が**見かけ上収束しても真の値から遠い**（実測: $10^{-10}$ の収束判定に 33,000 回、
そのときの $D$ はまだ $1.2 \times 10^{-7}$ ずれ、$K$ が $10^{-4}$ ずれる）。

`stationary_from_policy` は代わりに `transition_matrix` で $\Lambda$ を $(n_e n_a)^2 = 400^2$ の
密行列として作り、$(\Lambda' - I) D = 0$, $\mathbf{1}'D = 1$ を `np.linalg.solve` で一発で解く。
反復 33,000 回が線形方程式 1 回になり、残差は $10^{-16}$。
密行列を作るのはここだけで、1 回の `solve_household` につき 1 回しか呼ばれない。

---

## 4. 二分法の温かい再開 — 回数を減らす

`solve_general_equilibrium` は $K$ を二分法で振るたびに家計問題を解き直す。
価格が少ししか変わらないので、前回の $V_a$ から後ろ向き反復を始めれば、
初期値から始める約 2,000 回に対して **20 回程度**で収束する:

```python
warm: MarginalValue | None = None

def evaluate(K: float) -> SteadyState:
    nonlocal warm
    ss = solve_household(model, model.prices(K), Va_init=warm)
    warm = ss.Va
    return ss
```

細かい点が 2 つ:
- 収束判定は 10 反復に 1 回（`iterations % 10 == 0`）。判定自体が `max|Δa|` の配列演算なので、毎回やると無視できない。
- 下側の境界は `K_impatience_bound * (1 + 1e-8)`。それより下では $r \ge 1/\beta - 1$ で定常分布が存在せず、
  線形方程式が意味のない解（負の $K$）を返して二分法が壊れる。

---

## 5. くじ表現 — 前向きステップを疎にする

政策 $a_t$ は格子点の間に落ちるので、分布を進めるには「下の格子点に重み $\pi$、上に $1 - \pi$」と
振り分ける（`asset_lottery` → `Lottery(index, weight)`）。
これを使えば前向きステップは $O(n_e n_a)$ の散布で済み、$(n_e n_a)^2$ の $\Lambda$ を作る必要がない
（`forward_step` = 資産の散布 + $\Pi'$ の 2 × 2 積）。

フェイクニュース法の `expectation_vectors` も同じ `Lottery` を使って
$\mathcal{E}_t = \Lambda \mathcal{E}_{t-1}$ を密行列なしで計算している。

---

## 6. フェイクニュース法 — O(T²) を O(T) に

直接法は列ごとに移行経路を 1 本流すので、後ろ向き・前向きの回数は $2T \times T = O(T^2)$。
フェイクニュース法（第 8 章）は「1 期分の後ろ向きステップの微分」を $T$ 回と
「分布の痕跡の前向き」を $T$ 回で済ませる $O(T)$。$T = 5$ ではどちらも数 ms で差がないが、
$T = 80$ では直接法 O(T²) の差がはっきり出る（`02_experiments` の計算量の節）。
このリポジトリでは宿題が直接法なので、フェイクニュース法は答え合わせ用に置いてある。

---

## 7. 小さな入れ物は pydantic にしない

`BackwardStep` と `StationaryDistribution` は `NamedTuple`、`Lottery` は
`dataclass(frozen=True, slots=True)`。内側ループで何千回も作られるので、pydantic の
検証コスト（構築 1 回あたり約 8 µs。`NamedTuple` は 0.2 µs）を避ける。形の検査はそれらを作る関数側の `@typed` が担う。

---

## 8. 計測の仕方

```python
import time, cProfile, pstats
t0 = time.perf_counter(); ss = ks.solve_general_equilibrium(model); print(time.perf_counter() - t0)

pr = cProfile.Profile(); pr.enable()
ks.solve_household(model, ss.prices)
pr.disable(); pstats.Stats(pr).sort_stats("tottime").print_stats(8)
```

- 型検査の影響を見るときは `KS_TYPECHECK=0` と比べる。
- numba の初回コンパイルを含めないよう、計測前に 1 回だけ捨て呼び出しをする。
- kernel を変えたら、結果が変更前と **bit 単位で一致するか**（`np.array_equal`）を先に確かめる。
  速度は後。

## やっていないこと

- `backward_egm` 全体の numba 化（教材としての読みやすさを優先、§1）。
- 直接法の列ごとの並列化。$2T$ 本の移行経路は互いに独立なので `joblib` や
  `numba.prange` で並列にできるが、$T = 5$ では意味がない。
- 型検査を「入口だけ」にする最適化。`interpolate_y` は `backward_egm` からしか呼ばれないので
  内側の `@typed` を外せば検査が半分になるが、公開関数の仕様を一貫させる方を取っている。
