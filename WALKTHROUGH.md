# `ks` パッケージの歩き方

`ks.jacobian_direct(model, ss, T=5)` を呼んだとき、中で何が起きているのかを **実行される順番どおりに**、言葉と実際のコードを交互に並べて追う。

本文で引用しているコードはすべて実物（コメントとドックストリングは一部省略）。 `Q:` は読んでいて出た疑問、`**A:**` はその回答。回答中のコードには説明用の擬似例も含む。

---

## 全体像

やりたいことは1本の式に集約される。家計ブロックは価格の経路を受け取って集計量の経路を返す 写像であり、それを定常状態のまわりで一次微分したものがヤコビアン。

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

| モジュール          | 役割                                         |
| ------------------- | -------------------------------------------- |
| `ks/types.py`       | 語彙。配列の形と意味を型で表す               |
| `ks/calibration.py` | パラメータ、集計ショックの消去、価格と所得   |
| `ks/household.py`   | EGM・前向き・定常状態・移行経路              |
| `ks/jacobian.py`    | 直接法（第7章）とフェイクニュース法（第8章） |
| `ks/ssj.py`         | sequence-jacobian ライブラリ版               |
| `ks/_jit.py`        | numba の `njit(cache=True)` に型を付けたもの |

---

## 0. 語彙をそろえる — `ks/types.py`

異質的主体モデルで一番よく起きるバグは**配列の形の取り違え**である。 `np.ndarray` としか書いていないと、それが政策関数なのか分布なのかヤコビアンなのか、 軸がどちらが雇用状態でどちらが資産なのかが読み取れない。

そこで軸の名前まで型に書く。

```python
n_e : 個別の雇用状態の数。KS では 2（失業 / 就業）
n_a : 資産グリッドの点数
T   : 系列空間の切断期間（宿題では 5）
n_A : 集計状態の数。KS では 2（不況 / 好況）
      定常状態を作る過程でのみ登場し、SSJ の家計ブロックには残らない
```

`FloatArray` は `np.ndarray[tuple[int, ...], np.dtype[np.float64]]` の別名。裸の `np.ndarray` は静的には `ndarray[Any, Any]` に展開されて mypy の厳格モードで Any 扱いになるので、dtype まで固定してある。

```python
type PolicyFunction = Float[FloatArray, "n_e n_a"]
"""個別状態 (e_t, a_{t-1}) 上の政策関数。期末資産 a_t または消費 c_t。"""

type MarginalValue = Float[FloatArray, "n_e n_a"]
"""価値関数の資産微分 V_a。EGM の後ろ向き変数。形は政策関数と同じ。"""

type Distribution = Float[FloatArray, "n_e n_a"]
"""時点 t 冒頭の個別状態 (e_t, a_{t-1}) 上の分布 D_t。総和は 1。"""

type JacobianMatrix = Float[FloatArray, "T T"]
"""(t, s) 要素が dY_t / dX_s。行 t = 応答の時点、列 s = ショックの時点。"""
```

Q: "n_e n_a" とはどういう意味？こういう記法なの？記法なら何が記法でどこが自分で決められる場所？

**A:** jaxtyping の記法。`Float[np.ndarray, "軸1 軸2 ..."]` で「dtype が float、コンテナが `np.ndarray`、軸が2本」を表す。文字列はスペース区切りの軸リスト。

**ライブラリが決めている部分**

|                                  | 例                                                                                                 |
| -------------------------------- | -------------------------------------------------------------------------------------------------- |
| dtype の種類                     | `Float` / `Int` / `Bool` / `Shaped`                                                                |
| コンテナの型                     | `np.ndarray`（`jax.Array` や `torch.Tensor` も書ける）                                             |
| 文字列がスペース区切りであること | `"n_e n_a"` は2軸、`"n_a"` は1軸                                                                   |
| 特殊記法                         | `"*batch"` = 任意個の軸、`"n_a=200"` = 長さを固定、`"#n_a"` = ブロードキャスト可、`"_"` = 名前なし |

**自分で決める部分**

- **軸の名前そのもの**。`n_e` `n_a` `T` は私が決めた。`e a` でも `foo bar` でもよい
- その名前が何を指すかの取り決め（このコードでは `ks/types.py` の冒頭に書いてある）

一番大事なのは、**同じ名前は「同じ長さ」を意味する**こと。1回の関数呼び出しの中で `n_e` が2箇所に現れたら、その2つは同じでなければならない。

```python
def f(x: Float[np.ndarray, "n_e n_a"], y: Float[np.ndarray, "n_e"]): ...

f(np.zeros((2, 5)), np.ones(3))   # -> TypeCheckError（n_e が 2 と 3 で食い違う）
f(np.zeros((2, 5), dtype=int), np.ones(2))   # -> TypeCheckError（Float でない）
```

`PolicyFunction` と `MarginalValue` と `Distribution` は**形は同じ** `"n_e n_a"` だが名前が違う。 これは意図的で、シグネチャを見たときに「これは分布であって政策ではない」と分かるようにしている。

添字も名前で書けるようにする。

```python
class Employment(IntEnum):
    UNEMPLOYED = 0
    EMPLOYED = 1
```

これで `ss.a[0]` ではなく `ss.a[Employment.UNEMPLOYED]` と書ける。

型注釈は飾りではなく、実行時にも検査される。

Q: pydanticを使っていればこういうこと（typed関数の自作）ってしなくて良いのでは？

**A:** pydantic の検査が効くのは **BaseModel のフィールド**だけで、ふつうの関数の引数は見ない。

pydantic にも関数用の `@validate_call` があるので試してみたが、決定的な差があった。

| 検査したいこと                                  | `@validate_call` | `typed`（jaxtyped + beartype） |
| ----------------------------------------------- | ---------------- | ------------------------------ |
| 引数1本の次元数（1次元を渡すべきところに2次元） | 検出する         | 検出する                       |
| **引数をまたいだ軸長の一致**（`n_e` が 2 と 3） | **素通り**       | 検出する                       |

2行目が今回いちばん欲しかったもの。「資産グリッドは 200 点なのに政策は 150 点」のような取り違えは引数をまたぐので、`@validate_call` では捕まらない。

もう一つ実務的な理由として、`@validate_call` は引数を pydantic のバリデータに通すので呼び出しコストが高い。`backward_egm` は内側ループで何万回も呼ばれるため避けたかった。

そこで役割分担にしてある: **パラメータ（スカラーの妥当性）は pydantic、配列の形は jaxtyping**。

```python
_RUNTIME_TYPECHECK = os.environ.get("KS_TYPECHECK", "1").lower() not in ("0", "false", "no")

def typed[F: Callable[..., object]](fn: F) -> F:
    """引数と返り値の dtype・形状を実行時に検査するデコレータ。"""
    if not _RUNTIME_TYPECHECK:
        return fn
    return jaxtyped(typechecker=beartype)(fn)
```

同じ軸名（例えば `"n_a"`）が複数の引数に現れる場合、**その長さが一致していることまで**見る。 資産グリッドが 200 点なのに政策が 150 点、という取り違えはその場で落ちる。 重い計算で外したいときは `KS_TYPECHECK=0`（約2倍速くなる）。

---

## 1. パラメータを持つ — `KSCalibration`

既存コード `8_2_krusell_and_smith.py` の `Setting`（`replicate_model = 0`）を、 検証つきの pydantic モデルに置き換えたもの。**スカラーだけ**を持ち、配列は一切持たない。

Q: baseModel って何？

**A:** pydantic の基底クラス。これを継承すると、クラス本体に書いた型注釈から次が自動でついてくる。

1. `__init__` の自動生成（`KSCalibration(alpha=0.4)` が書けるようになる）
2. 渡された値の型検査。違えば `ValidationError`
3. 必要なら型変換（`"0.36"` → `0.36`）
4. `.model_dump()` で辞書化、`.model_copy(update=...)` で一部だけ差し替えたコピー

**dataclass に「実行時の検査」が付いたもの**、と思えばだいたい合っている。

Q: Configdict()って何？使うと何が良い？

**A:** そのモデルの**ふるまいの設定**をまとめたもの。`model_config = ConfigDict(...)` と書くとクラス全体に効く。このリポジトリで使っているのは2つだけ。

| 設定                           | 効果                                                                         |
| ------------------------------ | ---------------------------------------------------------------------------- |
| `frozen=True`                  | 生成後に属性を書き換えられなくする。`ss.K = 999` が `ValidationError` になる |
| `arbitrary_types_allowed=True` | pydantic が標準で知らない型（`np.ndarray`）をフィールドに持てるようにする    |

`KSCalibration` はスカラーだけなので `frozen=True` のみ。`KSModel` / `SteadyState` / `HouseholdJacobians` は配列を持つので両方いる。

`frozen=True` が効くのは、定常状態を解いたあとにパラメータが書き換わって「解いたときの前提と今の値が違う」という事故を防ぐため。

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

`gt=0.0, lt=1.0` が効くので、`KSCalibration(beta=1.5)` は `ValidationError` になる。 タイプミスで非現実的なパラメータのまま数時間走る、という事故が防げる。

Q: クラスが引数をとっていないように見えるけど、CofifDictと関係ある？

**A:** `ConfigDict` とは関係ない。別の話が2つ混ざっている。

- `class KSCalibration(BaseModel):` の丸括弧は継承元を書く場所であって、引数ではない。「`BaseModel` を継承する」という意味
- `__init__` が見えないのは pydantic が自動生成しているから。`alpha: float = Field(0.36, ...)` と書いた時点で `KSCalibration(alpha=0.4)` が使えるようになる

`model_config = ConfigDict(frozen=True)` も引数ではなく**クラス変数**。pydantic がこの名前の変数を特別扱いして設定として読む、という約束になっている。

Q: jaxtype って標準ライブラリのtyping とどう違うの？割とデファクトスタンダード？

**A:** 標準の `typing` には**配列の形を書く語彙がない**。書けるのはせいぜいこれだけ。

```python
def f(x: np.ndarray) -> np.ndarray:      # 形の情報はゼロ
```

numpy 公式の `numpy.typing.NDArray[np.float64]` でも dtype までで、次元数も軸長も表現できない。jaxtyping はその隙間を埋めるライブラリ。名前に jax とあるが **numpy 単体で問題なく使える**（このリポジトリは jax を入れていない）。

デファクトかというと、正直に言えば\*\*「その分野では」\*\*。JAX / PyTorch 系の研究コードでは広く使われているが、数値経済学のコードでは注釈なしか docstring で済ませる流儀がまだ多数派。ほかの選択肢は

- `numpy.typing.NDArray` … 標準寄りだが dtype まで
- `nptyping` … 似た目的だがメンテが停滞気味
- docstring に `(n_e, n_a)` と書くだけ … 検査はされない

今回は「形の取り違えが最大のバグ源」というこのモデル特有の事情があるので採用した。

Q: jaxtype と pydantic は何が違う？pydantic にjaxtypeを混ぜることの弊害はある？

**A:** 守備範囲が違う。

|                                  | jaxtyping                                | pydantic                         |
| -------------------------------- | ---------------------------------------- | -------------------------------- |
| 主な対象                         | 配列（dtype と形）                       | スカラー・辞書・ネストしたモデル |
| 検査のタイミング                 | 関数の呼び出し時（`typed` を付けたもの） | モデルの生成時                   |
| 値の範囲（$0 < \beta < 1$ など） | 見ない                                   | `Field(gt=0, lt=1)` で見る       |
| 引数をまたいだ軸長の一致         | 見る                                     | 見ない                           |

**混ぜること自体は公式にサポートされている。** jaxtyping の型は `__get_pydantic_core_schema__` を実装しているので、pydantic のフィールドに書けば生成時に形を検査してくれる。実測で `M(Pi=np.eye(2)[0])`（1次元を渡す）は `ValidationError` になる。

弊害として気にすべきだったのは配列がコピーされないかだが、実測すると **コピーされない**（`m.a is arr` が `True`）。メモリも増えないし、別名参照の落とし穴もない。

残る注意点は2つ。

- 検査は**生成時の1回だけ**。あとから `m.a[0] = ...` と中身を書き換えても再検査はされない（`frozen=True` は再代入を防ぐが、配列の中身の書き換えは防げない）
- pydantic のフィールドでは**モデル内の別フィールドとの軸長の一致は見ない**。`KSModel` の `Pi` と `a_grid` の整合は検査されていない

Q: gtとltとは何？変数の取りうる区間？

**A:** そのとおり、取りうる区間の指定。pydantic の `Field` に渡す制約。

|      | 意味                                                       |
| ---- | ---------------------------------------------------------- |
| `gt` | greater than（より大きい）。`gt=0.0` なら 0 そのものは不可 |
| `ge` | greater than or equal                                      |
| `lt` | less than                                                  |
| `le` | less than or equal                                         |

`Field(0.99, gt=0.0, lt=1.0)` は「既定値 0.99、取りうる範囲は開区間 $(0, 1)$」。範囲外なら生成時に `ValidationError`。

Q: delta はなぜ四半期？

**A:** モデル全体が四半期の頻度で組まれているから。KS(1998) がそう組んでいる。

| パラメータ                 | 四半期の値              | 年率に直すと                        |
| -------------------------- | ----------------------- | ----------------------------------- |
| `delta = 0.025`            | 資本減耗 2.5%/期        | $1-(1-0.025)^4 = 9.6\%$             |
| `beta = 0.99`              | 割引因子                | 時間選好率 $(1/0.99)^4 - 1 = 4.1\%$ |
| `dur_good = 8`             | 好況が平均8期続く       | 2年                                 |
| `dur_unemployed_bad = 2.5` | 不況時の失業が平均2.5期 | 約7.5ヶ月                           |

ヤコビアンの $t$ や $s$ も四半期。$T = 5$ は5四半期＝1年3ヶ月ぶんの経路。

`frozen=True` なので後から書き換えられない。定常状態を解いたあとにパラメータが 書き換わって整合性が壊れる、ということが起きない。

派生的な量はプロパティで持つ。

```python
    @property
    def r_impatience_bound(self) -> float:
        """r がこれ以上だと資産需要が発散する上限 1/beta - 1。"""
        return 1.0 / self.beta - 1.0

```

Q: なぜその値以上だと発散するの？

**A:** オイラー方程式から出る。

$$
u'(c_t) = \beta(1+r) E_t[u'(c_{t+1})]
$$

$\beta(1+r) = 1$ ちょうどのとき、これは $u'(c_t) = E_t[u'(c_{t+1})]$、つまり**限界効用がマルチンゲール**になる。限界効用は正なので、非負マルチンゲールは収束する。ところが個別の所得リスクが毎期入り続けるので、収束先は $u'(c) = 0$、すなわち $c \to \infty$ しかない。消費が無限に増えるには資産が無限に増えるしかないので、**定常分布が存在しない**。$\beta(1+r) > 1$ ならもっと露骨で、同じ結論。

Q: $\beta(1+r)>1$ のとき無限期間先の消費が少しでも0より大きかったらc_t が無限大に発散するという理解で正しい？

$$
$$

数値でも見える。境界 $K = 11.5564$ に近づくほど資産需要が爆発する。

```
K= 11.5565  r=0.0101009  資産需要= 358.7   ← 境界のすぐ上
K= 11.5700  r=0.0100747  資産需要=  32.1
K= 11.5800  r=0.0100553  資産需要=  20.4
K= 11.6000  r=0.0100166  資産需要=  13.1
K= 11.6094  r=0.0099985  資産需要=  11.6   ← 均衡（需要 = 供給）
K= 11.6500  r=0.0099203  資産需要=   8.5
```

---

## 2. 集計ショックを消す — `_build_markov_chains`

ここが KS を SSJ に持ち込むときの一番の関門。

**問題**: SSJ の定常状態には集計ショック $A$ が存在しない。だが KS の雇用遷移は $A$ に依存する （好況 $u = 0.04$、不況 $u = 0.10$）。$(A, l)$ の4状態チェーンを $l$ だけの2状態に潰す必要がある。

Q: この問題って直接法でもSSJライブラリを使っても同じように存在するってことであってる？

**A:** 合っている。これは**アルゴリズムの問題ではなく、モデルの立て方の問題**。

直接法もフェイクニュース法もライブラリの `.jacobian()` も、全部「集計ショックのない定常状態のまわりで線形化する」という同じ前提の上に乗っている。定常状態に $A$ がない以上、どの解法を使っても $(A, l)$ の4状態を潰す作業は避けられない。

避けたければ SSJ をやめて、元の KS(1998) のように $A$ を状態変数に持つ解法（$K$ グリッド + 予測式）に戻るしかない。

まず既存コードの `markov_KS` をそのまま移植して、4状態の同時チェーンを作る。 `dur` パラメータから確率を作るところは「2状態チェーンで留まる確率 $q$ の期待継続期間は $1/(1-q)$」という関係の逆算である。

```python
    # --- 集計状態の遷移。q = (D - 1) / D は「平均継続期間 D」と等価 ---
    p_gg = (c.dur_good - 1.0) / c.dur_good
    p_bb = (c.dur_bad - 1.0) / c.dur_bad
    aggregate = np.array([[p_bb, 1.0 - p_bb], [1.0 - p_gg, p_gg]])  # 行 = [不況, 好況]
```

`dur_good = dur_bad = 8` なので $q = 7/8 = 0.875$、行列は対称になる。 これが効いて $A$ の定常分布が $[0.5, 0.5]$ ちょうどになり、$u = 0.07$、$Z = 1.00$ が出る。

Q: 失業保険の額は外生的に与えられるのでは？

**A:** そのとおりで、失業保険 0.07 は外生的に置いたパラメータのまま、何も変わっていない。

**混乱の原因はおそらく「0.07 が2つある」こと。**

|                                          | 何を表すか                     | どこから来るか           |
| ---------------------------------------- | ------------------------------ | ------------------------ |
| 失業給付 `unemployment_insurance = 0.07` | 失業者がもらう**金額**         | 手で置いたパラメータ     |
| 定常失業率 `u = 0.07`                    | 人口の何**割**が失業しているか | `dur` から計算で出てくる |

**この2つはまったく無関係な量で、数値が一致しているのは偶然。** 紛らわしい書き方をしていた。

直前の本文で「$u = 0.07$ が出る」と書いたのは後者（失業率 7%）のこと。`dur_good = dur_bad = 8` の対称性から $A$ の定常分布が $[0.5, 0.5]$ になり、$u = 0.5 \times 0.10 + 0.5 \times 0.04 = 0.07$ と決まる、という話をしていた。

Q: Zって何だっけ？

**A:** 全要素生産性 (TFP)。生産関数 $Y = Z K^{\alpha} N^{1-\alpha}$ の $Z$。

既存コードでは集計ショック $A \in \{0.99, 1.01\}$ がこれにあたる（`Agrid`）。SSJ の定常状態には集計ショックがないので、その定常分布での平均を取って

$$Z = 0.5 \times 0.99 + 0.5 \times 1.01 = 1.00$$

を使っている。`dur` が対称だから 1.00 ちょうどになるだけで、非対称なら 1 からずれる。

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

Q: row, col って何？なぜindexの足し算掛け算をしている？

**A:** **2次元の状態 $(A, l)$ を1次元の通し番号に潰している**。$A$ が2通り、$l$ が2通りなので状態は4つあり、0〜3 の番号を振る。

| 番号 | 計算      | $(A, l)$     |
| ---- | --------- | ------------ |
| 0    | `2*0 + 0` | (不況, 失業) |
| 1    | `2*0 + 1` | (不況, 就業) |
| 2    | `2*1 + 0` | (好況, 失業) |
| 3    | `2*1 + 1` | (好況, 就業) |

`row = 2 * i_a + i_l` の `2` は「$l$ が2通りある」の 2。**$A$ を1つ進めると番号が2つ飛ぶ**、ということ。時刻を「60 × 時 + 分」で1つの数にするのと同じ発想。

`row` が今の状態の番号、`col` が来期の状態の番号。だから `joint[row, col]` は「今 `row` にいる人が来期 `col` に行く確率」になる。

Q: なぜfor文が4つも必要？

**A:** 埋めたいのが 4×4 の表で、**行の指定に添字2つ、列の指定に添字2つ**いるから。

```
for i_a      ← 今の集計状態      ┐ この2つで row（4通り）が決まる
  for i_l    ← 今の雇用状態      ┘
    for j_a  ← 来期の集計状態    ┐ この2つで col（4通り）が決まる
      for j_l ← 来期の雇用状態   ┘
```

$2 \times 2 \times 2 \times 2 = 16$ 回まわって、16個のマス目を1つずつ埋める。

`np.einsum` などで1行に圧縮することもできるが、$(A, l) \to (A', l')$ という対応がコードから見えなくなるので、ここは素直な4重ループにしてある（16回しか回らないので速度も問題にならない）。

同時遷移が $P(A'\mid A) \cdot P(l' \mid A, A', l)$ と積に分解できるのは KS の作り方による。 $P(A'\mid A)$ が $l$ に依存しないのは個人が無限小だから。

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

Q: marg_l が何かわからない

**A:** **周辺分布 (marginal distribution)** の `l` 版。「$A$ を無視したとき、人口の何割が失業／就業か」。

同時定常分布 `joint_stationary` は4つの数字だった。

```
[0.05,     0.45,     0.02,      0.48]
 不況&失業  不況&就業  好況&失業   好況&就業
```

ここから $A$ を足し上げて消すと

```
marg_l[0] = 0.05 + 0.02 = 0.07   ← 失業（不況の失業者 + 好況の失業者）
marg_l[1] = 0.45 + 0.48 = 0.93   ← 就業
```

コードの `joint_stationary[0] + joint_stationary[2]` がまさにこの足し算（添字 0 と 2 が失業の2通り）。

使い道は**割り算の分母**。$\Pi(l'\mid l) = \text{フロー} \div \text{ストック}$ の、ストックの方。`employment /= marg_l[:, None]` は「失業者の行は 0.07 で割り、就業者の行は 0.93 で割る」。

`[:, None]` は形を `(2,)` から `(2, 1)` にするためのもの。`(2,)` のままだと numpy のブロードキャストで**列ごとに**割ってしまい、意味が変わる。

これは数式で書くと

$$
\Pi(l' \mid l) = \frac{\sum_{A, A'} \pi(A, l), P(A' \mid A), P(l' \mid A, A', l)}{\pi(l)}
$$

で、やっているのは**条件付き確率の定義そのもの**である。分子は「実際に起きている $l \to l'$ のフロー」、分母は「実際に $l$ にいるストック」。$A$ は観測しないので 可能な $(A, A')$ の組を全部足し上げて消している。

最後の `employment /= marg_l[:, None]` の `[:, None]` が重要で、**行ごとに違う数で割っている**。 失業者は不況に偏在している（失業者の 71.4% が不況にいる。不況の頻度は 50% しかないのに）ので、 失業者の行だけ不況側の遷移行列が重くなる。共通の $\pi(A) = [0.5, 0.5]$ で潰すと 定常失業率が 0.0657 になり 0.07 から外れる。

得られた `Pi` は `[[0.507440, 0.492560], [0.037074, 0.962926]]`、定常失業率はちょうど 0.07。 検証は `02_experiments.ipynb` 節1 にある。

> **落ちているもの**: 元の KS 経済では $l$ 単独は厳密にはマルコフではない。 $\Pi$ は1期先の条件付き分布を平均的に再現するが、2期先以降はずれる（最大 6.9e-3）。 「不況では失業リスクが上がる」という連動も消える。

Q: もう少し説明が欲しい. Krusel and Smith では 4x4行列を外生的にあたえて失業保険も0.07と外生的だったけど、直接法では外生的じゃなくなるの？だから情報が落ちるということ？

**A:** いいえ、**外生性は何も変わっていない**。落ちているのは別のもの。

|                  | KS(1998)             | SSJ                                 |
| ---------------- | -------------------- | ----------------------------------- |
| 失業保険 0.07    | 外生                 | **外生のまま。1ミリも変えていない** |
| 遷移行列         | 4×4 を外生的に与える | **2×2 を外生的に与える**            |
| 集計ショック $A$ | 状態変数             | 状態変数ではない                    |

つまり「外生 → 内生」になったのではなく、**外生的に与える行列が 4×4 から 2×2 に差し替わった**だけ。

では何が落ちたのか。**「失業リスクの高さが景気に連動する」という性質**である。

- 4×4 は「今が不況なら失業を続ける確率は 0.60、好況なら 0.33」と書き分けられる
- 2×2 は景気を区別する軸を持たないので、平均した 0.507 しか書けない

これは確かに情報の欠落だが、**避けようがない**。SSJ の定常状態には $A$ が存在しないので、「$A$ に依存する遷移確率」を書く場所がそもそもない。$A$ を消すと決めた時点で決まってしまう。そして直接法のせいでもない（Q「この問題って直接法でも…」と同じ話）。

ただし SSJ の枠組みの中で埋め戻せる。SSJ では集計ショックは「状態変数」ではなく\*\*「入力経路への摂動」\*\*として復活する。$Z_t$ の経路を揺らし、それと連動して $\Pi_t$ も揺らせば、「不況で失業リスクが上がる」効果を $J^{K,\Pi}$ として取り込める。今回の宿題（第7章）でそこまでやっていないだけ。

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
        c = KSCalibration() if calibration is None else calibration
        return cls(
            calibration=c,
            chains=_build_markov_chains(c),
            a_grid=exponential_grid(c.a_min, c.a_max, c.n_a),
        )
```

Q: clsとは？クラスの依存注入？何クラスを想定している？

**A:** 依存注入ではない。`cls` は `@classmethod` の第一引数で、**そのクラス自身**が自動で入る。

```python
@classmethod
def build(cls, calibration=None) -> "KSModel":
    return cls(calibration=..., chains=..., a_grid=...)
```

`KSModel.build()` と呼ぶと `cls` は `KSModel` になるので、`cls(...)` は `KSModel(...)` と同じ。

`KSModel` と直書きせず `cls` にする理由は、将来 `class KSModelVariant(KSModel)` を作ったとき `KSModelVariant.build()` がちゃんと `KSModelVariant` を返すから。今すぐの実益はないが classmethod の慣習。

`self`（インスタンスが入る）との対比で覚えるとよい。`build` は**まだインスタンスがない状態から作る**メソッドなので `self` が使えず、`cls` になる。

Q: calibration or KSCalibration()のorの使い方を初めてみたんだけど、これはbooleanを返すの？

**A:** boolean は返さない。Python の `or` は**オペランドそのもの**を返す。

```python
a or b       # a が真なら a を、偽なら b を返す（True/False ではない）
```

なので `calibration or KSCalibration()` は「渡されていればそれを使い、`None` なら既定値で新しく作る」という意味になる。

**ただしこの書き方には落とし穴がある。** `or` が見るのは「`None` かどうか」ではなく「真か偽か」なので、`0` や `[]` や `""` のような**偽だが有効な値**が渡されると意図せず既定値に化ける。

今回は `calibration` が pydantic モデルか `None` のどちらかで、pydantic モデルは常に真（実測で `bool(KSCalibration())` は `True`）なので実害はなかった。とはいえ意図が明確な方がよいので、**この質問を受けて** `is None` に書き換えた。上に引用したコードが書き換え後のもの。

```python
c = calibration or KSCalibration()                          # 書き換え前
c = KSCalibration() if calibration is None else calibration  # 書き換え後
```

パッケージ内の同種の箇所（`Va_init`、`K_low`、`K_high`）はもともと `is not None` で書いてあり、これで全体が揃った。

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

`income` の失業側が `w` を掛けていないのが `replicate_model = 0` の特徴。 失業給付 0.07 は水準であって賃金比例ではない。

均衡がどこにあり得るかの上限も持っておく。

```python
    @cached_property
    def K_impatience_bound(self) -> float:
        """r(K) = 1/beta - 1 となる K。均衡 K はこれより大きくないと定常分布が存在しない。"""
```

このモデルでは 11.5564。均衡 $K = 11.6094$ はそのすぐ上にあり、 $\beta(1+r) = 0.99990$ と1に極端に近い。この事実がこの先ずっと効いてくる。

Q: 均衡の上限がなぜわかるのか？なぜその式なのか？

**A:** $r(K) = 1/\beta - 1$ を $K$ について解いただけ。$r(K)$ は $K$ の減少関数なので、$r$ の上限がそのまま $K$ の下限に翻訳される。

$$\alpha Z \left(\frac{K}{N}\right)^{\alpha-1} - \delta = \frac{1}{\beta} - 1 \qquad\Longrightarrow\qquad K = N\left\[\frac{\alpha Z}{1/\beta - 1 + \delta}\right\]^{\frac{1}{1-\alpha}}$$

これがコードの式そのもの。

```python
return self.N * ((c.alpha * self.Z) / (c.r_impatience_bound + c.delta)) ** (1.0 / (1.0 - c.alpha))
```

検算すると $r(11.55644489) = 0.0101010101$ で $1/\beta - 1$ にぴったり一致する。

なぜ $r$ に上限があるのかは、少し上の「なぜその値以上だと発散するの？」の答えを参照。

Q: モデルの均衡の値はなぜわかったのか？

**A:** **式では分からない。数値的に探して初めて出る値**。ここが上限との決定的な違い。

上限 11.5564 は解析的に解ける（$r$ の式を逆に解くだけ）。一方、均衡 $K = 11.6094$ は「家計の資産需要 = $K$」を満たす点で、資産需要は家計問題を解かないと分からない。

`solve_general_equilibrium` がやっているのは二分法。

1. ある $K$ を仮に置く
2. $r(K), w(K)$ を計算する
3. その価格のもとで家計問題を解き、資産需要を求める（`solve_household`）
4. 「資産需要 $- K$」の符号を見て、$K$ の区間を半分に狭める

超過需要が符号を変えるところが均衡。

```
K= 11.5800  資産需要= 20.4   超過需要= +8.79   ← 需要過剰、K を上げる
K= 11.6000  資産需要= 13.1   超過需要= +1.52
K= 11.6094  資産需要= 11.6   超過需要= -0.005  ← ほぼゼロ = 均衡
K= 11.6500  資産需要=  8.5   超過需要= -3.17   ← 供給過剰、K を下げる
```

二分法が使えるのは、超過需要が $K$ について単調減少だから（$r(K)$ が減少、資産需要が $r$ について増加）。

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

| 行                                                | 内容                                                                  |
| ------------------------------------------------- | --------------------------------------------------------------------- |
| `uc_nextgrid = beta * expected_Va`                | オイラー方程式の右辺。これが $u'(c_t)$ の満たすべき値                 |
| `c_nextgrid = uc_nextgrid ** (-eis)`              | 限界効用を逆にして消費に戻す。**内生グリッド上の**消費                |
| `coh = (1+r) * a_grid + y`                        | 手持ち資産 cash-on-hand。**外生グリッド上**                           |
| `interpolate_y(c_nextgrid + a_grid, coh, a_grid)` | 内生グリッド $c + a'$ を外生グリッド $coh$ に貼り替える。EGM の心臓部。中身は numba の kernel（`_interpolate_y_kernel`）で、昇順の $coh$ を前から走査して区間を見つける |
| `np.maximum(a, a_grid[0])`                        | 借入制約。オイラー方程式が成り立たない領域をここで潰す                |
| `Va = (1+r) * c ** (-1/eis)`                      | 次の後ろ向きステップに渡す量                                          |

引数の `expected_Va` は $E_t[V_{a,t+1}] = \Pi \cdot V_{a,t+1}$ で、 **期待を取る操作は呼び出し側の責任**にしてある。sequence-jacobian の HetBlock は これを代行するので、ライブラリ版のコードには `Pi @` が現れない。この対応関係は節10で見る。

返り値はタプルではなく名前つき。

```python
class BackwardStep(NamedTuple):
    Va: MarginalValue
    a: PolicyFunction
    c: PolicyFunction
```

> pydantic ではなく `NamedTuple` にしてあるのは、直接法の内側ループで何千回も 作られるため。注釈で形は明示してあるので読み取りには困らない。

---

## 5. 分布の1期分 — `forward_step`

講義ノート式 7.19 の $D_{t+1} = \Lambda_t^{\top} D_t$。

政策 $a_t$ はグリッド上の点に落ちるとは限らないので、まず「くじ」に直す。

```python
@dataclass(frozen=True, slots=True)
class Lottery:
    """政策 a_t をグリッド上の2点に振り分ける「くじ」表現。

        a_t = weight * a_grid[index] + (1 - weight) * a_grid[index + 1]

    NamedTuple にしていないのは、フィールド名 `index` が `tuple.index` と衝突するため。
    """
    index: LotteryIndex
    weight: LotteryWeight
```

そのうえで、**資産 → 雇用状態の順**に進める。

```python
@njit
def _forward_endogenous_kernel(D: FloatArray, index: IntArray, weight: FloatArray) -> FloatArray:
    n_e, n_a = D.shape
    out = np.zeros((n_e, n_a))
    for e in range(n_e):
        for i in range(n_a):  # 下側の格子点に weight、上側に 1 - weight を積む
            out[e, index[e, i]] += weight[e, i] * D[e, i]
        for i in range(n_a):
            out[e, index[e, i] + 1] += (1.0 - weight[e, i]) * D[e, i]
    return out


@typed
def forward_endogenous(D: Distribution, lottery: Lottery) -> Distribution:
    """資産の遷移だけを進める（雇用状態はまだ動かさない）。"""
    return _forward_endogenous_kernel(D, lottery.index, lottery.weight)


@typed
def forward_step(D: Distribution, Pi: EmploymentTransition, lottery: Lottery) -> Distribution:
    """D_{t+1} = Lambda_t' D_t。まず資産、次に雇用状態の順（SSJ の規約と同じ）。"""
    return Pi.T @ forward_endogenous(D, lottery)
```

`_forward_endogenous_kernel` と `_interpolate_y_kernel` が numba なのは速度のため。配列が 2 x 200 と小さく、numpy で書くと「小さな演算を何度も呼ぶオーバーヘッド」が支配的になる（`np.add.at` 版 5.8 µs → 0.8 µs、補間 16 µs → 1 µs）。形状の検査 `@typed` は外側の関数に残してあるので、呼び出し側から見た仕様は変わらない。

順番は規約の問題だが、**sequence-jacobian と合わせておかないと数値が一致しない**ので そちらに揃えてある。`D_t` は「時点 $t$ 冒頭、$e_t$ が判明したあと、$a_{t-1}$ を持っている状態」の分布。

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

`model.Pi @ step.Va` が節4で言った「期待を取る操作」。10回に1回しか収束判定しないのは 判定のコストを削るためで、sequence-jacobian も同じことをしている。

分布の方は**反復しない**。ここがこの実装の特徴。

```python
@typed
def stationary_from_policy(Pi: EmploymentTransition, lottery: Lottery) -> StationaryDistribution:
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

**なぜ反復しないのか。** $\beta(1+r) = 0.99990$ と1に極端に近いため、遷移行列 $\Lambda_{ss}$ の 第2固有値も1に近く、前向き反復が「見かけ上収束したのにまだ真の定常分布から遠い」状態になる。 実測では `max|D_new - D| < 1e-10` に 33,000 回かかり、そのときの $D$ はまだ真の値から 1.2e-7 ずれていて $K$ が 1.1e-4 ずれる。

状態数は $2 \times 200 = 400$ しかないので、密行列を作って `np.linalg.solve` に投げれば一発。 残差は 4.7e-16 まで落ちる。これが後で効いてくる（節8の「基準走行」）。

### 6.2 資本市場を清算する（一般均衡）

$r(K)$ は $K$ について減少、資産需要は $r$ について増加なので、超過需要は $K$ について減少する。 よって二分法で解ける。

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

- `Va = ss.Va` から始めて `reversed(range(T))` — 終端条件 $V_T = V_{ss}$ から後ろ向き。 ここが「時点 $s$ だけ解けばよいわけではない」の正体。$r$ を1点だけ動かしても、 そのループを通じて $t < s$ の政策が全部変わる。
- `D = ss.D.copy()` から `range(T)` — 初期条件 $D_0 = D_{ss}$ から前向き。
- `np.vdot(D, a_path[t])` — 集計。$K_t = a_t' D_t$。

方向が逆の2つのループが必要で、しかも両方とも長さ $T$ 必要。 これが直接法の1列あたりのコストで、$O(T)$。

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

理想的には `baseline.K` は $K_{ss}$ で平坦なはずだが、有限期間・補間・収束誤差のせいで 完全には平坦にならない。$Y_{ss}\mathbf{1}$ を引くとその誤差まで $1/\varepsilon$ 倍される。 節6.1 で分布を線形方程式で解いておいたおかげで、ここでは 1e-12 まで平坦になっている。

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

`up[s] += eps` の1行が「時点 $s$ だけを $\varepsilon$ 動かす」で、 `columns[...][:, s] = ...` が「その結果を第 $s$ 列に入れる」。 二重ループの外側が入力（$r$ と $w$）、内側が列。

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

`J["K,r"]` のような文字列キーではなく `J.K_r` で取れる。 `eps` と `scheme` を一緒に持たせてあるので、後から「この行列はどの設定で作ったのか」が分かる。

実装同士の突き合わせ用のメソッドもある。

```python
    @typed
    def max_difference(self, other: "HouseholdJacobians") -> float:
        """4本すべてを比べたときの最大の絶対差。実装同士の突き合わせに使う。"""
```

---

## 9. フェイクニュース法 — `jacobian_fake_news`

第8章の内容で、宿題には要らないが**直接法の答え合わせに使う**。 高価な計算が $O(T^2)$ から $O(T)$ に落ちる。

3つの部品からなる。

**①** `curly_Y[s]`, `curly_D[s]` — 「$s$ 期先のニュース」に対する時点0の反応。 $s$ 期先のショックへの時点0の政策は「ショック1回 + 定常ステップ $s$ 回」で得られる。

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
                up = steady_step(up.Va)
```

**②** `expectation_vectors` — 分布の痕跡を将来の集計量に変換するベクトル。

```python
    def expectation_vectors(outcome: PolicyFunction) -> Float[FloatArray, "T_minus_1 n_e n_a"]:
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

> **この方法も有限差分である**という点は押さえておきたい。省いているのは 「列ごとに同じ後ろ向き・前向き計算を繰り返す重複」であって、有限差分そのものではない。 ただし微分する対象が「経路全体の写像」ではなく「1期分の後ろ向きステップ」なので、 その誤差が対角方向の累積を通じて $J$ 全体に伝わる。だから既定を中央差分にしてある。

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

1. 引数が `Va_p`（すでに期待を取ったもの）になっている。`@het(exogenous="Pi", ...)` と 宣言すると、HetBlock が `backward_fun` を呼ぶ前に $\Pi \cdot V_a$ を計算してくれる。
2. 定常状態・移行経路・ヤコビアンのループはライブラリが持っている。

つまり自作版で書いた節5〜9は、ライブラリでは全部この1関数の外側に隠れている。 **逆に言えば、自作版を通したことで「隠れている部分に何があるか」が分かる。**

```python
def income(w, lbar, unemployment_insurance):
    y = np.array([unemployment_insurance, w * lbar])
    return y

hh_extended = hh.add_hetinputs([income])
```

> **落とし穴**: sequence-jacobian は `return` 文に書いた**変数名**を出力名として読み取る。 式をそのまま `return` してはいけない（必ず `return Va` のように名前付きの変数を返す）。

自作の型に詰め替えるヘルパも用意してある。同じ定常状態の上で両方の直接法を走らせると 差はゼロになる。

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

| 実装                 | 微分の対象              | 場所                                  |
| -------------------- | ----------------------- | ------------------------------------- |
| 自作 直接法          | 経路全体の写像 $H$      | `ks.jacobian_direct`                  |
| 自作 fake news       | 1期分の後ろ向きステップ | `ks.jacobian_fake_news`               |
| ライブラリ 直接法    | 経路全体の写像          | `ks.ssj.direct_jacobians_via_library` |
| ライブラリ fake news | 1期分の後ろ向きステップ | `ks.ssj.library_jacobians`            |

$T = 5$ での結果:

```
自作 直接法(中央)    vs ライブラリ fake news(中央) : 1.7e-07
自作 fake news(中央) vs ライブラリ fake news(中央) : 1.6e-07
自作 直接法(片側)    vs ライブラリ 直接法(片側)     : 0.0      ← 同一定常状態なら完全一致
同上、基準走行を差し引かない場合                    : 3.5e-06   ← ノート 7.7 の効果
ライブラリ fake news 片側 vs 中央                  : 2.6e-04   ← 差分幅の影響
```

さらに、予算制約から来る恒等式でも検算できる。

$$\frac{\partial C_0}{\partial r_0} + \frac{\partial K_0}{\partial r_0} = K\_{ss}, \qquad \frac{\partial C_0}{\partial w_0} + \frac{\partial K_0}{\partial w_0} = N$$

どちらも $s > 0$ ではゼロ。実測で 1e-11 のオーダーまで合う。

---

## どこから読むか

- **宿題だけ知りたい**: `01_homework_direct_method.ipynb` → このファイルの節7・8
- **実装を追いたい**: 節0（語彙）→ 節4（EGM）→ 節7（$H$）→ 節8（直接法）
- **キャリブレーションが気になる**: 節2 → `02_experiments.ipynb` 節1
- **ライブラリとの対応**: 節10 → `02_experiments.ipynb` 節5
