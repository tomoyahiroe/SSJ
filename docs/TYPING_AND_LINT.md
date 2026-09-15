# 型検査と Lint の仕組み

このリポジトリでは「型の安全性」を 3 つの層で分担している。どれも守備範囲が違い、
1 つで全部はカバーできない。

| 層 | 道具 | いつ動くか | 何を捕まえるか |
|---|---|---|---|
| 静的型検査 | mypy（strict + α） | コードを動かす前 | 型の不一致、Any の漏れ、None の未処理、到達しないコード |
| 実行時の形状検査 | jaxtyping + beartype（自作 `typed`） | 関数が呼ばれるたび | 配列の dtype・次元数・**軸の長さの不一致** |
| 値の検証 | pydantic | オブジェクトを作るとき | パラメータの範囲、未知の引数、不変性 |

Lint（Ruff）は別軸で、型ではなく「書き方」の問題を拾う。設定はすべて `pyproject.toml` にある。

---

## 1. 静的型検査 — mypy

`mypy --strict` を基準に、さらに `warn_unreachable`・`extra_checks`・
`disallow_any_unimported`・`disallow_any_decorated` を加えている。
対象は `ks/`・`tests/`・`tools/`。テストも同じ厳しさで検査する。

### 配列の型を Any にしない

numpy の裸の `np.ndarray` は、静的には `ndarray[tuple[Any, ...], dtype[Any]]` に展開される。
これを注釈に使うと、配列を受け渡すすべての関数シグネチャが Any を含み、strict の意味がなくなる。
そこで dtype まで固定した別名を 1 つ置き、配列の型はすべてこれを土台にする（`ks/types.py`）:

```python
type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
type IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]

type PolicyFunction = Float[FloatArray, "n_e n_a"]   # 政策関数
type Distribution = Float[FloatArray, "n_e n_a"]     # 分布
type JacobianMatrix = Float[FloatArray, "T T"]        # ヤコビアン
```

`Float[FloatArray, "n_e n_a"]` は jaxtyping の記法で、mypy からは単に `FloatArray` に見える
（軸名の文字列は静的には無視され、実行時にだけ使われる）。
つまり **静的には dtype まで、実行時には形まで** という分担になる。

### 妥協している点

- numpy の演算が Any を返すところ（`float ** float`、添字アクセスなど）は `float(...)` / `int(...)` で包んで返り値の型を確定させる。
- `sequence_jacobian` と `numba` には型情報がない。`ignore_missing_imports` にし、それを扱う
  `ks.ssj` と `test_ssj` だけ Any を含むことを許す（`[[tool.mypy.overrides]]`）。
- `disallow_any_explicit` は使わない。pydantic の `BaseModel` を継承する行が必ず引っかかるため。

---

## 2. 実行時の形状検査 — jaxtyping + beartype、自作の `typed`

mypy には配列の「形」が見えない。異質的主体モデルで一番多いバグは形の取り違え
（政策 `(n_e, n_a)` と分布 `(n_e, n_a)` を混ぜる、グリッドの点数が合っていない）なので、
これは実行時に検査する。

### 役割分担

- **jaxtyping** — 注釈の書き方を提供する。`Float[FloatArray, "n_e n_a"]` は
  「float の 2 次元配列、軸 0 の長さを `n_e`、軸 1 を `n_a` と呼ぶ」。
  同じ軸名が 1 回の呼び出しの中で複数回現れたら、その長さが一致することまで見る
  （引数どうし、引数と返り値）。
- **beartype** — 実際に値を注釈と照合するエンジン。全要素を走査せず、配列は形と dtype だけを
  見る $O(1)$ 設計で、1 回あたり 10 µs 前後。

### 自作デコレータ `typed`（`ks/types.py`）

```python
_RUNTIME_TYPECHECK = os.environ.get("KS_TYPECHECK", "1").lower() not in ("0", "false", "no")


def typed[F: Callable[..., object]](fn: F) -> F:
    if not _RUNTIME_TYPECHECK:
        return fn
    return jaxtyped(typechecker=beartype)(fn)
```

- 2 つのライブラリの組み合わせを 1 語にしただけだが、**型が `F -> F` のジェネリック**なので
  mypy から見た関数の型は装飾前と変わらない（`disallow_any_decorated` も通る）。
- 配列を受け取る・返す公開関数とメソッドに付ける（`ks/` で 29 か所）。
- 環境変数 `KS_TYPECHECK=0` で全部外れる。既定ではオンで、定常状態の計算では時間の
  半分以上がこの検査なので、重い実験のときだけ切る。
- 検査に失敗すると `jaxtyping.TypeCheckError`（中身は beartype の例外）が、
  どの引数の、どの軸が、いくつだったかを示して投げられる。

### pydantic モデルの中でも効く

pydantic のフィールドに `PolicyFunction` のような jaxtyping の型を書くと、
`arbitrary_types_allowed=True` のもとで pydantic は `isinstance` で検査し、jaxtyping の型は
`isinstance` で形と dtype を見る。したがって `SteadyState(a=..., D=...)` のようにオブジェクトを
作る時点でも配列の形が検査される。

### 動作の保証

`tests/test_types.py` が、次元数違い・dtype 違い・引数間の軸長不一致・返り値の形違いの
4 通りを `typed` が落とすことを固定している。

---

## 3. 値の検証 — pydantic

スカラーのパラメータと、計算結果の入れ物は pydantic の `BaseModel`。

- `KSCalibration`: すべてのパラメータに `Field(gt=..., lt=...)` の範囲制約。`frozen=True` で不変。
  範囲外の値や未知の引数は構築時に `ValidationError`。
- `KSModel` / `SteadyState` / `TransitionPath` / `HouseholdJacobians`: 配列を持つので
  `arbitrary_types_allowed=True`。`frozen=True` で不変。フィールドの型は §2 の仕組みで検査される。
- `pydantic.mypy` プラグイン（`init_typed`, `init_forbid_extra`）で、コンストラクタの
  キーワード引数も**静的に**検査される。名前の綴り間違いや型違いはコードを動かす前に分かる。

内側ループで何千回も作られる小さな入れ物（`BackwardStep`, `StationaryDistribution`）は
pydantic ではなく `NamedTuple`、`Lottery` は `dataclass(frozen=True, slots=True)`。
検証コストを避けるためで、形の検査はそれらを作る・受け取る関数側の `typed` が担う。

---

## 4. numba と型 — 自作の `njit`（`ks/_jit.py`）

数値計算の内側ループは numba でコンパイルする。numba にも型情報がなく、
`@numba.njit` をそのまま使うと mypy strict ではデコレートした関数が Any になる。
そこで `typed` と同じ形の薄い包みを置く:

```python
def njit[F: Callable[..., object]](fn: F) -> F:
    return cast("F", numba.njit(cache=True)(fn))
```

- kernel は `_interpolate_y_kernel` のようなプライベート関数で、注釈は `FloatArray` / `IntArray`
  だけ（njit の中では jaxtyping は使えない）。
- 形の検査は kernel を呼ぶ外側の公開関数に `typed` を付けて担保する。呼び出し側から見た仕様は
  numpy 版と変わらない。
- `cache=True` でコンパイル結果を `__pycache__/` に保存する（初回だけ 1〜2 秒）。

---

## 5. Lint — Ruff

`select = ["ALL"]` から、理由を書いて外す方式。外している理由は 6 種類しかない:

| 理由 | ルール |
|---|---|
| jaxtyping の軸名 `"n_e n_a"` を前方参照・引用符付き注釈と誤認する | F722, F821, UP037 |
| 経済学の記法（`K`, `Pi`, `Va`, `N`, `Z`）を大文字のまま使う。SIM300 は大文字属性を定数と誤認する | N802, N803, N806, N815, SIM300 |
| 日本語の docstring・全角記号（「。」で終わる、命令法・大文字化は当てはまらない） | RUF001–003, D400, D401, D403 |
| beartype が実行時に注釈を評価するので、型だけの import に移せない | TC001–003 |
| 例外メッセージは raise に直接書く / formatter と競合 / 著作権表示 | TRY003, EM101, EM102, COM812, CPY001 |

ファイルごとの例外も、それぞれ理由がある:

- `ks/ssj.py`: sequence-jacobian が `return y` の**変数名**を出力名として読むので `y = ...; return y` が必須（RET504）。
- `ks/__init__.py`: `__all__` を分野ごとにコメントで区切って並べている（RUF022）。
- `tests/`: assert とマジックナンバーが本体（S101, PLR2004）。
- `*.ipynb`: 探索的コードなので docstring・注釈・print・`a; b` などを求めない。Lint の対象ではある。
- 対象外: `8_2_*`（参照用の元コード）、`marimo/`（`.ipynb` からの生成物）。

その他: docstring は numpy 形式（`convention = "numpy"`）、引数は 7 個まで、複雑度は 15 まで。
整形は `ruff format`（`uv format`）。ノートブックは整形しない。

---

## 6. 検査の流れ

```bash
uv format                    # 整形
uv run ruff check .          # Lint（ノートブック含む）
uv run mypy                  # 静的型検査（ks, tests, tools）
uv run pytest                # テスト。実行時検査の動作もここで固定
```

どの誤りをどの層が捕まえるか:

| 誤り | 捕まえる層 |
|---|---|
| `float` のところに `str` を渡す | mypy（静的）、beartype（実行時） |
| `int64` の配列を渡す | jaxtyping / beartype |
| 次元数が違う、グリッドが 200 点なのに政策が 150 点 | jaxtyping / beartype |
| `beta=1.0` のような範囲外のパラメータ | pydantic |
| `KSCalibration(bta=0.99)` のような綴り間違い | pydantic.mypy（静的）、pydantic（実行時） |
| numpy の返り値が Any のまま漏れる | mypy strict（`no-any-return`） |
| 未使用 import、命名、docstring の欠落 | Ruff |
