"""numba のデコレータに型を付けたもの。

`ks` の kernel（内側ループ）は numba でコンパイルする。配列が 2 x 200 と小さいので、
numpy の演算そのものより「小さな演算を何度も呼ぶオーバーヘッド」が支配的で、
1本のコンパイル済みループに融合すると 10 倍前後速くなる（実測: interpolate_y 16 µs → 1 µs）。

numba には型情報がないので、`@numba.njit` をそのまま使うと mypy strict では
デコレートした関数が Any になる。ここで包んで「入れた関数と同じ型」を返すことにする。

`cache=True` でコンパイル結果を `__pycache__` に保存する。初回だけ 1〜2 秒かかる。
"""

from collections.abc import Callable
from typing import cast

import numba

__all__ = ["njit"]


def njit[F: Callable[..., object]](fn: F) -> F:
    """`numba.njit(cache=True)`。型は入れた関数のまま。"""
    return cast("F", numba.njit(cache=True)(fn))
