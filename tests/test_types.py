"""`ks.types` — 状態ラベルと、実行時の形状検査デコレータ `typed` の仕様。"""

import numpy as np
import pytest
from jaxtyping import TypeCheckError

from ks.types import AggregateState, AssetGrid, Employment, PolicyFunction, typed


@typed
def _scale(grid: AssetGrid, policy: PolicyFunction) -> PolicyFunction:
    """軸名 n_a を2つの引数で共有する関数。検査の題材。"""
    return policy * grid


@typed
def _wrong_return(grid: AssetGrid) -> PolicyFunction:
    """宣言は2次元なのに1次元を返す関数。返り値検査の題材。"""
    return grid


def test_employment_labels_are_axis0_indices() -> None:
    """Employment は配列の軸0 の添字で、失業 = 0、就業 = 1。"""
    assert Employment.UNEMPLOYED.value == 0
    assert Employment.EMPLOYED.value == 1
    assert AggregateState.BAD.value == 0
    assert AggregateState.GOOD.value == 1


def test_typed_accepts_matching_shapes() -> None:
    """`typed` は dtype と形が注釈どおりの引数をそのまま通す。"""
    out = _scale(np.ones(3), np.ones((2, 3)))
    assert out.shape == (2, 3)


def test_typed_rejects_wrong_ndim() -> None:
    """`typed` は次元数が注釈と違う引数（1次元のところに2次元）を TypeCheckError で落とす。"""
    with pytest.raises(TypeCheckError):
        _scale(np.ones((3, 1)), np.ones((2, 3)))


def test_typed_rejects_wrong_dtype() -> None:
    """`typed` は Float 注釈に整数配列を渡すと TypeCheckError で落とす。"""
    with pytest.raises(TypeCheckError):
        _scale(np.ones(3, dtype=np.int64), np.ones((2, 3)))


def test_typed_checks_axis_length_across_arguments() -> None:
    """同じ軸名 n_a が複数の引数に現れたら、その長さが一致していることまで検査する。

    グリッドが 4 点なのに政策が 3 点、という取り違えを呼び出しの時点で落とす。
    """
    with pytest.raises(TypeCheckError):
        _scale(np.ones(4), np.ones((2, 3)))


def test_typed_checks_return_value() -> None:
    """返り値の形も注釈と照合される。"""
    with pytest.raises(TypeCheckError):
        _wrong_return(np.ones(3))
