"""テスト全体で共有する定常状態。

定常状態の計算は数秒かかるので、セッションで1回だけ解いて使い回す。
"""

import pytest

import ks


@pytest.fixture(scope="session")
def model() -> ks.KSModel:
    """既定のキャリブレーション（KS 1998）から組み立てたモデル。"""
    return ks.KSModel.build()


@pytest.fixture(scope="session")
def ss(model: ks.KSModel) -> ks.SteadyState:
    """資本市場が清算する一般均衡の定常状態。"""
    return ks.solve_general_equilibrium(model)
