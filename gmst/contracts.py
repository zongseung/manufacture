"""Array, panel and prediction contracts shared by the plain-function models."""
from collections.abc import Callable
from datetime import date
from typing import NotRequired, TypedDict

import numpy as np
import polars as pl
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]
type BoolArray = NDArray[np.bool_]
type FlagArray = NDArray[np.int8]


class Panel(TypedDict):
    dates: list[date]
    Y: FloatArray
    X: dict[str, FloatArray]
    is_missing: BoolArray
    days: pl.DataFrame
    op: FlagArray
    hol: FlagArray
    dtype: FlagArray
    dow: FlagArray
    month: FlagArray


class Prediction(TypedDict):
    y_mean: FloatArray
    y_median: FloatArray
    q: FloatArray | None
    paths: FloatArray | None
    M_hat_median: float
    M_hat_mean: float
    peak_time_mode: int
    risk_raw: FloatArray
    Mq: NotRequired[FloatArray]


class Model[S](TypedDict):
    name: str
    fit: Callable[[Panel, str, FloatArray], S]
    predict: Callable[[S, Panel, int], Prediction]
    inner_state: Callable[[S], S | None]
