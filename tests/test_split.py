import numpy as np
import pandas as pd
import pytest

from dayahead.data import TZ
from dayahead.features import BASE_FEATURES
from dayahead.features import FORECASTS
from dayahead.features import TARGET
from dayahead.features import build_features
from dayahead.features import lag
from dayahead.metrics import evaluate
from dayahead.split import TEST_START
from dayahead.split import VAL_START
from dayahead.split import make_splits


@pytest.fixture
def raw():
    """Синтетика в формате load_raw: два года по часам, цена равна номеру строки."""
    idx = pd.date_range("2023-12-31 23:00", "2025-12-31 22:00", freq="h", tz="UTC").tz_convert(TZ)
    df = pd.DataFrame({c: 1.0 for c in FORECASTS}, index=idx)
    df[TARGET] = np.arange(len(df), dtype=float)
    return df


def test_lags_use_only_past(raw):
    feat = build_features(raw).dropna()
    assert (feat[TARGET] - feat["price_lag_24"] == 24).all()
    # окно 24 ч заканчивается за 24 ч до прогнозируемого часа
    assert (feat[TARGET] - feat["price_roll_mean_24"] == 24 + 11.5).all()


def test_short_lag_rejected(raw):
    with pytest.raises(ValueError):
        lag(raw[TARGET], 23)


def test_splits_ordered_and_complete(raw):
    s = make_splits(build_features(raw))
    assert list(s.X_train.columns) == BASE_FEATURES
    assert s.X_train.index.max() < VAL_START <= s.X_val.index.min()
    assert s.X_val.index.max() < TEST_START <= s.X_test.index.min()
    # validation и test без потерь строк: 181 и 184 суток
    assert len(s.X_val) == 181 * 24 - 1  # переход на летнее время
    assert len(s.X_test) == 184 * 24 + 1  # переход на зимнее время
    assert not s.X_train.isna().any().any()


def test_trainval_is_train_plus_validation(raw):
    s = make_splits(build_features(raw))
    assert len(s.X_trainval) == len(s.X_train) + len(s.X_val)
    assert s.X_trainval.index.equals(s.y_trainval.index)
    assert s.X_trainval.index.is_monotonic_increasing
    assert s.X_trainval.index.max() < TEST_START


def test_nan_in_validation_is_error(raw):
    feat = build_features(raw)
    feat["far_lag"] = feat[TARGET].shift(24 * 400)
    with pytest.raises(ValueError):
        make_splits(feat)


def test_holiday_flag(raw):
    feat = build_features(raw)
    assert feat.loc["2024-03-15", "is_holiday"].all()  # национальный праздник Венгрии
    assert not feat.loc["2024-03-14", "is_holiday"].any()


def test_evaluate():
    m = evaluate([-10, 0, 100, 400], [0, 0, 110, 380])
    assert m["MAE"] == 10
    assert m["RMSE"] == pytest.approx(np.sqrt(150))
    assert m["MAE_le0"] == 5
    assert m["MAE_gt300"] == 20
    assert np.isnan(evaluate([1, 2], [1, 2])["MAE_gt300"])
