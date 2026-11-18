"""Базовые признаки для моделей.

Аукцион Day-Ahead на сутки D проходит в D-1, поэтому из истории цены доступно
только то, что старше 24 ч. Прогнозы (спрос, солнце, ветер и т.д.) известны заранее.
"""

import holidays
import pandas as pd

TARGET = "price_eur_mwh"
MIN_LAG = 24

FORECASTS = [
    "demand_forecast_mwh",
    "solar_forecast_mwh",
    "wind_forecast_mwh",
    "nuclear_forecast_mwh",
    "temperature_forecast_c",
    "net_exchange_forecast_mwh",
]
# месяца нет намеренно: train это один год, месяц запоминает уровень цен 2024
CALENDAR = ["hour", "dow", "is_weekend", "is_holiday"]
DERIVED = ["residual_load"]
PRICE_LAGS = [
    "price_lag_24",
    "price_lag_48",
    "price_lag_168",
    "price_roll_mean_24",
    "price_roll_std_24",
    "price_roll_mean_168",
]
BASE_FEATURES = FORECASTS + CALENDAR + DERIVED + PRICE_LAGS


def lag(series: pd.Series, hours: int) -> pd.Series:
    """Сдвиг ряда назад на hours часов. Меньше 24 ч нельзя: это утечка из будущего."""
    if hours < MIN_LAG:
        raise ValueError(f"лаг {hours} ч < {MIN_LAG} ч: на момент прогноза значение неизвестно")
    return series.shift(hours)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Таблица с целью и базовыми признаками. На вход результат load_raw."""
    out = df.copy()
    idx = out.index

    out["hour"] = idx.hour
    out["dow"] = idx.dayofweek
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    hu = holidays.country_holidays("HU", years=sorted(set(idx.year)))
    out["is_holiday"] = pd.Index(idx.date).isin(list(hu)).astype(int)

    # спрос, который закрывает дорогая манёвренная генерация и импорт
    out["residual_load"] = (
        out["demand_forecast_mwh"]
        - out["solar_forecast_mwh"]
        - out["wind_forecast_mwh"]
        - out["nuclear_forecast_mwh"]
    )

    price = out[TARGET]
    for h in (24, 48, 168):
        out[f"price_lag_{h}"] = lag(price, h)
    known = lag(price, MIN_LAG)
    out["price_roll_mean_24"] = known.rolling(24).mean()
    out["price_roll_std_24"] = known.rolling(24).std()
    out["price_roll_mean_168"] = known.rolling(168).mean()
    return out
