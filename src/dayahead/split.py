"""Сплит по времени, общий для всех моделей.

train: 2024, validation: январь-июнь 2025, test: июль-декабрь 2025 (время Будапешта).
Гиперпараметры подбираем на validation, test используем один раз в конце.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dayahead.data import DEFAULT_RAW
from dayahead.data import TZ
from dayahead.data import load_raw
from dayahead.features import TARGET
from dayahead.features import build_features

VAL_START = pd.Timestamp("2025-01-01", tz=TZ)
TEST_START = pd.Timestamp("2025-07-01", tz=TZ)


@dataclass(frozen=True)
class Splits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

    # train + validation: на них переобучается итоговая модель перед оценкой на test
    @property
    def X_trainval(self) -> pd.DataFrame:
        return pd.concat([self.X_train, self.X_val])

    @property
    def y_trainval(self) -> pd.Series:
        return pd.concat([self.y_train, self.y_val])


def load_features(path: str | Path = DEFAULT_RAW) -> pd.DataFrame:
    """Полная таблица (цель + базовые признаки) до сплита. Сюда добавляют свои признаки."""
    return build_features(load_raw(path))


def make_splits(feat: pd.DataFrame) -> Splits:
    """Режет таблицу по времени. Строки не перемешиваются.

    Пропуски от лагов допустимы только в начале train, там строки отбрасываются.
    В validation и test состав строк у всех должен быть одинаковый, иначе метрики
    несравнимы, поэтому пропуск там считается ошибкой.
    """
    train = feat[feat.index < VAL_START].dropna()
    val = feat[(feat.index >= VAL_START) & (feat.index < TEST_START)]
    test = feat[feat.index >= TEST_START]
    for name, part in (("validation", val), ("test", test)):
        bad = part.columns[part.isna().any()].tolist()
        if bad:
            raise ValueError(f"пропуски в {name}: {bad}")

    def xy(part: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        return part.drop(columns=TARGET), part[TARGET]

    return Splits(*xy(train), *xy(val), *xy(test))


def load_splits(path: str | Path = DEFAULT_RAW) -> Splits:
    return make_splits(load_features(path))
