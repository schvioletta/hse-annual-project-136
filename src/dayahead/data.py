"""Загрузка сырых данных Day-Ahead.

Сырой CSV хранится только локально (data/raw/), в репозиторий не попадает.
"""

from pathlib import Path

import pandas as pd

TZ = "Europe/Budapest"
TIME_COL = "timestamp"


def load_raw(path: str | Path) -> pd.DataFrame:
    """Читает CSV и делает индекс по локальному времени Будапешта.

    В файле timestamp в UTC. Перевод в Europe/Budapest однозначен и учитывает
    переход на летнее время: локальные сутки содержат 23 или 25 часов.
    """
    df = pd.read_csv(path)
    ts = pd.to_datetime(df.pop(TIME_COL), utc=True).dt.tz_convert(TZ)
    df.index = pd.DatetimeIndex(ts, name=TIME_COL)
    return df.sort_index()
