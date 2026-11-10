import pandas as pd
import pytest

from dayahead.data import load_raw


def _write_csv(tmp_path, start_utc: str, periods: int):
    """Синтетический CSV в формате сырого файла, без реальных значений."""
    ts = pd.date_range(start_utc, periods=periods, freq="h", tz="UTC")
    df = pd.DataFrame({"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S%z"), "price_eur_mwh": 1.0})
    path = tmp_path / "synthetic.csv"
    df.to_csv(path, index=False)
    return path


@pytest.mark.parametrize(
    ("start_utc", "day", "hours"),
    [
        ("2024-03-30 23:00", "2024-03-31", 23),  # переход на летнее время
        ("2024-10-26 22:00", "2024-10-27", 25),  # переход на зимнее время
    ],
)
def test_load_raw_dst(tmp_path, start_utc, day, hours):
    df = load_raw(_write_csv(tmp_path, start_utc, hours))
    assert str(df.index.tz) == "Europe/Budapest"
    assert (df.index.date == pd.Timestamp(day).date()).all()
    assert len(df) == hours
