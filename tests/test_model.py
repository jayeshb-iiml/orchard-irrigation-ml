import pandas as pd
from model import temporal_split, ordinal_mae


def _toy_dataset():
    rows = []
    for block in ["B1", "B2"]:
        for month in range(1, 4):
            for day in range(1, 11):
                rows.append({
                    "block_id": block,
                    "date": pd.Timestamp(2024, month, day),
                    "label": (day + month) % 3,
                    "month": month,
                })
    return pd.DataFrame(rows)


def test_temporal_split_has_no_overlapping_rows():
    df = _toy_dataset()
    train, test = temporal_split(df)
    assert len(train) + len(test) == len(df)
    assert set(train.index).isdisjoint(set(test.index))


def test_temporal_split_keeps_test_dates_after_train_dates_within_each_block_month():
    df = _toy_dataset()
    train, test = temporal_split(df)
    for (block, month), test_group in test.groupby(["block_id", "month"]):
        train_group = train[(train["block_id"] == block) & (train["month"] == month)]
        if len(train_group) == 0:
            continue
        assert train_group["date"].max() < test_group["date"].min()


def test_temporal_split_is_roughly_80_20():
    df = _toy_dataset()
    train, test = temporal_split(df)
    fraction = len(test) / len(df)
    assert 0.15 <= fraction <= 0.25


def test_ordinal_mae_zero_for_perfect_predictions():
    assert ordinal_mae([0, 1, 2, 1], [0, 1, 2, 1]) == 0.0


def test_ordinal_mae_penalizes_distance():
    close_miss = ordinal_mae([2], [1])
    far_miss = ordinal_mae([2], [0])
    assert far_miss > close_miss
