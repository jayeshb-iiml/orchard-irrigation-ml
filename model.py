"""Baselines, the classifier, the temporal split, and evaluation.

Split note: a plain chronological tail-cut (last N% of days as test) was
tried first and rejected — with only one annual cycle of data, the tail
lands entirely inside a monsoon stretch where almost every block-day is
"No irrigation," leaving zero "Irrigate" examples in the test set and
making the class that matters most impossible to evaluate. Instead this
holds out the last ~20% of days *within each calendar month*, per block:
order is preserved at the scale that matters (no shuffling within a
month), and every season still appears in both train and test. That is
a real tradeoff, named here rather than hidden — it is not a pure
chronological holdout, and a true multi-year dataset would not need it.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report

import policy
import features
from blocks import block_params, BLOCKS

TEST_FRACTION_WITHIN_MONTH = 0.20
RANDOM_STATE = 42

BLOCK_LOOKUP = {b["id"]: b for b in BLOCKS}


def load_dataset(path="data/orchard_dataset.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def temporal_split(df):
    df = df.sort_values(["block_id", "date"]).copy()
    test_mask = pd.Series(False, index=df.index)

    for (_block, _month), group in df.groupby([df["block_id"], df["date"].dt.to_period("M")]):
        n_test = max(1, int(len(group) * TEST_FRACTION_WITHIN_MONTH))
        test_idx = group.sort_values("date").index[-n_test:]
        test_mask.loc[test_idx] = True

    return df[~test_mask].copy(), df[test_mask].copy()


def ordinal_mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def evaluate(y_true, y_pred, name):
    return {
        "name": name,
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "ordinal_mae": ordinal_mae(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "report": classification_report(y_true, y_pred, labels=[0, 1, 2],
                                         target_names=["None", "Monitor", "Irrigate"],
                                         zero_division=0, output_dict=True),
    }


def majority_class_baseline(train, test):
    majority = train["label"].mode()[0]
    preds = [majority] * len(test)
    return evaluate(test["label"], preds, "Majority class")


def season_only_baseline(train, test):
    """A sanity check against the calendar-proxy failure mode: if this
    does nearly as well as the full model, the task is mostly "what month
    is it," not a genuine read of conditions.
    """
    month_majority = train.groupby("month")["label"].agg(lambda s: s.mode()[0])
    fallback = train["label"].mode()[0]
    preds = test["month"].map(month_majority).fillna(fallback).astype(int)
    return evaluate(test["label"], preds, "Season-only")


def direct_rule_on_observed_baseline(test):
    """Applies the exact same policy logic the labels were generated from,
    but fed the OBSERVED (noisy) moisture instead of the true latent state
    the policy actually used. This isolates: if you only had what a person
    could actually check, how often would just applying the rule by hand
    already agree with the real decision? Recent rainfall is approximated
    from the 7-day trailing sum (the exact 2-day window isn't stored).
    """
    preds = []
    for _, row in test.iterrows():
        params = block_params(BLOCK_LOOKUP[row["block_id"]])
        obs_stress = (1.0 - min(1.0, row["observed_moisture"] / params["root_capacity_mm"]))
        obs_stress = float(np.clip(obs_stress * params["stress_sensitivity"], 0, 1))
        recent_rain_proxy = row["trailing_rainfall_7d_mm"] * (2 / 7)
        preds.append(policy.determine_irrigation_need(obs_stress, recent_rain_proxy))
    return evaluate(test["label"], preds, "Direct rule on observed features")


def train_classifier(train, feature_columns=None):
    X_train, feature_names = features.build_feature_matrix(train, feature_columns)
    y_train = train["label"]
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE, class_weight="balanced")
    clf.fit(X_train, y_train)
    return clf, feature_names


def evaluate_classifier(clf, feature_names, test, feature_columns=None, name="Random forest"):
    X_test, _ = features.build_feature_matrix(test, feature_columns)
    X_test = X_test.reindex(columns=feature_names, fill_value=0)
    preds = clf.predict(X_test)
    result = evaluate(test["label"], preds, name)
    importances = dict(sorted(zip(feature_names, clf.feature_importances_), key=lambda kv: -kv[1]))
    result["feature_importances"] = importances
    return result


def run_full_evaluation():
    features.assert_no_leakage()
    df = load_dataset()
    train, test = temporal_split(df)

    results = [
        majority_class_baseline(train, test),
        season_only_baseline(train, test),
        direct_rule_on_observed_baseline(test),
    ]

    clf, feature_names = train_classifier(train)
    results.append(evaluate_classifier(clf, feature_names, test, name="Random forest (full features)"))

    ablated_cols = [c for c in features.FEATURE_COLUMNS
                    if c not in ("observed_moisture", "observed_stress_rating", "days_since_last_irrigation")]
    clf_ablated, feature_names_ablated = train_classifier(train, feature_columns=ablated_cols)
    results.append(evaluate_classifier(clf_ablated, feature_names_ablated, test,
                                        feature_columns=ablated_cols, name="Random forest (weather+block only)"))

    return train, test, results, clf, feature_names


if __name__ == "__main__":
    train, test, results, clf, feature_names = run_full_evaluation()
    print(f"Train: {len(train)} rows, Test: {len(test)} rows\n")
    for r in results:
        print(f"{r['name']:40s} macro F1={r['macro_f1']:.3f}  ordinal MAE={r['ordinal_mae']:.3f}")
    print("\nTop feature importances (full model):")
    for name, imp in list(results[3]["feature_importances"].items())[:8]:
        print(f"  {name:30s} {imp:.3f}")
