import pandas as pd
import streamlit as st

import model
import policy
import project
from blocks import BLOCKS

st.set_page_config(page_title="Orchard Irrigation Outlook", layout="wide")

SPECIES_DISPLAY = {
    "coconut": "Coconut",
    "mango": "Mango",
    "chickoo": "Chickoo",
    "custard_apple": "Custard apple",
    "mahogany": "Mahogany",
    "imli": "Imli",
    "neem": "Neem",
}

STATUS_ICON = {policy.NONE: "\U0001F7E2", policy.MONITOR: "\U0001F7E1", policy.IRRIGATE: "\U0001F534"}


@st.cache_data(ttl=3600)
def get_forecast():
    return project.fetch_forecast_weather()


@st.cache_resource
def get_evaluation():
    return model.run_full_evaluation()


def compute_projection(starting_condition):
    forecast = get_forecast()
    fill_fraction = project.STARTING_CONDITIONS[starting_condition]
    rows = []
    for block in BLOCKS:
        rows.extend(project.project_block(block, forecast, fill_fraction))
    return pd.DataFrame(rows)


def classify_urgency(block_week):
    labels = block_week.sort_values("day_index")["label"].tolist()
    if labels[0] == policy.IRRIGATE:
        return "today"
    if policy.IRRIGATE in labels[1:]:
        first_day = next(i for i, l in enumerate(labels) if l == policy.IRRIGATE)
        return ("this_week", first_day)
    return "fine"


def render_block_card(block, block_week, tree_count):
    dates = block_week.sort_values("day_index")
    today_date = dates.iloc[0]["date"].strftime("%b %d")
    strip = []
    for _, row in dates.iterrows():
        day_label = "Today" if row["day_index"] == 0 else row["date"].strftime("%a")
        strip.append(f"{day_label} {STATUS_ICON[row['label']]}")
    species_name = SPECIES_DISPLAY[block["species"]]
    with st.container(border=True):
        st.markdown(f"**{species_name}** &nbsp;·&nbsp; block {block['id']} &nbsp;·&nbsp; {tree_count} trees")
        st.caption(" &nbsp; ".join(strip))


def render_today_tab():
    st.subheader("Today's outlook")
    st.caption(
        "Today's status and the week ahead, so water for the well or an outside supplier "
        "can be arranged before a block is actually short, not on the day it is."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        starting_condition = st.selectbox(
            "How do conditions feel right now, roughly?",
            list(project.STARTING_CONDITIONS.keys()),
            index=1,
        )
    with col2:
        water_available = st.radio("Water available today?", ["Yes", "Not sure", "No"], horizontal=True)

    with st.spinner("Pulling the latest weather forecast for Nashik..."):
        proj = compute_projection(starting_condition)

    groups = {"today": [], "this_week": [], "fine": []}
    for block in BLOCKS:
        block_week = proj[proj["block_id"] == block["id"]]
        urgency = classify_urgency(block_week)
        if urgency == "today":
            groups["today"].append((block, block_week))
        elif urgency[0] == "this_week":
            groups["this_week"].append((block, block_week, urgency[1]))
        else:
            groups["fine"].append((block, block_week))

    n_today = len(groups["today"])
    n_week = len(groups["this_week"])
    summary = f"**{n_today} block{'s' if n_today != 1 else ''} need attention today**"
    if n_week:
        summary += f", {n_week} more likely by the end of the week"
    st.markdown(summary + ".")
    if water_available == "No" and (n_today or n_week):
        st.warning("Water isn't marked available today — this is the point to arrange a tanker or check the well before it's urgent.")

    tree_counts = {b["id"]: b["tree_count"] for b in BLOCKS}

    if groups["today"]:
        st.markdown("### \U0001F534 Needs attention today")
        for block, block_week in groups["today"]:
            render_block_card(block, block_week, tree_counts[block["id"]])

    if groups["this_week"]:
        st.markdown("### \U0001F7E1 Likely needs water this week")
        for block, block_week, _ in groups["this_week"]:
            render_block_card(block, block_week, tree_counts[block["id"]])

    if groups["fine"]:
        st.markdown("### \U0001F7E2 Fine for now")
        for block, block_week in groups["fine"]:
            render_block_card(block, block_week, tree_counts[block["id"]])

    st.divider()
    st.caption(
        "Today's reading rests on real current weather and the condition picked above. "
        "Later days rest on a weather forecast and six days of simulated water balance on top of "
        "it — treat them as a trend to watch, not a guarantee, and recheck as the week goes on."
    )


def render_model_details_tab():
    st.subheader("How the prediction is built and how well it checks out")
    train, test, results, clf, feature_names = get_evaluation()

    st.markdown(
        "This tab is the evidence behind the same-day accuracy claim — validated on real "
        "historical data, not the live projection above (which has no sensor feed to validate "
        "against; see the note at the bottom of the Today tab)."
    )

    st.markdown("#### Evaluation results")
    summary_rows = [{"Model": r["name"], "Macro F1": round(r["macro_f1"], 3), "Ordinal MAE": round(r["ordinal_mae"], 3)}
                     for r in results]
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

    full_model = next(r for r in results if r["name"] == "Random forest (full features)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Confusion matrix (test set)")
        cm = pd.DataFrame(
            full_model["confusion_matrix"],
            index=["True: None", "True: Monitor", "True: Irrigate"],
            columns=["Pred: None", "Pred: Monitor", "Pred: Irrigate"],
        )
        st.dataframe(cm, use_container_width=True)
        st.caption("Zero two-step misses: the model never confuses 'no action needed' with 'irrigate.'")

    with col2:
        st.markdown("#### Feature importances")
        imp_df = pd.DataFrame(
            list(full_model["feature_importances"].items())[:8],
            columns=["Feature", "Importance"],
        )
        st.bar_chart(imp_df.set_index("Feature"))

    st.markdown("#### Example predictions from the test set")
    st.caption("Four random correct predictions plus one real miss, not a cherry-picked all-correct sample.")
    from features import build_feature_matrix
    X_test_all, _ = build_feature_matrix(test)
    X_test_all = X_test_all.reindex(columns=feature_names, fill_value=0)
    all_preds = clf.predict(X_test_all)
    test_with_preds = test.copy()
    test_with_preds["pred"] = all_preds
    correct = test_with_preds[test_with_preds["pred"] == test_with_preds["label"]]
    incorrect = test_with_preds[test_with_preds["pred"] != test_with_preds["label"]]

    sample_parts = [correct.sample(min(4, len(correct)), random_state=1)]
    if len(incorrect) > 0:
        sample_parts.append(incorrect.sample(1, random_state=1))
    sample = pd.concat(sample_parts)

    display_rows = []
    for _, row in sample.iterrows():
        display_rows.append({
            "Block": row["block_id"],
            "Date": row["date"].strftime("%Y-%m-%d"),
            "Species": SPECIES_DISPLAY[row["species"]],
            "True label": policy.LABELS[row["label"]],
            "Predicted": policy.LABELS[row["pred"]],
            "Agrees?": "Yes" if row["pred"] == row["label"] else "No",
        })
    st.dataframe(pd.DataFrame(display_rows), hide_index=True, use_container_width=True)

    st.markdown("#### What this does and doesn't establish")
    st.caption(
        "The model approximates the authored policy well from noisy, realistically-available "
        "observations, on held-out data from one simulated annual cycle. It does not establish "
        "that the policy reflects real orchard water needs, that these numbers hold on a second "
        "year of weather, or that anything here has been validated against real irrigation "
        "outcomes. See eval/eval_report.md for the full write-up."
    )


st.title("Orchard Irrigation Outlook")
st.caption(
    "A synthetic, single-cycle simulation of a 1,600-tree family orchard in Nashik, Maharashtra — "
    "built to show a classical ML workflow end to end, not to make real irrigation decisions."
)

tab_today, tab_model = st.tabs(["Today", "Model details"])
with tab_today:
    render_today_tab()
with tab_model:
    render_model_details_tab()
