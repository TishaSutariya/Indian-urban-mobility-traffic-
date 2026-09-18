import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Indian Urban Mobility Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

MODEL_FILE = MODEL_DIR / "accident_prediction_model.pkl"
RESULTS_FILE = MODEL_DIR / "model_results.csv"

# ============================================================
# SIMPLE PROFESSIONAL CSS
# No HTML dashboard components are used.
# ============================================================

st.markdown("""
<style>
.stApp {
    background-color: #0b0f14;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1.5rem;
    max-width: 1500px;
}

section[data-testid="stSidebar"] {
    background-color: #171b22;
}

h1 {
    font-size: 2.2rem !important;
    margin-bottom: 0.3rem !important;
}

h2 {
    font-size: 1.5rem !important;
    margin-top: 1rem !important;
    margin-bottom: 0.5rem !important;
}

h3 {
    margin-top: 0.5rem !important;
}

[data-testid="stMetric"] {
    background-color: #151a21;
    border: 1px solid #303640;
    border-radius: 14px;
    padding: 14px;
}

[data-testid="stMetricValue"] {
    font-size: 1.8rem;
}

[data-testid="stMetricLabel"] {
    color: #aeb5c0;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #303640;
    border-radius: 10px;
}

.stButton > button {
    border-radius: 8px;
}

hr {
    border-color: #303640;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# FUNCTIONS
# ============================================================

def load_csv(filename):
    path = DATA_DIR / filename

    if not path.exists():
        return pd.DataFrame()

    for encoding in ["utf-8", "latin1", "cp1252"]:
        try:
            df = pd.read_csv(path, encoding=encoding)
            df.columns = (
                df.columns.astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )
            return df
        except Exception:
            continue

    return pd.DataFrame()


def to_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )


def find_column(df, keywords):
    if df.empty:
        return None

    columns = list(df.columns)

    # Exact / starts-with style matching
    for keyword in keywords:
        keyword = keyword.lower().strip()

        for col in columns:
            col_clean = str(col).lower().strip()

            if col_clean == keyword:
                return col

    # Partial matching
    for keyword in keywords:
        keyword = keyword.lower().strip()

        for col in columns:
            col_clean = str(col).lower().strip()

            if keyword in col_clean:
                return col

    return None


def remove_total_rows(df, column):
    if df.empty or column is None:
        return df.copy()

    result = df.copy()

    mask = (
        result[column]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin([
            "total",
            "all india",
            "india total"
        ])
    )

    return result[~mask].copy()


def number(value):
    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value):,.0f}"


def load_all_data():
    return {
        "a1": load_csv("RA2019_A1_cleaned.csv"),
        "a2": load_csv("RA2019_A2_cleaned.csv"),
        "a47": load_csv("RA2019_A47_cleaned.csv"),
        "a48": load_csv("RA2019_A48_cleaned.csv"),
        "a49": load_csv("RA2019_A49_cleaned.csv"),
        "a50": load_csv("RA2019_A50_cleaned.csv"),
        "a51": load_csv("RA2019_A51_cleaned.csv"),
        "a52": load_csv("RA2019_A52_cleaned.csv"),
        "ml": load_csv("ml_accident_dataset.csv")
    }


def get_latest_values(a1):
    result = {
        "year": None,
        "accidents": None,
        "killed": None,
        "injured": None
    }

    if a1.empty:
        return result

    year_col = find_column(
        a1,
        ["years", "year"]
    )

    accident_col = find_column(
        a1,
        ["total number of road accidents"]
    )

    killed_col = find_column(
        a1,
        ["total number of persons killed"]
    )

    injured_col = find_column(
        a1,
        ["total number of persons injured"]
    )

    if year_col is None or accident_col is None:
        return result

    temp = a1.copy()

    temp[year_col] = to_numeric(temp[year_col])
    temp[accident_col] = to_numeric(temp[accident_col])

    if killed_col:
        temp[killed_col] = to_numeric(temp[killed_col])

    if injured_col:
        temp[injured_col] = to_numeric(temp[injured_col])

    temp = temp.dropna(
        subset=[year_col, accident_col]
    )

    if temp.empty:
        return result

    latest_year = int(temp[year_col].max())

    latest = temp[
        temp[year_col] == latest_year
    ]

    if latest.empty:
        return result

    row = latest.iloc[0]

    result["year"] = latest_year
    result["accidents"] = row[accident_col]

    if killed_col:
        result["killed"] = row[killed_col]

    if injured_col:
        result["injured"] = row[injured_col]

    return result


# ============================================================
# LOAD DATA
# ============================================================

data = load_all_data()

a1 = data["a1"]
a2 = data["a2"]
a47 = data["a47"]
a48 = data["a48"]
a49 = data["a49"]
a50 = data["a50"]
a51 = data["a51"]
a52 = data["a52"]
ml_df = data["ml"]

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🚦 Urban Mobility")
    st.caption("Indian Road Safety Intelligence")

    st.divider()

    st.subheader("Navigation")

    # ML FIRST, OVERVIEW LAST
    page = st.radio(
        "Go to",
        [
            "🤖 ML Prediction",
            "📈 Accident Trends",
            "🗺️ State Analysis",
            "⚠️ Accident Causes",
            "🛣️ Road Features",
            "🚦 Traffic Control",
            "🔀 Junction Analysis",
            "⚫ Black Spots",
            "📍 High Accident Locations",
            "📚 Data & Methodology",
            "🏠 Overview"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.caption("Data Source")
    st.caption("Government of India")
    st.caption("data.gov.in")

# ============================================================
# 1. ML PREDICTION
# ============================================================

if page == "🤖 ML Prediction":

    st.title("🤖 Machine Learning Prediction")

    st.caption(
        "State-level estimation of 2019 road accident counts "
        "using historical accident data from 2016–2018."
    )

    st.divider()

    st.subheader("Machine Learning Model")

    st.info(
        "Linear Regression uses historical State/UT-level accident "
        "information from 2016–2018 to estimate the reported accident "
        "count for 2019."
    )

    if not MODEL_FILE.exists():
        st.error(
            "Model file not found. Expected location: "
            f"{MODEL_FILE}"
        )
        st.stop()

    if ml_df.empty:
        st.error(
            "ML dataset not found. Please run feature_engineering.py."
        )
        st.stop()

    required_columns = [
        "State_UT",
        "Accidents_2016",
        "Accidents_2017",
        "Accidents_2018",
        "Accidents_2019",
        "Average_Accidents_2016_2018",
        "Accident_Change_2018_vs_2016",
        "Accident_Growth_2018_vs_2016"
    ]

    missing = [
        col for col in required_columns
        if col not in ml_df.columns
    ]

    if missing:
        st.error(
            "Missing ML columns: " + ", ".join(missing)
        )
        st.stop()

    try:
        model = joblib.load(MODEL_FILE)
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

    # --------------------------------------------------------
    # MODEL SUMMARY
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Model",
            "Linear Regression"
        )

    with c2:
        st.metric(
            "Input Features",
            "6"
        )

    with c3:
        st.metric(
            "Target",
            "2019"
        )

    with c4:
        st.metric(
            "Records",
            f"{len(ml_df):,}"
        )

    st.divider()

    # --------------------------------------------------------
    # STATE SELECTION
    # --------------------------------------------------------

    st.subheader("Select State / UT")

    states = (
        ml_df["State_UT"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    selected_state = st.selectbox(
        "State / Union Territory",
        states
    )

    selected = ml_df[
        ml_df["State_UT"].astype(str) == selected_state
    ]

    if selected.empty:
        st.warning("No data available for this State/UT.")
        st.stop()

    row = selected.iloc[0]

    # --------------------------------------------------------
    # HISTORICAL DATA
    # --------------------------------------------------------

    st.subheader("Historical Accident Counts")

    h1, h2, h3 = st.columns(3)

    with h1:
        st.metric(
            "2016",
            number(row["Accidents_2016"])
        )

    with h2:
        st.metric(
            "2017",
            number(row["Accidents_2017"])
        )

    with h3:
        st.metric(
            "2018",
            number(row["Accidents_2018"])
        )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    st.subheader("Calculated Features")

    f1, f2, f3 = st.columns(3)

    average_value = float(
        row["Average_Accidents_2016_2018"]
    )

    change_value = float(
        row["Accident_Change_2018_vs_2016"]
    )

    growth_value = float(
        row["Accident_Growth_2018_vs_2016"]
    )

    with f1:
        st.metric(
            "2016–2018 Average",
            number(average_value)
        )

    with f2:
        st.metric(
            "2018 − 2016 Change",
            number(change_value)
        )

    with f3:
        st.metric(
            "Growth 2016 → 2018",
            f"{growth_value:.2f}%"
        )

    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

    feature_columns = [
        "Accidents_2016",
        "Accidents_2017",
        "Accidents_2018",
        "Average_Accidents_2016_2018",
        "Accident_Change_2018_vs_2016",
        "Accident_Growth_2018_vs_2016"
    ]

    X_prediction = pd.DataFrame(
        [[
            float(row["Accidents_2016"]),
            float(row["Accidents_2017"]),
            float(row["Accidents_2018"]),
            average_value,
            change_value,
            growth_value
        ]],
        columns=feature_columns
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    try:
        prediction = float(
            model.predict(X_prediction)[0]
        )

        prediction = max(0, prediction)

    except Exception as e:
        st.error(
            f"Prediction failed: {e}"
        )
        st.stop()

    actual = float(
        row["Accidents_2019"]
    )

    error = abs(
        actual - prediction
    )

    st.divider()

    st.subheader(
        f"🎯 2019 Prediction — {selected_state}"
    )

    p1, p2, p3 = st.columns(3)

    with p1:
        st.metric(
            "Predicted 2019",
            number(prediction)
        )

    with p2:
        st.metric(
            "Actual 2019",
            number(actual)
        )

    with p3:
        st.metric(
            "Absolute Error",
            number(error)
        )

    # --------------------------------------------------------
    # ACTUAL VS PREDICTED
    # --------------------------------------------------------

    comparison = pd.DataFrame({
        "Type": [
            "Actual 2019",
            "Predicted 2019"
        ],
        "Accidents": [
            actual,
            prediction
        ]
    })

    fig = px.bar(
        comparison,
        x="Type",
        y="Accidents",
        text="Accidents",
        title=f"{selected_state}: Actual vs Predicted 2019"
    )

    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # ALL STATE PREDICTIONS
    # --------------------------------------------------------

    st.subheader("State-wise Predictions")

    input_df = ml_df[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    valid = ~input_df.isna().any(axis=1)

    valid_inputs = input_df[valid]

    try:
        predictions = model.predict(
            valid_inputs
        )

        all_predictions = ml_df.loc[
            valid,
            [
                "State_UT",
                "Accidents_2019"
            ]
        ].copy()

        all_predictions["Predicted_2019"] = predictions

        all_predictions["Absolute_Error"] = abs(
            all_predictions["Accidents_2019"]
            -
            all_predictions["Predicted_2019"]
        )

        chart_df = all_predictions.sort_values(
            "Predicted_2019",
            ascending=False
        ).head(15)

        fig2 = px.bar(
            chart_df.sort_values(
                "Predicted_2019"
            ),
            x="Predicted_2019",
            y="State_UT",
            orientation="h",
            title="State-wise Predicted 2019 Accident Counts"
        )

        fig2.update_layout(
            template="plotly_dark",
            height=600,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        table = all_predictions.copy()

        table["Accidents_2019"] = table[
            "Accidents_2019"
        ].map(number)

        table["Predicted_2019"] = table[
            "Predicted_2019"
        ].map(number)

        table["Absolute_Error"] = table[
            "Absolute_Error"
        ].map(number)

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:
        st.error(
            f"Could not generate State/UT predictions: {e}"
        )

    # --------------------------------------------------------
    # MODEL RESULTS
    # --------------------------------------------------------

    st.subheader("Model Evaluation")

    if RESULTS_FILE.exists():

        try:
            results = pd.read_csv(
                RESULTS_FILE
            )

            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:
            st.warning(
                f"Could not read model results: {e}"
            )

    st.warning(
        "Scope: this model estimates annual State/UT-level road "
        "accident counts. It is not a real-time traffic speed, "
        "density, congestion, or vehicle-flow prediction model."
    )

# ============================================================
# 2. ACCIDENT TRENDS
# ============================================================

elif page == "📈 Accident Trends":

    st.title("📈 Accident Trends")

    st.caption(
        "Historical trend of reported road accidents in India."
    )

    if a1.empty:
        st.error("A1 dataset not found.")
        st.stop()

    year_col = find_column(
        a1,
        ["years", "year"]
    )

    accident_col = find_column(
        a1,
        ["total number of road accidents"]
    )

    killed_col = find_column(
        a1,
        ["total number of persons killed"]
    )

    if not year_col or not accident_col:
        st.error("Required A1 columns were not found.")
        st.stop()

    trend = a1.copy()

    trend[year_col] = to_numeric(
        trend[year_col]
    )

    trend[accident_col] = to_numeric(
        trend[accident_col]
    )

    trend = trend.dropna(
        subset=[
            year_col,
            accident_col
        ]
    ).sort_values(
        year_col
    )

    latest = trend.iloc[-1]

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Latest Year",
            int(latest[year_col])
        )

    with c2:
        st.metric(
            "Reported Accidents",
            number(latest[accident_col])
        )

    fig = px.line(
        trend,
        x=year_col,
        y=accident_col,
        markers=True,
        title="India Road Accident Trend"
    )

    fig.update_layout(
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    if killed_col:

        trend[killed_col] = to_numeric(
            trend[killed_col]
        )

        fatality_df = trend.dropna(
            subset=[killed_col]
        )

        fig2 = px.line(
            fatality_df,
            x=year_col,
            y=killed_col,
            markers=True,
            title="Reported Persons Killed"
        )

        fig2.update_layout(
            template="plotly_dark",
            height=420
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

# ============================================================
# 3. STATE ANALYSIS
# ============================================================

elif page == "🗺️ State Analysis":

    st.title("🗺️ State / UT Analysis")

    st.caption(
        "Comparison of reported road accidents across States and UTs in 2019."
    )

    if a2.empty:
        st.error("A2 dataset not found.")
        st.stop()

    state_col = find_column(
        a2,
        ["state/ut", "states/uts", "state"]
    )

    accident_col = find_column(
        a2,
        [
            "state/ut-wise total number of road accidents during 2019",
            "total number of road accidents during 2019"
        ]
    )

    if not state_col or not accident_col:
        st.error("Required A2 columns were not found.")
        st.stop()

    state_df = remove_total_rows(
        a2,
        state_col
    )

    state_df[accident_col] = to_numeric(
        state_df[accident_col]
    )

    state_df = state_df.dropna(
        subset=[accident_col]
    )

    state_df = state_df.sort_values(
        accident_col,
        ascending=False
    )

    top_n = st.slider(
        "Number of States / UTs to display",
        5,
        min(20, len(state_df)),
        min(10, len(state_df))
    )

    chart_df = state_df.head(top_n)

    fig = px.bar(
        chart_df.sort_values(
            accident_col
        ),
        x=accident_col,
        y=state_col,
        orientation="h",
        text=accident_col,
        title=f"Top {top_n} States / UTs by Reported Accidents"
    )

    fig.update_layout(
        template="plotly_dark",
        height=max(450, top_n * 40)
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    display = state_df[
        [state_col, accident_col]
    ].copy()

    display[accident_col] = display[
        accident_col
    ].map(number)

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# 4. ACCIDENT CAUSES
# ============================================================

elif page == "⚠️ Accident Causes":

    st.title("⚠️ Accident Causes")

    st.caption(
        "Reported accident categories associated with selected contributing factors."
    )

    if a47.empty:
        st.error("A47 dataset not found.")
        st.stop()

    location_col = a47.columns[0]

    cause_columns = {
        "Over-Speeding": find_column(
            a47,
            ["over-speeding - number of accidents"]
        ),
        "Drunken Driving": find_column(
            a47,
            ["drunken driving/ consumption of alcohol and drug - number of accidents"]
        ),
        "Jumping Red Light": find_column(
            a47,
            ["jumping red light - number of accidents"]
        ),
        "Mobile Phone": find_column(
            a47,
            ["use of mobile phone - number of accidents"]
        )
    }

    totals = []

    clean = remove_total_rows(
        a47,
        location_col
    )

    for cause, column in cause_columns.items():

        if column:

            value = to_numeric(
                clean[column]
            ).sum()

            totals.append({
                "Cause": cause,
                "Reported Accidents": value
            })

    cause_df = pd.DataFrame(totals)

    if cause_df.empty:
        st.warning("No cause columns found.")
        st.stop()

    fig = px.bar(
        cause_df.sort_values(
            "Reported Accidents"
        ),
        x="Reported Accidents",
        y="Cause",
        orientation="h",
        text="Reported Accidents",
        title="Reported Accident Categories"
    )

    fig.update_layout(
        template="plotly_dark",
        height=430
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "These are reported accident classifications in the source data. "
        "The categories should not automatically be interpreted as mutually "
        "exclusive causal explanations."
    )

    overspeed = cause_columns["Over-Speeding"]

    if overspeed:

        temp = remove_total_rows(
            a47,
            location_col
        ).copy()

        temp[overspeed] = to_numeric(
            temp[overspeed]
        )

        temp = temp.dropna(
            subset=[overspeed]
        ).sort_values(
            overspeed,
            ascending=False
        ).head(10)

        fig2 = px.bar(
            temp.sort_values(overspeed),
            x=overspeed,
            y=location_col,
            orientation="h",
            title="Top Reported Locations — Over-Speeding Category"
        )

        fig2.update_layout(
            template="plotly_dark",
            height=500
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

# ============================================================
# 5. ROAD FEATURES
# ============================================================

elif page == "🛣️ Road Features":

    st.title("🛣️ Road Features")

    st.caption(
        "Reported accident counts across different road characteristics."
    )

    if a48.empty:
        st.error("A48 dataset not found.")
        st.stop()

    location_col = a48.columns[0]

    road_columns = {
        "Straight Road": find_column(
            a48,
            ["straight road - number of accidents"]
        ),
        "Curved Road": find_column(
            a48,
            ["curved road - number of accidents"]
        ),
        "Bridge": find_column(
            a48,
            ["bridge - number of accidents"]
        ),
        "Culvert": find_column(
            a48,
            ["culvert - number of accidents"]
        ),
        "Pot Holes": find_column(
            a48,
            ["pot holes - number of accidents"]
        ),
        "Steep Grade": find_column(
            a48,
            ["steep grade - number of accidents"]
        ),
        "Ongoing Road Works": find_column(
            a48,
            ["ongoing road works/under construction - number of accidents"]
        ),
        "Others": find_column(
            a48,
            ["others - number of accidents"]
        )
    }

    clean = remove_total_rows(
        a48,
        location_col
    )

    totals = []

    for feature, column in road_columns.items():

        if column:

            value = to_numeric(
                clean[column]
            ).sum()

            totals.append({
                "Road Feature": feature,
                "Reported Accidents": value
            })

    road_df = pd.DataFrame(totals)

    fig = px.bar(
        road_df.sort_values(
            "Reported Accidents"
        ),
        x="Reported Accidents",
        y="Road Feature",
        orientation="h",
        text="Reported Accidents",
        title="Reported Accidents by Road Feature"
    )

    fig.update_layout(
        template="plotly_dark",
        height=500
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ============================================================
# 6. TRAFFIC CONTROL
# ============================================================

elif page == "🚦 Traffic Control":

    st.title("🚦 Traffic Control")

    st.caption(
        "Reported accidents across different traffic control categories."
    )

    if a50.empty:
        st.error("A50 dataset not found.")
        st.stop()

    location_col = a50.columns[0]

    control_columns = {
        "Traffic Light Signal": find_column(
            a50,
            ["traffic light signal - number of accidents"]
        ),
        "Police Controlled": find_column(
            a50,
            ["police controlled - number of accidents"]
        ),
        "Stop Sign": find_column(
            a50,
            ["stop sign - number of accidents"]
        ),
        "Flashing Signal / Blinker": find_column(
            a50,
            ["flashing signal/blinker - number of accidents"]
        ),
        "Uncontrolled": find_column(
            a50,
            ["uncontrolled - number of accidents"]
        ),
        "Others": find_column(
            a50,
            ["others - number of accidents"]
        )
    }

    clean = remove_total_rows(
        a50,
        location_col
    )

    totals = []

    for category, column in control_columns.items():

        if column:

            value = to_numeric(
                clean[column]
            ).sum()

            totals.append({
                "Traffic Control": category,
                "Reported Accidents": value
            })

    control_df = pd.DataFrame(totals)

    fig = px.bar(
        control_df.sort_values(
            "Reported Accidents"
        ),
        x="Reported Accidents",
        y="Traffic Control",
        orientation="h",
        text="Reported Accidents",
        title="Reported Accidents by Traffic Control"
    )

    fig.update_layout(
        template="plotly_dark",
        height=500
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ============================================================
# 7. JUNCTION ANALYSIS
# ============================================================

elif page == "🔀 Junction Analysis":

    st.title("🔀 Junction Analysis")

    st.caption(
        "Reported accident counts across different junction types."
    )

    if a49.empty:
        st.error("A49 dataset not found.")
        st.stop()

    location_col = a49.columns[0]

    junction_columns = {
        "T-Junction": find_column(
            a49,
            ["t-junction - number of accidents"]
        ),
        "Y-Junction": find_column(
            a49,
            ["y-junction - number of accidents"]
        ),
        "Four-arm Junction": find_column(
            a49,
            ["four-arm junction - number of accidents"]
        ),
        "Staggered Junction": find_column(
            a49,
            ["staggered junction - number of accidents"]
        ),
        "Roundabout": find_column(
            a49,
            ["roundabout - number of accidents"]
        ),
        "Others": find_column(
            a49,
            ["others - number of accidents"]
        )
    }

    clean = remove_total_rows(
        a49,
        location_col
    )

    totals = []

    for junction, column in junction_columns.items():

        if column:

            value = to_numeric(
                clean[column]
            ).sum()

            totals.append({
                "Junction Type": junction,
                "Reported Accidents": value
            })

    junction_df = pd.DataFrame(totals)

    fig = px.bar(
        junction_df.sort_values(
            "Reported Accidents"
        ),
        x="Reported Accidents",
        y="Junction Type",
        orientation="h",
        text="Reported Accidents",
        title="Reported Accidents by Junction Type"
    )

    fig.update_layout(
        template="plotly_dark",
        height=500
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ============================================================
# 8. BLACK SPOTS
# ============================================================

elif page == "⚫ Black Spots":

    st.title("⚫ Black Spots")

    st.caption(
        "Locations reported as black spots on National Highways."
    )

    if a51.empty:
        st.error("A51 dataset not found.")
        st.stop()

    st.dataframe(
        a51,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "The dataset presents black-spot locations reported by "
        "States/UTs. This dashboard does not independently redefine "
        "the source classification."
    )

# ============================================================
# 9. HIGH ACCIDENT LOCATIONS
# ============================================================

elif page == "📍 High Accident Locations":

    st.title("📍 High Accident Locations")

    st.caption(
        "Location-level accident and fatality information."
    )

    if a52.empty:
        st.error("A52 dataset not found.")
        st.stop()

    search = st.text_input(
        "Search State, District, Police Station or Location",
        ""
    )

    filtered = a52.copy()

    if search.strip():

        mask = pd.Series(
            False,
            index=filtered.index
        )

        for column in filtered.columns:

            mask = (
                mask |
                filtered[column]
                .astype(str)
                .str.contains(
                    search,
                    case=False,
                    na=False
                )
            )

        filtered = filtered[mask]

    st.write(
        f"Records found: **{len(filtered):,}**"
    )

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# 10. DATA & METHODOLOGY
# ============================================================

elif page == "📚 Data & Methodology":

    st.title("📚 Data & Methodology")

    st.caption(
        "Datasets, preprocessing, analysis workflow and ML methodology."
    )

    st.subheader("Dataset Overview")

    overview_df = pd.DataFrame({
        "Dataset": [
            "A1",
            "A2",
            "A47",
            "A48",
            "A49",
            "A50",
            "A51",
            "A52"
        ],
        "Purpose": [
            "India accident trend",
            "State / UT accident statistics",
            "Accident contributing factors",
            "Road characteristics",
            "Junction types",
            "Traffic control",
            "Black spots",
            "High accident locations"
        ],
        "Rows": [
            len(a1),
            len(a2),
            len(a47),
            len(a48),
            len(a49),
            len(a50),
            len(a51),
            len(a52)
        ]
    })

    st.dataframe(
        overview_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Machine Learning Features")

    feature_df = pd.DataFrame({
        "Feature": [
            "Accidents_2016",
            "Accidents_2017",
            "Accidents_2018",
            "Average_Accidents_2016_2018",
            "Accident_Change_2018_vs_2016",
            "Accident_Growth_2018_vs_2016"
        ],
        "Description": [
            "Reported accidents during 2016",
            "Reported accidents during 2017",
            "Reported accidents during 2018",
            "Average accidents during 2016–2018",
            "2018 accident count minus 2016 accident count",
            "Percentage growth from 2016 to 2018"
        ]
    })

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Workflow")

    st.write(
        "1. Collect official road accident datasets."
    )

    st.write(
        "2. Clean column names, text values and numerical fields."
    )

    st.write(
        "3. Identify missing values and duplicate records."
    )

    st.write(
        "4. Perform exploratory data analysis."
    )

    st.write(
        "5. Create historical ML features from 2016–2018 accident counts."
    )

    st.write(
        "6. Train the existing Linear Regression model."
    )

    st.write(
        "7. Use the trained model to estimate 2019 State/UT accident counts."
    )

    st.subheader("Important Scope")

    st.warning(
        "The current project is based mainly on road accident and road "
        "safety data. It does not currently provide real-time traffic "
        "speed, density, congestion or live vehicle-flow prediction."
    )

    st.warning(
        "The ML dataset contains a relatively small number of State/UT "
        "observations. Therefore, the model results should be interpreted "
        "within the scope of this project dataset."
    )

# ============================================================
# 11. OVERVIEW — LAST
# ============================================================

elif page == "🏠 Overview":

    latest = get_latest_values(a1)

    st.title("🚦 Indian Urban Mobility Intelligence")

    st.caption(
        "Interactive analysis of Indian road accident patterns, "
        "contributing factors, road characteristics, traffic control "
        "systems, junction types, high-accident locations and "
        "state-level machine learning prediction."
    )

    st.divider()

    st.subheader("National Road Safety Snapshot")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Total Road Accidents",
            number(latest["accidents"])
        )

    with c2:
        st.metric(
            "Persons Killed",
            number(latest["killed"])
        )

    with c3:
        st.metric(
            "Persons Injured",
            number(latest["injured"])
        )

    with c4:
        st.metric(
            "States / UTs",
            "37"
        )

    if latest["year"]:
        st.caption(
            f"National figures shown above are from {latest['year']} "
            "in the A1 dataset."
        )

    st.divider()

    st.subheader("Project Modules")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.info(
            "📈 Accident Trends\n\n"
            "Historical accident patterns across India."
        )

    with m2:
        st.info(
            "🗺️ State Analysis\n\n"
            "State and Union Territory comparisons."
        )

    with m3:
        st.info(
            "⚠️ Accident Causes\n\n"
            "Reported contributing-factor categories."
        )

    m4, m5, m6 = st.columns(3)

    with m4:
        st.info(
            "🛣️ Road Features\n\n"
            "Road characteristics associated with reported accidents."
        )

    with m5:
        st.info(
            "🚦 Traffic Control\n\n"
            "Reported accidents by traffic-control category."
        )

    with m6:
        st.info(
            "🤖 ML Prediction\n\n"
            "State-level 2019 accident estimation."
        )

    st.divider()

    # --------------------------------------------------------
    # LONG TERM TREND
    # --------------------------------------------------------

    if not a1.empty:

        year_col = find_column(
            a1,
            ["years", "year"]
        )

        accident_col = find_column(
            a1,
            ["total number of road accidents"]
        )

        if year_col and accident_col:

            trend = a1.copy()

            trend[year_col] = to_numeric(
                trend[year_col]
            )

            trend[accident_col] = to_numeric(
                trend[accident_col]
            )

            trend = trend.dropna(
                subset=[
                    year_col,
                    accident_col
                ]
            ).sort_values(
                year_col
            )

            fig = px.line(
                trend,
                x=year_col,
                y=accident_col,
                markers=True,
                title="India Road Accident Trend"
            )

            fig.update_layout(
                template="plotly_dark",
                height=430,
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.divider()

    st.subheader("About This Project")

    st.write(
        "Indian Urban Mobility Intelligence is a data-driven road "
        "safety analytics project built using official Government of "
        "India road accident datasets. The dashboard combines "
        "exploratory data analysis, State/UT comparisons, accident "
        "factor analysis, road and junction analysis, and a "
        "state-level machine learning model."
    )

    st.caption(
        "Data source: Government of India Open Government Data Platform — data.gov.in"
    )