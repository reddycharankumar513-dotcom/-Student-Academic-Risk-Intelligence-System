"""
Student Academic Risk Intelligence System
File: app.py (Streamlit Dashboard)
"""

import os
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

# --- Page configuration ---
st.set_page_config(
    page_title="Student Academic Risk Intelligence System",
    layout="wide",
    page_icon="🎓",
)


@st.cache_data
def load_and_prepare_data(filepath):
    """Load Maths.csv and apply the same feature engineering as analysis.py."""
    df = pd.read_csv(filepath)

    # G3 = 0 -> Dropout (not a zero score), 1-9 -> Fail, 10-20 -> Pass
    def classify_result(g3):
        if g3 == 0:
            return "Dropout"
        elif 1 <= g3 <= 9:
            return "Fail"
        else:
            return "Pass"

    df["Result"] = df["G3"].apply(classify_result)
    df["Percentage"] = (df["G3"] / 20) * 100
    df["avg_alcohol"] = (df["Dalc"] + df["Walc"]) / 2
    df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2
    df["grade_trend"] = df["G3"] - df["G1"]

    support_cols = ["schoolsup", "famsup", "paid"]
    df["total_support"] = df[support_cols].apply(
        lambda row: sum(1 for val in row if val == "yes"), axis=1
    )

    df["risk_score"] = (
        (df["failures"] * 2)
        + (df["absences"] / 10)
        + df["avg_alcohol"]
        - df["studytime"]
    )

    df["g1_g2_avg"] = (df["G1"] + df["G2"]) / 2

    return df


# --- Load data ---
data_path = Path(
    os.getenv("MATHS_DATASET_PATH", r"C:\Users\chara\Downloads\Maths (1).csv")
)
if not data_path.is_file():
    st.error(
        f"Maths dataset not found at '{data_path}'. "
        "Set MATHS_DATASET_PATH to the CSV location."
    )
    st.stop()

df = load_and_prepare_data(data_path)

# --- Main title ---
st.title("🎓 Student Academic Risk Intelligence System")

# --- KPI Metric Cards (4 in one row) ---
non_dropout = df[df["G3"] != 0]

total_students = len(df)
class_avg_g3 = round(non_dropout["G3"].mean(), 2) if len(non_dropout) else 0.0
pass_count = (non_dropout["G3"] >= 10).sum()
pass_rate = round((pass_count / len(non_dropout)) * 100, 1) if len(non_dropout) else 0.0
at_risk_count = ((df["G3"] >= 1) & (df["G3"] <= 9)).sum()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Students", total_students)
kpi2.metric("Class Average G3", class_avg_g3)
kpi3.metric("Pass Rate %", f"{pass_rate}%")
kpi4.metric("At-Risk Count", at_risk_count)


# --- Performance Charts (side by side) ---
st.subheader("📊 Performance Charts")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    color_map = {"Pass": "green", "Fail": "red", "Dropout": "grey"}
    fig_scatter = px.scatter(
        df,
        x="studytime",
        y="G3",
        color="Result",
        color_discrete_map=color_map,
        hover_data=["absences", "G1", "G2"],
        title="Study Time vs Final Grade",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with chart_col2:
    avg_g3_by_internet = df.groupby("internet", as_index=False)["G3"].mean()
    fig_bar = px.bar(
        avg_g3_by_internet,
        x="internet",
        y="G3",
        title="Average G3 by Internet Access",
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# --- Student Analysis Table with filter ---
st.subheader("🚨 Student Analysis Table")

result_filter = st.selectbox(
    "Filter by Result",
    options=["All", "Pass", "Fail", "Dropout"],
)

if result_filter == "All":
    filtered_df = df
else:
    filtered_df = df[df["Result"] == result_filter]

display_cols = [
    "G1", "G2", "G3", "Result", "Percentage",
    "absences", "studytime", "failures", "risk_score",
]
st.dataframe(filtered_df[display_cols])

# --- At-Risk Students section ---
st.subheader("⚠️ At-Risk Students")

at_risk_df = df[(df["G3"] >= 1) & (df["G3"] <= 9)].sort_values("G3", ascending=True)
at_risk_display_cols = ["G1", "G2", "G3", "absences", "studytime", "failures"]
st.dataframe(at_risk_df[at_risk_display_cols])

st.write(f"Total at-risk students: {len(at_risk_df)}")
