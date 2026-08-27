"""
Student Academic Risk Intelligence System
File: analysis.py

Loads student performance data, engineers risk-related features,
computes summary statistics, and generates static (Matplotlib) and
interactive (Plotly) visualizations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


def load_and_prepare_data(filepath):
    """
    Load the UCI Student Performance CSV and engineer risk-related
    features used throughout the analysis.

    Parameters
    ----------
    filepath : str
        Path to the Maths.csv file.

    Returns
    -------
    pd.DataFrame
        The original data plus engineered columns.
    """
    # Load the raw CSV into a DataFrame
    df = pd.read_csv(filepath)

    # --- Result: categorize students based on final grade (G3) ---
    # G3 = 0 means the student dropped out, NOT that they scored zero.
    # G3 between 1 and 9 (inclusive) means the student failed.
    # G3 between 10 and 20 (inclusive) means the student passed.
    def classify_result(g3):
        if g3 == 0:
            return "Dropout"
        elif 1 <= g3 <= 9:
            return "Fail"
        else:
            return "Pass"

    df["Result"] = df["G3"].apply(classify_result)

    # --- Percentage: convert G3 (out of 20) to a percentage ---
    df["Percentage"] = (df["G3"] / 20) * 100

    # --- avg_alcohol: average of workday (Dalc) and weekend (Walc) alcohol use ---
    df["avg_alcohol"] = (df["Dalc"] + df["Walc"]) / 2

    # --- parent_edu_avg: average education level of mother (Medu) and father (Fedu) ---
    df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2

    # --- grade_trend: change in performance from first period (G1) to final (G3) ---
    df["grade_trend"] = df["G3"] - df["G1"]

    # --- total_support: count of "yes" answers across support-related columns ---
    support_cols = ["schoolsup", "famsup", "paid"]
    df["total_support"] = df[support_cols].apply(
        lambda row: sum(1 for val in row if val == "yes"), axis=1
    )

    # --- risk_score: composite score combining failures, absences,
    #     alcohol use, and study time into a single risk indicator ---
    df["risk_score"] = (
        (df["failures"] * 2)
        + (df["absences"] / 10)
        + df["avg_alcohol"]
        - df["studytime"]
    )

    # --- g1_g2_avg: average of first and second period grades ---
    df["g1_g2_avg"] = (df["G1"] + df["G2"]) / 2

    # Note: the UCI Student Performance dataset has no missing values,
    # so no null-handling logic is required here.

    return df


def calculate_statistics(df):
    """
    Compute high-level NumPy-based statistics from the prepared DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by load_and_prepare_data().

    Returns
    -------
    dict
        Dictionary of summary statistics.
    """
    # Exclude dropouts (G3 == 0) for grade-based statistics, since a
    # dropout's G3 of 0 is not a genuine academic score.
    non_dropout = df[df["G3"] != 0]

    # 1. class_avg_g3: mean G3 among non-dropout students
    class_avg_g3 = float(np.mean(non_dropout["G3"].values)) if len(non_dropout) else 0.0

    # 2. pass_rate: % of non-dropout students who passed (G3 >= 10)
    pass_count = int(np.sum(non_dropout["G3"].values >= 10))
    pass_rate = (pass_count / len(non_dropout)) * 100 if len(non_dropout) else 0.0

    # 3. dropout_count: number of students with G3 == 0
    dropout_count = int(np.sum(df["G3"].values == 0))

    # 4. at_risk_count: number of students with G3 between 1 and 9 inclusive
    at_risk_count = int(np.sum((df["G3"].values >= 1) & (df["G3"].values <= 9)))

    # 5. correlation_matrix: correlation between G1, G2, G3 (non-dropouts only)
    correlation_matrix = np.corrcoef(
        [non_dropout["G1"].values, non_dropout["G2"].values, non_dropout["G3"].values]
    )

    stats = {
        "total_students": len(df),
        "class_avg_g3": class_avg_g3,
        "pass_rate": pass_rate,
        "dropout_count": dropout_count,
        "at_risk_count": at_risk_count,
        "correlation_matrix": correlation_matrix,
    }

    return stats


def generate_static_charts(df):
    """
    Create and save two static Matplotlib charts:
    1. Bar chart of average G3 by study time level.
    2. Pie chart of Pass/Fail/Dropout distribution.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by load_and_prepare_data().
    """
    # Ensure the output folder exists before saving any charts
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # --- Chart 1: Bar chart - Average G3 by Study Time ---
    avg_g3_by_studytime = df.groupby("studytime")["G3"].mean().sort_index()

    plt.figure(figsize=(8, 6))
    plt.bar(avg_g3_by_studytime.index.astype(str), avg_g3_by_studytime.values, color="steelblue")
    plt.title("Average G3 by Study Time")
    plt.xlabel("Study Time (1=<2hrs, 2=2-5hrs, 3=5-10hrs, 4=>10hrs)")
    plt.ylabel("Average G3")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avg_g3_by_studytime.png"))
    plt.close()  # Close the figure to free memory / avoid overlapping plots

    # --- Chart 2: Pie chart - Result distribution ---
    result_counts = df["Result"].value_counts()

    plt.figure(figsize=(8, 6))
    plt.pie(
        result_counts.values,
        labels=result_counts.index,
        autopct="%1.1f%%",  # Show percentages on each slice
        colors=["#4CAF50", "#F44336", "#9E9E9E"][: len(result_counts)],
    )
    plt.title("Student Result Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pass_fail_dropout_pie.png"))
    plt.close()


def generate_interactive_charts(df):
    """
    Create two interactive Plotly charts:
    1. Scatter plot of study time vs. final grade (G3), colored by Result.
    2. Bar chart of average G3 by internet access.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by load_and_prepare_data().
    """
    # --- Chart 1: Scatter plot - Study Time vs Final Grade ---
    color_map = {"Pass": "green", "Fail": "red", "Dropout": "grey"}

    fig1 = px.scatter(
        df,
        x="studytime",
        y="G3",
        color="Result",
        color_discrete_map=color_map,
        hover_data=["absences", "G1", "G2"],
        title="Study Time vs Final Grade (G3)",
    )
    fig1.show()

    # --- Chart 2: Bar chart - Average G3 by Internet Access ---
    avg_g3_by_internet = df.groupby("internet", as_index=False)["G3"].mean()

    fig2 = px.bar(
        avg_g3_by_internet,
        x="internet",
        y="G3",
        color="internet",
        title="Average G3 by Internet Access",
    )
    fig2.show()


def print_summary(stats):
    """
    Print a clean, formatted summary of the analysis statistics.

    Parameters
    ----------
    stats : dict
        Dictionary returned by calculate_statistics().
    """
    print("=" * 48)
    print("STUDENT ACADEMIC RISK INTELLIGENCE SYSTEM")
    print("ANALYSIS SUMMARY")
    print("=" * 48)
    print(f"Total Students        : {stats['total_students']}")
    print(f"Class Average G3      : {stats['class_avg_g3']:.2f}")
    print(f"Pass Rate             : {stats['pass_rate']:.2f}%")
    print(f"At-Risk Count         : {stats['at_risk_count']}")
    print(f"Dropout Count         : {stats['dropout_count']}")
    print("=" * 48)


if __name__ == "__main__":
    # 1. Load and prepare the data
    df = load_and_prepare_data("data/Maths.csv")

    # 2. Calculate summary statistics
    stats = calculate_statistics(df)

    # 3. Generate and save static charts
    generate_static_charts(df)

    # 4. Generate and display interactive charts
    generate_interactive_charts(df)

    # 5. Print the summary table
    print_summary(stats)

    print("Analysis complete. Charts saved to output/ folder")
