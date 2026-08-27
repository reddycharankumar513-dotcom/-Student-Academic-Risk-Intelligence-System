"""
Student Academic Risk Intelligence System
File: main.py (FastAPI REST API)
"""

import os
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import uvicorn

# Create the FastAPI application instance with metadata
app = FastAPI(
    title="Student Academic Risk Intelligence System API",
    description="API for analyzing student performance data",
    version="1.0.0",
)


def load_data(filepath=None):
    """
    Load the Maths CSV and apply the same feature engineering used in
    analysis.py.

    Parameters
    ----------
    filepath : str or os.PathLike, optional
        Path to the dataset. If omitted, ``MATHS_DATASET_PATH`` is used,
        falling back to the supplied local dataset path.

    Returns
    -------
    pd.DataFrame
        The prepared DataFrame.
    """
    dataset_path = Path(
        filepath
        or os.getenv("MATHS_DATASET_PATH", r"C:\Users\chara\Downloads\Maths (1).csv")
    )
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Maths dataset not found at '{dataset_path}'. "
            "Set MATHS_DATASET_PATH to the CSV location."
        )

    df = pd.read_csv(dataset_path)

    # Result: classify students based on final grade (G3)
    # G3 = 0 -> Dropout (not a zero score), 1-9 -> Fail, 10-20 -> Pass
    def classify_result(g3):
        if g3 == 0:
            return "Dropout"
        elif 1 <= g3 <= 9:
            return "Fail"
        else:
            return "Pass"

    df["Result"] = df["G3"].apply(classify_result)

    # Percentage of final grade out of 20
    df["Percentage"] = (df["G3"] / 20) * 100

    # Average alcohol consumption (weekday + weekend)
    df["avg_alcohol"] = (df["Dalc"] + df["Walc"]) / 2

    # Average parental education level
    df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2

    # Grade trend from G1 to G3
    df["grade_trend"] = df["G3"] - df["G1"]

    # Count of "yes" answers across support-related columns
    support_cols = ["schoolsup", "famsup", "paid"]
    df["total_support"] = df[support_cols].apply(
        lambda row: sum(1 for val in row if val == "yes"), axis=1
    )

    # Composite risk score
    df["risk_score"] = (
        (df["failures"] * 2)
        + (df["absences"] / 10)
        + df["avg_alcohol"]
        - df["studytime"]
    )

    # Average of G1 and G2
    df["g1_g2_avg"] = (df["G1"] + df["G2"]) / 2

    return df


# Load the data once at startup and keep it in memory for all requests
df = load_data()


# ---------------------------------------------------------------------------
# GET Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """Root endpoint with basic API information."""
    return {
        "message": "Student Academic Risk Intelligence System API",
        "docs": "Visit /docs for full API documentation",
        "version": "1.0.0",
    }


@app.get("/summary")
def get_summary():
    """
    Return high-level summary statistics for the whole dataset.
    Class average and pass rate are calculated on non-dropout students only.
    """
    non_dropout = df[df["G3"] != 0]

    total_students = len(df)
    class_average_g3 = round(float(non_dropout["G3"].mean()), 2) if len(non_dropout) else 0.0
    pass_count = int((non_dropout["G3"] >= 10).sum())
    pass_rate_percent = (
        round((pass_count / len(non_dropout)) * 100, 2) if len(non_dropout) else 0.0
    )
    at_risk_count = int(((df["G3"] >= 1) & (df["G3"] <= 9)).sum())
    dropout_count = int((df["G3"] == 0).sum())

    return {
        "total_students": total_students,
        "class_average_g3": class_average_g3,
        "pass_rate_percent": pass_rate_percent,
        "at_risk_count": at_risk_count,
        "dropout_count": dropout_count,
    }


@app.get("/at-risk")
def get_at_risk_students():
    """
    Return students at risk (G3 between 1 and 9 inclusive),
    sorted by G3 ascending (worst performing first).
    """
    at_risk_df = df[(df["G3"] >= 1) & (df["G3"] <= 9)].sort_values("G3", ascending=True)

    results = [
        {
            "student_index": int(idx),
            "G1": float(row["G1"]),
            "G2": float(row["G2"]),
            "G3": float(row["G3"]),
            "absences": int(row["absences"]),
        }
        for idx, row in at_risk_df.iterrows()
    ]
    return results


@app.get("/top-students")
def get_top_students():
    """
    Return the top 5 students by G3 (excluding dropouts),
    sorted by G3 descending.
    """
    non_dropout = df[df["G3"] != 0]
    top5 = non_dropout.sort_values("G3", ascending=False).head(5)

    results = [
        {
            "student_index": int(idx),
            "G1": float(row["G1"]),
            "G2": float(row["G2"]),
            "G3": float(row["G3"]),
        }
        for idx, row in top5.iterrows()
    ]
    return results


# ---------------------------------------------------------------------------
# POST Endpoint with Pydantic validation
# ---------------------------------------------------------------------------

class StudentInput(BaseModel):
    G1: float = Field(..., ge=0, le=20, description="First period grade must be between 0 and 20")
    G2: float = Field(..., ge=0, le=20, description="Second period grade must be between 0 and 20")
    studytime: int = Field(..., ge=1, le=4, description="Study time must be between 1 and 4")
    absences: int = Field(..., ge=0, le=100, description="Absences must be between 0 and 100")
    failures: int = Field(..., ge=0, le=4, description="Number of past failures must be between 0 and 4")


@app.post("/predict-result")
def predict_result(student: StudentInput):
    """
    Estimate a student's final grade (G3) and predict their outcome
    based on a simple weighted formula.
    """
    # Calculate estimated final grade using the weighted formula
    estimated_g3 = (
        (student.G1 * 0.3)
        + (student.G2 * 0.6)
        + (student.studytime * 0.3)
        - (student.failures * 1.5)
        - (student.absences * 0.05)
    )

    # Clamp the estimate to a valid grade range (0-20)
    estimated_g3 = max(0.0, min(20.0, estimated_g3))

    # Determine the prediction category
    if estimated_g3 == 0:
        prediction = "Dropout Risk"
    elif estimated_g3 < 10:
        prediction = "Fail"
    else:
        prediction = "Pass"

    # Determine confidence level based on G1/G2 consistency
    if student.G1 > 12 and student.G2 > 12:
        confidence = "High"
    elif student.G1 < 8 and student.G2 < 8:
        confidence = "High"
    else:
        confidence = "Medium"

    return {
        "estimated_g3": round(estimated_g3, 2),
        "prediction": prediction,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Uvicorn runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
