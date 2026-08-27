"""
ICT233 - Assignment Solution
Dataset: public-covid-19-cases-canada.csv
https://github.com/LiuFang00/ICT233/blob/master/public-covid-19-cases-canada.csv

Run top to bottom (e.g. as cells in Jupyter, or `python covid_solution.py`).
Requires: pandas, sqlalchemy, matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)


# ============================================================
# QUESTION 1 - EXTRACT, LOAD, TRANSFORM (Python + Pandas)
# ============================================================

# ---------- Q1(a)(i) Load the .csv file ----------
RAW_URL = "https://raw.githubusercontent.com/LiuFang00/ICT233/master/public-covid-19-cases-canada.csv"
df = pd.read_csv(RAW_URL)          # can also read from a local copy: pd.read_csv("public-covid-19-cases-canada.csv")
print("Q1(a)(i) - Loaded shape:", df.shape)
print(df.head(), "\n")


# ---------- Q1(a)(ii) Summarize the dataset ----------
print("=== Q1(a)(ii) Dataset summary ===")
print("Rows x Columns:", df.shape)
print("\nColumn dtypes:\n", df.dtypes)
print("\nColumn-by-column profile:")
for col in df.columns:
    n_unique = df[col].nunique(dropna=True)
    n_null = df[col].isna().sum()
    print(f"  - {col:20s} unique={n_unique:6d}  nulls={n_null:6d}")

print("\nKey feature value ranges:")
print("  age categories        :", sorted(df["age"].dropna().unique()))
print("  sex categories         :", df["sex"].unique())
print("  province categories    :", df["province"].unique())
print("  has_travel_history     :", df["has_travel_history"].unique())
print("  locally_acquired       :", df["locally_acquired"].unique())
print("  date_report range      :", df["date_report"].min(), "to", df["date_report"].max())
print("  report_week range      :", df["report_week"].min(), "to", df["report_week"].max())

print("\nCounts of 'useless' / placeholder values discovered:")
print("  age == 'Not Reported'          :", (df["age"] == "Not Reported").sum())
print("  sex == 'Not Reported'          :", (df["sex"] == "Not Reported").sum())
print("  case_id all null?              :", df["case_id"].isna().all())
print("  has_travel_history nulls       :", df["has_travel_history"].isna().sum())
print("  locally_acquired nulls         :", df["locally_acquired"].isna().sum())

# Findings written up as comments (also required in a short report):
#   - case_id: 100% NaN -> useless column, safe to drop entirely.
#   - age: ~97% "Not Reported"; the remaining values are a mix of clean bins
#     ("50-59"), truncated bins ("<18", "<10", "<20", "<1") and raw single
#     ages ("2", "50", "61") that need to be re-binned.
#   - sex: mostly Male/Female, small number of "Not Reported" -> those rows
#     should be dropped since sex cannot be inferred.
#   - has_travel_history / locally_acquired: mostly missing, but still useful
#     for Q2/Q3(b)(iii) -> keep the column, treat NaN as "unknown"/"f".
#   - case_source: free-text URLs, not analytically useful -> drop.


# ---------- Q1(a)(iii) Two potential insights ----------
print("\n=== Q1(a)(iii) Potential insights ===")
print("""
1) Provincial concentration of cases: grouping by `province` shows the vast
   majority of confirmed cases are concentrated in Quebec and Ontario, which
   suggests the outbreak trajectory (and any resource allocation) was highly
   uneven across Canada rather than spread proportionally by population.

2) Reporting-time trend: grouping cases by `date_report`/`report_week` shows
   case counts were low in Jan-Feb 2020 and rose sharply through March into
   April 2020, consistent with the documented onset of Canada's first pandemic
   wave and the timing of related public-health interventions.
""")


# ============================================================
# Q1(b) DATA PRE-PROCESSING
# ============================================================

# ---------- Q1(b)(i) Remove useless values ----------
clean = df.copy()

# Drop columns that are useless for every row (100% missing / non-analytic free text)
clean = clean.drop(columns=["case_id", "case_source"])

# Drop rows where sex is not usable
clean = clean[clean["sex"] != "Not Reported"]

# locally_acquired / has_travel_history: keep column, standardise missing -> 'Unknown'
clean["has_travel_history"] = clean["has_travel_history"].fillna("Unknown")
clean["locally_acquired"] = clean["locally_acquired"].fillna("Unknown")
# normalise inconsistent capitalisation, e.g. "Close Contact" vs "close contact"
clean["locally_acquired"] = clean["locally_acquired"].str.title()

# age == 'Not Reported' cannot be interpolated -> drop those rows (per Q1(b) instructions)
clean = clean[clean["age"] != "Not Reported"]

print("Q1(b)(i) - shape after removing useless rows/columns:", clean.shape)


# ---------- Q1(b)(ii) Reformat age into standard groups ----------
def normalise_age_group(value: str) -> str:
    """Map any raw age value to one of {'0-19','20-29',...,'90-99'}."""
    value = str(value).strip()

    # already a clean 10-year bin, e.g. "50-59"
    if "-" in value:
        return value

    # truncated / less-than style values, e.g. "<18", "<10", "<1", "<20"
    if value.startswith("<"):
        n = int(value[1:])
        return "0-19" if n <= 20 else f"{(n // 10) * 10}-{(n // 10) * 10 + 9}"

    # a raw single age, e.g. "2", "50", "61"
    n = int(value)
    lower = (n // 10) * 10
    return f"{lower}-{lower + 9}"


clean["age"] = clean["age"].apply(normalise_age_group)
print("Q1(b)(ii) - age groups after reformatting:", sorted(clean["age"].unique()))


# ---------- Q1(b)(iii) Total infected persons per age group ----------
age_group_counts = clean["age"].value_counts().sort_index()
print("\nQ1(b)(iii) - infections per age group:")
print(age_group_counts)


# ---------- Q1(c) Save cleaned dataset ----------
clean.to_csv("cleaned_covid_canada.csv", index=False)
print("\nQ1(c) - cleaned dataset saved to cleaned_covid_canada.csv, shape:", clean.shape)


# ============================================================
# QUESTION 2 - LOAD STEP: ORM + DATABASE
# ============================================================
from sqlalchemy import create_engine, Column, Integer, String, Date
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import func, extract

Base = declarative_base()


# ---------- Q2(a) ORM table class + load ----------
class CovidCase(Base):
    """One row = one confirmed COVID-19 case record."""
    __tablename__ = "covid_cases"

    id = Column(Integer, primary_key=True, autoincrement=True)   # surrogate key
    provincial_case_id = Column(Integer)
    age_group = Column(String(10))
    sex = Column(String(20))
    health_region = Column(String(100))
    province = Column(String(50))
    country = Column(String(50))
    date_report = Column(Date)
    report_week = Column(Date)
    has_travel_history = Column(String(10))
    locally_acquired = Column(String(30))


engine = create_engine("sqlite:///covid_canada.db", echo=False)
Base.metadata.drop_all(engine)   # clean slate on re-run
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

load_df = pd.read_csv("cleaned_covid_canada.csv", parse_dates=["date_report", "report_week"])

records = [
    CovidCase(
        provincial_case_id=int(r.provincial_case_id),
        age_group=r.age,
        sex=r.sex,
        health_region=r.health_region,
        province=r.province,
        country=r.country,
        date_report=r.date_report.date(),
        report_week=r.report_week.date(),
        has_travel_history=r.has_travel_history,
        locally_acquired=r.locally_acquired,
    )
    for r in load_df.itertuples(index=False)
]
session.bulk_save_objects(records)
session.commit()
print("Q2(a) - rows loaded into SQLite via ORM:", session.query(CovidCase).count())


# ---------- Q2(b)(i) Male/female infectors per month ----------
monthly_sex = (
    session.query(
        extract("year", CovidCase.date_report).label("year"),
        extract("month", CovidCase.date_report).label("month"),
        CovidCase.sex,
        func.count(CovidCase.id).label("total"),
    )
    .group_by("year", "month", CovidCase.sex)
    .order_by("year", "month")
    .all()
)
print("\nQ2(b)(i) - male/female infectors per month:")
for row in monthly_sex:
    print(f"  {int(row.year)}-{int(row.month):02d}  {row.sex:8s} {row.total}")


# ---------- Q2(b)(ii) Age groups sorted by female infectors, descending ----------
female_by_age = (
    session.query(CovidCase.age_group, func.count(CovidCase.id).label("total"))
    .filter(CovidCase.sex == "Female")
    .group_by(CovidCase.age_group)
    .order_by(func.count(CovidCase.id).desc())
    .all()
)
print("\nQ2(b)(ii) - age groups ranked by female infectors:")
for row in female_by_age:
    print(f"  {row.age_group:8s} {row.total}")


# ---------- Q2(b)(iii) No travel history: top 2 months for age > 50 ----------
# "older than 50" = age groups whose lower bound is >= 50, i.e. 50-59 upward
older_than_50 = ["50-59", "60-69", "70-79", "80-89", "90-99"]

no_travel_top_months = (
    session.query(
        extract("year", CovidCase.date_report).label("year"),
        extract("month", CovidCase.date_report).label("month"),
        func.count(CovidCase.id).label("total"),
    )
    .filter(CovidCase.has_travel_history.in_(["f", "Unknown"]))
    .filter(CovidCase.age_group.in_(older_than_50))
    .group_by("year", "month")
    .order_by(func.count(CovidCase.id).desc())
    .limit(2)
    .all()
)
print("\nQ2(b)(iii) - top 2 months, no travel history, age > 50:")
for row in no_travel_top_months:
    print(f"  {int(row.year)}-{int(row.month):02d}  total={row.total}")


# ============================================================
# QUESTION 3 - EDA WITH PANDAS + VISUALIZATION
# ============================================================

# ---------- Q3(a) Same 3 queries as Q2(b), but with Pandas ----------
eda = pd.read_csv("cleaned_covid_canada.csv", parse_dates=["date_report", "report_week"])
eda["year_month"] = eda["date_report"].dt.to_period("M")

# (i) male/female infectors per month
q3_i = eda.groupby(["year_month", "sex"]).size().unstack(fill_value=0)
print("\nQ3(a)(i) - male/female infectors per month (pandas):")
print(q3_i)

# (ii) age groups by female infectors, descending
q3_ii = (
    eda[eda["sex"] == "Female"]
    .groupby("age")
    .size()
    .sort_values(ascending=False)
)
print("\nQ3(a)(ii) - age groups ranked by female infectors (pandas):")
print(q3_ii)

# (iii) no travel history, top 2 months for age > 50
no_travel_mask = eda["has_travel_history"].isin(["f", "Unknown"])
older_mask = eda["age"].isin(older_than_50)
q3_iii = (
    eda[no_travel_mask & older_mask]
    .groupby("year_month")
    .size()
    .sort_values(ascending=False)
    .head(2)
)
print("\nQ3(a)(iii) - top 2 months, no travel history, age > 50 (pandas):")
print(q3_iii)


# ---------- Q3(b)(i) Top 3 provinces per month ----------
def top3_provinces_per_month(data: pd.DataFrame) -> pd.DataFrame:
    """Return top-3 provinces by case count for every calendar month present."""
    counts = data.groupby(["year_month", "province"]).size().reset_index(name="cases")
    return (
        counts.sort_values(["year_month", "cases"], ascending=[True, False])
        .groupby("year_month")
        .head(3)
        .reset_index(drop=True)
    )


top3 = top3_provinces_per_month(eda)
print("\nQ3(b)(i) - top 3 provinces per month:")
print(top3)


# ---------- Q3(b)(ii) Figure: total cases per province ----------
province_totals = eda["province"].value_counts()
plt.figure(figsize=(10, 5))
province_totals.plot(kind="bar", color="#2c7fb8")
plt.title("Total COVID-19 Cases by Province")
plt.xlabel("Province")
plt.ylabel("Number of Cases")
plt.tight_layout()
plt.savefig("q3_b_ii_cases_by_province.png", dpi=150)
plt.close()
print("\nQ3(b)(ii) - figure saved: q3_b_ii_cases_by_province.png")


# ---------- Q3(b)(iii) Figure: age distribution by gender for top province ----------
top_province = province_totals.idxmax()
sub = eda[eda["province"] == top_province]
age_gender = sub.groupby(["age", "sex"]).size().unstack(fill_value=0)
age_order = sorted(age_gender.index, key=lambda x: int(x.split("-")[0]))
age_gender = age_gender.loc[age_order]

age_gender.plot(kind="bar", figsize=(10, 5))
plt.title(f"Age Distribution by Gender - {top_province} (highest case count)")
plt.xlabel("Age Group")
plt.ylabel("Number of Cases")
plt.legend(title="Sex")
plt.tight_layout()
plt.savefig("q3_b_iii_age_gender_top_province.png", dpi=150)
plt.close()
print(f"Q3(b)(iii) - figure saved: q3_b_iii_age_gender_top_province.png (province={top_province})")


# ---------- Q3(c)(i) Function: day of week for a date ----------
def get_day_of_week(date_value) -> str:
    """Return abbreviated weekday name (Mon, Tue, ...) for a date/date-string."""
    if not isinstance(date_value, (pd.Timestamp, datetime)):
        date_value = pd.to_datetime(date_value)
    return date_value.strftime("%a")


eda["report_week"] = eda["date_report"].apply(get_day_of_week)
print("\nQ3(c)(i) - sample of updated 'report_week' (day-of-week) column:")
print(eda[["date_report", "report_week"]].head())


# ---------- Q3(c)(ii) Top 3 days cases were detected ----------
top3_days = eda["report_week"].value_counts().head(3)
print("\nQ3(c)(ii) - top 3 days of week with most cases:")
print(top3_days)


# ---------- Q3(c)(iii) Figure: cases per gender per month ----------
gender_month = eda.groupby(["year_month", "sex"]).size().unstack(fill_value=0)
gender_month.plot(kind="bar", figsize=(10, 5))
plt.title("COVID-19 Cases per Gender by Month")
plt.xlabel("Month")
plt.ylabel("Number of Cases")
plt.legend(title="Sex")
plt.tight_layout()
plt.savefig("q3_c_iii_cases_gender_month.png", dpi=150)
plt.close()
print("\nQ3(c)(iii) - figure saved: q3_c_iii_cases_gender_month.png")

print("\nAll tasks completed.")
