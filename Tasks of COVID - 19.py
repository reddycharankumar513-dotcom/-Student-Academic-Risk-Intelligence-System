#!/usr/bin/env python
# coding: utf-8

# In[1]:


##TASK 1 


import numpy as np
import pandas as pd

data = pd.read_csv(r"D:\DEML\cleaned1_covid-19_cases_canada.csv")

print(data.head())
print(data.columns)
print(data.shape)
print(data.size)
print(data.info())   
print(data.isnull().sum())


data.drop(columns=["case_id"], inplace=True, errors="ignore")
print(data.columns)
print(data.dtypes)

print(data["has_travel_history"].value_counts(dropna=False))

data["has_travel_history"] = data["has_travel_history"].replace({
    "t": "Yes",
    "f": "No"
})

data["has_travel_history"] = data["has_travel_history"].fillna("Unknown")
print(data["locally_acquired"].value_counts(dropna=False))

data["locally_acquired"] = data["locally_acquired"].fillna("Unknown")
print(data.isnull().sum())
print(data.duplicated().sum())

# Explain TWO (2) potential insights that can be derived from the dataset.

print(data['province'].value_counts())
# INSIGHT 1
# Here the cases are not equal in all the cities or states of Canada Highest no of cases are in the Quebec and Ontario

print(data.isnull().sum())
# INSIGHT 2
# Here has_travel_history and locally_aquired has the more number of null values


#i)
data=data[data['age']!='Not Reported']#Removed the rows where age is not reported
data = data.dropna(subset=["age"]) #Removing the null values

#ii)
def clean_age(age):
    age=str(age)

    if age in ["<1", "<10", "<18", "<20"]:
        return "0-19"
    if "-" in age:
        return age

    try:
        age = int(age)
        if age<=19:
            return "0-19"
        elif age<=29:
            return "20-29"
        elif age<=39:
            return "30-39"
        elif age<=49:
            return "40-49"
        elif age<=59:
            return "50-59"
        elif age<=69:
            return "60-69"
        elif age<=79:
            return "70-79"
        elif age<=89:
            return "80-89"
        else:
            return "90-99"

    except:
        return None
data["age"] = data["age"].apply(clean_age)
data = data.dropna(subset=["age"])

#iii)
print(data['age'].value_counts())

#c)
data.to_csv("cleaned1_covid-19_cases_canada.csv", index=False)


# In[3]:


import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Date, func
from sqlalchemy.orm import declarative_base, sessionmaker

# Load data
data = pd.read_csv(r"D:\DEML\cleaned1_covid-19_cases_canada.csv")
print("CSV records:", len(data))
print(data.columns)

# Convert dates (explicit format to avoid warnings)
data["date_report"] = pd.to_datetime(data["date_report"], format="%d-%m-%Y", errors="coerce").dt.date
data["report_week"] = pd.to_datetime(data["report_week"], format="%d-%m-%Y", errors="coerce").dt.date

print(data[["date_report", "report_week"]].head())
print("Missing date_report:", data["date_report"].isnull().sum())
print("Missing report_week:", data["report_week"].isnull().sum())

# Database setup
engine = create_engine("sqlite:///covid_q2.db", echo=False)
Base = declarative_base()

class CovidCase(Base):
    __tablename__ = "covid_cases"
    id = Column(Integer, primary_key=True, autoincrement=True)
    provincial_case_id = Column(String)
    age = Column(String)
    sex = Column(String)
    health_region = Column(String)
    province = Column(String)
    country = Column(String)
    date_report = Column(Date)
    report_week = Column(Date)
    has_travel_history = Column(String)
    locally_acquired = Column(String)
    case_source = Column(String)

# Reset table
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)
print("Fresh table created.")

# Session
Session = sessionmaker(bind=engine)
session = Session()

# Insert rows
cases = []
for _, row in data.iterrows():
    case = CovidCase(
        provincial_case_id=str(row.get("provincial_case_id", "")),
        age=str(row.get("age", "")),
        sex=str(row.get("sex", "")),
        health_region=str(row.get("health_region", "")),
        province=str(row.get("province", "")),
        country=str(row.get("country", "")),
        date_report=row["date_report"],
        report_week=row["report_week"],
        has_travel_history=str(row.get("has_travel_history", "")),
        locally_acquired=str(row.get("locally_acquired", "")),
        case_source=str(row.get("case_source", ""))
    )
    cases.append(case)

session.add_all(cases)
session.commit()

print("Data inserted successfully!")

# Queries (keep session open)
total_records = session.query(func.count(CovidCase.id)).scalar()
print("CSV records:", len(data))
print("Database records:", total_records)

dates = session.query(CovidCase.date_report).limit(5).all()
for row in dates:
    print(row)

result = (
    session.query(
        func.strftime("%Y-%m", CovidCase.date_report).label("Month"),
        CovidCase.sex,
        func.count(CovidCase.id).label("Total")
    )
    .filter(CovidCase.sex.in_(["Male", "Female"]))
    .group_by(func.strftime("%Y-%m", CovidCase.date_report), CovidCase.sex)
    .order_by(func.strftime("%Y-%m", CovidCase.date_report))
    .all()
)

for row in result:
    print(row)

result = (
    session.query(
        CovidCase.age,
        func.count(CovidCase.id).label("Female Cases")
    )
    .filter(CovidCase.sex == "Female")
    .group_by(CovidCase.age)
    .order_by(func.count(CovidCase.id).desc())
    .all()
)

for row in result:
    print(row)

older_age_groups = ["50-59", "60-69", "70-79", "80-89", "90-99"]

result = (
    session.query(
        func.strftime("%Y-%m", CovidCase.date_report).label("Month"),
        func.count(CovidCase.id).label("Total Cases")
    )
    .filter(CovidCase.has_travel_history == "No")
    .filter(CovidCase.age.in_(older_age_groups))
    .group_by(func.strftime("%Y-%m", CovidCase.date_report))
    .order_by(func.count(CovidCase.id).desc())
    .limit(2)
    .all()
)

for row in result:
    print(row)

# Close session at the end
session.close()


# In[4]:


## TASK 3
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"D:\DEML\cleaned1_covid-19_cases_canada.csv")

print(data.head())
print("Number of records:", len(data))

# i.)
data["date_report"] = pd.to_datetime(
    data["date_report"],
    format="%d-%m-%Y",
    errors="coerce"
)
print(data["date_report"].head())

data["Month"] = data["date_report"].dt.to_period("M").astype(str)

gender_month = (
    data[data["sex"].isin(["Male", "Female"])]
    .groupby(["Month", "sex"])
    .size()
    .reset_index(name="Total Cases")
    .sort_values(["Month", "sex"])
)

print(gender_month)

# ii.)
female_age = (
    data[data["sex"] == "Female"]
    .groupby("age")
    .size()
    .reset_index(name="Female Cases")
    .sort_values("Female Cases", ascending=False)
)

display(female_age)

#iii.)
older_than_50 = [
    "50-59",
    "60-69",
    "70-79",
    "80-89",
    "90-99"
]

top_two_months = (
    data[
        (data["has_travel_history"] == "No") &
        (data["age"].isin(older_than_50))
    ]
    .groupby("Month")
    .size()
    .reset_index(name="Total Cases")
    .sort_values("Total Cases", ascending=False)
    .head(2)
)

display(top_two_months)

#b.)
#i.)
def top_three_provinces_each_month(df):

    result = (
        df.groupby(["Month", "province"])
        .size()
        .reset_index(name="Total Cases")
    )

    result = result.sort_values(
        ["Month", "Total Cases"],
        ascending=[True, False]
    )

    top_three = (
        result.groupby("Month")
        .head(3)
        .reset_index(drop=True)
    )

    return top_three

top_provinces = top_three_provinces_each_month(data)

display(top_provinces)

#b.)
#ii.)
province_cases = (
    data.groupby("province")
    .size()
    .reset_index(name="Total Cases")
    .sort_values("Total Cases", ascending=False)
)

display(province_cases)

plt.figure(figsize=(12, 6))

plt.bar(
    province_cases["province"],
    province_cases["Total Cases"]
)

plt.xlabel("Province")
plt.ylabel("Total COVID-19 Cases")
plt.title("Total COVID-19 Cases by Province")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#iii.)
highest_province = province_cases.iloc[0]["province"]

print("Province with highest cases:", highest_province)

highest_province_data = data[
    data["province"] == highest_province
]

age_gender = (
    highest_province_data[
        highest_province_data["sex"].isin(["Male", "Female"])
    ]
    .groupby(["age", "sex"])
    .size()
    .reset_index(name="Total Cases")
)

age_gender_pivot = age_gender.pivot(
    index="age",
    columns="sex",
    values="Total Cases"
).fillna(0)

display(age_gender_pivot)

age_gender_pivot.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.xlabel("Age Group")
plt.ylabel("Number of COVID-19 Cases")
plt.title(
    f"Age Distribution of COVID-19 Cases by Gender - {highest_province}"
)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#3.
#c.)
#i.)
def get_week_day(date):
    return date.strftime("%a")

print(get_week_day(pd.Timestamp("2020-01-25")))

data["report_week"] = pd.to_datetime(data["date_report"], errors="coerce").dt.day_name()
print(data[["date_report", "report_week"]].head(10))


#c.)
#ii.)
top_three_days = (
    data.groupby("report_week")
    .size()
    .reset_index(name="Total Cases")
    .sort_values("Total Cases", ascending=False)
    .head(3)
)

display(top_three_days)

#c.)
#iii.)
data["Month"]

monthly_gender = (
    data[
        data["sex"].isin(["Male", "Female"])
    ]
    .groupby(["Month", "sex"])
    .size()
    .reset_index(name="Total Cases")
)

monthly_gender_pivot = monthly_gender.pivot(
    index="Month",
    columns="sex",
    values="Total Cases"
).fillna(0)

import matplotlib.pyplot as plt

# Check that monthly_gender_pivot exists
print(monthly_gender_pivot.head())

# Plot
monthly_gender_pivot.plot(
    kind='bar',
    figsize=(12, 6)
)

plt.xlabel("Month")
plt.ylabel("Number of COVID-19 Cases")
plt.title("COVID-19 Cases per Gender for Each Month")
plt.xticks(rotation=45)
plt.legend(title="Gender")
plt.tight_layout()
plt.show()



# In[ ]:




