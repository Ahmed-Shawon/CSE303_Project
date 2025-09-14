# -*- coding: utf-8 -*-
"""CSE303_final_project_group_streamlit.py"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler

# ===============================
# Page title
# ===============================
st.title("CSE303 Student Survey Analysis")

# ===============================
# Load CSV
# ===============================
df = pd.read_csv("student_survey.csv")

# ===============================
# Encode categorical columns
# ===============================
df["Year_of_Study"] = df["Year_of_Study"].map({"1st": 1, "2nd": 2, "3rd": 3, "4th": 4})
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Extracurricular_pariticipation"] = df["Extracurricular_pariticipation"].map({"Yes": 1, "No": 0})
df["PartTime_Job_Status"] = df["PartTime_Job_Status"].map({"Yes": 1, "No": 0})

# Handle missing values
df = df.dropna(subset=["Year_of_Study"])
df["CGPA"] = df["CGPA"].fillna(df["CGPA"].mean())
df["Extracurricular_Hours"] = df["Extracurricular_Hours"].fillna(0)
df["PartTime_Job_Hours"] = df["PartTime_Job_Hours"].fillna(0)

# ===============================
# Numeric columns
# ===============================
numeric_cols = [
    "Age", "CGPA", "Year_of_Study", "Study_Hours_per_Week",
    "Sleep_Quality", "Sleep_Hours", "Anxiety_Level",
    "Stress_Level", "PartTime_Job_Hours", "Extracurricular_Hours",
    "Credits_Enrolled"
]

# ===============================
# Sidebar Filters
# ===============================
st.sidebar.header("Filters")
year_filter = st.sidebar.selectbox("Year of Study", ["All"] + sorted(df["Year_of_Study"].unique().tolist()))
gender_filter = st.sidebar.selectbox("Gender", ["All", "Male", "Female"])
job_filter = st.sidebar.selectbox("PartTime Job Status", ["All", "Yes", "No"])

filtered_df = df.copy()
if year_filter != "All":
    filtered_df = filtered_df[filtered_df["Year_of_Study"] == year_filter]
if gender_filter != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == (1 if gender_filter == "Male" else 0)]
if job_filter != "All":
    filtered_df = filtered_df[filtered_df["PartTime_Job_Status"] == (1 if job_filter == "Yes" else 0)]

# ===============================
# EDA: Histograms with Boxplots
# ===============================
st.header("EDA: Histograms with Boxplots")
for col in numeric_cols:
    st.subheader(f"Distribution of {col}")
    fig = px.histogram(filtered_df, x=col, nbins=20, marginal="box", title=f"{col} Distribution")
    st.plotly_chart(fig)

# ===============================
# Scatter Plots with Regression
# ===============================
st.header("Scatter Plots with Regression Trendline")
relationships = [
    ("Stress_Level", "Year_of_Study"),
    ("CGPA", "Gender"),
    ("Stress_Level", "PartTime_Job_Status"),
    ("Stress_Level", "Study_Hours_per_Week"),
    ("Sleep_Quality", "Anxiety_Level"),
    ("Sleep_Hours", "Anxiety_Level"),
    ("Anxiety_Level", "Year_of_Study")
]

for y, x in relationships:
    st.subheader(f"{y} vs {x}")
    fig = px.scatter(filtered_df, x=x, y=y, trendline="ols", title=f"{y} vs {x}")
    st.plotly_chart(fig)

# ===============================
# Boxplots for Outlier Detection
# ===============================
st.header("Boxplots for Outlier Detection")
for col in numeric_cols:
    st.subheader(f"Box Plot for {col}")
    fig = px.box(filtered_df, y=col, title=f"Box Plot for {col}")
    st.plotly_chart(fig)

# ===============================
# Remove Outliers using IQR Capping
# ===============================
for col in numeric_cols:
    Q1, Q3 = filtered_df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    filtered_df[col] = np.where(filtered_df[col] < lower, lower, filtered_df[col])
    filtered_df[col] = np.where(filtered_df[col] > upper, upper, filtered_df[col])

# ===============================
# Standardization
# ===============================
scaler = StandardScaler()
filtered_df[numeric_cols] = scaler.fit_transform(filtered_df[numeric_cols])

# ===============================
# Correlation Heatmap
# ===============================
st.header("Correlation Heatmap")
corr = filtered_df[numeric_cols].corr()
fig = ff.create_annotated_heatmap(
    z=corr.values,
    x=list(corr.columns),
    y=list(corr.index),
    annotation_text=np.round(corr.values, 2),
    colorscale="RdBu",
    showscale=True
)
st.plotly_chart(fig)

# ===============================
# Regression Models
# ===============================
simple_models = [
    ('Year_of_Study', 'Age'),
    ('Extracurricular_Hours', 'Extracurricular_pariticipation'),
    ('PartTime_Job_Hours', 'PartTime_Job_Status'),
    ('Anxiety_Level', 'Stress_Level'),
    ('Sleep_Quality', 'Sleep_Hours'),
    ('Year_of_Study', 'Stress_Level'),
    ('Anxiety_Level', 'Gender'),
    ('Sleep_Hours', 'Stress_Level'),
    ('CGPA', 'Age'),
    ('Sleep_Quality', 'Gender')
]

multiple_models = [
    ('Stress_Level', ['Anxiety_Level','Year_of_Study','Sleep_Hours']),
    ('Sleep_Quality', ['Sleep_Hours','Anxiety_Level','Stress_Level']),
    ('Anxiety_Level', ['Stress_Level','Sleep_Hours','Gender']),
    ('PartTime_Job_Hours', ['PartTime_Job_Status','Age','Sleep_Quality']),
    ('Extracurricular_Hours', ['Extracurricular_pariticipation','Year_of_Study','PartTime_Job_Status']),
    ('Year_of_Study', ['Age','Stress_Level','PartTime_Job_Status']),
    ('Sleep_Hours', ['Stress_Level','Anxiety_Level','CGPA']),
    ('CGPA', ['Study_Hours_per_Week','Age','Sleep_Quality']),
    ('Credits_Enrolled', ['Year_of_Study','PartTime_Job_Status','Stress_Level']),
    ('Extracurricular_pariticipation', ['Extracurricular_Hours','Gender','Sleep_Hours'])
]

def fit_and_print(formula, df):
    model = smf.ols(formula=formula, data=df).fit()
    st.text(f"=== Model: {formula} ===")
    st.text(model.summary())
    st.text(f"R-squared: {model.rsquared:.3f}")
    return model

st.header("Simple Linear Regression Models")
simple_fitted = {}
for target, pred in simple_models:
    formula = f"{target} ~ {pred}"
    simple_fitted[formula] = fit_and_print(formula, filtered_df)

st.header("Multiple Linear Regression Models")
multiple_fitted = {}
for target, preds in multiple_models:
    formula = f"{target} ~ {' + '.join(preds)}"
    multiple_fitted[formula] = fit_and_print(formula, filtered_df)

# ===============================
# Hypothesis Testing
# ===============================
st.header("Hypothesis Testing")
alpha = 0.05

# 1. Independent t-test
male_cgpa = filtered_df[filtered_df["Gender"] == 0]["CGPA"]
female_cgpa = filtered_df[filtered_df["Gender"] == 1]["CGPA"]
t_stat, p_val = stats.ttest_ind(male_cgpa, female_cgpa, nan_policy="omit")
st.subheader("1. Independent t-test: CGPA by Gender")
st.write(f"t-statistic = {t_stat}, p-value = {p_val}")
st.write("Result:", "Reject H0" if p_val < alpha else "Fail to reject H0")

# 2. One-way ANOVA
anova_groups = [group["Stress_Level"].dropna() for _, group in filtered_df.groupby("Year_of_Study")]
f_stat, p_val = stats.f_oneway(*anova_groups)
st.subheader("2. One-way ANOVA: Stress_Level ~ Year_of_Study")
st.write(f"F-statistic = {f_stat}, p-value = {p_val}")
st.write("Result:", "Reject H0" if p_val < alpha else "Fail to reject H0")

# 3. Correlation test
corr_coef, p_val = stats.pearsonr(filtered_df["CGPA"], filtered_df["Anxiety_Level"])
st.subheader("3. Correlation test: CGPA vs Anxiety_Level")
st.write(f"Correlation coefficient = {corr_coef}, p-value = {p_val}")
st.write("Result:", "Reject H0" if p_val < alpha else "Fail to reject H0")

# 4. Proportion test (Chi-square)
filtered_df["Anxiety_High"] = (filtered_df["Anxiety_Level"] > filtered_df["Anxiety_Level"].mean()).astype(int)
contingency_table = pd.crosstab(filtered_df["Gender"], filtered_df["Anxiety_High"])
chi2, p_val, dof , expected = stats.chi2_contingency(contingency_table)
st.subheader("4. Proportion test (Chi-square): Gender vs High Anxiety")
st.write(contingency_table)
st.write(f"Chi2 = {chi2}, p-value = {p_val}")
st.write("Result:", "Reject H0" if p_val < alpha else "Fail to reject H0")
