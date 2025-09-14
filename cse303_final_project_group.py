# -*- coding: utf-8 -*-
"""CSE303_final_project_group.ipynb"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from pyngrok import ngrok

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
# EDA: Histograms with boxplots
# ===============================
for col in numeric_cols:
    fig = px.histogram(df, x=col, nbins=20, title=f"Distribution of {col}", marginal="box")
    fig.show()

# ===============================
# Scatter Plots (with regression trendline)
# ===============================
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
    fig = px.scatter(df, x=x, y=y, title=f"{y} vs {x}", trendline="ols")
    fig.show()

# ===============================
# Boxplots for outlier detection
# ===============================
for col in numeric_cols:
    fig = px.box(df, y=col, title=f"Box Plot for {col}")
    fig.show()

# ===============================
# Remove outliers using IQR capping
# ===============================
for col in numeric_cols:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])

# ===============================
# Standardization
# ===============================
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# ===============================
# Correlation Heatmap
# ===============================
corr = df[numeric_cols].corr()
fig = ff.create_annotated_heatmap(
    z=corr.values,
    x=list(corr.columns),
    y=list(corr.index),
    annotation_text=np.round(corr.values, 2),
    colorscale="RdBu",
    showscale=True
)
fig.update_layout(title="Correlation Heatmap")
fig.show()

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

# Function to fit model
def fit_and_print(formula, df):
    model = smf.ols(formula=formula, data=df).fit()
    print("=== Model:", formula, "===")
    print(model.summary())
    print("\nR-squared: {:.3f}\n".format(model.rsquared))
    return model

# Fit simple models
simple_fitted = {}
print("\n### SIMPLE LINEAR MODELS ###\n")
for target, pred in simple_models:
    formula = f"{target} ~ {pred}"
    simple_fitted[formula] = fit_and_print(formula, df)

# Fit multiple models
multiple_fitted = {}
print("\n### MULTIPLE LINEAR MODELS ###\n")
for target, preds in multiple_models:
    formula = f"{target} ~ {' + '.join(preds)}"
    multiple_fitted[formula] = fit_and_print(formula, df)

# ===============================
# Visualize simple regression (plotly)
# ===============================
for (target, pred), model in zip(simple_models, simple_fitted.values()):
    fig = px.scatter(df, x=pred, y=target, trendline="ols", title=f"Regression: {target} vs {pred}")
    fig.show()

# ===============================
# Visualize multiple regression (Actual vs Predicted)
# ===============================
for (target, preds), model in zip(multiple_models, multiple_fitted.values()):
    predicted = model.predict(df)
    fig = px.scatter(x=predicted, y=df[target], labels={'x':'Predicted','y':'Actual'},
                     title=f"Multiple Regression: {target} ~ {' + '.join(preds)}")
    fig.add_shape(type='line', x0=predicted.min(), x1=predicted.max(),
                  y0=df[target].min(), y1=df[target].max(), line=dict(color='red', dash='dash'))
    fig.show()

# ===============================
# Hypothesis Testing
# ===============================
alpha = 0.05

# 1. Independent t-test (CGPA by Gender)
male_cgpa = df[df["Gender"] == 0]["CGPA"]
female_cgpa = df[df["Gender"] == 1]["CGPA"]
t_stat, p_val = stats.ttest_ind(male_cgpa, female_cgpa, nan_policy="omit")
print("\n1. Independent t-test: CGPA by Gender")
print(f"t-statistic = {t_stat}, p-value = {p_val}")
print("Result:", "Reject H0" if p_val < alpha else "Fail to reject H0")

# 2. One-way ANOVA (Stress_Level ~ Year_of_Study)
anova_groups = [group["Stress_Level"].dropna() for _, group in df.groupby("Year_of_Study")]
f_stat, p_val = stats.f_oneway(*anova_groups)
print("\n2. One-way ANOVA: Stress_Level ~ Year_of_Study")
print(f"F-statistic = {f_stat}, p-value = {p_val}")
print("Result:", "Reject H0" if p_val < alpha else "Fail to reject H0")

# 3. Correlation test (CGPA vs Anxiety_Level)
corr_coef, p_val = stats.pearsonr(df["CGPA"], df["Anxiety_Level"])
print("\n3. Correlation test: CGPA vs Anxiety_Level")
print(f"Correlation coefficient = {corr_coef}, p-value = {p_val}")
print("Result:", "Reject H0" if p_val < alpha else "Fail to reject H0")

# 4. Proportion test (Chi-square: Gender vs High Anxiety)
df["Anxiety_High"] = (df["Anxiety_Level"] > df["Anxiety_Level"].mean()).astype(int)
contingency_table = pd.crosstab(df["Gender"], df["Anxiety_High"])
chi2, p_val, dof , expected = stats.chi2_contingency(contingency_table)
print("\n4. Proportion test (Chi-square): Gender vs High Anxiety")
print(contingency_table)
print(f"Chi2 = {chi2}, p-value = {p_val}")
print("Result:", "Reject H0" if p_val < alpha else "Fail to reject H0")

# ===============================
# Streamlit + ngrok setup
# ===============================
# ngrok.kill()
# NGROK_AUTH_TOKEN = "32eN4jnYOTeqprDbxQMwVwMvE9c_CD4exrJFA55RG12RMrfo"
# ngrok.set_auth_token(NGROK_AUTH_TOKEN)
# public_url = ngrok.connect(8501)
# print("Streamlit app will be available at:", public_url)
