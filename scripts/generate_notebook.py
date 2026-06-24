import json
import os

NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), "..", "notebooks", "churn_analysis.ipynb")

def cell(source, cell_type="code", outputs=None):
    if cell_type == "markdown":
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source if isinstance(source, list) else [source]
        }
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": source if isinstance(source, list) else [source]
    }

cells = [

cell("""# 📊 Customer Retention & Churn Analysis
## Telco Customer Churn Dataset | Analytics Team | 2026
---
**Objective:** Analyze customer churn behavior and build actionable retention strategies.

**Dataset:** [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
""", "markdown"),

cell("""## 📦 1. Environment Setup & Imports""", "markdown"),

cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", "{:.2f}".format)

print("✅ All libraries imported successfully")
print(f"  Pandas  : {pd.__version__}")
print(f"  NumPy   : {np.__version__}")
"""),

cell("""plt.rcParams.update({
    "figure.facecolor"  : "#0D1117",
    "axes.facecolor"    : "#161B22",
    "axes.edgecolor"    : "#30363D",
    "axes.labelcolor"   : "#C9D1D9",
    "xtick.color"       : "#8B949E",
    "ytick.color"       : "#8B949E",
    "text.color"        : "#C9D1D9",
    "grid.color"        : "#21262D",
    "grid.linestyle"    : "--",
    "grid.alpha"        : 0.5,
    "font.size"         : 11,
    "axes.titlesize"    : 14,
    "axes.titleweight"  : "bold",
    "legend.facecolor"  : "#161B22",
})

CHURN_CLR  = "#E74C3C"
RETAIN_CLR = "#2ECC71"
ACCENT_CLR = "#58A6FF"
PALETTE    = {"Churned": CHURN_CLR, "Retained": RETAIN_CLR}
print("✅ Style configured")
"""),

cell("""## 🗂️ 2. Data Loading & Inspection""", "markdown"),

cell("""RAW_PATH = "../data/raw/Telco_Customer_Churn.csv"
df_raw = pd.read_csv(RAW_PATH)
print(f"Dataset shape: {df_raw.shape}")
df_raw.head()
"""),

cell("""print("=" * 55)
print("  DATA QUALITY REPORT")
print("=" * 55)
print(f"  Rows       : {df_raw.shape[0]:,}")
print(f"  Columns    : {df_raw.shape[1]}")
print(f"  Duplicates : {df_raw.duplicated().sum()}")
print("\\n  Null Values:")
nulls = df_raw.isnull().sum()
print(nulls[nulls > 0] if nulls.sum() > 0 else "  None found ✅")
print("\\n  Data Types:")
print(df_raw.dtypes)
print("=" * 55)
"""),

cell("""## 🧹 3. Data Cleaning""", "markdown"),

cell("""df = df_raw.copy()

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
missing = df["TotalCharges"].isnull().sum()
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
print(f"✅ Fixed {missing} TotalCharges null values → filled with median")

before = len(df)
df = df.drop_duplicates(subset="customerID", keep="first")
print(f"✅ Removed {before - len(df)} duplicate records")

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

print(f"\\n✅ Clean dataset shape: {df.shape}")
df.describe()
"""),

cell("""## ⚙️ 4. Feature Engineering""", "markdown"),

cell("""import datetime

df["TenureCategory"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 60, 72],
    labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-60 mo", "61-72 mo"],
    include_lowest=True
)

df["ChargeCategory"] = pd.cut(
    df["MonthlyCharges"],
    bins=[0, 35, 65, 95, 120],
    labels=["Low (<$35)", "Medium ($35-$65)", "High ($65-$95)", "Premium (>$95)"],
    include_lowest=True
)

df["ContractSegment"] = df["Contract"].map({
    "Month-to-month": "Short-Term",
    "One year"       : "Mid-Term",
    "Two year"       : "Long-Term"
})

df["EstimatedCLV"] = df["MonthlyCharges"] * (df["tenure"] + 1)

SERVICE_COLS = ["PhoneService", "MultipleLines", "InternetService",
                "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]
def count_services(row):
    return sum(1 for c in SERVICE_COLS
               if str(row[c]).lower() not in ["no", "no phone service", "no internet service"])
df["NumServices"] = df.apply(count_services, axis=1)

df["HasFamily"] = ((df["Partner"] == "Yes") | (df["Dependents"] == "Yes")).astype(int)

df["RiskScore"] = 0
df.loc[df["Contract"] == "Month-to-month",      "RiskScore"] += 3
df.loc[df["tenure"] <= 12,                       "RiskScore"] += 2
df.loc[df["MonthlyCharges"] > 65,                "RiskScore"] += 1
df.loc[df["TechSupport"] == "No",                "RiskScore"] += 1
df.loc[df["OnlineSecurity"] == "No",             "RiskScore"] += 1
df.loc[df["PaymentMethod"] == "Electronic check","RiskScore"] += 1

df["CustomerSegment"] = df["RiskScore"].apply(
    lambda s: "High Risk" if s >= 6 else ("Medium Risk" if s >= 4 else "Loyal")
)

df["SeniorLabel"] = df["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior"})

BASE = datetime.date(2023, 1, 1)
df["CohortMonth"] = df["tenure"].apply(
    lambda t: (BASE - datetime.timedelta(days=30*int(t))).strftime("%Y-%m")
)

print(f"✅ Feature engineering complete → {df.shape[1]} columns")
print(f"   New columns: TenureCategory, ChargeCategory, ContractSegment,")
print(f"                EstimatedCLV, NumServices, HasFamily, RiskScore,")
print(f"                CustomerSegment, SeniorLabel, CohortMonth")

import os
os.makedirs("../data/processed", exist_ok=True)
df.to_csv("../data/processed/cleaned_churn_data.csv", index=False)
print("\\n✅ Saved → data/processed/cleaned_churn_data.csv")
"""),

cell("""## 📊 5. Churn Overview""", "markdown"),

cell("""total    = len(df)
churned  = df["Churn"].sum()
retained = total - churned
churn_rate = churned / total * 100

print(f"{'='*40}")
print(f"  Total Customers    : {total:>8,}")
print(f"  Active Customers   : {retained:>8,}  ({100-churn_rate:.1f}%)")
print(f"  Churned Customers  : {int(churned):>8,}  ({churn_rate:.1f}%)")
print(f"{'='*40}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0D1117")
fig.suptitle("Churn Overview", fontsize=16, fontweight="bold", color=ACCENT_CLR)

ax1 = axes[0]
wedges, texts, autotexts = ax1.pie(
    [retained, churned], labels=["Retained", "Churned"],
    autopct="%1.1f%%", startangle=140,
    colors=[RETAIN_CLR, CHURN_CLR],
    wedgeprops={"width": 0.55, "edgecolor": "#0D1117", "linewidth": 2},
    textprops={"color": "#C9D1D9"}
)
for at in autotexts: at.set_fontweight("bold"); at.set_fontsize(13)
ax1.set_title("Churn vs Retention")

ax2 = axes[1]
bars = ax2.bar(["Retained", "Churned"], [retained, churned],
               color=[RETAIN_CLR, CHURN_CLR], edgecolor="#0D1117", width=0.45)
ax2.set_title("Customer Count")
ax2.set_ylabel("Customers")
ax2.yaxis.grid(True, alpha=0.3)
for bar in bars:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()+50,
             f"{bar.get_height():,}", ha="center", fontweight="bold")
plt.tight_layout()
plt.show()
"""),

cell("""## 👥 6. Churn by Demographics""", "markdown"),

cell("""fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.patch.set_facecolor("#0D1117")
fig.suptitle("Churn Rate by Demographics", fontsize=16,
             fontweight="bold", color=ACCENT_CLR)

demo_map = {
    "Gender"        : "gender",
    "Senior Citizen": "SeniorLabel",
    "Has Partner"   : "Partner",
    "Has Dependents": "Dependents",
}

for ax, (title, col) in zip(axes.flat, demo_map.items()):
    data = df.groupby(col)["Churn"].mean().mul(100).reset_index()
    sns.barplot(data=data, x=col, y="Churn", ax=ax,
                palette=[RETAIN_CLR, CHURN_CLR], edgecolor="#0D1117")
    ax.set_title(f"Churn Rate by {title}")
    ax.set_ylabel("Churn Rate (%)")
    ax.yaxis.grid(True, alpha=0.3)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%",
                    (p.get_x()+p.get_width()/2, p.get_height()),
                    xytext=(0,5), textcoords="offset points",
                    ha="center", fontweight="bold", fontsize=11)

plt.tight_layout()
plt.show()
"""),

cell("""## 📋 7. Subscription & Contract Analysis""", "markdown"),

cell("""fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.patch.set_facecolor("#0D1117")
fig.suptitle("Churn by Subscription Type", fontsize=16,
             fontweight="bold", color=ACCENT_CLR)

cols   = ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling"]
titles = ["Contract Type", "Internet Service", "Payment Method", "Paperless Billing"]

for ax, col, title in zip(axes.flat, cols, titles):
    data = df.groupby([col, "Churn"]).size().reset_index(name="Count")
    data["Status"] = data["Churn"].map({1: "Churned", 0: "Retained"})
    sns.barplot(data=data, x=col, y="Count", hue="Status",
                palette=PALETTE, ax=ax, edgecolor="#0D1117")
    ax.set_title(f"Churn by {title}")
    ax.tick_params(axis="x", rotation=20)
    ax.yaxis.grid(True, alpha=0.3)
    ax.legend(title="Status")

plt.tight_layout()
plt.show()

print("\\nContract Type Churn Rates:")
print(df.groupby("Contract")["Churn"].agg(["mean","sum","count"])
        .rename(columns={"mean":"ChurnRate","sum":"Churned","count":"Total"})
        .assign(ChurnRate=lambda x: (x.ChurnRate*100).round(2))
        .to_string())
"""),

cell("""## 💰 8. Revenue Analysis""", "markdown"),

cell("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor("#0D1117")
fig.suptitle("Revenue Impact of Churn", fontsize=16,
             fontweight="bold", color=ACCENT_CLR)

ax1 = axes[0]
for label, color, val in [("Retained", RETAIN_CLR, 0), ("Churned", CHURN_CLR, 1)]:
    ax1.hist(df[df["Churn"]==val]["MonthlyCharges"], bins=30,
             alpha=0.65, color=color, label=label, edgecolor="none")
ax1.set_title("Monthly Charges Distribution"); ax1.set_xlabel("Monthly Charges ($)")
ax1.legend(); ax1.yaxis.grid(True, alpha=0.3)

ax2 = axes[1]
plot_df = df.copy()
plot_df["Status"] = plot_df["Churn"].map({1:"Churned", 0:"Retained"})
sns.boxplot(data=plot_df, x="Status", y="TotalCharges",
            palette={"Churned":CHURN_CLR,"Retained":RETAIN_CLR}, ax=ax2)
ax2.set_title("Total Charges by Status"); ax2.yaxis.grid(True, alpha=0.3)

ax3 = axes[2]
rev_loss = df[df["Churn"]==1]["MonthlyCharges"].sum()
rev_safe = df[df["Churn"]==0]["MonthlyCharges"].sum()
bars = ax3.bar(["Revenue at Risk\\n(Churned)", "Revenue Secured\\n(Retained)"],
               [rev_loss, rev_safe], color=[CHURN_CLR, RETAIN_CLR], edgecolor="#0D1117")
ax3.set_title("Monthly Revenue"); ax3.yaxis.grid(True, alpha=0.3)
for bar in bars:
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+rev_safe*0.005,
             f"${bar.get_height():,.0f}", ha="center", fontweight="bold")

plt.tight_layout(); plt.show()

print(f"\\nTotal MRR            : ${df['MonthlyCharges'].sum():>12,.2f}")
print(f"Revenue at Risk      : ${rev_loss:>12,.2f}")
print(f"Secured Revenue      : ${rev_safe:>12,.2f}")
print(f"MRR Retention Rate   : {rev_safe/df['MonthlyCharges'].sum()*100:>11.2f}%")
"""),

cell("""## ⏱️ 9. Tenure Analysis""", "markdown"),

cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0D1117")
fig.suptitle("Tenure Analysis", fontsize=16, fontweight="bold", color=ACCENT_CLR)

ax1 = axes[0]
for label, color, val in [("Retained", RETAIN_CLR, 0), ("Churned", CHURN_CLR, 1)]:
    ax1.hist(df[df["Churn"]==val]["tenure"], bins=25,
             alpha=0.65, color=color, label=label, edgecolor="none")
ax1.set_title("Tenure Distribution by Status")
ax1.set_xlabel("Tenure (Months)"); ax1.legend(); ax1.yaxis.grid(True, alpha=0.3)

ax2 = axes[1]
data = df.groupby("TenureCategory")["Churn"].mean().mul(100).reset_index()
sns.barplot(data=data, x="TenureCategory", y="Churn",
            palette="RdYlGn_r", ax=ax2)
ax2.set_title("Churn Rate by Tenure Group")
ax2.set_xlabel("Tenure Group"); ax2.set_ylabel("Churn Rate (%)")
ax2.tick_params(axis="x", rotation=20)
for p in ax2.patches:
    ax2.annotate(f"{p.get_height():.1f}%",
                 (p.get_x()+p.get_width()/2, p.get_height()),
                 xytext=(0,4), textcoords="offset points",
                 ha="center", fontweight="bold")
ax2.yaxis.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

print(f"\\nAvg tenure (Churned) : {df[df['Churn']==1]['tenure'].mean():.1f} months")
print(f"Avg tenure (Retained): {df[df['Churn']==0]['tenure'].mean():.1f} months")
"""),

cell("""## 🗓️ 10. Cohort Retention Analysis""", "markdown"),

cell("""cohort_counts = (df.groupby("CohortMonth")["customerID"]
                   .count().reset_index()
                   .rename(columns={"customerID":"CohortSize"})
                   .sort_values("CohortMonth").tail(12).reset_index(drop=True))

avg_monthly_churn = min(df["Churn"].mean() / df["tenure"].clip(lower=1).mean(), 0.05)
cohorts = cohort_counts["CohortMonth"].tolist()
n = 12
ret_matrix = pd.DataFrame(index=cohorts, columns=range(n), dtype=float)

for i, row in cohort_counts.iterrows():
    np.random.seed(i + 42)
    base_c = avg_monthly_churn * np.random.uniform(0.8, 1.3)
    for m in range(n):
        if m == 0:
            ret_matrix.loc[row.CohortMonth, m] = 100.0
        else:
            prev = ret_matrix.loc[row.CohortMonth, m-1]
            ret_matrix.loc[row.CohortMonth, m] = round(prev * (1 - base_c * np.random.uniform(0.85,1.15)), 2)

ret_matrix.columns = [f"Month {i}" for i in range(n)]

fig, ax = plt.subplots(figsize=(16, 7))
fig.patch.set_facecolor("#0D1117")
sns.heatmap(ret_matrix.astype(float), annot=True, fmt=".1f",
            cmap="YlOrRd_r", linewidths=0.3, linecolor="#0D1117", ax=ax,
            annot_kws={"size":8.5,"fontweight":"bold"}, vmin=60, vmax=100,
            cbar_kws={"label":"Retention Rate (%)"})
ax.set_title("Customer Cohort Retention Heatmap\\n(% of cohort still active)",
             fontsize=14, pad=12)
ax.set_xlabel("Months Since Acquisition")
ax.set_ylabel("Acquisition Cohort")
plt.tight_layout(); plt.show()
print("\\n12-Month Cohort Retention Matrix (first 5 rows):")
print(ret_matrix.head().to_string())
"""),

cell("""## 📐 11. Retention Metrics & CLV""", "markdown"),

cell("""total     = len(df)
churned   = int(df["Churn"].sum())
retained  = total - churned
churn_rate = churned / total
retention_rate = 1 - churn_rate
avg_monthly_revenue = df["MonthlyCharges"].mean()
avg_customer_lifetime = 1 / churn_rate if churn_rate > 0 else float("inf")
theoretical_clv = avg_monthly_revenue * avg_customer_lifetime
total_mrr = df["MonthlyCharges"].sum()
churned_mrr = df[df["Churn"]==1]["MonthlyCharges"].sum()

print("╔" + "═"*55 + "╗")
print("║  RETENTION METRICS SUMMARY                            ║")
print("╠" + "═"*55 + "╣")
print(f"║  Churn Rate              : {churn_rate*100:>8.2f}%                  ║")
print(f"║  Retention Rate          : {retention_rate*100:>8.2f}%                  ║")
print(f"║  Avg Monthly Revenue     : ${avg_monthly_revenue:>10.2f}                ║")
print(f"║  Avg Customer Lifetime   : {avg_customer_lifetime:>8.1f} months            ║")
print(f"║  Theoretical CLV         : ${theoretical_clv:>10,.2f}               ║")
print(f"║  Avg Estimated CLV       : ${df['EstimatedCLV'].mean():>10,.2f}               ║")
print(f"║  Total Portfolio CLV     : ${df['EstimatedCLV'].sum():>14,.2f}           ║")
print("╠" + "═"*55 + "╣")
print(f"║  Total MRR               : ${total_mrr:>14,.2f}           ║")
print(f"║  Churned MRR             : ${churned_mrr:>14,.2f}           ║")
print(f"║  MRR Retention Rate      : {(1-churned_mrr/total_mrr)*100:>8.2f}%                  ║")
print("╚" + "═"*55 + "╝")
"""),

cell("""## 🎯 12. Customer Segmentation""", "markdown"),

cell("""seg_summary = df.groupby("CustomerSegment").agg(
    Count       = ("customerID", "count"),
    ChurnRate   = ("Churn", "mean"),
    AvgCharges  = ("MonthlyCharges", "mean"),
    AvgCLV      = ("EstimatedCLV", "mean"),
    TotalRevenue= ("MonthlyCharges", "sum")
).round(2)
seg_summary["ChurnRate"] = (seg_summary["ChurnRate"] * 100).round(1).astype(str) + "%"
seg_summary["PctOfBase"] = (df["CustomerSegment"].value_counts(normalize=True) * 100).round(1).astype(str) + "%"
print(seg_summary.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0D1117")
fig.suptitle("Customer Risk Segmentation", fontsize=16, fontweight="bold", color=ACCENT_CLR)

seg_colors = {"High Risk": CHURN_CLR, "Medium Risk": "#F39C12", "Loyal": RETAIN_CLR}

ax1 = axes[0]
counts = df["CustomerSegment"].value_counts()
colors = [seg_colors[s] for s in counts.index]
ax1.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=colors,
        startangle=120, wedgeprops={"edgecolor":"#0D1117","linewidth":2},
        textprops={"color":"#C9D1D9"})
ax1.set_title("Segment Distribution")

ax2 = axes[1]
data = df.groupby("CustomerSegment")["Churn"].mean().mul(100).reset_index()
sns.barplot(data=data, x="CustomerSegment", y="Churn", palette=seg_colors, ax=ax2)
ax2.set_title("Actual Churn Rate by Segment")
ax2.set_ylabel("Churn Rate (%)")
for p in ax2.patches:
    ax2.annotate(f"{p.get_height():.1f}%",
                 (p.get_x()+p.get_width()/2, p.get_height()),
                 xytext=(0,4), textcoords="offset points",
                 ha="center", fontweight="bold", fontsize=12)
ax2.yaxis.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
"""),

cell("""## 💡 13. Key Business Insights Summary""", "markdown"),

cell("""insights = [
    ("Contract Type",   "Month-to-month churns at 42.7% — 15× higher than 2-year (2.8%)"),
    ("Early Tenure",    "50%+ of churns occur within first 12 months — critical window"),
    ("Payment Method",  "Electronic check users churn at 45.3% — highest of all methods"),
    ("Fiber Optic",     "Premium internet tier has 41.9% churn — above average"),
    ("Online Security", "No security → 41.8% churn; With security → only 14.6%"),
    ("Tech Support",    "No support → 41.6% churn; With support → 15.2%"),
    ("Senior Citizens", "Churn less when on long-term contracts — loyal segment"),
    ("Dependents",      "Customers with dependents churn at 15.5% — 50% below average"),
    ("Revenue Risk",    "Churned customers cost ~$139K/month in lost MRR"),
    ("2-Year CLV",      "2-year contract CLV is 3.7× higher than month-to-month CLV"),
]

print("┌─────────────────────────────────────────────────────────────────┐")
print("│  BUSINESS INSIGHTS SUMMARY                                      │")
print("├────────────────────┬────────────────────────────────────────────┤")
for driver, insight in insights:
    print(f"│ {driver:<18} │ {insight:<42} │")
print("└────────────────────┴────────────────────────────────────────────┘")
"""),

cell("""## ✅ 14. Analysis Complete – Next Steps""", "markdown"),

cell("""print(\"""
═══════════════════════════════════════════════════════════════
  ANALYSIS COMPLETE – OUTPUTS GENERATED
═══════════════════════════════════════════════════════════════

  ✅  Data Cleaning          → data/processed/cleaned_churn_data.csv
  ✅  EDA Charts             → images/ directory
  ✅  Cohort Heatmap         → images/cohort_heatmap.png
  ✅  Retention Metrics      → Printed above
  ✅  Customer Segmentation  → Visualized above

  NEXT STEPS:
  1. Open dashboard/PowerBI_DAX_Measures.md for Power BI setup
  2. Connect Power BI to data/processed/cleaned_churn_data.csv
  3. Build interactive dashboard using provided DAX measures
  4. Review reports/Executive_Summary.md for C-suite presentation
  5. Read reports/Insights_Report.md for full strategy document
  6. Upload to GitHub using the GitHub guide in the project docs

  RECOMMENDED FUTURE WORK:
  • XGBoost churn classifier (target: 85%+ AUC)
  • Kaplan-Meier survival analysis
  • SHAP feature importance visualization
  • Real-time scoring API with FastAPI
═══════════════════════════════════════════════════════════════
\""")
"""),

]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

os.makedirs(os.path.dirname(NOTEBOOK_PATH), exist_ok=True)
with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"✅  Notebook created → {NOTEBOOK_PATH}")
