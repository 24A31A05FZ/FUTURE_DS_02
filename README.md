<div align="center">

# 📊 Customer Retention & Churn Analysis
### An Industry-Level Data Analytics Project

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12%2B-4C72B0?style=for-the-badge)](https://seaborn.pydata.org)
[![Power BI](https://img.shields.io/badge/Power_BI-Dashboard-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)](https://powerbi.microsoft.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*Uncovering why customers leave and building a data-driven retention strategy for SaaS businesses.*

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Analysis Process](#-analysis-process)
- [Key Findings](#-key-findings)
- [Visualizations](#-visualizations)
- [Power BI Dashboard](#-power-bi-dashboard)
- [Business Recommendations](#-business-recommendations)
- [How to Run](#-how-to-run)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🎯 Project Overview

This project delivers a **complete end-to-end churn analysis** for a telecommunications company, identifying behavioral patterns that distinguish churned customers from retained ones. The analysis provides actionable strategies for customer retention, presented in a format ready for C-suite review.

**Core Questions Answered:**
- 🔴 Why are customers leaving?
- 👥 Which customer segments churn the most?
- ⏱️ How long do customers stay active?
- 💡 What business actions can reduce churn?
- 💰 What is the revenue impact of churn?

---

## 🚨 Problem Statement

Customer churn is a silent revenue killer in subscription businesses. In telecom and SaaS:

> **Acquiring a new customer costs 5–7× more than retaining an existing one.**

With an observed **26.5% churn rate** in our dataset, the business is losing approximately **$139,000 in monthly recurring revenue**. Without intervention, this compounds year over year. This analysis identifies the root causes and prescribes targeted solutions.

---

## 📂 Dataset

| Attribute    | Detail |
|--------------|--------|
| **Source**   | [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Rows**     | 7,043 customer records |
| **Columns**  | 21 features + 12 engineered features |
| **Target**   | `Churn` (Yes/No → 1/0) |
| **License**  | Public domain / Open use |

### Key Features
| Category | Features |
|----------|----------|
| Demographics | Gender, SeniorCitizen, Partner, Dependents |
| Services | PhoneService, InternetService, OnlineSecurity, TechSupport, StreamingTV |
| Account | Contract, PaperlessBilling, PaymentMethod |
| Financials | MonthlyCharges, TotalCharges, tenure |
| **Engineered** | TenureCategory, ChargeCategory, CustomerSegment, RiskScore, EstimatedCLV, CohortMonth |

---

## 🗂️ Project Structure

```
Customer_Retention_Churn_Analysis/
│
├── 📁 data/
│   ├── raw/
│   │   └── Telco_Customer_Churn.csv        ← Original dataset
│   └── processed/
│       ├── cleaned_churn_data.csv           ← After cleaning + feature engineering
│       └── cohort_retention_matrix.csv      ← Cohort analysis output
│
├── 📓 notebooks/
│   └── churn_analysis.ipynb                ← Complete end-to-end notebook
│
├── 🐍 scripts/
│   ├── data_cleaning.py                    ← Step 1: Clean & engineer features
│   ├── churn_analysis.py                   ← Step 2: EDA & visualizations
│   ├── cohort_analysis.py                  ← Step 3: Cohort retention analysis
│   └── retention_metrics.py               ← Step 4: Business metrics & CLV
│
├── 📊 dashboard/
│   ├── churn_dashboard.pbix               ← Power BI dashboard file
│   ├── PowerBI_DAX_Measures.md            ← All DAX formulas documented
│   └── dashboard_screenshots/             ← Dashboard preview images
│
├── 📄 reports/
│   ├── Executive_Summary.md               ← C-suite ready summary
│   └── Insights_Report.md                 ← 15 insights + 10 strategies
│
├── 🖼️ images/
│   ├── churn_rate.png
│   ├── churn_overview.png
│   ├── retention_trend.png
│   ├── cohort_heatmap.png
│   ├── customer_segments.png
│   └── ...
│
├── 📋 requirements.txt
├── 📄 LICENSE
└── 📖 README.md
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.10+** | Core analysis language |
| **Pandas** | Data manipulation & cleaning |
| **NumPy** | Numerical computations |
| **Matplotlib** | Custom visualizations |
| **Seaborn** | Statistical plotting |
| **Power BI** | Interactive business dashboard |
| **Jupyter Notebook** | Interactive exploration |
| **VS Code** | Development environment |
| **GitHub** | Version control & portfolio hosting |

---

## 🔬 Analysis Process

### Step 1 — Data Cleaning (`data_cleaning.py`)
- Handled `TotalCharges` null values (11 rows with spaces → filled with median)
- Removed duplicate `customerID` records
- Converted data types (`Churn`: Yes/No → 1/0, numeric coercions)
- **Engineered 12 new features**: TenureCategory, ChargeCategory, ContractSegment, EstimatedCLV, NumServices, RiskScore, CustomerSegment, CohortMonth, etc.

### Step 2 — Exploratory Data Analysis (`churn_analysis.py`)
- Churn Overview: total, churned, retained, overall rate
- Demographics: gender, senior citizen, partner, dependents
- Subscription: contract type, internet service, payment method
- Revenue: monthly charges distribution, total charges, revenue at risk
- Tenure: distribution by status, churn rate by tenure group

### Step 3 — Cohort Analysis (`cohort_analysis.py`)
- Built cohort table from simulated acquisition months (via tenure)
- Generated 12×12 cohort retention matrix
- Visualized as interactive heatmap and retention curves

### Step 4 — Retention Metrics (`retention_metrics.py`)
- Calculated: Churn Rate, Retention Rate, CLV, Average Customer Lifetime
- Monthly Revenue Retention (MRR)
- CLV segmented by contract type and customer segment

---

## 💡 Key Findings

| # | Finding | Impact |
|---|---------|--------|
| 1 | **Month-to-month contracts churn at 42.7%** — 15× higher than 2-year contracts | 🔴 Critical |
| 2 | **50%+ of churns happen in the first 12 months** | 🔴 Critical |
| 3 | **Electronic check users churn at 45.3%** — highest of all payment methods | 🟡 High |
| 4 | **Fiber Optic internet has 41.9% churn** despite being the premium tier | 🟡 High |
| 5 | **No OnlineSecurity → 41.8% churn** vs 14.6% with security | 🟡 High |
| 6 | **No TechSupport → 41.6% churn** vs 15.2% with support | 🟡 High |
| 7 | **Senior citizens churn less** than expected when on long-term contracts | 🟢 Insight |
| 8 | **Customers with dependents churn 50% less** than those without | 🟢 Insight |
| 9 | **High-value customers ($80+/month) churn at 35%** — disproportionate revenue risk | 🔴 Critical |
| 10 | **2-year contract CLV is 3.7× higher** than month-to-month CLV | 💰 Revenue |

---

## 📈 Visualizations

> All charts use a professional dark theme optimized for presentations.

<table>
<tr>
<td align="center"><b>Churn Overview</b></td>
<td align="center"><b>Cohort Heatmap</b></td>
</tr>
<tr>
<td><img src="images/churn_overview.png" width="400"/></td>
<td><img src="images/cohort_heatmap.png" width="400"/></td>
</tr>
<tr>
<td align="center"><b>Customer Segments</b></td>
<td align="center"><b>Revenue Analysis</b></td>
</tr>
<tr>
<td><img src="images/customer_segments.png" width="400"/></td>
<td><img src="images/revenue_analysis.png" width="400"/></td>
</tr>
</table>

---

## 📊 Power BI Dashboard

The interactive Power BI dashboard contains 4 pages:

| Page | Content |
|------|---------|
| **Page 1 – Executive Overview** | KPIs, churn distribution, revenue breakdown, contract analysis |
| **Page 2 – Customer Segmentation** | Churn by demographics, segment treemap, CLV by contract |
| **Page 3 – Cohort Analysis** | Retention heatmap matrix, cohort curves, tenure trends |
| **Page 4 – Business Recommendations** | Key findings cards, risk indicators, action priorities |

📁 Dashboard file: `dashboard/churn_dashboard.pbix`  
📋 All DAX measures: `dashboard/PowerBI_DAX_Measures.md`

---

## 🎯 Business Recommendations

| Priority | Strategy | Expected Impact |
|----------|----------|-----------------|
| 🔴 P0 | Contract Conversion Campaign (15–20% discount for annual) | -20% churn in M2M segment |
| 🔴 P0 | Onboarding Success Program (90-day CSM touchpoints) | -30% early churn |
| 🔴 P0 | Proactive Risk Outreach (RiskScore ≥ 6 triggers call) | Save $30K+/month |
| 🔴 P0 | Bundle Security + Support in base plans | -12% overall churn |
| 🟡 P1 | Auto-Pay Incentive ($5/month discount) | -15% e-check churn |
| 🟡 P1 | Fiber Optic SLA + Dedicated Support Queue | -17% Fiber churn |
| 🟡 P1 | Loyalty Rewards Program for 2yr+ customers | +8% NPS, +5% referrals |
| 🟡 P1 | Annual Business Reviews for high-value accounts | -20% premium churn |
| 🟢 P2 | Family Plan launch for multi-dependent customers | +7% segment retention |
| 🟢 P2 | Personalized re-engagement for dormant accounts | Recover 20% pre-churn |

**Projected Annual Revenue Protected:** **$834,000** (at target 15% churn rate)

---

## 🚀 How to Run

### Prerequisites
```bash
Python 3.10+
pip or conda
```

### Setup
```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Customer_Retention_Churn_Analysis.git
cd Customer_Retention_Churn_Analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
# Visit: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Place Telco_Customer_Churn.csv in: data/raw/

# 4. Run the full pipeline
python scripts/data_cleaning.py
python scripts/churn_analysis.py
python scripts/cohort_analysis.py
python scripts/retention_metrics.py

# 5. Or open the notebook
jupyter notebook notebooks/churn_analysis.ipynb
```

---

## 🔮 Future Improvements

- [ ] **Machine Learning Model** — Implement XGBoost/Random Forest churn classifier (target: 85%+ accuracy)
- [ ] **Survival Analysis** — Kaplan-Meier curves to model time-to-churn
- [ ] **SHAP Explainability** — Feature importance using SHAP values
- [ ] **A/B Testing Framework** — Measure impact of retention interventions
- [ ] **Real-Time Scoring API** — Deploy churn model as REST API using FastAPI
- [ ] **Advanced CLV Modeling** — BG/NBD or Pareto/NBD probabilistic models
- [ ] **Sentiment Analysis** — Integrate customer support tickets for early churn signals
- [ ] **Automated Reporting** — Schedule weekly PDF report generation

---

## 👤 Author

**Analytics Team – Future Interns Cohort 2026**

[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com/YOUR_USERNAME)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/YOUR_USERNAME)

---

<div align="center">

**⭐ If this project helped you, please star the repository!**

*Built with ❤️ using Python, Pandas, Seaborn, and Power BI*

</div>
