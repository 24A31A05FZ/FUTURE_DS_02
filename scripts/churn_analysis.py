import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cleaned_churn_data.csv")
IMG_DIR    = os.path.join(os.path.dirname(__file__), "..", "images")
os.makedirs(IMG_DIR, exist_ok=True)

PALETTE    = {"Churned": "#E74C3C", "Retained": "#2ECC71"}
CHURN_CLR  = "#E74C3C"
RETAIN_CLR = "#2ECC71"

plt.rcParams.update({
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
    "font.family"       : "DejaVu Sans",
    "font.size"         : 11,
    "axes.titlesize"    : 14,
    "axes.titleweight"  : "bold",
    "legend.facecolor"  : "#161B22",
    "legend.edgecolor"  : "#30363D",
})

def save_fig(name: str):
    path = os.path.join(IMG_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=plt.rcParams["figure.facecolor"])
    plt.close()
    print(f"  💾  Saved → {path}")


def _bar_pct(ax, container):
    total = sum(p.get_height() for p in container)
    for p in container:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f"{h/total*100:.1f}%",
                xy=(p.get_x() + p.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=9, color="#C9D1D9",
            )


def plot_churn_overview(df: pd.DataFrame):
    print("\n📊  [1/8] Churn Overview")

    total     = len(df)
    churned   = df["Churn"].sum()
    retained  = total - churned
    churn_rate = churned / total * 100

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle("Customer Churn Overview", fontsize=18, fontweight="bold", color="#58A6FF", y=1.02)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    kpis = [
        ("Total Customers", f"{total:,}",  "#58A6FF"),
        ("Churned",         f"{churned:,}", CHURN_CLR),
        ("Retained",        f"{retained:,}", RETAIN_CLR),
    ]
    for i, (label, value, color) in enumerate(kpis):
        ax = fig.add_subplot(gs[i])
        ax.set_facecolor("#1C2128")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.5, 0.62, value, ha="center", va="center",
                fontsize=36, fontweight="bold", color=color)
        ax.text(0.5, 0.35, label, ha="center", va="center",
                fontsize=13, color="#8B949E")
        if i == 1:
            ax.text(0.5, 0.14, f"({churn_rate:.1f}% churn rate)", ha="center",
                    fontsize=10, color=CHURN_CLR)
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(2)
        ax.set_visible(True)

    plt.tight_layout()
    save_fig("churn_rate.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0D1117")

    ax = axes[0]
    ax.set_facecolor("#161B22")
    wedges, texts, autotexts = ax.pie(
        [retained, churned],
        labels=["Retained", "Churned"],
        autopct="%1.1f%%",
        startangle=140,
        colors=[RETAIN_CLR, CHURN_CLR],
        wedgeprops={"width": 0.55, "edgecolor": "#0D1117", "linewidth": 2},
        textprops={"color": "#C9D1D9", "fontsize": 12},
    )
    for at in autotexts:
        at.set_fontsize(13); at.set_fontweight("bold")
    ax.set_title("Churn vs Retention Distribution", pad=12)

    ax2 = axes[1]
    bars = ax2.bar(
        ["Retained", "Churned"], [retained, churned],
        color=[RETAIN_CLR, CHURN_CLR], width=0.45,
        edgecolor="#0D1117", linewidth=1.5,
    )
    ax2.set_title("Customer Count by Churn Status")
    ax2.set_ylabel("Number of Customers")
    ax2.yaxis.grid(True, alpha=0.3)
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 50,
                 f"{bar.get_height():,}",
                 ha="center", fontsize=12, fontweight="bold", color="#C9D1D9")

    plt.suptitle("Churn Overview Dashboard", fontsize=16, fontweight="bold",
                 color="#58A6FF", y=1.02)
    plt.tight_layout()
    save_fig("churn_overview.png")


def plot_demographics(df: pd.DataFrame):
    print("\n📊  [2/8] Customer Demographics")

    demo_cols = {
        "Gender"       : "gender",
        "Senior Citizen": "SeniorLabel",
        "Has Partner"  : "Partner",
        "Has Dependents": "Dependents",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor("#0D1117")
    fig.suptitle("Churn Analysis by Customer Demographics",
                 fontsize=18, fontweight="bold", color="#58A6FF")

    for ax, (title, col) in zip(axes.flat, demo_cols.items()):
        churn_by = df.groupby(col)["Churn"].mean().mul(100).reset_index()
        churn_by.columns = [col, "ChurnRate"]
        sns.barplot(
            data=churn_by, x=col, y="ChurnRate", ax=ax,
            palette=[RETAIN_CLR, CHURN_CLR], edgecolor="#0D1117",
        )
        ax.set_title(f"Churn Rate by {title}")
        ax.set_ylabel("Churn Rate (%)")
        ax.set_xlabel(title)
        ax.yaxis.grid(True, alpha=0.3)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}%",
                        (p.get_x() + p.get_width() / 2, p.get_height()),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    save_fig("customer_segments.png")


def plot_subscription_analysis(df: pd.DataFrame):
    print("\n📊  [3/8] Subscription Analysis")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor("#0D1117")
    fig.suptitle("Churn by Subscription & Service Type",
                 fontsize=18, fontweight="bold", color="#58A6FF")

    cols = ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling"]
    titles = ["Contract Type", "Internet Service", "Payment Method", "Paperless Billing"]

    for ax, col, title in zip(axes.flat, cols, titles):
        counts = df.groupby([col, "Churn"]).size().reset_index(name="Count")
        counts["Status"] = counts["Churn"].map({1: "Churned", 0: "Retained"})
        sns.barplot(data=counts, x=col, y="Count", hue="Status",
                    palette=PALETTE, ax=ax, edgecolor="#0D1117")
        ax.set_title(f"Churn by {title}")
        ax.set_xlabel(title)
        ax.set_ylabel("Customers")
        ax.tick_params(axis="x", rotation=20)
        ax.yaxis.grid(True, alpha=0.3)
        ax.legend(title="Status")

    plt.tight_layout()
    save_fig("subscription_analysis.png")


def plot_revenue_analysis(df: pd.DataFrame):
    print("\n📊  [4/8] Revenue Analysis")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0D1117")
    fig.suptitle("Revenue Impact Analysis", fontsize=18,
                 fontweight="bold", color="#58A6FF")

    ax1 = axes[0]
    for label, color, val in [("Retained", RETAIN_CLR, 0), ("Churned", CHURN_CLR, 1)]:
        subset = df[df["Churn"] == val]["MonthlyCharges"]
        ax1.hist(subset, bins=30, alpha=0.65, color=color, label=label, edgecolor="none")
    ax1.set_title("Monthly Charges Distribution")
    ax1.set_xlabel("Monthly Charges ($)")
    ax1.set_ylabel("Count")
    ax1.legend()
    ax1.yaxis.grid(True, alpha=0.3)

    ax2 = axes[1]
    plot_df = df.copy()
    plot_df["Status"] = plot_df["Churn"].map({1: "Churned", 0: "Retained"})
    sns.boxplot(data=plot_df, x="Status", y="TotalCharges",
                palette={"Churned": CHURN_CLR, "Retained": RETAIN_CLR}, ax=ax2)
    ax2.set_title("Total Charges by Status")
    ax2.set_xlabel("Customer Status")
    ax2.set_ylabel("Total Charges ($)")
    ax2.yaxis.grid(True, alpha=0.3)

    ax3 = axes[2]
    revenue_loss = df[df["Churn"] == 1]["MonthlyCharges"].sum()
    revenue_safe = df[df["Churn"] == 0]["MonthlyCharges"].sum()
    bars = ax3.bar(["Revenue at Risk\n(Churned)", "Revenue Secured\n(Retained)"],
                   [revenue_loss, revenue_safe],
                   color=[CHURN_CLR, RETAIN_CLR],
                   edgecolor="#0D1117", linewidth=1.5)
    ax3.set_title("Monthly Revenue Distribution")
    ax3.set_ylabel("Revenue ($)")
    ax3.yaxis.grid(True, alpha=0.3)
    for bar in bars:
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + revenue_safe * 0.01,
                 f"${bar.get_height():,.0f}",
                 ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    save_fig("revenue_analysis.png")


def plot_tenure_analysis(df: pd.DataFrame):
    print("\n📊  [5/8] Tenure Analysis")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0D1117")
    fig.suptitle("Customer Tenure Analysis", fontsize=18,
                 fontweight="bold", color="#58A6FF")

    ax1 = axes[0]
    for label, color, val in [("Retained", RETAIN_CLR, 0), ("Churned", CHURN_CLR, 1)]:
        subset = df[df["Churn"] == val]["tenure"]
        ax1.hist(subset, bins=25, alpha=0.65, color=color, label=label, edgecolor="none")
    ax1.set_title("Tenure Distribution by Status")
    ax1.set_xlabel("Tenure (Months)")
    ax1.set_ylabel("Count")
    ax1.legend()
    ax1.yaxis.grid(True, alpha=0.3)

    ax2 = axes[1]
    tenure_churn = df.groupby("TenureCategory")["Churn"].mean().mul(100).reset_index()
    sns.barplot(data=tenure_churn, x="TenureCategory", y="Churn",
                palette="RdYlGn_r", ax=ax2)
    ax2.set_title("Churn Rate by Tenure Group")
    ax2.set_xlabel("Tenure Group")
    ax2.set_ylabel("Churn Rate (%)")
    ax2.tick_params(axis="x", rotation=20)
    ax2.yaxis.grid(True, alpha=0.3)
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.1f}%",
                     (p.get_x() + p.get_width() / 2, p.get_height()),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=10, fontweight="bold")

    ax3 = axes[2]
    clv_seg = df.groupby("CustomerSegment")["EstimatedCLV"].mean().reset_index()
    colors = {"Loyal": RETAIN_CLR, "Medium Risk": "#F39C12", "High Risk": CHURN_CLR}
    sns.barplot(data=clv_seg, x="CustomerSegment", y="EstimatedCLV",
                palette=colors, ax=ax3)
    ax3.set_title("Average CLV by Customer Segment")
    ax3.set_xlabel("Segment")
    ax3.set_ylabel("Estimated CLV ($)")
    ax3.yaxis.grid(True, alpha=0.3)
    for p in ax3.patches:
        ax3.annotate(f"${p.get_height():,.0f}",
                     (p.get_x() + p.get_width() / 2, p.get_height()),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    save_fig("retention_trend.png")


def plot_correlation_heatmap(df: pd.DataFrame):
    print("\n📊  [6/8] Correlation Heatmap")

    numeric_df = df.select_dtypes(include=[np.number]).drop(
        columns=["RiskScore", "HasFamily", "RevenueLostIfChurned"], errors="ignore"
    )

    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("#0D1117")
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", linewidths=0.5,
        cmap="coolwarm", center=0, ax=ax,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 9},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=15, pad=12)
    plt.tight_layout()
    save_fig("correlation_heatmap.png")


def plot_segment_breakdown(df: pd.DataFrame):
    print("\n📊  [7/8] Customer Segment Breakdown")

    seg_counts = df["CustomerSegment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0D1117")
    fig.suptitle("Customer Risk Segmentation",
                 fontsize=18, fontweight="bold", color="#58A6FF")

    colors = [
        "#E74C3C" if s == "High Risk" else
        "#F39C12" if s == "Medium Risk" else
        "#2ECC71"
        for s in seg_counts["Segment"]
    ]

    ax1 = axes[0]
    wedges, texts, autotexts = ax1.pie(
        seg_counts["Count"],
        labels=seg_counts["Segment"],
        autopct="%1.1f%%",
        colors=colors,
        startangle=120,
        wedgeprops={"edgecolor": "#0D1117", "linewidth": 2},
        textprops={"color": "#C9D1D9"},
    )
    for at in autotexts:
        at.set_fontsize(12); at.set_fontweight("bold")
    ax1.set_title("Segment Distribution")

    ax2 = axes[1]
    churn_by_seg = df.groupby("CustomerSegment")["Churn"].mean().mul(100).reset_index()
    seg_colors = {
        "High Risk"   : CHURN_CLR,
        "Medium Risk" : "#F39C12",
        "Loyal"       : RETAIN_CLR,
    }
    sns.barplot(data=churn_by_seg, x="CustomerSegment", y="Churn",
                palette=seg_colors, ax=ax2)
    ax2.set_title("Actual Churn Rate by Segment")
    ax2.set_xlabel("Customer Segment")
    ax2.set_ylabel("Churn Rate (%)")
    ax2.yaxis.grid(True, alpha=0.3)
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.1f}%",
                     (p.get_x() + p.get_width() / 2, p.get_height()),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    save_fig("segment_breakdown.png")


def plot_services_impact(df: pd.DataFrame):
    print("\n📊  [8/8] Services Impact")

    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]

    churn_rates = {}
    for col in service_cols:
        subset = df[df[col].isin(["Yes", "No"])]
        rate = subset.groupby(col)["Churn"].mean().mul(100)
        churn_rates[col] = {"Has Service": rate.get("Yes", 0),
                            "No Service" : rate.get("No",  0)}

    rate_df = pd.DataFrame(churn_rates).T.reset_index()
    rate_df.columns = ["Service", "Has Service", "No Service"]

    rate_long = rate_df.melt(id_vars="Service", var_name="Status", value_name="ChurnRate")

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#0D1117")
    sns.barplot(data=rate_long, x="Service", y="ChurnRate", hue="Status",
                palette={"Has Service": RETAIN_CLR, "No Service": CHURN_CLR}, ax=ax)
    ax.set_title("Churn Rate: With vs Without Each Service",
                 fontsize=15, fontweight="bold")
    ax.set_xlabel("Service")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=20)
    ax.yaxis.grid(True, alpha=0.3)
    ax.legend(title="Subscription Status")

    plt.tight_layout()
    save_fig("services_impact.png")


def main():
    print("\n" + "=" * 60)
    print("  CHURN ANALYSIS – EDA & VISUALIZATIONS")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"✅  Loaded cleaned data → {df.shape}\n")

    plot_churn_overview(df)
    plot_demographics(df)
    plot_subscription_analysis(df)
    plot_revenue_analysis(df)
    plot_tenure_analysis(df)
    plot_correlation_heatmap(df)
    plot_segment_breakdown(df)
    plot_services_impact(df)

    print("\n🎉  EDA complete – all charts saved to /images/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
