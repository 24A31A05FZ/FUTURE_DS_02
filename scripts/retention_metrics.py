import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cleaned_churn_data.csv")
IMG_DIR   = os.path.join(os.path.dirname(__file__), "..", "images")
os.makedirs(IMG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor" : "#0D1117",
    "axes.facecolor"   : "#161B22",
    "axes.edgecolor"   : "#30363D",
    "axes.labelcolor"  : "#C9D1D9",
    "xtick.color"      : "#8B949E",
    "ytick.color"      : "#8B949E",
    "text.color"       : "#C9D1D9",
    "font.size"        : 11,
    "axes.titlesize"   : 14,
    "axes.titleweight" : "bold",
})

CHURN_CLR  = "#E74C3C"
RETAIN_CLR = "#2ECC71"
ACCENT_CLR = "#58A6FF"


def save_fig(name: str):
    path = os.path.join(IMG_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=plt.rcParams["figure.facecolor"])
    plt.close()
    print(f"  💾  Saved → {path}")


def calculate_churn_rate(df: pd.DataFrame) -> dict:
    total    = len(df)
    churned  = df["Churn"].sum()
    retained = total - churned

    overall_churn_rate    = (churned  / total) * 100
    overall_retention_rate = (retained / total) * 100

    contract_churn = (
        df.groupby("Contract")["Churn"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "Churned", "count": "Total"})
    )
    contract_churn["ChurnRate"]     = (contract_churn["Churned"] / contract_churn["Total"]) * 100
    contract_churn["RetentionRate"] = 100 - contract_churn["ChurnRate"]

    return {
        "total"                 : total,
        "churned"               : int(churned),
        "retained"              : int(retained),
        "overall_churn_rate"    : overall_churn_rate,
        "overall_retention_rate": overall_retention_rate,
        "contract_churn"        : contract_churn,
    }


def calculate_clv(df: pd.DataFrame) -> dict:
    monthly_churn_rate    = df["Churn"].mean()
    avg_customer_lifetime = 1 / monthly_churn_rate if monthly_churn_rate > 0 else np.inf

    avg_monthly_revenue   = df["MonthlyCharges"].mean()
    clv_formula           = avg_monthly_revenue * avg_customer_lifetime

    seg_clv = df.groupby("CustomerSegment")["EstimatedCLV"].agg(["mean", "sum"]).round(2)

    contract_clv = df.groupby("Contract")["EstimatedCLV"].mean().round(2)

    return {
        "monthly_churn_rate"    : monthly_churn_rate,
        "avg_customer_lifetime" : avg_customer_lifetime,
        "avg_monthly_revenue"   : avg_monthly_revenue,
        "theoretical_clv"       : clv_formula,
        "avg_estimated_clv"     : df["EstimatedCLV"].mean(),
        "total_clv"             : df["EstimatedCLV"].sum(),
        "seg_clv"               : seg_clv,
        "contract_clv"          : contract_clv,
    }


def calculate_revenue_retention(df: pd.DataFrame) -> dict:
    total_mrr    = df["MonthlyCharges"].sum()
    churned_mrr  = df[df["Churn"] == 1]["MonthlyCharges"].sum()
    retained_mrr = total_mrr - churned_mrr

    mrr_retention_rate = (retained_mrr / total_mrr) * 100
    revenue_at_risk    = churned_mrr

    revenue_contract = df.groupby("Contract")["MonthlyCharges"].sum()
    churn_rev_contract = df[df["Churn"] == 1].groupby("Contract")["MonthlyCharges"].sum()

    return {
        "total_mrr"          : total_mrr,
        "churned_mrr"        : churned_mrr,
        "retained_mrr"       : retained_mrr,
        "mrr_retention_rate" : mrr_retention_rate,
        "revenue_at_risk"    : revenue_at_risk,
        "revenue_contract"   : revenue_contract,
        "churn_rev_contract" : churn_rev_contract,
    }


def print_metrics_report(churn_metrics, clv_metrics, revenue_metrics):
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║   RETENTION METRICS REPORT                              ║")
    print("╠" + "═" * 58 + "╣")

    cm = churn_metrics
    print(f"║  Total Customers       : {cm['total']:>10,}                    ║")
    print(f"║  Churned Customers     : {cm['churned']:>10,}                    ║")
    print(f"║  Retained Customers    : {cm['retained']:>10,}                    ║")
    print(f"║  Overall Churn Rate    : {cm['overall_churn_rate']:>9.2f}%                    ║")
    print(f"║  Overall Retention Rate: {cm['overall_retention_rate']:>9.2f}%                    ║")
    print("╠" + "═" * 58 + "╣")

    cv = clv_metrics
    print(f"║  Monthly Churn Rate    : {cv['monthly_churn_rate']:>9.4f}                     ║")
    print(f"║  Avg Customer Lifetime : {cv['avg_customer_lifetime']:>9.1f} months              ║")
    print(f"║  Avg Monthly Revenue   : ${cv['avg_monthly_revenue']:>9.2f}                   ║")
    print(f"║  Theoretical CLV       : ${cv['theoretical_clv']:>9,.2f}                  ║")
    print(f"║  Avg Estimated CLV     : ${cv['avg_estimated_clv']:>9,.2f}                  ║")
    print(f"║  Total Portfolio CLV   : ${cv['total_clv']:>12,.2f}               ║")
    print("╠" + "═" * 58 + "╣")

    rv = revenue_metrics
    print(f"║  Total MRR             : ${rv['total_mrr']:>12,.2f}               ║")
    print(f"║  Retained MRR          : ${rv['retained_mrr']:>12,.2f}               ║")
    print(f"║  Revenue at Risk       : ${rv['churned_mrr']:>12,.2f}               ║")
    print(f"║  MRR Retention Rate    : {rv['mrr_retention_rate']:>9.2f}%                    ║")
    print("╚" + "═" * 58 + "╝\n")

    print("  Contract-level Churn Rates:")
    print(cm["contract_churn"][["ChurnRate", "RetentionRate"]].to_string())
    print()

    print("  CLV by Customer Segment:")
    print(cv["seg_clv"].to_string())
    print()


def plot_metrics_dashboard(df, churn_metrics, clv_metrics, revenue_metrics):
    print("📊  Plotting Metrics Dashboard")

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0D1117")
    fig.suptitle("Customer Retention Metrics Dashboard",
                 fontsize=20, fontweight="bold", color=ACCENT_CLR, y=1.01)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    seg_clv = clv_metrics["seg_clv"].reset_index()
    colors = {
        "High Risk"  : CHURN_CLR,
        "Medium Risk": "#F39C12",
        "Loyal"      : RETAIN_CLR,
    }
    bar_colors = [colors.get(s, ACCENT_CLR) for s in seg_clv["CustomerSegment"]]
    bars = ax1.bar(seg_clv["CustomerSegment"], seg_clv["mean"],
                   color=bar_colors, edgecolor="#0D1117")
    ax1.set_title("Avg CLV by Segment")
    ax1.set_ylabel("Avg CLV ($)")
    ax1.yaxis.grid(True, alpha=0.3)
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 20,
                 f"${bar.get_height():,.0f}",
                 ha="center", fontsize=9, fontweight="bold")

    ax2 = fig.add_subplot(gs[0, 1])
    rv = revenue_metrics
    cats   = ["Total MRR", "Revenue at Risk", "Retained MRR"]
    values = [rv["total_mrr"], rv["churned_mrr"], rv["retained_mrr"]]
    bar_c  = [ACCENT_CLR, CHURN_CLR, RETAIN_CLR]
    bars2  = ax2.bar(cats, values, color=bar_c, edgecolor="#0D1117")
    ax2.set_title("Monthly Revenue Breakdown")
    ax2.set_ylabel("Revenue ($)")
    ax2.tick_params(axis="x", rotation=15)
    ax2.yaxis.grid(True, alpha=0.3)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + rv["total_mrr"] * 0.005,
                 f"${bar.get_height():,.0f}",
                 ha="center", fontsize=9, fontweight="bold")

    ax3 = fig.add_subplot(gs[0, 2])
    contract_clv = clv_metrics["contract_clv"].reset_index()
    sns.barplot(data=contract_clv, x="Contract", y="EstimatedCLV",
                palette="viridis", ax=ax3)
    ax3.set_title("Avg CLV by Contract Type")
    ax3.set_ylabel("Avg CLV ($)")
    ax3.tick_params(axis="x", rotation=15)
    ax3.yaxis.grid(True, alpha=0.3)
    for p in ax3.patches:
        ax3.annotate(f"${p.get_height():,.0f}",
                     (p.get_x() + p.get_width() / 2, p.get_height()),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=9, fontweight="bold")

    ax4 = fig.add_subplot(gs[1, 0])
    contract_ch = churn_metrics["contract_churn"].reset_index()
    sns.barplot(data=contract_ch, x="Contract", y="ChurnRate",
                palette=[CHURN_CLR, "#F39C12", RETAIN_CLR], ax=ax4)
    ax4.set_title("Churn Rate by Contract")
    ax4.set_ylabel("Churn Rate (%)")
    ax4.tick_params(axis="x", rotation=15)
    ax4.yaxis.grid(True, alpha=0.3)
    for p in ax4.patches:
        ax4.annotate(f"{p.get_height():.1f}%",
                     (p.get_x() + p.get_width() / 2, p.get_height()),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=10, fontweight="bold")

    ax5 = fig.add_subplot(gs[1, 1])
    charge_churn = df.groupby("ChargeCategory")["Churn"].mean().mul(100).reset_index()
    sns.barplot(data=charge_churn, x="ChargeCategory", y="Churn",
                palette="RdYlGn_r", ax=ax5)
    ax5.set_title("Churn Rate by Charge Category")
    ax5.set_xlabel("Monthly Charge Tier")
    ax5.set_ylabel("Churn Rate (%)")
    ax5.tick_params(axis="x", rotation=25)
    ax5.yaxis.grid(True, alpha=0.3)
    for p in ax5.patches:
        ax5.annotate(f"{p.get_height():.1f}%",
                     (p.get_x() + p.get_width() / 2, p.get_height()),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=9, fontweight="bold")

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor("#1C2128")
    ret_rate = churn_metrics["overall_retention_rate"]
    ax6.set_xlim(0, 1); ax6.set_ylim(0, 1)
    ax6.axis("off")

    theta = np.linspace(np.pi, 0, 200)
    ax6.plot(0.5 + 0.38 * np.cos(theta), 0.15 + 0.38 * np.sin(theta),
             color="#30363D", linewidth=18, solid_capstyle="round")
    filled = int(len(theta) * (ret_rate / 100))
    ax6.plot(0.5 + 0.38 * np.cos(theta[:filled]),
             0.15 + 0.38 * np.sin(theta[:filled]),
             color=RETAIN_CLR, linewidth=18, solid_capstyle="round")

    ax6.text(0.5, 0.52, f"{ret_rate:.1f}%",
             ha="center", fontsize=30, fontweight="bold", color=RETAIN_CLR,
             transform=ax6.transAxes)
    ax6.text(0.5, 0.33, "Retention Rate",
             ha="center", fontsize=12, color="#8B949E",
             transform=ax6.transAxes)
    ax6.set_title("Overall Retention Rate")

    plt.tight_layout()
    save_fig("retention_metrics_dashboard.png")


def main():
    print("\n" + "=" * 60)
    print("  CHURN ANALYSIS – RETENTION METRICS")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"✅  Loaded cleaned data → {df.shape}\n")

    churn_metrics   = calculate_churn_rate(df)
    clv_metrics     = calculate_clv(df)
    revenue_metrics = calculate_revenue_retention(df)

    print_metrics_report(churn_metrics, clv_metrics, revenue_metrics)
    plot_metrics_dashboard(df, churn_metrics, clv_metrics, revenue_metrics)

    print("\n🎉  Retention metrics analysis complete!")
    print("=" * 60 + "\n")

    return churn_metrics, clv_metrics, revenue_metrics


if __name__ == "__main__":
    main()
