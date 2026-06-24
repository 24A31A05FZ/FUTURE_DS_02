import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

RAW_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "raw",       "Telco_Customer_Churn.csv")
OUT_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cleaned_churn_data.csv")

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"✅  Loaded data  → Shape: {df.shape}")
    print(f"    Columns     : {list(df.columns)}\n")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("  DATA INSPECTION REPORT")
    print("=" * 60)
    print(f"  Rows       : {df.shape[0]:,}")
    print(f"  Columns    : {df.shape[1]}")
    print(f"  Duplicates : {df.duplicated().sum()}")
    print("\n  Null values per column:")
    nulls = df.isnull().sum()
    print(nulls[nulls > 0].to_string() if nulls.sum() else "  → None found ✅")
    print("\n  Data types:")
    print(df.dtypes.to_string())
    print("=" * 60 + "\n")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing_tc = df["TotalCharges"].isnull().sum()
    if missing_tc:
        median_tc = df["TotalCharges"].median()
        df["TotalCharges"].fillna(median_tc, inplace=True)
        print(f"✅  TotalCharges: filled {missing_tc} nulls with median ({median_tc:.2f})")

    before = len(df)
    df.dropna(subset=["customerID"], inplace=True)
    print(f"✅  Dropped {before - len(df)} rows with null customerID")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset="customerID", keep="first")
    print(f"✅  Removed {before - len(df)} duplicate customerIDs\n")
    return df


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print("✅  Data types converted successfully")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    bins   = [0, 12, 24, 48, 60, 72]
    labels = ["0-12 mo", "13-24 mo", "25-48 mo", "49-60 mo", "61-72 mo"]
    df["TenureCategory"] = pd.cut(
        df["tenure"], bins=bins, labels=labels, include_lowest=True
    )

    df["ChargeCategory"] = pd.cut(
        df["MonthlyCharges"],
        bins=[0, 35, 65, 95, 120],
        labels=["Low (<$35)", "Medium ($35-$65)", "High ($65-$95)", "Premium (>$95)"],
        include_lowest=True,
    )

    contract_map = {
        "Month-to-month": "Short-Term",
        "One year"       : "Mid-Term",
        "Two year"       : "Long-Term",
    }
    df["ContractSegment"] = df["Contract"].map(contract_map)

    df["EstimatedCLV"] = df["MonthlyCharges"] * (df["tenure"] + 1)

    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    def _count_services(row):
        count = 0
        for col in service_cols:
            val = str(row[col]).strip().lower()
            if val not in ["no", "no phone service", "no internet service"]:
                count += 1
        return count

    df["NumServices"] = df.apply(_count_services, axis=1)

    df["SeniorLabel"] = df["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior"})

    df["HasFamily"] = (
        (df["Partner"] == "Yes") | (df["Dependents"] == "Yes")
    ).astype(int)

    df["RevenueLostIfChurned"] = df["MonthlyCharges"] * df["Churn"]

    import datetime
    BASE_DATE = datetime.date(2023, 1, 1)
    df["CohortMonth"] = df["tenure"].apply(
        lambda t: (BASE_DATE - datetime.timedelta(days=30 * int(t))).strftime("%Y-%m")
    )

    df["RiskScore"] = 0
    df.loc[df["Contract"] == "Month-to-month",                          "RiskScore"] += 3
    df.loc[df["tenure"] <= 12,                                          "RiskScore"] += 2
    df.loc[df["MonthlyCharges"] > 65,                                   "RiskScore"] += 1
    df.loc[df["TechSupport"] == "No",                                   "RiskScore"] += 1
    df.loc[df["OnlineSecurity"] == "No",                                "RiskScore"] += 1
    df.loc[df["PaymentMethod"] == "Electronic check",                   "RiskScore"] += 1

    def _segment(score):
        if score >= 6:
            return "High Risk"
        elif score >= 4:
            return "Medium Risk"
        else:
            return "Loyal"

    df["CustomerSegment"] = df["RiskScore"].apply(_segment)

    print("✅  Feature engineering complete – added 12 new columns\n")
    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅  Cleaned data saved → {path}")
    print(f"    Final shape: {df.shape}")


def main():
    print("\n" + "=" * 60)
    print("  CUSTOMER CHURN ANALYSIS – DATA CLEANING PIPELINE")
    print("=" * 60 + "\n")

    df = load_data(RAW_PATH)
    inspect_data(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = convert_data_types(df)
    df = engineer_features(df)
    save_data(df, OUT_PATH)

    print("\n🎉  Data cleaning pipeline completed successfully!")
    print("=" * 60 + "\n")
    return df


if __name__ == "__main__":
    df = main()
