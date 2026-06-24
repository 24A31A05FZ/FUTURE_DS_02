import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 7043

def generate_telco_dataset(n=N):

    cust_ids = [f"CUST-{i:05d}" for i in range(1, n + 1)]

    gender         = np.random.choice(["Male", "Female"], n)
    senior_citizen = np.random.binomial(1, 0.16, n)
    partner        = np.random.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents     = np.where(
        partner == "Yes",
        np.random.choice(["Yes", "No"], n, p=[0.4, 0.6]),
        np.random.choice(["Yes", "No"], n, p=[0.1, 0.9])
    )

    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]
    )

    tenure = np.where(
        contract == "Month-to-month", np.random.randint(1, 30, n),
        np.where(contract == "One year", np.random.randint(12, 48, n),
                 np.random.randint(24, 72, n))
    )

    internet = np.random.choice(
        ["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]
    )

    phone = np.where(internet == "No",
                     np.random.choice(["Yes", "No"], n, p=[0.9, 0.1]),
                     np.full(n, "Yes"))

    multiple_lines = np.where(
        phone == "Yes",
        np.random.choice(["Yes", "No", "No phone service"], n, p=[0.42, 0.42, 0.16]),
        np.full(n, "No phone service")
    )

    def inet_service(p_yes):
        return np.where(
            internet != "No",
            np.random.choice(["Yes", "No"], n, p=[p_yes, 1 - p_yes]),
            np.full(n, "No internet service")
        )

    online_security  = inet_service(0.29)
    online_backup    = inet_service(0.34)
    device_protect   = inet_service(0.34)
    tech_support     = inet_service(0.29)
    streaming_tv     = inet_service(0.38)
    streaming_movies = inet_service(0.39)

    paperless = np.random.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment   = np.random.choice(
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21]
    )

    base_charge = np.where(internet == "Fiber optic", 70, np.where(internet == "DSL", 45, 20))
    monthly_charges = (
        base_charge
        + (phone == "Yes").astype(int) * 10
        + (online_security  == "Yes").astype(int) * 8
        + (online_backup    == "Yes").astype(int) * 8
        + (device_protect   == "Yes").astype(int) * 8
        + (tech_support     == "Yes").astype(int) * 8
        + (streaming_tv     == "Yes").astype(int) * 8
        + (streaming_movies == "Yes").astype(int) * 8
        + np.random.normal(0, 3, n)
    ).clip(18, 119).round(2)

    total_charges = (monthly_charges * tenure + np.random.normal(0, 10, n)).clip(0).round(2)

    churn_prob = (
        0.10
        + (contract == "Month-to-month").astype(float) * 0.25
        + (tenure <= 12).astype(float) * 0.15
        + (internet == "Fiber optic").astype(float) * 0.10
        + (payment == "Electronic check").astype(float) * 0.08
        + (online_security  == "No").astype(float) * 0.06
        + (tech_support     == "No").astype(float) * 0.06
        + (paperless == "Yes").astype(float) * 0.04
        - (contract == "Two year").astype(float) * 0.10
        - (tenure > 36).astype(float) * 0.08
    ).clip(0.02, 0.90)

    churn = np.random.binomial(1, churn_prob, n)

    blank_idx = np.random.choice(n, 11, replace=False)
    total_charges_str = total_charges.astype(str)
    total_charges_str[blank_idx] = " "

    df = pd.DataFrame({
        "customerID"      : cust_ids,
        "gender"          : gender,
        "SeniorCitizen"   : senior_citizen,
        "Partner"         : partner,
        "Dependents"      : dependents,
        "tenure"          : tenure,
        "PhoneService"    : phone,
        "MultipleLines"   : multiple_lines,
        "InternetService" : internet,
        "OnlineSecurity"  : online_security,
        "OnlineBackup"    : online_backup,
        "DeviceProtection": device_protect,
        "TechSupport"     : tech_support,
        "StreamingTV"     : streaming_tv,
        "StreamingMovies" : streaming_movies,
        "Contract"        : contract,
        "PaperlessBilling": paperless,
        "PaymentMethod"   : payment,
        "MonthlyCharges"  : monthly_charges,
        "TotalCharges"    : total_charges_str,
        "Churn"           : np.where(churn == 1, "Yes", "No"),
    })

    return df


if __name__ == "__main__":
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "raw", "Telco_Customer_Churn.csv"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = generate_telco_dataset()
    df.to_csv(out_path, index=False)

    churn_rate = (df["Churn"] == "Yes").mean() * 100
    print(f"[OK] Dataset generated: {df.shape}")
    print(f"     Churn rate: {churn_rate:.1f}%")
    print(f"     Saved to: {out_path}")
