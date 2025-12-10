import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATA_PATH = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
PLOTS_DIR = "plots"

def ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at: {path}")
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    return df


def basic_overview(df: pd.DataFrame):
    print("\n=== HEAD (first 5 rows) ===")
    print(df.head())

    print("\n=== INFO ===")
    print(df.info())

    print("\n=== DESCRIBE (numeric) ===")
    print(df.describe())

    print("\n=== DESCRIBE (include=object) ===")
    print(df.describe(include="object"))

    print("\n=== MISSING VALUES PER COLUMN ===")
    print(df.isna().sum())


def plot_churn_distribution(df: pd.DataFrame):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Churn")
    plt.title("Churn vs Non-Churn Distribution")
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "churn_distribution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_churn_by_gender(df: pd.DataFrame):
    if "gender" not in df.columns:
        return

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="gender", hue="Churn")
    plt.title("Churn by Gender")
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "churn_by_gender.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_churn_by_senior_citizen(df: pd.DataFrame):
    if "SeniorCitizen" not in df.columns:
        return

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="SeniorCitizen", hue="Churn")
    plt.title("Churn by Senior Citizen (0 = No, 1 = Yes)")
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "churn_by_senior_citizen.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_churn_by_contract(df: pd.DataFrame):
    if "Contract" not in df.columns:
        return

    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x="Contract", hue="Churn")
    plt.title("Churn by Contract Type")
    plt.xticks(rotation=20)
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "churn_by_contract.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_tenure_distribution(df: pd.DataFrame):
    if "tenure" not in df.columns:
        return

    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x="tenure", bins=30, kde=True)
    plt.title("Tenure Distribution")
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "tenure_distribution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_tenure_by_churn(df: pd.DataFrame):
    if "tenure" not in df.columns:
        return

    plt.figure(figsize=(7, 4))
    sns.kdeplot(data=df, x="tenure", hue="Churn", common_norm=False)
    plt.title("Tenure Distribution by Churn")
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "tenure_by_churn.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_monthly_charges_by_churn(df: pd.DataFrame):
    if "MonthlyCharges" not in df.columns:
        return

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
    plt.title("Monthly Charges by Churn")
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "monthly_charges_by_churn.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_internet_service_by_churn(df: pd.DataFrame):
    if "InternetService" not in df.columns:
        return

    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x="InternetService", hue="Churn")
    plt.title("Churn by Internet Service Type")
    plt.xticks(rotation=20)
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "churn_by_internet_service.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_payment_method_by_churn(df: pd.DataFrame):
    if "PaymentMethod" not in df.columns:
        return

    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="PaymentMethod", hue="Churn")
    plt.title("Churn by Payment Method")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "churn_by_payment_method.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def run_eda():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Cleaning data...")
    df = clean_data(df)

    print("Basic overview:")
    basic_overview(df)

    print("\nGenerating plots...")
    ensure_plots_dir()

    plot_churn_distribution(df)
    plot_churn_by_gender(df)
    plot_churn_by_senior_citizen(df)
    plot_churn_by_contract(df)
    plot_tenure_distribution(df)
    plot_tenure_by_churn(df)
    plot_monthly_charges_by_churn(df)
    plot_internet_service_by_churn(df)
    plot_payment_method_by_churn(df)

    print("\nEDA complete. Plots saved in:", PLOTS_DIR)


if __name__ == "__main__":
    run_eda()
