import pandas as pd

def feature_engineering(df):
    df = df.copy()

    # فقط اگر وجود داشت حذف کن (امن‌تر)
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Binary encoding
    binary_cols = [
        "gender", "Partner", "Dependents",
        "PhoneService", "PaperlessBilling"
    ]

    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({
                "Yes": 1,
                "No": 0,
                "Male": 1,
                "Female": 0
            })

    # One-hot encoding
    multi_cols = [
        "MultipleLines", "InternetService",
        "Contract", "PaymentMethod",
        "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]

    df = pd.get_dummies(df, columns=[c for c in multi_cols if c in df.columns])

    print("Final shape after encoding:", df.shape)

    return df


if __name__ == "__main__":
    from src.data.clean_data import clean_data

    df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # clean step جدا (درست)
    df = clean_data(df)

    # feature engineering جدا
    df = feature_engineering(df)

    print(df.head())