import pandas as pd

def clean_data(df):
    # 1. حذف ID (هیچ ارزشی برای مدل نداره)
    df = df.drop("customerID", axis=1)

    # 2. تبدیل TotalCharges به عدد
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 3. پر کردن missing هایی که ایجاد شد
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # 4. تبدیل target به 0/1
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    print("Missing after cleaning:")
    print(df.isnull().sum().sum())

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df_clean = clean_data(df)

    print(df_clean.head())