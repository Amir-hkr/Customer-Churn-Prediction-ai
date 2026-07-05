import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    print(df.head())
    return df


if __name__ == "__main__":
    path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = load_data(path)