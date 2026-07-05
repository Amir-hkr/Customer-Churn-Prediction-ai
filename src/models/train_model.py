import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# -------------------------
# Load Data
# -------------------------
def load_data():
    df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    return df


# -------------------------
# Preprocess
# -------------------------
def preprocess(df):
    df = df.copy()

    # Drop ID
    df = df.drop("customerID", axis=1)

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Target
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    # Binary encoding
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]

    for col in binary_cols:
        df[col] = df[col].map({
            "Yes": 1,
            "No": 0,
            "Male": 1,
            "Female": 0
        })

    # One-hot encoding
    df = pd.get_dummies(df)

    return df


# -------------------------
# Train
# -------------------------
def train(df):

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # SAVE feature columns (IMPORTANT FOR STREAMLIT)
    feature_columns = X.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model
    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    # SAVE EVERYTHING
    joblib.dump(model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(feature_columns, "models/features.pkl")

    print("✅ Model + scaler + features saved")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    df = load_data()
    df = preprocess(df)
    train(df)