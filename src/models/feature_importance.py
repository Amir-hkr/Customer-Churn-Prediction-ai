import pandas as pd
import joblib
import matplotlib.pyplot as plt


# -------------------------
# Load model + features
# -------------------------
model = joblib.load("models/logistic_model.pkl")
features = joblib.load("models/features.pkl")


# -------------------------
# فقط برای Logistic Regression
# -------------------------
if hasattr(model, "coef_"):
    importance = model.coef_[0]

    df_imp = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    })

    df_imp["abs"] = df_imp["Importance"].abs()
    df_imp = df_imp.sort_values("abs", ascending=False).head(15)

    print(df_imp[["Feature", "Importance"]])

    # -------------------------
    # Plot
    # -------------------------
    plt.figure(figsize=(10, 6))
    plt.barh(df_imp["Feature"], df_imp["Importance"])
    plt.gca().invert_yaxis()
    plt.title("Top 15 Feature Importance (Logistic Regression)")
    plt.show()

else:
    print("Model does not support feature importance")

