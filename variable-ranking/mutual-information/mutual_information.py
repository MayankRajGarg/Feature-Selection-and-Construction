import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

data = load_diabetes()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Calculate Mutual Information
mi_scores = mutual_info_regression(
    X_train,
    y_train,
    random_state=42
)

# Convert into Series
mi_ranking = pd.Series(
    mi_scores,
    index=X.columns
).sort_values(ascending=False)

print("Mutual Information Ranking:")
print(mi_ranking)

results_mi = []

for i in range(1, 11):

    top_features = mi_ranking.head(i).index.tolist()
    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]

    model = LinearRegression()
    model.fit(X_train_top, y_train)
    y_pred = model.predict(X_test_top)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    results_mi.append({
        "Number of Features": i,
        "Features": top_features,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

results_mi_df = pd.DataFrame(results_mi)
print(results_mi_df)

model_all = LinearRegression()
model_all.fit(X_train, y_train)

y_pred_all = model_all.predict(X_test)

baseline = [
    mean_absolute_error(y_test, y_pred_all),
    mean_squared_error(y_test, y_pred_all),
    np.sqrt(mean_squared_error(y_test, y_pred_all)),
    r2_score(y_test, y_pred_all)
]

print("\nBaseline Model (All 10 Features):")
print(f"MAE: {baseline[0]:.4f}")
print(f"MSE: {baseline[1]:.4f}")
print(f"RMSE: {baseline[2]:.4f}")
print(f"R2 Score: {baseline[3]:.4f}")


metrics = ["MAE", "MSE", "RMSE", "R2 Score"]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for ax, metric in zip(axes.flatten(), metrics):
    ax.plot(
        results_mi_df["Number of Features"],
        results_mi_df[metric],
        marker="o",
        label="Mutual Information"
    )
    ax.axhline(
        y=baseline[metrics.index(metric)],
        linestyle="--",
        label="All 10 Features"
    )
    ax.set_xlabel("Number of Features")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs Number of Features")
    ax.set_xticks(range(1, 11))
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

# Store Results in a Text File
with open("./variable-ranking/mutual-information/model_results.txt", "w") as file:
    file.write("MUTUAL INFORMATION FEATURE SELECTION\n")
    file.write("=" * 60 + "\n\n")
    file.write("Mutual Information Ranking:\n")
    file.write(str(mi_ranking))
    file.write("\n\nMODEL PERFORMANCE (Top K Features)\n")
    file.write("=" * 60 + "\n\n")
    file.write(results_mi_df.to_string(index=False))
    file.write("\n\nBASELINE MODEL (All 10 Features)\n")
    file.write("=" * 60 + "\n")
    file.write(f"MAE: {baseline[0]:.4f}\n")
    file.write(f"MSE: {baseline[1]:.4f}\n")
    file.write(f"RMSE: {baseline[2]:.4f}\n")
    file.write(f"R2 Score: {baseline[3]:.4f}\n")


print("Results successfully saved to model_results.txt")