import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

# Calculate Pearson Correlation
correlations = {}

for feature in X.columns:
    x = X[feature]
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))

    r = numerator / denominator
    correlations[feature] = abs(r)

# Rank Features
correlations = pd.Series(correlations)
ranking = correlations.sort_values(ascending=False)

print("Pearson Correlation Ranking:")
print(ranking)


# Helper Evaluation Function
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2


results = []
for i in range(1, 10):
    top_features = ranking.head(i).index.tolist()

    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]

    model = LinearRegression()
    model.fit(X_train_top, y_train)

    y_pred = model.predict(X_test_top)
    mae, mse, rmse, r2 = evaluate_model(y_test, y_pred)

    results.append({
        "Number of Features": i,
        "Features": top_features,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })


# Convert results into DataFrame
results_df = pd.DataFrame(results)

print("\nModel Performance:")
print(results_df.to_string(index=False))


# Model with all 10 features (baseline)
model_all = LinearRegression()
model_all.fit(X_train, y_train)
y_pred_all = model_all.predict(X_test)
baseline = evaluate_model(y_test, y_pred_all)

print("\nAll 10 Features:")
print(f"MAE: {baseline[0]:.4f}")
print(f"MSE: {baseline[1]:.4f}")
print(f"RMSE: {baseline[2]:.4f}")
print(f"R2 Score: {baseline[3]:.4f}")


# Plot Results
metrics = ["MAE", "MSE", "RMSE", "R2 Score"]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for ax, metric in zip(axes.flatten(), metrics):
    ax.plot(
        results_df["Number of Features"],
        results_df[metric],
        marker="o",
        label="Top N Features"
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
with open("./variable-ranking/model_results.txt", "w") as file:
    file.write("PEARSON CORRELATION FEATURE SELECTION\n")
    file.write("=" * 60 + "\n\n")
    file.write("Feature Ranking:\n")
    file.write(str(ranking))
    file.write("\n\n")
    file.write("MODEL PERFORMANCE (Top K Features)\n")
    file.write("=" * 60 + "\n\n")
    file.write(results_df.to_string(index=False))
    file.write("\n\n")
    file.write("BASELINE MODEL (All 10 Features)\n")
    file.write("=" * 60 + "\n")
    file.write(f"MAE: {baseline[0]:.4f}\n")
    file.write(f"MSE: {baseline[1]:.4f}\n")
    file.write(f"RMSE: {baseline[2]:.4f}\n")
    file.write(f"R2 Score: {baseline[3]:.4f}\n")

print("Results successfully saved to model_results.txt")