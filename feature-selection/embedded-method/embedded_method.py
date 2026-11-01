import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

selection_model = DecisionTreeRegressor(
    max_depth=3,
    random_state=42
)

selection_model.fit(X_train, y_train)

importance = pd.Series(
    selection_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("Feature Importance Ranking:")
print(importance)

# Select Features with Non-Zero Importance
selected_features = importance[importance > 0].index.tolist()

print("\nSelected Features:")
print(selected_features)

print("Number of Selected Features:", len(selected_features))


def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return [mae, mse, rmse, r2]


model_all = DecisionTreeRegressor(
    max_depth=3,
    random_state=42
)
model_all.fit(X_train, y_train)
predictions_all = model_all.predict(X_test)
baseline = evaluate_model(y_test, predictions_all)


model_selected = DecisionTreeRegressor(
    max_depth=3,
    random_state=42
)
model_selected.fit(X_train[selected_features], y_train)
predictions_selected = model_selected.predict(X_test[selected_features])
selected_results = evaluate_model(y_test, predictions_selected)


results = pd.DataFrame(
    [baseline, selected_results],
    columns=["MAE", "MSE", "RMSE", "R2 Score"],
    index=[
        "Baseline (All 10 Features)",
        "Embedded Method (Selected Features)"
    ]
)

print("\nModel Performance:")
print(results)


with open("./feature-selection/embedded-method/embedded_method_results.txt","w") as file:
    file.write("EMBEDDED FEATURE SELECTION\n")
    file.write("=" * 70 + "\n\n")
    file.write("FEATURE IMPORTANCE RANKING\n")
    file.write("-" * 70 + "\n")
    file.write(importance.to_string())
    file.write("\n\nSELECTED FEATURES\n")
    file.write("-" * 70 + "\n")
    file.write(str(selected_features))
    file.write(f"\n\nNumber of Selected Features: {len(selected_features)}")
    file.write("\n\nMODEL PERFORMANCE\n")
    file.write("=" * 70 + "\n")
    file.write(results.to_string())

print("\nResults saved to embedded_method_results.txt")