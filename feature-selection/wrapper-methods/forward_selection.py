import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import KFold, train_test_split
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

model = LinearRegression()
cv = KFold(n_splits=5, shuffle=True, random_state=42)
selector = SequentialFeatureSelector(
    model,
    n_features_to_select=7,
    direction="forward",
    scoring="neg_mean_squared_error",
    cv=cv
)

selector.fit(X_train, y_train)
selected_features = X_train.columns[selector.get_support()]
print("Selected Features:")
print(selected_features.tolist())


def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return [mae, mse, rmse, r2]

model_all = LinearRegression()
model_all.fit(X_train, y_train)
y_pred_all = model_all.predict(X_test)
baseline = evaluate_model(y_test, y_pred_all)

model_selected = LinearRegression()
model_selected.fit(X_train[selected_features], y_train)
y_pred_selected = model_selected.predict(X_test[selected_features])
selected_results = evaluate_model(y_test, y_pred_selected)

results = pd.DataFrame(
    [baseline, selected_results],
    columns=["MAE", "MSE", "RMSE", "R2 Score"],
    index=["Baseline (All 10 Features)", "Forward Selection (7 Features)"]
)
print("\nModel Performance:")
print(results)

with open("./feature-selection/wrapper-methods/forward_selection_results.txt", "w") as file:
    file.write("FORWARD FEATURE SELECTION\n")
    file.write("=" * 70 + "\n")
    file.write("Selected Features:\n")
    file.write(str(selected_features.tolist()))
    file.write("\n\nMODEL PERFORMANCE\n")
    file.write("=" * 70 + "\n")
    file.write(results.to_string())

print("\nResults saved to forward_selection_results.txt")