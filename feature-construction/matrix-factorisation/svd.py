import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = load_diabetes()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svd = TruncatedSVD(n_components=5, random_state=42)

X_train_svd = svd.fit_transform(X_train_scaled)
X_test_svd = svd.transform(X_test_scaled)
print("Original shape:", X_train_scaled.shape)
print("Transformed shape:", X_train_svd.shape)

explained_variance = svd.explained_variance_ratio_

for i, variance in enumerate(explained_variance):
    print(f"Component {i+1}: {variance:.4f}")

print("Total explained variance:", sum(explained_variance))

# Original features
model_original = LinearRegression()
model_original.fit(X_train_scaled, y_train)
pred_original = model_original.predict(X_test_scaled)


# SVD features
model_svd = LinearRegression()
model_svd.fit(X_train_svd, y_train)
pred_svd = model_svd.predict(X_test_svd)


# Evaluation
print("Original Features")
print("MSE:", mean_squared_error(y_test, pred_original))
print("R2:", r2_score(y_test, pred_original))

print("\nSVD Features")
print("MSE:", mean_squared_error(y_test, pred_svd))
print("R2:", r2_score(y_test, pred_svd))

with open("./feature-construction/matrix-factorisation/svd_results.txt", "w") as file:

    file.write("FEATURE REDUCTION USING SVD\n")
    file.write("=" * 70 + "\n\n")

    file.write(f"Reduced Number of Features: {X_train_svd.shape[1]}\n")
    file.write("\nEXPLAINED VARIANCE\n")
    for i, variance in enumerate(explained_variance):
        file.write(f"Component {i+1}: {variance:.4f}\n")

    file.write(f"\nTotal Explained Variance: {sum(explained_variance):.4f}\n")

    # Model Performance
    file.write("\nMODEL PERFORMANCE COMPARISON\n")
    file.write("=" * 70 + "\n\n")

    file.write("Original Features (10 Features)\n")
    file.write(f"MSE: {mean_squared_error(y_test, pred_original):.4f}\n")
    file.write(f"R2 Score: {r2_score(y_test, pred_original):.4f}\n")

    file.write("\nSVD Features (5 Components)\n")
    file.write(f"MSE: {mean_squared_error(y_test, pred_svd):.4f}\n")
    file.write(f"R2 Score: {r2_score(y_test, pred_svd):.4f}\n")

print("Results successfully saved to svd_results.txt")