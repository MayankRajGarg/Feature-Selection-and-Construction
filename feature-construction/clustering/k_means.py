import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = load_diabetes()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X.columns,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X.columns,
    index=X_test.index
)

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

kmeans.fit(X_train_scaled.T)
print(kmeans.labels_)

cluster_labels = kmeans.labels_

feature_clusters = {}
for i, feature in enumerate(X.columns):
    cluster = cluster_labels[i]

    if cluster not in feature_clusters:
        feature_clusters[cluster] = []
    feature_clusters[cluster].append(feature)

print("Feature Groups:")

for cluster, features in feature_clusters.items():
    print(f"Cluster {cluster}: {features}")


def create_cluster_features(X_data, feature_clusters):
    new_features = pd.DataFrame(index=X_data.index)

    for cluster, features in feature_clusters.items():
        new_features[f"Cluster_{cluster}"] = (
            X_data[features].mean(axis=1)
        )

    return new_features


X_train_new = create_cluster_features(X_train_scaled, feature_clusters)
X_test_new = create_cluster_features(X_test_scaled, feature_clusters)

print(X_train_new.head())
print("New shape:", X_train_new.shape)

# Model using original features
model_original = LinearRegression()
model_original.fit(X_train_scaled, y_train)

y_pred_original = model_original.predict(X_test_scaled)

# Model using constructed features
model_cluster = LinearRegression()
model_cluster.fit(X_train_new, y_train)

y_pred_cluster = model_cluster.predict(X_test_new)


# Evaluation
print("Original Features")
print("MSE:", mean_squared_error(y_test, y_pred_original))
print("R2:", r2_score(y_test, y_pred_original))

print("\nConstructed Features")
print("MSE:", mean_squared_error(y_test, y_pred_cluster))
print("R2:", r2_score(y_test, y_pred_cluster))



with open("./feature-construction/clustering/clustering_results.txt", "w") as file:

    file.write("FEATURE SELECTION USING K-MEANS CLUSTERING\n")
    file.write("=" * 70 + "\n\n")

    file.write("FEATURE GROUPS\n")
    file.write("-" * 70 + "\n")
    for cluster, features in feature_clusters.items():
        file.write(f"Cluster {cluster}: {features}\n")

    file.write("\nCONSTRUCTED FEATURES\n")
    file.write("-" * 70 + "\n")

    file.write(f"Original Number of Features: {X.shape[1]}\n")
    file.write(f"New Number of Features: {X_train_new.shape[1]}\n")

    file.write("\nConstructed Feature Names:\n")
    file.write(str(X_train_new.columns.tolist()))

    file.write("\n\nMODEL PERFORMANCE COMPARISON\n")
    file.write("=" * 70 + "\n\n")

    file.write("Original Features (10 Features)\n")
    file.write("-" * 70 + "\n")
    file.write(f"MSE: {mean_squared_error(y_test, y_pred_original):.4f}\n")
    file.write(f"R2 Score: {r2_score(y_test, y_pred_original):.4f}\n")

    file.write("\nConstructed Features (Cluster Features)\n")
    file.write("-" * 70 + "\n")
    file.write(f"MSE: {mean_squared_error(y_test, y_pred_cluster):.4f}\n")
    file.write(f"R2 Score: {r2_score(y_test, y_pred_cluster):.4f}\n")

print("Results successfully saved to clustering_results.txt")