import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_regression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#----------------- Functions for MRMR ----------------------
def calculate_mi(X, Y):
    X = np.asarray(X).reshape(-1, 1)
    Y = np.asarray(Y)
    mi = mutual_info_regression(X, Y,random_state=42)
    return mi[0]

def calculate_redundancy_matrix(X):
    features = X.columns
    redundancy = pd.DataFrame(np.zeros((len(features), len(features))), index=features, columns=features)
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            feature_a = features[i]
            feature_b = features[j]

            mi = calculate_mi(X[feature_a], X[feature_b])

            redundancy.loc[feature_a, feature_b] = mi
            redundancy.loc[feature_b, feature_a] = mi
    return redundancy

def calculate_relevance(X, y):
    relevance = {}
    for feature in X.columns:
        mi = calculate_mi(X[feature],y)
        relevance[feature] = mi

    return pd.Series(relevance, name="Relevance")

def calculate_mrmr(relevance, redundancy):
    n = len(relevance)
    mrmr_scores = {}
    for feature in relevance.index:
        relevance_score = relevance[feature]
        redundancy_score = (redundancy.loc[feature].sum()/(n - 1))

        mrmr_score = relevance_score - redundancy_score
        mrmr_scores[feature] = mrmr_score

    return pd.Series(mrmr_scores, name="mRMR")


#----------------- Calculation ----------------------
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


redundancy = calculate_redundancy_matrix(X_train)
relevance = calculate_relevance(X_train, y_train)
mrmr_scores = calculate_mrmr(relevance, redundancy)
ranking = mrmr_scores.sort_values(ascending=False)

print("mRMR Ranking:")
print(ranking)

#--------------------- Evaluation -----------------------

results = []
for k in range(1, len(X.columns) + 1):

    selected_features = ranking.index[:k].tolist()

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    model = LinearRegression()
    model.fit(X_train_selected, y_train)
    predictions = model.predict(X_test_selected)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    results.append({
        "Number of Features": k,
        "Selected Features": ", ".join(selected_features),
        "MSE": mse,
        "R2 Score": r2
    })

results_df = pd.DataFrame(results)

print("\nEvaluation Results:")
print(results_df.to_string(index=False))

#-------------------------- Saving --------------------------------
with open("variable-ranking/min-redundancy-max-relevance/mrmr_results.txt", "w") as file:
    file.write("MRMR FEATURE RANKING\n")
    file.write("=" * 60 + "\n\n")
    file.write(ranking.to_string())
    file.write("\n\n")
    file.write("LINEAR REGRESSION EVALUATION\n")
    file.write("=" * 60 + "\n\n")
    
    for _, row in results_df.iterrows():
        file.write(f"Number of Features: {row['Number of Features']}\n")
        file.write(f"Selected Features: {row['Selected Features']}\n")

        file.write(f"MSE: {row['MSE']:.4f}\n")
        file.write(f"R2 Score: {row['R2 Score']:.4f}\n")
        file.write("-" * 60 + "\n")

print("Results saved successfully!")


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(
    results_df["Number of Features"],
    results_df["MSE"],
    marker="o"
)
axes[0].set_xlabel("Number of Features")
axes[0].set_ylabel("Mean Squared Error")
axes[0].set_title("MSE vs Number of Features")
axes[0].set_xticks(range(1, len(X.columns) + 1))
axes[0].grid(True)


axes[1].plot(
    results_df["Number of Features"],
    results_df["R2 Score"],
    marker="o"
)

axes[1].set_xlabel("Number of Features")
axes[1].set_ylabel("R2 Score")
axes[1].set_title("R2 Score vs Number of Features")
axes[1].set_xticks(range(1, len(X.columns) + 1))
axes[1].grid(True)

plt.tight_layout()
plt.show()