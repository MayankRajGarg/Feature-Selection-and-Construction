# Wrapper Methods for Feature Selection

Implementating of Wrapper-Based Feature Selection methods.

### Dataset
* This project uses the Diabetes dataset. 
* It has 442 samples with 10 numerical features. 
* The target variable(y) represents disease progression.

### Methods Implemented

**1. Forward Feature Selection**
* Starts with an empty feature set.
* Iteratively adds the feature that provides the best improvement in model performance.
* Stops when the specified number of features is selected.

**2. Backward Feature Selection**
* Starts with all available features.
* Iteratively removes the feature whose removal gives the best model performance.
* Stops when the specified number of features remains.

### Methodology
* Uses Linear Regression as the estimator.
* Applies 5-Fold Cross-Validation to evaluate feature combinations.
* Uses Negative Mean Squared Error (`neg_mean_squared_error`) as the selection criterion.
* Selects 7 features using both Forward and Backward Selection.
* Compares the selected-feature models against a baseline model using all 10 features.

### Evaluation Metrics
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

### Technologies Used
* Python
* Pandas
* NumPy
* Scikit-learn
