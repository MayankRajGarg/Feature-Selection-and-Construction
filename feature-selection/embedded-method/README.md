# Embedded Feature Selection

Implementating Embedded Feature Selection using a Decision Tree Regressor.

### Dataset
* This project uses the Diabetes dataset. 
* It has 442 samples with 10 numerical features. 
* The target variable(y) represents disease progression.

### Methodology
* Train a Decision Tree Regressor with a maximum depth of 3.
* Calculate feature importance based on each feature's contribution to reducing impurity.
* Rank features according to their importance scores.
* Select features with non-zero importance.
* Train a model using only the selected features.
* Compare its performance against a baseline model trained using all 10 features.

### Feature Importance Calculation
* Decision Trees calculate feature importance based on the total reduction in impurity contributed by each feature.
* For regression, impurity is measured using Mean Squared Error (MSE).
* Features contributing to zero impurity reduction receive an importance score of zero.

### Evaluation Metrics
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

### Output
* Feature importance ranking.
* List of selected features.
* Performance comparison between the baseline and selected-feature models.
* Results saved in `embedded_method_results.txt`.

### Technologies Used
* Python
* Pandas
* NumPy
* Scikit-learn
