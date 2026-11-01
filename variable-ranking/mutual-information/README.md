# Feature Selection using Mutual Information

Comparing feature selection methods using Mutual Information.

### Dataset
* This project uses the Diabetes dataset.
* It has 442 samples with 10 numerical features.
* The target variable(y) represents disease progression.

### Methodology
* Calculate Mutual Information between each feature and the target variable.
* Rank features based on their Mutual Information scores.
* Train Linear Regression models using the Top 1 to Top 10 features.
* Compare model performance against a model using all 10 features.

### How is Mutual Information Calculated?
* It's calculated using the joint and individual probability distributions of the variables.
* Since the dataset has only continuous numerical features, a K-Nearest Neighbors (KNN)-based estimator is used.
* The estimator uses distances b/w neighbouring observations to estimate local probability densities and calculate Mutual Information scores.
* Scikit-learn's `mutual_info_regression()` implements this estimation approach.

The mathematical definition is:

$$
I(X;Y)=\sum_{x,y}P(x,y)\log\frac{P(x,y)}{P(x)P(y)}
$$

### Evaluation Metrics
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

### Output
* Model performance comparison.
* `Performance_with_Mutual_Information.png` contains the performance comparison plots.
* Results saved in `mi_results.txt`.

### Results
* Top 5 feature from the Ranking givea a Model Similar to the Whole Model. So the last 5 are not much relevent according to this Analysis.

### Technologies Used
* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
