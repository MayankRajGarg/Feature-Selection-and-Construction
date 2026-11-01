# Feature Selection with Ranking using Pearson Correlation

Comparing feature selections using the Pearson Correlation Coefficient'S Ranking.

### Dataset
* This project uses the Diabetes dataset. 
* It has 442 samples with 10 numerical features. 
* The target variable(y) represents disease progression.

### Methodology
* Calculate Pearson correlation for each feature and the target variable.
* Rank them based on their absolute correlation values.
* Train Linear Regression models using the Top 1 to Top 10 features.
* Compare model performance against a model using all 10 features.

### Evaluation Metrics
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

### Output
* Model performance comparison.
* `Performance_with_Correlation_Coefficient.png` has the Plot to compare.
* Results saved in `model_results.txt`.

### Results
* Top 3 to 4 feature from the Ranking give the Model Similar to the Whole Model.


### Technologies Used
* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
