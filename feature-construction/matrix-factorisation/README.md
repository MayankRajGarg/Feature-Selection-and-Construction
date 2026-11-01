# Feature Reduction using Matrix Factorisation

Implimenting dimensional reduction using Singular Value Decomposition (SVD).

### Dataset
* This project uses the Diabetes dataset. 
* It has 442 samples with 10 numerical features. 
* The target variable(y) represents disease progression.

### Methodology

* Split the dataset into training and testing sets.
* Standardize features using StandardScaler.
* Apply SVD to reduce the feature dimensions from 10 to 5.
* Transform both training and testing data using the fitted SVD model.
* Train Linear Regression models using original features and SVD components.


### Evaluation Metrics

* Mean Squared Error (MSE)
* R² Score

### Output
* Original and transformed feature dimensions.
* Explained variance for each of the 5 components.
* Total explained variance.
* Performance comparison between original features and SVD components.
* Results saved in `svd_results.txt`.

### Technologies Used
* Python
* Pandas
* NumPy
* Scikit-learn
