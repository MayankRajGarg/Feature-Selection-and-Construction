# Variable Selection and Feature Engineering

Implementing different techniques for Variable & Feature Selection and Feature Construction. It compares different approaches based on ability to reduce features while maintaining performance.

### Dataset
I have uses the **Diabetes Dataset** available in Scikit-learn, originally derived from a study of diabetes patients.

* Contains **442 patient records** with 10 numerical features.
* The target variable represents a measure of disease progression.

**Features:**
| Feature | Description                              |
| ------- | ---------------------------------------- |
| age     | Age of the patient                       |
| sex     | Sex of the patient (numerically encoded) |
| bmi     | Body Mass Index                          |
| bp      | Average blood pressure                   |
| s1–s6   | Six blood serum measurements             |


### Objective
To explore and compare different variable selection and feature construction techniques and understand their impact on regression model performance.

### Methods Implemented

#### 1. Variable Ranking
Ranks individual features based on their relationship with the target variable.
* **Pearson Correlation:** Ranks features based on their linear correlation with the target.
* **Mutual Information:** Ranks features based on the information they provide about the target.

#### 2. Feature Selection
Selects a subset of relevant features from the original dataset.
* **Wrapper Methods:**
  * Forward/Backward Feature Selection - Select a the subset with a greedy Approch
* **Embedded Method:**
  * Letting the Model decide the weightage/ importance of the features and remove less usefull ones.

#### 3. Feature Construction
Creates new features or components from existing features to reduce dimensionality.
* **K-Means Clustering:** Groups similar features and constructs new features using cluster-wise averages.
* **Matrix Factorisation:** Transforms the original features into a smaller set of components with SVD.

### Methodology
* Load and preprocess the Diabetes dataset.
* Apply different variable selection and feature construction techniques.
* Train Linear Regression models using selected or constructed features.
* Compare their performance against using all original features.
* Evaluate the results using standard regression metrics.

### Evaluation Metrics

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

### Reference

The primary reference for this project is:
Guyon, I., & Elisseeff, A. (2003). *An Introduction to Variable and Feature Selection*. Journal of Machine Learning Research, 3, 1157–1182.


### Project Structure

```text
variable-selection/
│
├── feature-construction/
│   ├── clustering/
│   └── matrix-factorisation/
│
├── feature-selection/
│   ├── embedded-method/
│   └── wrapper-methods/
│
└── variable-ranking/
    ├── correlation-coefficient/
    └── mutual-information/
```

### Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

