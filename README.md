# Data Science Mini-Project | EDA and Multiple Linear Regression

This mini-proyect analyses a students performance dataset and train a ML model to predict performance index of students using python, performaning exploratory data analysis (EDA) and data modeling. It was applied multiple linear regression algorithm using two implementations, one from sklearn and the other from an own implementation. The dataset consists on features that describes students such as sleep hours, hours studied, etc, and a column called performance index which we want to predict.

With the help of EDA we could select the appropriate features to use to train the models. Two datasets were used to train each model, one with all features, and the other one applying feature selection. Both models showed nearly the same performance of 98% of R2 score. Finally, the models built using different implementations had the same high performance.

## Data
**[data/Student_Performance.csv](data/Student_Performance.csv)**  
Dataset from Kaggle: https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression

## Notebooks

1. **[student_performance.ipynb](student_performance.ipynb)**  
- Applies descriptive statistics and visualization.
- Assess correlation between variables.
- Applies feature selection and normalization.
- Trains and evaluates models using SGDRegressor from sklearn and from an own implementation.

2. **[linear_regression_details.ipynb](linear_regression_details.ipynb)**  
- Explains briefly the multiple linear regression of an own implementation.

## Modules
The `modules`  directory contains one Python script **[`modules/linear_regression.py`](modules/linear_regression.py)** with the implementation of a linear model and some utility functions used throughout the project.
