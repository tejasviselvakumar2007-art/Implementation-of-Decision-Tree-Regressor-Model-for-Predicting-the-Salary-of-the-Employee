# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and check for missing values.
2. Encode categorical data (convert text to numbers using LabelEncoder).
3. Split the dataset into training and testing sets using train_test_split.
4. Train the DecisionTreeRegressor model, make predictions, and evaluate using MSE and R² score  

## Program:
```
import pandas as pd

# Load dataset
data = pd.read_csv("Salary.csv")

# Check dataset
print(data.head())
print(data.info())
print(data.isnull().sum())

# Convert Position (text) to numeric
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])

print(data.head())

# Features and Target
x = data[["Position","Level"]]
y = data["Salary"]

# Split dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

# Prediction
y_pred = dt.predict(x_test)

# Evaluation
from sklearn import metrics
mse = metrics.mean_squared_error(y_test, y_pred)
print("MSE:", mse)

r2 = metrics.r2_score(y_test, y_pred)
print("R2 Score:", r2)

# Predict new value
print("Predicted Salary:", dt.predict([[5,6]]))

```

## Output:
<img src="https://img.sanishtech.com/u/3fe0e20071498461e414f00b63f90426.png" alt="Screenshot 2026-03-19 105021" width="1919" height="956" loading="lazy" style="max-width:100%;height:auto;">


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
