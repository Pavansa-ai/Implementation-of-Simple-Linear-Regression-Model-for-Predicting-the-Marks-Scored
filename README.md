# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries such as pandas, numpy, matplotlib, and scikit-learn.

2.Load the dataset containing student study hours and corresponding scores using read_csv().

3.Display the first and last few records of the dataset to understand its structure.

4.Separate the independent variable (X) as study hours and dependent variable (Y) as scores.

5.Split the dataset into training and testing sets using the train_test_split() method, with one-third of the data used for testing.

6.Create a Linear Regression model using the LinearRegression() class.

7.Train the regression model using the training dataset (X_train, Y_train).

8.Predict the output values for the testing dataset using the trained model.

9.Compare the predicted values with the actual test values.

Plot the training set results by displaying the scatter plot of actual values and the regression line.

10.Plot the testing set results using the same regression line for comparison.

11.Calculate error metrics such as:

Mean Squared Error (MSE)

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

12.Display the error values to evaluate model performance.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: S.PAVAN
RegisterNumber:  25001333/212225040296
*/

#EXP2 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn .model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("student_scores.csv")

print("First 5 rows of the dataset:")
print(df.head())

print("Last 5 rows of the dataset:")
print(df.tail())

X = df.iloc[:, :-1].values
Y = df.iloc[:, 1].values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)


model = LinearRegression()
model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)

print("Predicted values:")
print(Y_pred)
print("Actual Values")
print(Y_test)

plt.scatter(X_train,Y_train, color="blue", label ="Actual scores")
plt.title("Hours vs Scores - Training Set")
plt.xlabel("Hours Studied")
plt.ylabel("Scores Achieved")
plt.legend()
plt.show()

plt.scatter(X_test, Y_test, color='green', label="Actual Scores")
plt.plot(X_train, model.predict(X_train), color='red', label="Fitted Line")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours Studied")
plt.ylabel("Scores Achieved")
plt.legend()
plt.show()


mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print('Mean Squared Error (MSE) =', mse)
print('Mean Absolute Error (MAE) =', mae)
print('Root Mean Squared Error (RMSE) =', rmse)

```

## Output:
<img width="673" height="792" alt="Screenshot 2026-02-02 111305" src="https://github.com/user-attachments/assets/55c5530a-a536-4a72-aaf2-03c19808ca89" />

<img width="835" height="556" alt="Screenshot 2026-02-02 111317" src="https://github.com/user-attachments/assets/df2708e8-05ae-44eb-a60c-cde152471287" />




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
