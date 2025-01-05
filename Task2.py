import pandas as pd
import numpy as np
import matplotlib.pyplot as mpt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
#dataset
data = pd.read_csv("salary.csv")
data_sorted = data.sort_values(by='exp(in months)')
#dependent and independent variables
x = data[['exp(in months)']]
y = data[['salary(in thousands)']]
#data splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)
#model
model = LinearRegression()
model.fit(x_train, y_train)
#prediction
y_pred = model.predict(x_test)
#model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error of the Model:', mse)
print('R-squared Score of the Model:', r2)
#coefficient and intercept in linear equation
print('Coefficient(a) of the linear equation:', model.coef_)
print('Intercept(b) of the equation:', model.intercept_)
#visualisation
mpt.scatter(x,y, label = 'Data Points')
mpt.plot(x_test, y_pred, color='red', label = 'Regression Line')
mpt.xlabel("experience (in months)")
mpt.ylabel("salary(in thousands)")
mpt.title("Linear Regression Plot")
mpt.legend(loc = 'lower right')
mpt.show()
