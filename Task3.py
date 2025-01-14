import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sb

train_data = pd.read_csv("HouseData.csv")
test_data = pd.read_csv("HouseTestData.csv")

#training data
train_data.info()
train_data['lot_size'].unique()# 3 different categories: sqft,acre,NaN
acre_Rows = (train_data['lot_size_units'] == 'acre')
train_data.loc[acre_Rows, 'lot_size'] *= 43560
train_data.loc[acre_Rows, 'lot_size_units'] = 'sqft'# 2 different categories : sqft,NaN
print(train_data.isnull().sum())# gives total number of rows with null values
train_data.drop(['lot_size_units','zip_code','size_units'],inplace = True, axis = 1)# drop unnecessary columns
median = train_data['lot_size'].median()
train_data.fillna( {'lot_size':median}, inplace=True)# replace NaN with median
train_data.drop_duplicates(inplace=True)# drop duplicate values
print("\nNull values in training data\n")
print(train_data.isnull().sum(),'\n')# gives total number of rows with null values

#test data
test_data.info()
test_data['lot_size'].unique()# 3 different categories: sqft,acre,NaN
acre_Rows = (test_data['lot_size_units'] == 'acre')
test_data.loc[acre_Rows, 'lot_size'] *= 43560
test_data.loc[acre_Rows, 'lot_size_units'] = 'sqft'# 2 different categories : sqft,NaN
print(test_data.isnull().sum())# gives total number of rows with null values
test_data.drop(['lot_size_units','zip_code','size_units'],inplace = True, axis = 1)# drop unnecessary columns
median = test_data['lot_size'].median()
test_data.fillna( {'lot_size':median}, inplace=True)# replace NaN with median
test_data.drop_duplicates(inplace=True)# drop duplicate values
print("\nNull values in test data\n") 
print(test_data.isnull().sum())# gives total number of rows with null values

# data splitting into training and test data
x_train = train_data.drop(['price','lot_size'], axis = 1)
y_train = train_data['price']
x_test = test_data.drop(['price','lot_size'], axis = 1)
y_test = test_data['price']

model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("\nThe value of b,a0,a1,a2 in y = a0x0 + a1x1 + a2x2 + b : \n")
print("Intercept (b):", model.intercept_)
print("Coefficient (a0),(a1),(a2):", model.coef_)

print("\nmodel evaluation: ")
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error: ",mse)
r2 = r2_score(y_test, y_pred)
print("\nR-squared Score: ", r2)

# training data visualisation
plt.figure(figsize=(15,10))
#beds vs price
plt.subplot(2,2,1)
sb.regplot(x="beds",y="price",data=train_data, color = 'green')
plt.title("No. of Beds VS Price")
#baths vs price
plt.subplot(2,2,2)
sb.regplot(x="baths",y="price",data=train_data, color = 'red')
plt.title("No. of Bathrooms VS Price")
#size vs price
plt.subplot(2,2,3)
sb.regplot(x="size",y="price",data=train_data, color = 'blue')
plt.title("House_Size VS Price")
#lot_size vs price
plt.subplot(2,2,4)
sb.regplot(x="lot_size",y="price",data=train_data, color = 'orange')
plt.title("Plot_size VS Price")
plt.tight_layout()

# test data visualization
plt.figure(figsize=(15,10))
#beds vs price
plt.subplot(2,2,1)
sb.regplot(x="beds",y="price",data=test_data, color = 'green')
plt.title("No. of Beds VS Price")
#baths vs price
plt.subplot(2,2,2)
sb.regplot(x="baths",y="price",data=test_data, color = 'red')
plt.title("No. of Bathrooms VS Price")
#size vs price
plt.subplot(2,2,3)
sb.regplot(x="size",y="price",data=test_data, color = 'orange')
plt.title("House_Size VS Price")
plt.tight_layout()
plt.show()
