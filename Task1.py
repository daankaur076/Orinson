import pandas as p
import numpy as nm
import seaborn as snb
import matplotlib.pyplot as plot
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

print("The Iris dataset is loaded")
iris = load_iris()
x = iris.data
y = iris.target

print("Categories in 'Species' target: ")
print(iris.target_names)

# pandas DataFrame for better visualization
df = p.DataFrame(data=x, columns=iris.feature_names)
df['target'] = y
df['target_names'] = df['target'].map(dict(zip(range(len(iris.target_names)), iris.target_names)))

# Visualisation
snb.pairplot(df, hue='target_names')
plot.show()

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Creating a KNN classifier
knn = KNeighborsClassifier(n_neighbors=4) 
# Training the classifier
knn.fit(x_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(x_test)
print("Predicted Classes for Test Data:")
print(y_pred)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model's Accuracy:", accuracy) 

# Predict the class of a new data point
new_data = [[5.0, 3.0, 1.6, 0.2]] 
print("New Data Point:")
print(new_data)
predicted_class = knn.predict(new_data)
print("Predicted class:", predicted_class)
