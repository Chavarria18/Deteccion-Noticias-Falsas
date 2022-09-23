import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, model_selection

# Dataset: https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset
# 
# Column 11 - quantitative measure of disease progression one year after baseline
X, y = datasets.load_diabetes(return_X_y=True)

# Shape & 1st data element
print(X.shape)
print("Data Set Characteristics: ",X[4])
print("Target: ",y[0])

# Generate a line between values in this data index = 2 bmi body mass index
X = X[:, np.newaxis, 2]

# Split X and y into test and training sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)


model = linear_model.LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Scaled BMIs')
plt.ylabel('Disease Progression')
plt.title('Diabetes Progression Against TCH')
plt.show()