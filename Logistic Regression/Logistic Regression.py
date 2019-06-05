# Imoport Libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Load Dataset
data = pd.read_csv("Marks Dataset.csv", header=None)
print(data.head(4))
print(data.shape)
print(type(data))


# X = feature values, all the columns except the last column
x = data.iloc[:, :-1].values
# y = target values, last column of the data frame
y = data.iloc[:, -1].values


# filter out the applicants that got admitted and that din't get admission
admitted = data.loc[y == 1]
not_admitted = data.loc[y == 0]


# plots
plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
plt.legend()
plt.show()


# Split dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# Training the Model
model = LogisticRegression()
model = model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)
print("Model Coefficient: ", model.coef_)
print('Intercept', model.intercept_)


coe = str(model.coef_[0])
coe = coe[1:-1]
coef = list(map(float, coe.split(' ')))

x1 = - model.intercept_/coef[0]
y2 = - model.intercept_/coef[1]

x_plt = [x1, 0]
y_plt = [0, y2]


# plots
plt.plot(x_plt, y_plt, label='Decision Boundary')
plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
plt.xlabel('Marks in 1st Exam')
plt.ylabel('Marks in 2nd Exam')
plt.legend()
plt.show()
