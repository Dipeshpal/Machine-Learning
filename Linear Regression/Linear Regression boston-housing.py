# Dataset Description-

# The Boston data frame has 506 rows and 14 columns.
# This data frame contains the following columns:

# crim: per capita crime rate by town.
# zn: proportion of residential land zoned for lots over 25,000 sq.ft.
# indus: proportion of non-retail business acres per town.
# chas: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
# nox: nitrogen oxides concentration (parts per 10 million).
# rm: average number of rooms per dwelling.
# age: proportion of owner-occupied units built prior to 1940.
# dis: weighted mean of distances to five Boston employment centres.
# rad: index of accessibility to radial highways.
# tax: full-value property-tax rate per \$10,000.
# ptratio: pupil-teacher ratio by town.
# black: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.
# lstat: lower status of the population (percent).
# medv: median value of owner-occupied homes in \$1000s.


# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

bos1 = pd.read_csv('train.csv')

# for col in bos1.columns:
#     print(col)
#
x = bos1["tax"]

# print(x)
y = bos1["medv"]


# Correlations Matrix (Visualize Relations between Data)
# From this we can find which param has more relations
correlations = bos1.corr()
sns.heatmap(correlations, square=True, cmap="YlGnBu")
plt.show()


# from pandas import DataFrame
# df = DataFrame(bos1, "tax")
#
# plt.scatter(df['rad'], df['tax'], color='red')
# plt.title('rad Price Vs price', fontsize=14)
# plt.xlabel('rad Rate', fontsize=14)
# plt.ylabel('price', fontsize=14)
# plt.grid(True)
# plt.show()


a = x.as_matrix()
b = y.as_matrix()

print(type(a))
print(type(b))

print(a.shape)
print(b.shape)

p = np.reshape(a, (-1, 1))
q = np.reshape(b, (-1, 1))
print(p.shape)

# with sklearn
r = 201
from sklearn import linear_model
regr = linear_model.LinearRegression()
print(p[r], q[r])
regr.fit(p, q)

print(regr.predict([p[r]]))
