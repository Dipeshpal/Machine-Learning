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

x = bos1["rad"]

print(x)
y = bos1["medv"]


# Correlations Matrix (Visualize Relations between Data)
# From this we can find which param has more relations
correlations = bos1.corr()
sns.heatmap(correlations, square=True, cmap="YlGnBu")
# plt.yticks(rotation=0)
# plt.xticks(rotation=45)
plt.show()
# print(bos1.shape)
# print(bos1.head())


# Mean x and Y
mean_x = np.mean(x)
mean_y = np.mean(y)

# Total number of values
m = len(x)

numer = 0
denom = 0

for i in range(m):
    numer += (x[i] - mean_x) * (y[i] - mean_y)
    denom += (x[i] - mean_x) ** 2

b1 = numer/denom
b0 = mean_y - (b1*mean_x)

print(b1, b0)


plt.plot(x,  y, color='#58b970', label='Regression Line')
plt.scatter(x, y, c='#ef5424', label='Scatter Plot')

plt.legend()
plt.show()
