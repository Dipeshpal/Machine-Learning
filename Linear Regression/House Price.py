# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

bos1 = pd.read_csv('dataset.csv')

x = bos1["area"]

# print(x)
y = bos1["price"]

# Correlations Matrix (Visualize Relations between Data)
# From this we can find which param has more relations
correlations = bos1.corr()
sns.heatmap(correlations, square=True, cmap="YlGnBu")
# plt.yticks(rotation=0)
# plt.xticks(rotation=45)
plt.show()
# print(bos1.shape)
# print(bos1.head())


# y = mx + c
# where, m = summation[(x-mean_x)(y-mean_y)]%summation[(x-mean_x)**2]


# Mean x and Y
mean_x = np.mean(x)
mean_y = np.mean(y)

# Total number of values
l = len(x)

numer = 0
denom = 0

for i in range(l):
    numer += (x[i] - mean_x) * (y[i] - mean_y)
    denom += (x[i] - mean_x) ** 2

m = numer / denom
c = mean_y - (m * mean_x)

# print(b1, b0)


max_x = np.max(x)
min_y = np.min(y)

X = np.linspace(max_x, min_y, 20)
Y = m*X + c

plt.plot(X, Y, color='#58b970', label='Regression Line')
plt.scatter(x, y, c='#ef5424', label='Scatter Plot')
plt.scatter(X, Y, c='#58b970', label='Scatter Plot 2')


plt.legend()
plt.show()


# Calculate R**2
# R_square = Summation[(y_pred-y_mean)**2]%Summation[(y-y_mean)]**2


ss_t = 0
ss_r = 0
for i in range(l):
    y_pred = m * x[i] + c
    ss_t += (y[i] - mean_y) ** 2
    ss_r += (y[i] - y_pred) ** 2
r2 = 1 - (ss_r / ss_t)
print(r2)
