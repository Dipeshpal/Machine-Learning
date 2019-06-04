# Linear Regression

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# load dataset
dataset = pd.read_csv('headbrain.csv')
# dropping ALL duplicate values
dataset.drop_duplicates(keep=False, inplace=True)
print("Dataset head: ", dataset.head())
print("Dataset shape: ", dataset.shape)


# Correlations Matrix (Visualize Relations between Data)
# From this we can find which param has more relations
correlations = dataset.corr()
sns.heatmap(correlations, square=True, cmap="YlGnBu")
plt.title("Correlations")
plt.show()


# Getting feature (x) and label(y)
# From correlations matrix we found Head Size(cm^3) and Brain Weight(grams) are most co-related data
x = dataset["Head Size(cm^3)"].values
y = dataset["Brain Weight(grams)"].values


# Fitting Line (Model) y = mx + c
# where, m = summation[(x-mean_x)(y-mean_y)]%summation[(x-mean_x)**2]
# c =  y - mx
mean_x = np.mean(x)
mean_y = np.mean(y)

# Total number of features
l = len(x)

# numerator = summation[(x-mean_x)(y-mean_y)
# denominator = summation[(x-mean_x)**2
numerator = 0
denominator = 0
for i in range(l):
    numerator += (x[i] - mean_x) * (y[i] - mean_y)
    denominator += (x[i] - mean_x) ** 2

# m is gradient
m = numerator / denominator

# c is intercept
c = mean_y - (m * mean_x)

print("m: ", m)
print("c: ", c)

# for better visualization (Scaling of data) get max and min point of x
max_x = np.max(x) + 100
min_x = np.min(x) - 100

# X is data points (between max_x and min_y)
X = np.linspace(max_x, min_x, 10)

# model here (we know m and c, already calculated above on sample dataset)
Y = m*X + c

# plotting graph for model
plt.plot(X, Y, color='#58b970', label='Regression Line')
plt.scatter(x, y, c='#ef5424', label='Scatter Plot:n Given Data')
plt.legend()
plt.show()


# Calculate R Square
sst = 0
ssr = 0
for i in range(l):
    y_pred = m * x[i] + c
    sst += (y[i] - mean_y) ** 2
    ssr += (y[i] - y_pred) ** 2

# print("Sum of Squared Total: ", sst)
# print("Sum of Squared due to Regression: ", ssr)
r2 = 1 - (ssr / sst)
print("R Squared: ", r2)

