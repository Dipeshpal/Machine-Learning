# Linear Regression with Multiple Variable

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# load dataset
dataset = pd.read_csv('train.csv')
# dropping ALL duplicate values
dataset.drop_duplicates(keep=False, inplace=True)
print("Dataset Head Brain-")
print(dataset.head())
print(dataset.shape)
# print(len(dataset.columns))


# Correlations Matrix (Visualize Relations between Data)
# From this we can find which param has more relations
correlations = dataset.corr()
sns.heatmap(correlations, square=True, cmap="YlGnBu")
plt.title("Correlations, Single Feature here (Area)")
plt.show()


# Getting feature (x) and label(y)
# From correlations matrix we found Head Size(cm^3) and Brain Weight(grams) are most co-related data
# x = dataset[[10, 11]].values
# x = x.reshape((-1, 1))
# y = dataset["medv"].values
# y = y.reshape((-1, 1))

x = dataset.iloc[:, 9:11].values
y = dataset.iloc[:, 14].values
y = y.reshape(1, -1)
y = y.T
print("x: ", x.shape)
print("y: ", y.shape)
print("lx: ", len(x))
print("lx: ", len(y))

reg = LinearRegression()

reg = reg.fit(x, y)

Y_pred = reg.predict(x)

# plotting graph for model
plt.plot(x, Y_pred, color='#58b970', label='Regression Line')
plt.scatter(x, y, c='#ef5424', label='Scatter Plot of Given Data')
plt.legend()
plt.show()

r2_score = reg.score(x, y)
print("m", reg.coef_)
print("c: ", reg.intercept_)
print("r2_score:", r2_score)
