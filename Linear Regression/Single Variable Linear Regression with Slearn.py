# Linear Regression with Multiple Variable

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# load dataset
dataset = pd.read_csv('headbrain.csv')
# dropping ALL duplicate values
dataset.drop_duplicates(keep=False, inplace=True)
print("Dataset Head Brain-")
print(dataset.head())
print(dataset.shape)


# Correlations Matrix (Visualize Relations between Data)
# From this we can find which param has more relations
correlations = dataset.corr()
sns.heatmap(correlations, square=True, cmap="YlGnBu")
plt.title("Correlations, Single Feature here (Area)")
plt.show()


# Getting feature (x) and label(y)
# From correlations matrix we found Head Size(cm^3) and Brain Weight(grams) are most co-related data
x = dataset["Head Size(cm^3)"].values
x = x.reshape((-1, 1))
y = dataset["Brain Weight(grams)"].values
y = y.reshape((-1, 1))

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
