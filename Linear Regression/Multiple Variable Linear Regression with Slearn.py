# Linear Regression with Multiple Variable

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# load dataset
dataset = pd.read_csv('Boston House Dataset.csv')
# dropping ALL duplicate values
dataset.drop_duplicates(keep=False, inplace=True)


# Correlations Matrix (Visualize Relations between Data)
correlations = dataset.corr()
sns.heatmap(correlations, square=True, cmap="YlGnBu")
plt.title("Correlations, Single Feature here (Area)")
plt.show()


# Getting feature (x) and label(y)
x = dataset.iloc[:, 4:-1].values
y = dataset.iloc[:, 14].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print(type(x_train), x_train.shape)
print(type(y_train), y_train.shape)


# Model Fit
reg = LinearRegression()
reg = reg.fit(x_train, y_train)

Y_pred = reg.predict(x_test)

r2_score = reg.score(x, y)
print("r2_score:", r2_score)

plt.scatter(y_test, Y_pred)
plt.plot(y_test, Y_pred, color='#58b970', label='Regression Line')
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()
