## 2.1. Linear Regression with Multiple Variable (Multivariate Linear Regression)-

In Multivariate Linear Regression we have multiple features and their respected labels.

**Explore Dataset-** 

![enter image description here](https://raw.githubusercontent.com/Dipeshpal/Machine-Learning/master/Raw%20Images/Boston%20House%20Dataset.PNG)
Explore dataset in [Kaggle](https://www.kaggle.com/c/boston-housing)

[Click here](https://github.com/Dipeshpal/Machine-Learning/blob/master/Linear%20Regression/Boston%20House%20Dataset.csv) to Download or View

### Housing Values in Suburbs of Boston-

The  **medv**  variable is the target variable.

### Data description-

The Boston data frame has 506 rows and 14 columns.

This data frame contains the following columns:

**_crim_**: per capita crime rate by town.

**_zn_**: proportion of residential land zoned for lots over 25,000 sq.ft.

_**indus**_: proportion of non-retail business acres per town.

_**chas**_: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).

_**nox**_: nitrogen oxides concentration (parts per 10 million).

_**rm**_: average number of rooms per dwelling.

_**age**_: proportion of owner-occupied units built prior to 1940.

_**dis**_: weighted mean of distances to five Boston employment centres.

_**rad**_: index of accessibility to radial highways.

_**tax**_: full-value property-tax rate per  $10,000.

_**ptratio**_: pupil-teacher ratio by town.

_**black**_: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.

_**lstat**_: lower status of the population (percent).

_**medv**_: median value of owner-occupied homes in  $1000s.

### Source

Harrison, D. and Rubinfeld, D.L. (1978) Hedonic prices and the demand for clean air. J. Environ. Economics and Management 5, 81–102.

Belsley D.A., Kuh, E. and Welsch, R.E. (1980) Regression Diagnostics. Identifying Influential Data and Sources of Collinearity. New York: Wiley.

---------------------

**Code-**

[Click here](https://github.com/Dipeshpal/Machine-Learning/blob/master/Linear%20Regression/Multiple%20Variable%20Linear%20Regression%20with%20Slearn.py) to view source code.

```
# Import Libraries  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split
```

```
# load dataset  
dataset = pd.read_csv('Boston House Dataset.csv')  
# dropping ALL duplicate values  
dataset.drop_duplicates(keep=False, inplace=True)
```

```
# Correlations Matrix (Visualize Relations between Data)  
correlations = dataset.corr()  
sns.heatmap(correlations, square=True, cmap="YlGnBu")  
plt.title("Correlations)")  
plt.show()
```

**Output-**

![Correlations](https://raw.githubusercontent.com/Dipeshpal/Machine-Learning/master/Raw%20Images/Correlations%20Boston%20House.PNG)

```
# Getting feature (x) and label(y)  
x = dataset.iloc[:, 4:-1].values  
y = dataset.iloc[:, 14].values  
  
# Splitting into Training and Testing dataset  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)  
  
print(type(x_train), x_train.shape)  
print(type(y_train), y_train.shape)
```

**Output-**

![Output 1 Boston House](https://raw.githubusercontent.com/Dipeshpal/Machine-Learning/master/Raw%20Images/Boston%20House%20Output.PNG)

**Model Fitting-**

```
# Model Fit  
reg = LinearRegression()  
reg = reg.fit(x_train, y_train)  
  
Y_pred = reg.predict(x_test)  
  
r2_score = reg.score(x, y)  
print("r2_score:", r2_score)
```

**Output-**

![Final Output](https://raw.githubusercontent.com/Dipeshpal/Machine-Learning/master/Raw%20Images/Boston%20House%20Final%20Output.PNG)


-------

(Next)[https://github.com/Dipeshpal/Machine-Learning#algorithms-] Logistic Regression
