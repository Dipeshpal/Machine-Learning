## Python Implementation with sklearn-

### Exploring dataset-

It has 4 columns and 237 rows . "Gender" and "Age Range" is unwanted (useless) features, "Head Size(cm^3)" is feature that we will use. Our label is "Brain Weight(grams)".

[Download](https://github.com/Dipeshpal/Machine-Learning/blob/master/Linear%20Regression/headbrain.csv) or [View](https://github.com/Dipeshpal/Machine-Learning/blob/master/Linear%20Regression/headbrain.csv) dataset.

![headbrain dataset](https://raw.githubusercontent.com/Dipeshpal/Machine-Learning/master/Raw%20Images/headbrain%20dataset.png)


### 1.2. Python Implementation with sklearn-

**Code-** 

[Click here](https://github.com/Dipeshpal/Machine-Learning/blob/master/Linear%20Regression/Single%20Variable%20Linear%20Regression%20with%20Slearn.py) to Dowanlod or view source code

**Let's Start-**

```
# Import Libraries  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.linear_model import LinearRegression
```

```
# load dataset  
dataset = pd.read_csv('headbrain.csv')  
# dropping ALL duplicate values  
dataset.drop_duplicates(keep=False, inplace=True)  
print("Dataset head: ", dataset.head())  
print("Dataset shape: ", dataset.shape)
```

**Output-**

![Output 1](https://raw.githubusercontent.com/Dipeshpal/Machine-Learning/master/Raw%20Images/Output%201.png)


```
# Correlations Matrix (Visualize Relations between Data)  
# From this we can find which param has more relations  
correlations = dataset.corr()  
sns.heatmap(correlations, square=True, cmap="YlGnBu")  
plt.title("Correlations")  
plt.show()
```

**Output-**

![Output 2](https://raw.githubusercontent.com/Dipeshpal/Machine-Learning/master/Raw%20Images/Output%202.PNG)


**We will use Head Size(cm^3) as feature and Brain Weight(grams) labels. Discard “Gender” and “Age Range” because it is useless.**

```
# Getting feature (x) and label(y)  
# From correlations matrix we found Head Size(cm^3) and Brain Weight(grams) are most co-related data  
x = dataset["Head Size(cm^3)"].values  
x = x.reshape((-1, 1))  
y = dataset["Brain Weight(grams)"].values  
y = y.reshape((-1, 1))
```

**Fit Model: Now we will calculate slope ( m ) and intercept ( c )-** 

```
reg = LinearRegression()  
reg = reg.fit(x, y)  
Y_pred = reg.predict(x)  
   
r2_score = reg.score(x, y)  
print("m:", reg.coef_)  
print("c: ", reg.intercept_)  
print("r2_score:", r2_score)
```

**Output-**

![Output 3.1](https://raw.githubusercontent.com/Dipeshpal/Machine-Learning/master/Raw%20Images/Output%203.1.PNG)

```
# plotting graph for model  
plt.plot(x, Y_pred, color='#58b970', label='Regression Line')  
plt.scatter(x, y, c='#ef5424', label='Scatter Plot of Given Data')  
plt.legend()  
plt.show()
```

**Output-**

![Output 4.1](https://raw.githubusercontent.com/Dipeshpal/Machine-Learning/master/Raw%20Images/Output%204.1.PNG)
