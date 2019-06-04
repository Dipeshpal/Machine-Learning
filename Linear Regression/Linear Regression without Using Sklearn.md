## Python Implementation without using sklearn-

### Exploring dataset-

It has 4 columns and 237 rows . "Gender" and "Age Range" is unwanted (useless) features, "Head Size(cm^3)" is feature that we will use. Our label is "Brain Weight(grams)".

[Download](https://github.com/Dipeshpal/Machine-Learning/blob/master/Linear%20Regression/headbrain.csv) or [View](https://github.com/Dipeshpal/Machine-Learning/blob/master/Linear%20Regression/headbrain.csv) dataset.

![headbrain dataset](https://raw.githubusercontent.com/Dipeshpal/Machine-Learning/master/Raw%20Images/headbrain%20dataset.png)

### 1.1. Python Implementation without using sklearn-

**Code-** 
[Click here](https://github.com/Dipeshpal/Machine-Learning/blob/master/Linear%20Regression/Single%20Variable%20Linear%20Regression%20Manually.py) to view source code file

**Let's Start-**

```
# Import Libraries  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns
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

![Output 1
](https://raw.githubusercontent.com/Dipeshpal/Machine-Learning/master/Raw%20Images/Output%201.png)

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

**We will use Head Size(cm^3) as feature and Brain Weight(grams) labels. Discard "Gender" and "Age Range" because it is useless.**

```
# Getting feature (x) and label(y)  
# From correlations matrix we found Head Size(cm^3) and Brain Weight(grams) are most co-related data  
x = dataset["Head Size(cm^3)"].values  
y = dataset["Brain Weight(grams)"].values
```

**Fit Model: Now we will calculate slope ( m ) and intercept ( c )-** 

```
# Fitting Line (Model) y = mx + c  
# where, m = summation[(x-mean_x)(y-mean_y)]%summation[(x-mean_x)**2]  
# c =  y - mx  
mean_x = np.mean(x)  
mean_y = np.mean(y)
```

```
# Total number of features  
l = len(x)  
  
# numerator = summation[(x-mean_x)(y-mean_y)  
# denominator = summation[(x-mean_x)**2  
numerator = 0  
denominator = 0  

for i in range(l):  
    numerator += (x[i] - mean_x) * (y[i] - mean_y)  
    denominator += (x[i] - mean_x) ** 2
```

** From above lines, we get numerator and denominator**, now let's calculate mean and intercept-

![mean formula](https://camo.githubusercontent.com/fbb26758e5f73103a8c9c09ae85eef06947da750/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f313630302f312a43783159656a397a4c5649314f313649336d4f4471412e706e67)

```
# m is gradient  
m = numerator / denominator  
  
# c is intercept  
c = mean_y - (m * mean_x)  
  
print("m: ", m)  
print("c: ", c)
```

**Output-**

![Output 3](https://raw.githubusercontent.com/Dipeshpal/Machine-Learning/master/Raw%20Images/Output%203.PNG)


**Let's see the fitness of the model-**

```
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
```

**Output-**
Perfectly fitting the dataset

![Output 4](https://raw.githubusercontent.com/Dipeshpal/Machine-Learning/master/Raw%20Images/Output%204.PNG)

**Now let's check Error-**

```
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
```

**Output-**

![Output 5](https://raw.githubusercontent.com/Dipeshpal/Machine-Learning/master/Raw%20Images/Output%205.PNG)

R-squared = Explained variation / Total variation

R-squared is always between 0 and 100%:

-   0% indicates that the model explains none of the variability of the response data around its mean.
-   100% indicates that the model explains all the variability of the response data around its mean

[Next](https://github.com/Dipeshpal/Machine-Learning/blob/master/Linear%20Regression/Linear%20Regression%20with%20Sklearn.md) Python Implementation with sklearn.
