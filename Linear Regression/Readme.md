
# Linear regression
Linear regression is useful for finding relationship between two continuous variables. One is predictor or independent variable and other is response or dependent variable. It looks for statistical relationship but not deterministic relationship. Relationship between two variables is said to be deterministic if one variable can be accurately expressed by the other.

Linear regression is useful for finding relationship between two continuous variables. One is predictor or independent variable and other is response or dependent variable. It looks for statistical relationship but not deterministic relationship. Relationship between two variables is said to be deterministic if one variable can be accurately expressed by the other.

**The core idea is to obtain a line that best fits the data. The best fit line is the one for which total prediction error (all data points) are as small as possible. Error is the distance between the point to the regression line.**

## Behind the Mathematics-
We have a dataset which contains information about relationship between ‘area of house’ and ‘price’. Check this [House.csv](https://github.com/Dipeshpal/Machine-Learning/blob/master/Linear%20Regression/dataset.csv) file for more details.
This will be our training data. Goal is to design a model that can predict price if given the area of house. Using the training data, a regression line is obtained which will give minimum error. This linear equation is then used for any new data. That is, if we give area of house as an input, our model should predict their price with minimum error.

**Line to be fit-**

Y(pred) = b0 + b1*x

or

h(Θ) = Θ1 + Θ2*x

or

Y = c + m*x

The values b0 and b1 (or Θ1 and Θ2 or c and m) must be chosen so that they minimize the error. If sum of squared error is taken as a metric to evaluate the model, then goal to obtain a line that best reduces the error.

**Error Calculations-**

![enter image description here](https://cdn-images-1.medium.com/max/1600/1*Utp8sgyLk7H39qOQY9pf1A.png)

If we don’t square the error, then positive and negative point will cancel out each other.

For model with one predictor,

**Intercept Calculation-**

![enter image description here](https://cdn-images-1.medium.com/max/1600/1*1evY0PuCUENCpDP_QRplig.png)

or c = Y_mean - m*x_mean

**Co-efficient Formula-**

![enter image description here](https://cdn-images-1.medium.com/max/1600/1*Cx1Yej9zLVI1O16I3mODqA.png)

or m = summation[(x-mean_x)(y-mean_y)] % summation[(x-mean_x)**2]

**_Exploring ‘b1’_**

-   If b1 > 0, then x(predictor) and y(target) have a positive relationship. That is increase in x will increase y.
-   If b1 < 0, then x(predictor) and y(target) have a negative relationship. That is increase in x will decrease y.

**_Exploring ‘b0’_**

-   If the model does not include x=0, then the prediction will become meaningless with only b0. For example, we have a dataset that relates height(x) and weight(y). Taking x=0(that is height as 0), will make equation have only b0 value which is completely meaningless as in real-time height and weight can never be zero. This resulted due to considering the model values beyond its scope.
-   If the model includes value 0, then ‘b0’ will be the average of all predicted values when x=0. But, setting zero for all the predictor variables is often impossible.
-   The value of b0 guarantee that residual have mean zero. If there is no ‘b0’ term, then regression will be forced to pass over the origin. Both the regression co-efficient and prediction will be biased.

## Residual Analysis-

Let’s explain the concept of residue through an example. Consider, we have a dataset which predicts sales of juice when given a temperature of place. Value predicted from regression equation will always have some difference with the actual value. Sales will not match exactly with the true output value. This difference is called as residue.

Residual plot helps in analyzing the model using the values of residues. It is plotted between predicted values and residue. Their values are standardized. The distance of the point from 0 specifies how bad the prediction was for that value. If the value is positive, then the prediction is low. If the value is negative, then the prediction is high. 0 value indicates prefect prediction. Detecting residual pattern can improve the model.

**_R-Squared value_**

This value ranges from 0 to 1. Value ‘1’ indicates predictor perfectly accounts for all the variation in Y. Value ‘0’ indicates that predictor ‘x’ accounts for no variation in ‘y’.

1. **Regression sum of squares (SSR)**

This gives information about how far estimated regression line is from the horizontal ‘no relationship’ line (average of actual output).

![](https://cdn-images-1.medium.com/max/800/1*eXRB9iStTLFtrPkSfbWEHg.png)


2. **Sum of Squared error (SSE)**

How much the target value varies around the regression line (predicted value).

![](https://cdn-images-1.medium.com/max/800/1*M7ukJZNTvPd6tQqNXxGGzQ.png)


3. **Total sum of squares (SSTO)**

This tells how much the data point move around the mean.

![](https://cdn-images-1.medium.com/max/800/1*LXAc7FPLOgB1L3IqSUKl5A.png)


# Implementations with Python-
1. **Single Variable Linear Regression:** First we will predict **Weight of Brain** according to **Size of Brain**. We will solve this problem **Manually** and **Sklearn**.
2. **Multivariate Linear Regression-** Multiple Input (area of house, distance of house from highway etc.) and Single Output. By using **Sklearn**.

## 1. **Single Variable Linear Regression-**
Single Variable Linear Regression in the type of regression in which only one feature to train the model.
We will fit our model with dataset "headbrain"-
### Exploring dataset-

It has 4 columns and 237 rows . "Gender" and "Age Range" is unwanted (useless) features, "Head Size(cm^3)" is feature that we will use. Our label is "Brain Weight(grams)".

[Download](https://github.com/Dipeshpal/Machine-Learning/blob/master/Linear%20Regression/headbrain.csv) or [View](https://github.com/Dipeshpal/Machine-Learning/blob/master/Linear%20Regression/headbrain.csv) dataset.

![headbrain dataset](https://raw.githubusercontent.com/Dipeshpal/Machine-Learning/master/Raw%20Images/headbrain%20dataset.png)

### 1.1. Python Implementation Manually (without using sklearn)-

**Code-** 
[Click here](https://github.com/Dipeshpal/Machine-Learning/blob/master/Linear%20Regression/Single%20Variable%20Linear%20Regression%20Manually.py) to view source code file

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

**Now we will calculate slope (m) and intercept ( c )-** 

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
