# Logistic regression-
Logistic Regression produce results in a binary format which is used to predict the outcome od a categorical dependent variabel. So the outcome should be discrete/categorical such as:

 - 0 or 1
 
 - yes or No
 
 - True or False
   
 -  High or Low

When the number of possible outcomes is only two it is called **Binary Logistic Regression**.

## Why not Linear Regression?
Because in Linear Regression we have values between -∞ to +∞. But in Logistic Regression we have discrete values (0 or 1).

**Linear Regression** is used for **Regression** problems and **Logistic Regression** is used for **Classification** problems.

![enter image description here](http://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1534281070/linear_vs_logistic_regression_edxw03.png)

In Logistic Regression we have curve like "S". In Logistic Regression we map any real-valued number into a value between 0 and 1.

So, we use **Sigmoid Function**. Sigmoid Function map any real-valued number into a value between 0 and 1.
We have threshold value here which helps for classifications.

In this figure you  can see that  threshold value  is 0.5. All the values above 0.5 belongs to class 1 and all the values less then 0.5 belongs to class 0.

![Logistic Regression](https://cdn-images-1.medium.com/max/800/1*zfH9946AssCx4vzjaizWeg.png)

## Working of Logistic Regression-

In Linear Regression, the output is the weighted sum of inputs. **Logistic Regression is a generalized Linear Regression in the sense that we don’t output the weighted sum of inputs directly, but we pass it through a function that can map any real value between 0 and 1.**

**If we take the weighted sum of inputs as the output as we do in Linear Regression, the value can be more than 1 but we want a value between 0 and 1. That’s why Linear Regression can’t be used for classification tasks**.

We can see from the below figure that the output of the linear regression is passed through an  **activation** function that can map any real value between 0 and 1.

![](https://cdn-images-1.medium.com/max/800/1*8q9ztX9dGVCv7e0DmH_IVA.png)

The activation function that is used is known as the  **sigmoid** function. The plot of the sigmoid function looks like

![](https://cdn-images-1.medium.com/max/800/1*yKvimZ3MCAX-rwMX2n87nw.png)

sigmoid function

We can see that the value of the sigmoid function always lies between 0 and 1. The value is exactly 0.5 at X=0. We can use 0.5 as the probability threshold to determine the classes. If the probability is greater than 0.5, we classify it as  **Class-1(Y=1)**  or else as  **Class-0(Y=0)**.

Before we build our model let’s look at the assumptions made by Logistic Regression

-   The dependent variable must be categorical
-   The independent variables(features) must be independent (to avoid multicollinearity).

#### Let's understand Dataset first then we will understand Hypothesis and Cost function-

The data used in this blog has been taken from Andrew Ng’s  [Machine Learning](https://www.coursera.org/learn/machine-learning)  course on Coursera. The data can be downloaded from  [here](https://github.com/Dipeshpal/Machine-Learning/blob/master/Logistic%20Regression/Marks%20Dataset.csv). **The data consists of marks of two exams for 100 applicants. The target value takes on binary values 1,0. 1 means the applicant was admitted to the university whereas 0 means the applicant didn't get an admission. The objective is to build a classifier that can predict whether an application will be admitted to the university or not.**

**Code-**

Let’s load the data into `pandas` Dataframe using the `read_csv` function. We will also split the data into `admitted` and `non-admitted` to visualize the data.


```
# Imoport Libraries  
import matplotlib.pyplot as plt  
import pandas as pd  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score  
from sklearn.model_selection import train_test_split
```

```
# Load Dataset
data = pd.read_csv("Marks Dataset.csv", header=None)  
print(data.head(4))  
print(data.shape)  
print(type(data))
```
![LR Output 1.1](https://raw.githubusercontent.com/Dipeshpal/Machine-Learning/master/Raw%20Images/LR%20Output%201.1.PNG)

```
# X = feature values, all the columns except the last column  
x = data.iloc[:, :-1].values  
# y = target values, last column of the data frame  
y = data.iloc[:, -1].values
```

```
# filter out the applicants that got admitted and that din't get admission  
admitted = data.loc[y == 1]  
not_admitted = data.loc[y == 0]  
  
# plots  
plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')  
plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')  
plt.legend()  
plt.show()
```

![LR Output 3](https://raw.githubusercontent.com/Dipeshpal/Machine-Learning/master/Raw%20Images/LR%20Output%203.PNG)



Now that we have a clear understanding of the problem and the data, let’s go ahead and build our model.

#### Training the model using Sklearn-

**Let’s first prepare the data for our model.**

```
# Split dataset into train and test  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
```
**Training-**
```
# Training the Model  
model = LogisticRegression()  
model = model.fit(x_train, y_train)  
y_pred = model.predict(x_test)  
accuracy = accuracy_score(y_test, y_pred)  
print("Accuracy: ", accuracy)  
print("Model Coefficient: ", model.coef_)  
print('Intercept', model.intercept_)
```

![Output 3](https://raw.githubusercontent.com/Dipeshpal/Machine-Learning/master/Raw%20Images/LR%20Output%201.2%20New.PNG)


```
coe = str(model.coef_[0])  
coe = coe[1:-1]  
coef = list(map(float, coe.split(' ')))  
  
x1 = - model.intercept_/coef[0]  
y2 = - model.intercept_/coef[1]  
  
x_plt = [x1, 0]  
y_plt = [0, y2]  
  
  
# plots  
plt.plot(x_plt, y_plt, label='Decision Boundary')  
plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')  
plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')  
plt.xlabel('Marks in 1st Exam')  
plt.ylabel('Marks in 2nd Exam')  
plt.legend()  
plt.show()
```

**Output-**

![enter image description here](https://raw.githubusercontent.com/Dipeshpal/Machine-Learning/master/Raw%20Images/LR%20Output%201.3.PNG)

#### Hypothesis and Cost Function

Till now we have understood how Logistic Regression can be used to classify the instances into different classes. In this section, we will define the hypothesis and the cost function.

A Linear Regression model can be represented by the equation.

![](https://cdn-images-1.medium.com/max/800/1*HsoXveMFMW46v9oKV3QikQ.png)

We then apply the sigmoid function to the output of the linear regression

![](https://cdn-images-1.medium.com/max/800/1*AlIJXuiC19cucDZ_1kE1pg.png)

where the sigmoid function is represented by,

![](https://cdn-images-1.medium.com/max/800/1*IO5RjkmyCq6t1VmZkW8Sxg.png)

The hypothesis for logistic regression then becomes,

![](https://cdn-images-1.medium.com/max/800/1*L9a6phB1ZzjRhb-VI3W1YQ.png)

![](https://cdn-images-1.medium.com/max/800/1*jStEeKa6l6KgQxbS8iGzrw.png)

If the weighted sum of inputs is greater than zero, the predicted class is 1 and vice-versa. So the decision boundary separating both the classes can be found by setting the weighted sum of inputs to 0.

#### **Cost Function**

Like Linear Regression, we will define a cost function for our model and the objective will be to minimize the cost.

The cost function for a single training example can be given by:

![](https://cdn-images-1.medium.com/max/800/1*qKfAYUsI0VPcIXVBbEdPEg.png)



