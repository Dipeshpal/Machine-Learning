
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
1. **Single Variable Linear Regression:** First we will predict Price of the Houses according to Area of House (One Input and One Output). We will solve this by **Manually** and by using **Sklearn** also.
2. **Multivariate Linear Regression-** Multiple Input (area of house, distance of house from highway etc.) and Single Output. By using **Sklearn**.
