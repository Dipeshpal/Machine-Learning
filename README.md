# Machine Learning
Machine learning is an application of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it learn for themselves.
![Machine Learning](https://cdn-images-1.medium.com/max/1200/1*ZkZS46p7Lbw-PDBtPMfEEw.jpeg)

## Features of Machine Learning-
1. It uses the data to detect patterns in a dataset and adjust program actions accordingly.
2. It focuses on the developement of computer program that can teach themselves to grow and change when exposed to new data.
3. It enables computers to find hidden insights using iterative alogrith without being explicitly programmed.
4. Machine Learning is a method of data analysis that automates analytical model building.

## Some machine learning methods-
Machine learning algorithms are often categorized as supervised or unsupervised.

1. **Supervised machine learning** algorithms can apply what has been learned in the past to new data using labeled examples to predict future events. Starting from the analysis of a known training dataset, the learning algorithm produces an inferred function to make predictions about the output values. The system is able to provide targets for any new input after sufficient training. The learning algorithm can also compare its output with the correct, intended output and find errors in order to modify the model accordingly.

2. **Unsupervised machine learning** algorithms are used when the information used to train is neither classified nor labeled. Unsupervised learning studies how systems can infer a function to describe a hidden structure from unlabeled data. The system doesn’t figure out the right output, but it explores the data and can draw inferences from datasets to describe hidden structures from unlabeled data.

3. **Semi-supervised machine learning** algorithms fall somewhere in between supervised and unsupervised learning, since they use both labeled and unlabeled data for training – typically a small amount of labeled data and a large amount of unlabeled data. The systems that use this method are able to considerably improve learning accuracy. Usually, semi-supervised learning is chosen when the acquired labeled data requires skilled and relevant resources in order to train it / learn from it. Otherwise, acquiringunlabeled data generally doesn’t require additional resources.

4. **Reinforcement machine learning** algorithms is a learning method that interacts with its environment by producing actions and discovers errors or rewards. Trial and error search and delayed reward are the most relevant characteristics of reinforcement learning. This method allows machines and software agents to automatically determine the ideal behavior within a specific context in order to maximize its performance. Simple reward feedback is required for the agent to learn which action is best; this is known as the reinforcement signal.

## How it works?

![enter image description here](https://cdn-images-1.medium.com/max/1600/0*i7crGI3BrI_Xkd3l)

It learns from given data and find insights. According to data it makes changes in program.

## Steps-

![Steps of Machine Learning](https://raw.githubusercontent.com/Dipeshpal/Machine-Learning/master/Raw%20Images/Steps%20of%20ML.PNG)

## Examples-

 1. Google Maps: Find your historical patterns etc.
 2. Uber: Find location, find your patterns etc.
 3. Social Media (Tagging): Facebook dface project for tagging people on image.
 4. Virtual Assistant: Google Assistant, Alexa, Cortana etc
 5. Product Recommendations: Amazon
 6. Self Driving Car: Tesla
 7. **Applications using ML**: Netfilx, Amazon, Hulu etc.
  
## Machine Learning Life Cycle-
1. Collecting Data: Collecting data from various sources
2. Data Wrangling: Formatting raw data. 
3. Analyse Data: Analysis on data
4. Train Algorithm: Fit model
5. Test Algorithm: Test model
6. Deployment: Deploy in real system

## Important Python Libraries for Machine Learning-

 1. [Scikit-learn](https://scikit-learn.org/stable/)
 2. [Seaborn](https://seaborn.pydata.org/)
 3. [Matplotlib](https://matplotlib.org/)
 4. [Pandas](https://pandas.pydata.org/)
 5. [Numpy](https://www.numpy.org/)


## Types of Problems in Machine Learning-
1.  **Supervised Learning**-
	
	1.1. **Classification**
	
	1.2. **Regression**
	
2. **Unsupervised Learning**
	
	2.1. **Clustering**
	
	2.2. **Association**

**Lets Understand in Detail-**

####  1. Supervised Learning-
Supervised learning problems can be further grouped into regression and classification problems.

Supervised learning is where you have input variables (x) and an output variable (Y) and you use an algorithm to learn the mapping function from the input to the output.

Y = f(X)

The goal is to approximate the mapping function so well that when you have new input data (x) that you can predict the output variables (Y) for that data.

It is called supervised learning because the process of an algorithm learning from the training dataset can be thought of as a teacher supervising the learning process. We know the correct answers, the algorithm iteratively makes predictions on the training data and is corrected by the teacher. Learning stops when the algorithm achieves an acceptable level of performance.

-   **Classification**: A classification problem is when the output variable is a category, such as “red” or “blue” or “disease” and “no disease”.
-   **Regression**: A regression problem is when the output variable is a real value, such as “dollars” or “weight”.

####  2. Unsupervised Learning-
Unsupervised learning is where you only have input data (X) and no corresponding output variables.

The goal for unsupervised learning is to model the underlying structure or distribution in the data in order to learn more about the data.

These are called unsupervised learning because unlike supervised learning above there is no correct answers and there is no teacher. Algorithms are left to their own devises to discover and present the interesting structure in the data.

Unsupervised learning problems can be further grouped into clustering and association problems.

-   **Clustering**: A clustering problem is where you want to discover the inherent groupings in the data, such as grouping customers by purchasing behavior.
-   **Association**: An association rule learning problem is where you want to discover rules that describe large portions of your data, such as people that buy X also tend to buy Y.


## Algorithms-
We wil talk about following algorithm. We will also talk about Mathematics behind it and Implementation in Python.
 1. **Supervised Learning-**
 
	  1.1. **[Linear Regression](https://github.com/Dipeshpal/Machine-Learning/tree/master/Linear%20Regression)**

	  1.2. **Logistic Regression**

	  1.3. **Decision Tree**

	  1.4. **Random Forest**

	  1.5. **Naive Bayes Classifier**
	  
	  1.6. **Support vector machines** for classification problems.
	 
2. **Unsupervised learning-** 

	2.1. **k-means** for clustering problems

	2.2. **Apriori algorithm** for association rule learning problems

3. **Semi-Supervised Learning**
4. **Reinforcement Learning**
