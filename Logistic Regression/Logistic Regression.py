# Imoport Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("Marks Dataset.csv", header=None)

print(data.head(4))
print(data.shape)
print(type(data))

# X = feature values, all the columns except the last column
x = data.iloc[:, :-1]

# y = target values, last column of the data frame
y = data.iloc[:, -1]

print(x.shape)
print(y.shape)

# filter out the applicants that got admitted
admitted = data.loc[y == 1]

# filter out the applicants that din't get admission
not_admitted = data.loc[y == 0]

# plots
plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
plt.legend()
plt.show()

