# Sheet1: Examples

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

#Loading the IRIS/iris dataset
iris = datasets.load_iris()


#Creating a DataFrame:
data_frame = pd.DataFrame(data=iris.data, columns=iris.feature_names)

#Adding a column for the species name:
data_frame['species'] = iris.target

#test:
print(data_frame.head(150))
print(data_frame.describe())


# Create the scatter plot:(What is the meaning of sepal and petal?, what is the target property?, Ask TA/Atef):
plt.scatter(data_frame['sepal length (cm)'], data_frame['sepal width (cm)'], c=iris.target)

# Labels:
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Sepal Analysis')

# Show the scatter plot:
plt.show()


#Compute the mean of the sepal length of all species:
averages = data_frame.groupby('species')['sepal length (cm)'].mean()

print(averages)