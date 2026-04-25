# Sheet1: Assignments

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

#Loading the IRIS/iris dataset
iris = datasets.load_iris()


#Creating a DataFrame:
data_frame = pd.DataFrame(data=iris.data, columns=iris.feature_names)

#Adding a column for the species name:
data_frame['species'] = iris.target


# plt.scatter(data_frame['petal length (cm)'], data_frame['petal width (cm)'], c=iris.target)

# # Labels:
# plt.xlabel('Petal Length')
# plt.ylabel('Petal Width')
# plt.title('Petal Analysis')

plt.hist(data_frame['petal width (cm)'], bins=20, color='purple', alpha=0.7)

plt.xlabel('Petal Width')
plt.ylabel('Frequency')
plt.title('Petal Width Distribution')

# Show the hist plot:
plt.show()


