# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# # --- Q1: Load and Train ---
# data = load_breast_cancer()
# X = data.data
# y = data.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = LogisticRegression(max_iter=10000)
# model.fit(X_train, y_train)

# y_prediction = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_prediction)

# print("--- Q1: Basic Logistic Regression ---")
# print(f"Accuracy: {accuracy * 100:.2f}%")

# # --- Q2: Confusion Matrix & Metrics Analysis ---

# # 1. Generate the Confusion Matrix numbers
# # This shows: [TN, FP] / [FN, TP]
# cm = confusion_matrix(y_test, y_prediction)

# # 2. Generate the Detailed Report (Precision, Recall, F1)
# # target_names helps us see 'malignant' vs 'benign' instead of 0 vs 1
# report = classification_report(y_test, y_prediction, target_names=data.target_names)

# print("\n--- Q2: Confusion Matrix ---")
# print(cm)
# print("\n--- Detailed Classification Report ---")
# print(report)

# # 3. Create a Visual Plot of the Matrix
# # This makes it easier to see where the "Doctor" made mistakes
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
# disp.plot(cmap='Blues')
# plt.title("Confusion Matrix: Breast Cancer Detection")

# # Save the plot so you can view it on your HP Pavilion
# plt.savefig('confusion_matrix.png')
# print("\n[Visual Matrix saved as confusion_matrix.png]")



from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data  # 4 features: sepal length/width, petal length/width
y = iris.target # 3 classes: 0, 1, 2

# 2. Split into Training and Testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the Multi-class Logistic Regression model
# multi_class='multinomial' tells the model to use Softmax logic
# solver='lbfgs' is a standard algorithm that handles multiclass well
model_iris = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
model_iris.fit(X_train, y_train)

# 4. Predict
y_pred_iris = model_iris.predict(X_test)

# 5. Evaluate
print("--- Q3: Multi-Class Iris Classification ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_iris) * 100:.2f}%")

# 6. Detailed Report
# The target_names here will be ['setosa', 'versicolor', 'virginica']
print("\nClassification Report:")
print(classification_report(y_test, y_pred_iris, target_names=iris.target_names))

# 7. Confusion Matrix (It will be 3x3 this time!)
print("\nConfusion Matrix (3x3):")
print(confusion_matrix(y_test, y_pred_iris))