import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

# 1. Load the data
path = "/home/electronica/ML_Assignments"
file_path = os.path.join(path, "employee_salary_data.csv")
df = pd.read_csv(file_path)

# 2. Prepare X and y
# We will use all features for a better prediction
X = df[['YearsExperience', 'Age', 'WorkingHoursPerWeek']]
y = df['Salary']

# 3. Split the data (70% Train, 30% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #42 is the seed

# 4. Train the model using ONLY the training set
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions on the Testing set (The "Exam")
y_prediction = model.predict(X_test)

# 6. Calculate the "Grade" (MSE)
mse = mean_squared_error(y_test, y_prediction)

print(f"--- Assignment Part 1 Results ---")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"Model MSE: {mse:.2f}")


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_prediction, color='blue', label='Predictions')

# 7. Add a "Perfect Match" line (Diagonal line)
# This line shows where Actual = Predicted
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect Fit Line')

# 8. Add Labels and Title
plt.xlabel('Actual Salaries ($)')
plt.ylabel('Predicted Salaries ($)')
plt.title('Assignment Part 2: Actual vs Predicted Salaries')
plt.legend()
plt.grid(True)

# 9. Save the result
plt.savefig('actual_vs_predicted.png')
print("\n[Assignment Complete: Visualization saved as actual_vs_predicted.png]")