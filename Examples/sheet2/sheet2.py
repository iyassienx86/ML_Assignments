
import pandas as pd
import numpy as np
import os

# folder path
path = ("/home/electronica/ML_Assignments")
# print("Path to folder:", path)

#Just shortening the argument to be thrown
file_path = os.path.join(path, "employee_salary_data.csv")

# Prepare
data_frame = pd.read_csv(file_path)

# Test :)
# print("\n--- Data IS READY ---")
# print(data_frame.head(2))

# Here we go 


# Load your data

# Step A: Use Experienceyears to predict Salary
X = data_frame['YearsExperience'].values
path = ("/home/electronica/ML_Assignments")
file_path = os.path.join(path, "employee_salary_data.csv")

data_frame = pd.read_csv(file_path)

X = data_frame['YearsExperience'].values
y = data_frame['Salary'].values

def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m = 0
    b = 0
    n = len(y)
    for i in range(iterations):
        y_predicted = m * X + b
        cost = (1/n) * sum([val**2 for val in (y - y_predicted)])
        m_gradient = -(2/n) * sum(X * (y - y_predicted))
        b_gradient = -(2/n) * sum(y - y_predicted)
        m = m - (learning_rate * m_gradient)
        b = b - (learning_rate * b_gradient)
        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost:.2f}, m {m:.2f}, b {b:.2f}")
            
    return m, b

final_m, final_b = gradient_descent(X, y)

print(f"\n--- Training Complete ---")
print(f"Final Slope (m): {final_m}")
print(f"Final Intercept (b): {final_b}")

years = 10
predicted_salary = final_m * years + final_b
print(f"Predicted Salary for {years} years: ${predicted_salary:,.2f}")

# Step B: Implement Gradient Descent (from scratch)
def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m = 0  # Starting Slope (Weight)
    b = 0  # Starting Intercept (Bias)
    n = len(y) # Number of data points

    for i in range(iterations):
        # 1. Make a prediction using the current line (y = mx + b)
        y_predicted = m * X + b
        
        # 2. Calculate the Error (Cost)
        cost = (1/n) * sum([val**2 for val in (y - y_predicted)])
        
        # 3. Calculate the "Gradients" (The direction to move)
        # These are derived from calculus to find the slope of the error curve
        m_gradient = -(2/n) * sum(X * (y - y_predicted))
        b_gradient = -(2/n) * sum(y - y_predicted)
        
        # 4. Update the numbers (The "Learning" step)
        m = m - (learning_rate * m_gradient)
        b = b - (learning_rate * b_gradient)
        
        # Optional: Print progress every 100 steps
        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost:.2f}, m {m:.2f}, b {b:.2f}")
            
    return m, b

# Run the training
final_m, final_b = gradient_descent(X, y)

print(f"\n--- Training Complete ---")
print(f"Final Slope (m): {final_m}")
print(f"Final Intercept (b): {final_b}")

# Testing a prediction
years = 10
predicted_salary = final_m * years + final_b
print(f"Predicted Salary for {years} years: ${predicted_salary:,.2f}")