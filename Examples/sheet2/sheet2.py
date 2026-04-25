import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import os

# 1. Setup paths and load data
path = "/home/electronica/ML_Assignments"
file_path = os.path.join(path, "employee_salary_data.csv")
data_frame = pd.read_csv(file_path)

# --- PART A: MANUAL GRADIENT DESCENT ---
# We use .values to get flat 1D arrays for the manual math
X_manual = data_frame['YearsExperience'].values
y_manual = data_frame['Salary'].values

def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m = 0
    b = 0
    n = len(y)
    for i in range(iterations):
        y_predicted = m * X + b
        # Now y and y_predicted are both 1D, so this works!
        cost = (1/n) * np.sum((y - y_predicted)**2)
        
        m_gradient = -(2/n) * np.sum(X * (y - y_predicted))
        b_gradient = -(2/n) * np.sum(y - y_predicted)
        
        m = m - (learning_rate * m_gradient)
        b = b - (learning_rate * b_gradient)
        
        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost:.2f}, m {m:.2f}, b {b:.2f}")
            
    return m, b

# Run the manual training
final_m, final_b = gradient_descent(X_manual, y_manual)

print(f"\n--- Manual Training Complete ---")
print(f"Final Slope (m): {final_m:.2f}")
print(f"Final Intercept (b): {final_b:.2f}")

# --- PART B: SCIKIT-LEARN (Linear, Ridge, Lasso) ---
# We use the DataFrame directly for X so it stays 2D and keeps column names
X_sk = data_frame[['YearsExperience']] # Note the double brackets
y_sk = data_frame['Salary']

lin_reg = LinearRegression().fit(X_sk, y_sk)
ridge_reg = Ridge(alpha=1.0).fit(X_sk, y_sk)
lasso_reg = Lasso(alpha=1.0).fit(X_sk, y_sk)

print("\n--- Model Comparison ---")
print(f"{'Feature':<15} | {'Linear':<10} | {'Ridge':<10} | {'Lasso':<10}")
print("-" * 50)

# We use X_sk.columns because it is a DataFrame and has names
for i, feature in enumerate(X_sk.columns):
    print(f"{feature:<15} | {lin_reg.coef_[i]:.2f} | {ridge_reg.coef_[i]:.2f} | {lasso_reg.coef_[i]:.2f}")