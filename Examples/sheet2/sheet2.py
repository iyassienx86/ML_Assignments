import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import os

# 1. Setup paths and load data
path = "/home/electronica/ML_Assignments"
file_path = os.path.join(path, "employee_salary_data.csv")
data_frame = pd.read_csv(file_path)

# --- PART A: MANUAL GRADIENT DESCENT ---
# X_manual = data_frame['YearsExperience'].values
# y_manual = data_frame['Salary'].values

# # We modify the function to track the cost history
# def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
#     m = 0
#     b = 0
#     n = len(y)
#     history = []  # <--- Step 1: Create the "diary" list
    
#     for i in range(iterations):
#         y_predicted = m * X + b
#         cost = (1/n) * np.sum((y - y_predicted)**2)
#         history.append(cost) # <--- Step 2: Save current cost to the diary
        
#         m_gradient = -(2/n) * np.sum(X * (y - y_predicted))
#         b_gradient = -(2/n) * np.sum(y - y_predicted)
        
#         m = m - (learning_rate * m_gradient)
#         b = b - (learning_rate * b_gradient)
        
#         if i % 100 == 0:
#             print(f"Iteration {i}: Cost {cost:.2f}, m {m:.2f}, b {b:.2f}")
            
#     return m, b, history # <--- Step 3: Return the diary along with m and b

# # Run the manual training and UNPACK the history
# final_m, final_b, history = gradient_descent(X_manual, y_manual)

# print(f"\n--- Manual Training Complete ---")
# print(f"Final Slope (m): {final_m:.2f}")
# print(f"Final Intercept (b): {final_b:.2f}")

# # --- PART B: SCIKIT-LEARN (Linear, Ridge, Lasso) ---
# X_sk = data_frame[['YearsExperience']] 
# y_sk = data_frame['Salary']

# lin_reg = LinearRegression().fit(X_sk, y_sk)
# ridge_reg = Ridge(alpha=1.0).fit(X_sk, y_sk)
# lasso_reg = Lasso(alpha=1.0).fit(X_sk, y_sk)

# print("\n--- Model Comparison ---")
# print(f"{'Feature':<15} | {'Linear':<10} | {'Ridge':<10} | {'Lasso':<10}")
# print("-" * 50)
# for i, feature in enumerate(X_sk.columns):
#     print(f"{feature:<15} | {lin_reg.coef_[i]:.2f} | {ridge_reg.coef_[i]:.2f} | {lasso_reg.coef_[i]:.2f}")

# # --- PART C: VISUALIZATION ---
# # Now 'history' exists, so this will work!
# plt.figure(figsize=(10,6))
# plt.plot(range(len(history)), history, color='red', linewidth=2)
# plt.title('How the Error Dropped (MSE vs Iterations)')
# plt.xlabel('Iteration Number')
# plt.ylabel('Mean Squared Error')
# plt.grid(True)
# plt.savefig('mse_drop.png')
# print("\n[Plot successfully saved as mse_drop.png]")




X = data_frame[['YearsExperience', 'Age', 'WorkingHoursPerWeek']]
y = data_frame['Salary']

model_fitted = LinearRegression().fit(X, y)

print("--- Multiple Linear Regression Results ---")
print(f"Intercept (Base Salary b): {model_fitted.intercept_:.2f}")

for feature, slope in zip(X.columns, model_fitted.coef_):
    print(f"Slope for {feature} (m): {slope:.2f}")

person = [[5, 30, 40]]
predicted = model_fitted.predict(person)
print(f"\nPredicted Salary for the example person: ${predicted[0]:,.2f}")