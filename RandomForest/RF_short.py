import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from rf_regressor import RandomForestRegressor, mean_squared_error

# === Load and prepare your dataset ===
dane = dane.apply(pd.to_numeric)

# target variable - cena lotu (zł)
X = dane.drop(columns="Price")
y = dane["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()



# === Quick MSE test for different number of trees ===
n_trees_list = [1, 5, 10, 20, 30, 50, 100]
mse_list = []

print("Quick MSE test — does performance improve as trees increase?\n")

for n_trees in n_trees_list:
    rf = RandomForestRegressor(n_estimators=n_trees, max_depth=5, max_features='sqrt', bootstrap=True)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_list.append(mse)
    print(f"n_estimators = {n_trees:3} | Test MSE = {mse:.4f}")

# === Optional: Plot ===
plt.plot(n_trees_list, mse_list, marker='o')
plt.xlabel("Number of Trees")
plt.ylabel("Test MSE")
plt.title("MSE vs Number of Trees")
plt.grid(True)
plt.show()