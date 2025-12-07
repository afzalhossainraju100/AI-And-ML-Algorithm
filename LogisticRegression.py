# -----------------------------------------
# 1. Load Dataset
# -----------------------------------------
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

import sys

# Load data. Try to get a DataFrame; if pandas is not installed, fall back to numpy arrays.
try:
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
except Exception as e:
    # sklearn raises ImportError with message 'requires pandas' when pandas is missing
    msg = str(e)
    if "requires pandas" in msg or "pandas" in msg:
        print("pandas is not available. Falling back to numpy arrays.")
        print("To use DataFrame output, install pandas: python -m pip install pandas")
    else:
        # other exceptions: print and re-raise
        print("Warning while fetching dataset:", msg)
    # Fallback: fetch without as_frame and use numpy arrays
    data = fetch_california_housing(as_frame=False)
    X = data.data
    y = data.target

# -----------------------------------------
# 2. Select Features & Split Data
# -----------------------------------------
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------
# 3. Train the Model
# -----------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------------------
# 4. Make Predictions
# -----------------------------------------
y_pred = model.predict(X_test)

# -----------------------------------------
# 5. Evaluate Model
# -----------------------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# -----------------------------------------
# 6. Visualization: Actual vs Predicted
# -----------------------------------------
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Actual vs Predicted House Prices")
plt.show()

# -----------------------------------------
# 7. Colored Plot for Better Understanding
# -----------------------------------------
errors = y_test - y_pred

plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, c=errors, cmap="coolwarm")
plt.colorbar(label="Prediction Error")
plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Colored Error Plot: Actual vs Predicted")
plt.show()
