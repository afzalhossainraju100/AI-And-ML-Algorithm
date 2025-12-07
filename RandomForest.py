from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# 1. Load Dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# 2. Select Features & Target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Make Predictions
y_pred = model.predict(X_test)

# 5. Evaluate Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Calculate Errors for colored plot
errors = abs(y_test - y_pred)

# 6. Visualization: Actual vs Predicted
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted (Linear Regression)")
plt.show()

# 7. Colored Plot
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, c=errors, cmap="viridis")
plt.colorbar(label="Prediction Error")
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Colored Error Plot")
plt.show()
