import numpy as np
from pylearn.linear import Ridge_Regression
from pylearn.metrics import MAE, MSE, RMSE, R2_Score

# Generate some sample data for Ridge Regression
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Split data into training and testing sets (simplified for example)
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# Initialize and train the Ridge Regression model
ridge_model = Ridge_Regression(alpha=0.1)  # You can adjust the alpha value
ridge_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = ridge_model.predict(X_test)

# Display coefficients and intercept
print(f"Ridge Regression Weights: {ridge_model.weights}")
print(f"Ridge Regression Intercept: {ridge_model.intercept}")

# Calculate and display different accuracy scores
print("\n--- Regression Metrics ---")
print(f"Mean Absolute Error (MAE): {MAE().cal(y_test, y_pred)}")
print(f"Mean Squared Error (MSE): {MSE().cal(y_test, y_pred)}")
print(f"Root Mean Squared Error (RMSE): {RMSE().cal(y_test, y_pred)}")
print(f"R2 Score: {R2_Score().cal(y_test, y_pred)}")
    