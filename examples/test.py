import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
features_data = np.random.randint(10, 100, size=(100, 2))
target_data = (
    3 * features_data[:, 0] 
    + 2 * features_data[:, 1] 
    + np.random.normal(0, 20, 100)  # Increased noise
)

df = pd.DataFrame(features_data, columns=['Feature_A', 'Feature_B'])
df['Target'] = target_data

X = df[['Feature_A', 'Feature_B']]
y = df['Target']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Scale target variable
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# Import necessary classes
from pylearn.linear import Linear_Reg
from pylearn.gradient_descent import Batch_gradient_Descent, MiniBatch_Gradient_Descent
from pylearn.metrics import Regression_Metrics

# --- Linear Regression with Batch Gradient Descent ---
print("\n=== Linear Regression with Batch Gradient Descent ===")
bgd = Batch_gradient_Descent(learning_rate=0.0001, epochs=1000)
bgd.fit(X_train_scaled, y_train_scaled)
y_pred_batch = bgd.predict(X_test_scaled)
y_pred_batch_original = scaler_y.inverse_transform(y_pred_batch.reshape(-1, 1)).flatten()

print("y_test:", y_test.values)
print("y_pred_batch:", y_pred_batch_original)
print("Sklearn R2 Score (Batch):", r2_score(y_test, y_pred_batch_original))

# --- Linear Regression with Minibatch Gradient Descent ---
print("\n=== Linear Regression with Minibatch Gradient Descent ===")
minibatch_size = 10
minibatch_gd = MiniBatch_Gradient_Descent(learning_rate=0.0001, epochs=1000, batch_size=minibatch_size)
minibatch_gd.fit(X_train_scaled, y_train_scaled)
y_pred_minibatch = minibatch_gd.predict(X_test_scaled)
y_pred_minibatch_original = scaler_y.inverse_transform(y_pred_minibatch.reshape(-1, 1)).flatten()

print("y_pred_minibatch:", y_pred_minibatch_original)
print("Sklearn R2 Score (Minibatch):", r2_score(y_test, y_pred_minibatch_original))

# --- Polynomial Regression ---
print("\n=== Polynomial Regression (Degree 2) ===")
from pylearn.linear import Polynomial_Regression

poly_reg = Polynomial_Regression(degree=2)
poly_reg.fit(X_train_scaled, y_train_scaled)
y_pred_poly = poly_reg.predict(X_test_scaled)
y_pred_poly_original = scaler_y.inverse_transform(y_pred_poly.reshape(-1, 1)).flatten()

print("y_pred_poly:", y_pred_poly_original)
print("Sklearn R2 Score (Polynomial):", r2_score(y_test, y_pred_poly_original))

# --- Compare Results ---
print("\n=== Comparison of Results ===")
print("Batch Gradient Descent R2 Score:", r2_score(y_test, y_pred_batch_original))
print("Minibatch Gradient Descent R2 Score:", r2_score(y_test, y_pred_minibatch_original))
print("Polynomial Regression R2 Score:", r2_score(y_test, y_pred_poly_original))
