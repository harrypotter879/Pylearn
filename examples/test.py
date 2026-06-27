import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Increase noise standard deviation to ensure variance
features_data = np.random.randint(10, 100, size=(10, 2))
target_data = (
    3 * features_data[:, 0]
    + 2 * features_data[:, 1]
    + np.random.normal(0, 20, 10)  # Increased noise from 10 to 20
)

df = pd.DataFrame(features_data, columns=['Feature_A', 'Feature_B'])
df['Target'] = target_data

X = df[['Feature_A', 'Feature_B']]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

# Scale features
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# Scale target variable
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

from pylearn.linear import Linear_Reg
from pylearn.gradient_descent import Batch_gradient_Descent
from pylearn.metrics import Regression_Metrics
from sklearn.metrics import r2_score

# Adjust hyperparameters for better convergence
bgd = Batch_gradient_Descent(learning_rate=0.0001, epochs=1000)
bgd.fit(X_train, y_train_scaled)
y_pred_scaled = bgd.predict(X_test)

# Inverse transform predictions to original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Debug: Print y_test and y_pred
print("y_test:", y_test.values)
print("y_pred:", y_pred)

# Calculate R2 using sklearn for comparison
sklearn_r2 = r2_score(y_test, y_pred)
print("Sklearn R2 Score:", sklearn_r2)

# Safe R2 calculation to avoid numerical instability
ss_res = np.sum((y_test - y_pred) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
if ss_tot == 0:
    print("Warning: y_test has zero variance. R2_Score cannot be computed.")
else:
    safe_r2 = 1 - (ss_res / ss_tot)
    print("Safe R2 Score:", safe_r2)

# Use pylearn's R2_Score for comparison
rm = Regression_Metrics()
pylearn_r2 = rm.R2_Score(y_test, y_pred)
print("Pylearn R2 Score:", pylearn_r2)