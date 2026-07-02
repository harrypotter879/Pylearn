import numpy as np
from pylearn.gradient_descent import MSELoss ,LinearModel ,Batch_gradient_Descent,RidgeLoss

# 1. Create classification data (0 for small numbers, 1 for large numbers)
X_train_class = np.array([[1], [2], [5], [6]])
y_train_class = np.array([0, 0, 1, 1]) # Binary labels

# 2. Grab your "cartridges" for Logistic Regression
logistic_model = LinearModel()
logistic_loss = MSELoss()

# 3. Slide them into the EXACT SAME gradient descent engine
classifier = Batch_gradient_Descent(
    learning_rate=0.001, 
    epochs=1000, 
    loss_fn=logistic_loss, 
    model=logistic_model
)

# 4. Train it exactly the same way
classifier.fit(X_train_class, y_train_class)

# 5. Predict the probability for a new value (X = 5.5 should be close to 1)
X_new_class = np.array([[5.5]])
prob = classifier.predict(X_new_class)

print(f"Probability of being class 1: {prob[0]:.4f}")