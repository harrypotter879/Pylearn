from pylearn.linear import Linear_Reg
from pylearn.gradient_descent import Batch_gradient_Descent
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(42)

features_data = np.random.randint(10, 100, size=(10, 2))  
target_data = np.random.randint(0, 2, size=(10, 1))       

df = pd.DataFrame(features_data, columns=['Feature_A', 'Feature_B'])
df['Target'] = target_data


X = df[['Feature_A', 'Feature_B']]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

bgd = Batch_gradient_Descent(learning_rate=0.001,epochs=100)
bgd.fit(X_train,y_train)
print(bgd.predict(X_test))