from pylearn.linear import Linear_Reg
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

lr = Linear_Reg()
lr.fit_predict(X_train,y_train,X_test)