import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Dummy dataset (ya apna actual CSV use karo)
data = {
    'area': [1000, 1500, 2000],
    'bedrooms': [2, 3, 4],
    'bathrooms': [1, 2, 3],
    'price': [300000, 500000, 700000]
}

df = pd.DataFrame(data)

X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']

model = LinearRegression()
model.fit(X, y)

# ✅ Save proper .pkl model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
