import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Example data
data = {
    'area': [1000, 1500, 2000, 2500],
    'bedrooms': [2, 3, 4, 4],
    'bathrooms': [1, 2, 3, 3],
    'price': [300000, 500000, 700000, 850000]
}
df = pd.DataFrame(data)

X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save as .pkl
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ model.pkl saved successfully.")
