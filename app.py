from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))  # Make sure this file exists

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        prediction = model.predict([features])
        return render_template('index.html', prediction_text=f"Predicted Price: ₹{round(prediction[0], 2)}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
