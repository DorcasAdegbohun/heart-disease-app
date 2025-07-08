from flask import Flask, render_template, request, redirect
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('heart_disease_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form.get(f)) for f in [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]]
        prediction = model.predict(np.array([features]))[0]
        result = 'Positive for Heart Disease' if prediction == 1 else 'Negative for Heart Disease'
        print("Prediction result:", result)
        return render_template('result.html', prediction=result)
    except Exception as e:
        print("❌ Error during prediction:", e)
        return "Something went wrong. Please check the form inputs."

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
        