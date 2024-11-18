'''
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the pre-trained model
with open('./models/rf_model_final.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get form data
        try:
            age = float(request.form["age"])
            sex = float(request.form["sex"])
            cp = float(request.form["cp"])
            trestbps = float(request.form["trestbps"])
            chol = float(request.form["chol"])
            fbs = float(request.form["fbs"])
            restecg = float(request.form["restecg"])
            thalach = float(request.form["thalach"])
            exang = float(request.form["exang"])
            oldpeak = float(request.form["oldpeak"])
            slope = float(request.form["slope"])
            ca = float(request.form["ca"])
            thal = int(request.form["thal"])
            
            # Prepare input for model prediction
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            pred_proba = model.predict_proba(input_data)[0][1]
            prediction = "Yes" if pred_proba > 0.5 else "No"

            return render_template("index.html", prediction=prediction, probability=f"{pred_proba:.2f}")

        except ValueError:
            return render_template("index.html", error="Please enter valid values.")
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)
'''
'''
from flask import Flask, request, jsonify
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with open('./models/xgb_model_final.pkl', 'rb') as file:
    xgb_model = pickle.load(file)

# Route for predicting hypertension
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([data['exang'], data['ca'], data['oldpeak'], data['cp'], data['thalach'], data['thal']]).reshape(1, -1)
    prediction = xgb_model.predict(features)
    prediction_proba = xgb_model.predict_proba(features)[0][1]  # Probability of positive case

    response = {
        'prediction': int(prediction[0]),
        'probability': prediction_proba
    }
    return jsonify(response)

# Home route
@app.route('/')
def home():
    return "Welcome to the Hypertension Prediction API!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
'''
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the pre-trained model
with open('./models/xgb_model_final.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get form data
        try:
            #age = float(request.form["age"])
            #sex = float(request.form["sex"])
            cp = float(request.form["cp"])
            #trestbps = float(request.form["trestbps"])
            #chol = float(request.form["chol"])
            #fbs = float(request.form["fbs"])
            #restecg = float(request.form["restecg"])
            thalach = float(request.form["thalach"])
            exang = float(request.form["exang"])
            oldpeak = float(request.form["oldpeak"])
            #slope = float(request.form["slope"])
            ca = float(request.form["ca"])
            thal = int(request.form["thal"])
            
            # Prepare input for model prediction
            input_data = np.array([[cp, thalach, exang, oldpeak, ca, thal]])
            pred_proba = model.predict_proba(input_data)[0][1]
            prediction = "Yes" if pred_proba > 0.5 else "No"

            return render_template("index.html", prediction=prediction, probability=f"{pred_proba:.2f}")

        except ValueError:
            return render_template("index.html", error="Please enter valid values.")
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)
