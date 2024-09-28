from flask import Flask,render_template,request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

try:
    sleep_model = joblib.load('sleeping_disorder_predict1')
except Exception as e:
    print(f"Error while loading the model: {e}")



@app.route("/",methods=['GET'])
def Home():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    
    try:
        float_features = [float(x) for x in request.form.values()]
    

        features = [np.array(float_features)]
        prediction = sleep_model.predict(features)
        if prediction[0] == 0:
            sleeping_status = 'INSOMNIA'
        elif prediction[0] == 1:
            sleeping_status = 'NORMAL:NO SLEEPING DISORDER'
        else:
            sleeping_status = 'SLEEP APNEA'

        return render_template("pred.html", prediction_text=f"{sleeping_status}")
    except Exception as e:
        print("Hello")
        print(f"Error making prediction: {e}")
        return render_template("pred.html", prediction_text="Error making prediction.")

if __name__ == "__main__":
    app.run(debug=True)