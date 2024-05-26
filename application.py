import pickle
from flask import Flask, request, jsonify, render_template 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler





# @1st: Making Flask's app name
application = Flask(__name__)
app = application
# Deploy karne ke liye hume app.py ko rename kiya application.py mai




# This is of no use atleast for this ML Project
# @app.route("/")
# def hello_world(): 
#     return "<h1>Hello, World!</h1>"




# @2nd:  Yahe karna hai
## import ridge regressor model and standard scaler pickle file
# Means .pkl file ko load karenge

ridge_model = pickle.load(open('Models/ridge.pkl','rb'))
standard_scaler = pickle.load(open('Models/scaler.pkl','rb'))





## Route for Home Page: 
# @3rd:
@app.route('/')
def index():
    return render_template('index.html')





# @4th:   isske sath home.html bhi hai 
# By the way hume frontend engineer wala kaam nhi karna hai, data scientist wala kaam karna hai :)
@app.route('/predictdata', methods=['GET', 'POST'])  # essa karke mera ek page open ho jana chahiye. Jo I/P hume model ko dena hai, voh sabhi hum yaha pe likhenge. ## means jo I/P hum denge usska text field aa gay aur Submit button aa gay yahe 'predictdata' wala page khul ke
# This post method is for giving data in backend
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region= float(request.form.get('Region'))
        # Yahe same order mai hona chahiye (v. imp)

        # Scaling
        new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_data_scaled)

        # aab result pass karke home.html mai dikhane ke liye:
        return render_template('home.html', result= result[0])
    else:
        return render_template('home.html')





# This is the Entry point
# This is the most imp thing to do among all the above
if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)
