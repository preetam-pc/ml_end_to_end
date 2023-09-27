import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import json

app=Flask(__name__)

regmodel=pickle.load(open('regmodel.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))

@app.route('/')
def home():
    return  render_template('home.html')


@app.route('/predict_api', methods=['Post'])
def predict_api():
    data= request.get_json('data')['data']
    print(type(data))
    print(data.keys())
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaler.transform( np.array(list(data.values())).reshape(1,-1))
    pred=regmodel.predict(new_data)
    print(pred[0])
    return jsonify(pred[0])

if __name__=="__main__":
    app.run(debug=True)    