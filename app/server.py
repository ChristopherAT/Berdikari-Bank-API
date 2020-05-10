#
# IYKRA Data Fellowship - Data Engineer
# Practise Case Week 5: Model Deployment
# By: Christopher Aldora Tjitrabudi
#

import os
import pandas as pd
# from sklearn.externals import joblib
from flask import Flask, jsonify, request, render_template
import json 
import dill as pickle

app = Flask(__name__)

def load_pk():
    with open('../models/model_xgb.pk', 'rb') as file:
        model = pickle.load(file)
    with open('../models/category.pk', 'rb') as file:
        category = pickle.load(file)
    with open('../models/column.pk', 'rb') as file:
        columns = pickle.load(file)
    return (model,category,columns)

def json2df(json_data):
    print("   Converting JSON to DataFrame....")
    df = pd.DataFrame()
    try:
        # If the json data consist of more than 2 records
        for col in json_data:
            for row in json_data[col]:
                df.loc[int(row),col] = json_data[col][row]        
    except:
        # If the json_data is 1 record only
        for col in json_data:
            df.loc[1,col] = json_data[col]
    finally:
        print(df)
        return df
	
@app.route("/")
def start():
    return render_template("home.html") # returns the home page

@app.route('/predict', methods=['POST'])
def apicall():
    try:
        json_ = request.json
        test = json2df(json_)
    except Exception as e:
        raise e
    if test.empty:
        return jsonify({})
    else:
        print("   Predicting....")
        predictions = model.predict(test)
        responses = jsonify({"prediction": str(predictions.tolist())})
        responses.status_code = 200
        return responses
		
if __name__ == "__main__":
    print("   Loading model....")
    (model,category,columns) = load_pk()
    print("   Running flask....")
    app.run(host='0,0,0,0', port=5000)