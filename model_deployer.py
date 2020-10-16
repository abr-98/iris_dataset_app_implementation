import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
import sys
import os
from flask import Flask,request
import flask

with open('./model.pkl', 'rb') as model_p:
    model_lr=pickle.load(model_p)    

def getParameters():
    parameters = []
    parameters.append(float(request.args.get('SL')))
    parameters.append(float(request.args.get('SW')))
    parameters.append(float(request.args.get('PL')))
    parameters.append(float(request.args.get('PW')))
    return parameters
    
def sendResponse(responseObj):
    response = flask.jsonify(responseObj)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET')
    response.headers.add('Access-Control-Allow-Headers', 'accept,content-type,Origin,X-Requested-With,Content-Type,access_token,Accept,Authorization,source')
    response.headers.add('Access-Control-Allow-Credentials', True)
    return response

app=Flask(__name__)
@app.route('/iris_predict',methods=['GET'])
def iris_predict():
    parameters = getParameters()
    print(parameters)
    inputFeature = np.array([parameters])
    print(inputFeature)
    #print(model_lr)
    prediction=model_lr.predict(inputFeature)
    print(prediction[0])
    return sendResponse(str(prediction[0]))

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000)



