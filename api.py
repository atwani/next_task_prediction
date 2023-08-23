import sys
import flask
from flask import request
import io 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
#import tensorflow as tf
import joblib
import json
#import jsonify
app = flask.Flask(__name__)
model = joblib.load(r'model_linear2.pkl')

@app.route('/predict',methods=['POST'])
def predict():
	features = flask.request.get_json(force=True)
	data= pd.DataFrame([features])
	data1 = data.to_numpy()
	scaler=StandardScaler()
	scaler_datax=scaler.fit_transform(data1)
	prediction=model.predict(scaler_datax)
	ts=np.array2string(prediction)
	
	return(prediction)
	#event=np.array(event)
	#data={'Total1':[0],'File':[0],'File submissions':[0],'System':[0]}
	


if __name__=='__main__':
	app.run(debug=True)
