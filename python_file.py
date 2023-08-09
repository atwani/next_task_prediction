import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle

pickled_model = pickle.load(open('model_linear2.pkl', 'rb'))
data = np.array([[sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5] ]])
prediction=pickled_model.predict(data);
#print(prediction)
