from flask import Flask,render_template,url_for,request, jsonify, json
from flask.wrappers import Response
import pandas as pd
# from sklearn.externals import joblib
import numpy as np
import keras
from werkzeug.wrappers import response
app = Flask(__name__)

@app.route('/')
def home():
    return "Hello world!"
   

@app.route('/predict',methods=['POST'])
def predict():
    classifier = keras.models.load_model(r"C:\Users\ishan\Desktop\SC")
    #print(parameters)
    keys = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    print(request.json , '\n \n \n \n \n')
    print(type(request.json))

#         inputFeature = np.asarray(parameters).reshape(1,-1)
    X_user = pd.DataFrame(
    request.json,index=[0])
#         my_prediction = clfr.predict(inputFeature)
    my_prediction = classifier.predict(X_user)
    prediction = float(my_prediction[0])*100
    print(prediction)
    return jsonify(prediction), 200

if __name__ == '__main__':
    app.run(debug=True)






