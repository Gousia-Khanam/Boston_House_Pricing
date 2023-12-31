'''
#command to get my ip address:'curl ifconfig.me'
#178.153.0.20

import json
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


app=Flask(__name__)    # name is the starting point of application where it will run
#load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))
Scalar=pickle.load(open('Scaling_LR.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data=request.json['data']  #get the data
    print(data)
    print(np.array(list(data.values()))).reshape(1,-1)
    new_data=Scalar.transform(np.array(list(data.values()))).reshape(1,-1)
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify( output[0])


#getting data from the fields of form 
@app.route('/predict',method=['POST'])

def predict():
    data=[float(x)for x in (request.form.values())]  #here we are taking for loop to get all values from the form and making it a float
    final_input=Scalar.transform(np.array(list(data.values()))).reshape(1,-1)
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The Predicted price of House is {}".format(output))


if __name__=='__main__':
    app.run(debug=True)
'''
import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('Scaling_LR.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['GET'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))



if __name__=="__main__":
    app.run(debug=True)
   
     