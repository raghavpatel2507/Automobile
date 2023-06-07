import pickle
from flask import Flask,render_template,request,jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
# import ridge and scaler model
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

# home route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prdictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        cylinders=float(request.form.get('cylinders'))
        displacement=float(request.form.get('displacement'))
        horsepowers=float(request.form.get('horsepowers'))
        weight=float(request.form.get('weight'))
        acceleration=float(request.form.get('acceleration'))
        model_year=float(request.form.get('model_year'))
        origin1=float(request.form.get('origin1'))

        new_data_scaled=standard_scaler.transform([[cylinders,displacement,horsepowers,weight,acceleration,model_year,origin1]])
        result=ridge_model.predict(new_data_scaled)

        return render_template('home.html',result=result[0])
    else:
        return render_template('home.html')

    
    


if __name__=="__main__":
    app.run(host="0.0.0.0")
