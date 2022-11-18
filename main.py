import plistlib
from pyexpat import features
from flask import Flask, render_template, request,jsonify
import numpy as np
import pandas as pd
import pickle
import os


app=Flask(__name__)
pipe = pickle.load(open("XGmodel.pkl", "rb"))
picFolder = os.path.join('static','pics')

app.config['UPLOAD_FOLDER'] = picFolder


@app.route('/')
def index():
    pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'pile.JPG')
    return render_template('index.html',user_image = pic1)

@app.route('/predict', methods=['POST'])
def predict():
    Dia = int(request.form.get('Dia'))
    x1 = float(request.form.get('1layer'))
    x2 = float(request.form.get('2layer'))
    x3 = float(request.form.get('3layer'))
    Piletopelv = float(request.form.get('Pile top elev'))
    Groundelv = float(request.form.get('Ground Elevation'))
    xPiletop = float(request.form.get('Extra Pile Top Elevation'))
    piletip = float(request.form.get('Pile Tip Elevation'))
    sptshaft = float(request.form.get('SPT Blow Count at Pile Shaft'))
    spttip= float(request.form.get('SPT Blow Count at Pile tip'))

    #print(Dia,x1,x2,x3,Piletopelv,Groundelv,xPiletop,piletip,sptshaft,spttip)
    input = pd.DataFrame([[Dia,x1,x2,x3,Piletopelv,Groundelv,xPiletop,piletip,sptshaft,spttip]],columns=['D','X1','X2','X3','Xp','Xg','Xt','Xm','Ns','Nt'])
    prediction = pipe.predict(input)[0]


    return str(np.round(prediction,2))

if __name__=="__main__":
    app.run(debug=True, port=5005)