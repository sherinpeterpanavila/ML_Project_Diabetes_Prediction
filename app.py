#import relevant libraries for flask, html rendering and loading the ML model
from flask import Flask,request, url_for, redirect, render_template
import pickle
import pandas as pd
import joblib

app = Flask(__name__)

#model = pickle.load(open('model.pkl',"rb"))
#scale = pickle.load(open('scale.pkl',"rb"))

model = joblib.load(open('model.pkl',"rb"))
scale = joblib.load(open('scale.pkl',"rb"))
@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predict",methods=['POST'])
def predict():

    pregnancies = request.form['1']
    glucose = request.form['2']
    bloodPressure = request.form['3']
    skinThickness = request.form['4']
    insulin = request.form['5']
    bmi = request.form['6']
    dpf = request.form['7']
    age = request.form['8']

    rowPdf = pd.DataFrame([pd.Series([pregnancies,glucose,bloodPressure,skinThickness,insulin, bmi, dpf,age])])
    rowPdf_new = pd.DataFrame(scale.transform(rowPdf))

    print(rowPdf_new)

    #Model prediction

    prediction = model.predict_proba(rowPdf_new)
    
    print("Predicted value is",prediction)
    if(prediction[0][1]  > 0.5):        
        valPred = round(prediction[0][1],3)
        return render_template('result.html',pred=f'You have a chance of having diabetes.\n\n Probability of you being a diabetic is {valPred*100:.2f}%.\n\nAdvice : Exercise Regularly')
    else:
        valPred = round(prediction[0][0],3)
        return render_template('result.html',pred=f'Congratulations!!!, You are in a Safe Zone.\n\n Probability of you being a non-diabetic is {valPred*100:.2f}%.\n\n Advice : Exercise Regularly and maintain like this..!')

if __name__ == '__main__':
    app.run(debug=True)