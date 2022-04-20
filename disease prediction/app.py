import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
model2=pickle.load(open('model2.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/diabetes')
def diabetes():
    return render_template('index2.html')

@app.route('/predict-heart',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]
    text=""
    if output==1:
        text="you have got a heart disease.You must consider going to a doctor"
    else:
        text="You do not have a heart disease"

    return render_template('index.html', prediction_text=text)

@app.route('/predict-diabetes',methods=['POST'])
def predict2():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model2.predict(final_features)

    output = prediction[0]
    text=""
    if output==1:
        text="you might have diabetes.You must consider going to a doctor"
    else:
        text="You do not have a diabetes"

    return render_template('index2.html', prediction_text=text)


if __name__ == "__main__":
    app.run(debug=True)


    # 63,1,3,145,233,1,0,150,0,2.3,0,1
    # 57	1	0	110	335	0	1	143	1	3.0	1	3	