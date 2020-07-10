import numpy as np
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    if request.method == 'POST':
        area = float(request.form["area"])
        room = float(request.form["room"])
        age = float(request.form["age"])
        #print(area, room, age)
        
        pred = model.predict([[area, room, age]])
    #features = str(features)
    #last = model.predict([[features]])
    #print(features2)

    return render_template("index.html", prediction_text = "price is{}".format(pred))

if __name__ == '__main__':
    app.run(debug=True)
    