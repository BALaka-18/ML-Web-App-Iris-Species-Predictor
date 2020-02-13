from flask import Flask, render_template, request, json, url_for
import numpy as np
import pandas as pd
import joblib as jb


app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/preview')
def preview():
    data = pd.read_csv('D&M/iris.csv')
    return render_template("preview.html", data_view = data)

@app.route('/', methods=['POST'])
def analyze():
    if request.method == 'POST':
        sepal_length = request.form['sepal_length']
        sepal_width = request.form['sepal_width']
        petal_length = request.form['petal_length']
        petal_width = request.form['petal_width']
        model_choice = request.form['model_choice']

        # Cleaning the data from unicode to float
        sample_data = [sepal_length,sepal_width,petal_length,petal_width]
        cleaned = [float(i) for i in sample_data]

        # Reshape the data
        data = np.array(cleaned).reshape(1,-1)

        # Load the model
        if model_choice == 'lgrmod':
            lr_model = jb.load('D&M/log_reg.pkl')
            pred = lr_model.predict(data)
        elif model_choice == 'knnmod':
            knn_model = jb.load('D&M/knn.pkl')
            pred = knn_model.predict(data)

    return render_template("index.html", sepal_length = sepal_length,sepal_width = sepal_width,petal_length = petal_length,petal_width = petal_width,data = data,pred = pred[0],model_selected = model_choice)



if __name__ == "__main__":
    app.run(debug=True)





