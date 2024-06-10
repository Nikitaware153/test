from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle



with open('test\lr_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
 
    hours_studied = float(request.form['hours_studied'])
    previous_scores = float(request.form['previous_scores'])
    extracurricular_activities = float(request.form['extracurricular_activities'])
    sleep_hours = float(request.form['sleep_hours'])
    sample_question_papers_practiced = float(request.form['sample_question_papers_practiced'])

    
    prediction = model.predict([[hours_studied, previous_scores, extracurricular_activities, sleep_hours, sample_question_papers_practiced]])

    
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True, port=8080)






