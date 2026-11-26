import numpy as np
import pickle
from pathlib import Path
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(BASE_DIR / 'template'))
with (BASE_DIR / 'titanic_model.pkl').open('rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    pclass = int(request.form['pclass'])
    sex = request.form['sex']
    age = float(request.form['age'])
    sibsp = int(request.form['sibsp'])
    parch = int(request.form['parch'])
    fare = float(request.form['fare'])

    final_feature = [pclass, sex, age, sibsp, parch, fare]
    prediction = model.predict([final_feature])
    return render_template('index.html', result='Survived' if prediction[0] == 1 else 'Did not survive')
                                      
if __name__ == '__main__':
    app.run(debug=True)








if __name__ == '__main__':
    app.run(debug=True)
