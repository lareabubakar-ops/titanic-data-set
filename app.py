
import numpy as np
import pickle
import csv
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__, template_folder=str(BASE_DIR / 'template'))

# Load the trained model
with (BASE_DIR / 'titanic_model.pkl').open('rb') as model_file:
    model = pickle.load(model_file)


@app.route('/')
def home():
    with (BASE_DIR / 'Titanic-Dataset.csv').open(newline='', encoding='utf-8') as dataset_file:
        passengers = list(csv.DictReader(dataset_file))

    def survival_rate(rows):
        return round(100 * sum(int(row['Survived']) for row in rows) / len(rows), 1) if rows else 0

    def group_stats(key, values):
        groups = [[row for row in passengers if row.get(key) == value] for value, _ in values]
        return {
            'labels': [label for _, label in values],
            'rates': [survival_rate(group) for group in groups],
            'counts': [len(group) for group in groups],
        }

    age_bands = [('0-17', 'Child (0–17)', lambda age: age < 18),
                 ('18-34', '18–34', lambda age: 18 <= age < 35),
                 ('35-54', '35–54', lambda age: 35 <= age < 55),
                 ('55+', '55+', lambda age: age >= 55)]
    ages = [row for row in passengers if row['Age']]
    age_stats = {'labels': [], 'rates': [], 'counts': []}
    for _, label, condition in age_bands:
        group = [row for row in ages if condition(float(row['Age']))]
        age_stats['labels'].append(label)
        age_stats['rates'].append(survival_rate(group))
        age_stats['counts'].append(len(group))

    total = len(passengers)
    survived = sum(int(row['Survived']) for row in passengers)
    stats = {
        'total': total,
        'survived': survived,
        'survival_rate': round(100 * survived / total, 1),
        'average_age': round(sum(float(row['Age']) for row in ages) / len(ages), 1),
        'class': group_stats('Pclass', [('1', 'First'), ('2', 'Second'), ('3', 'Third')]),
        'sex': group_stats('Sex', [('female', 'Women'), ('male', 'Men')]),
        'embarked': group_stats('Embarked', [('S', 'Southampton'), ('C', 'Cherbourg'), ('Q', 'Queenstown')]),
        'ages': age_stats,
    }
    return render_template('index.html', stats=stats, result=request.args.get('result'))

 
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    pclass = int(request.form['pclass'])
    sex = request.form['sex']
    if sex == 'male':
        sex = 0
    else:
        sex = 1
    age = float(request.form['age'])

    sibsp = int(request.form['sibsp'])
    parch = int(request.form['parch'])
    fare = float(request.form['fare'])
    embarked = request.form['embarked']
    if embarked == 'C':
        embarked = 0
    elif embarked == 'Q':
        embarked = 1
    else:
        embarked = 2

    final_feature = np.array([pclass, sex, age, sibsp, parch, fare, embarked])

    prediction = model.predict([final_feature])

    result = 'Survived' if prediction[0] == 1 else 'Did not survive'

    return redirect(url_for('home', result=result))


if __name__ == '__main__':
    app.run(debug=True)
