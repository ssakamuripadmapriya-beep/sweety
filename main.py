from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Read dataset
data = pd.read_csv("student_fees.csv")

# Inputs
X = data[['roll_no', 'previous_due', 'stu_year']]

# Output
y = data['fee_amount']

# Train model
model = LinearRegression()
model.fit(X, y)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction
@app.route('/predict', methods=['POST'])
def predict():

    roll_no = float(request.form['roll_no'])
    previous_due = float(request.form['previous_due'])
    stu_year= float(request.form['stu_year'])

    new_data = [[roll_no, previous_due, stu_year]]

    prediction = model.predict(new_data)

    predicted_fee = round(prediction[0], 2)

    return render_template('index.html', prediction=predicted_fee)

if __name__ == '__main__':
    app.run(debug=True)