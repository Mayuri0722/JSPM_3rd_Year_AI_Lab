# Expense Tracker AI Application
# This application allows users to track their expenses, categorize them using a machine learning model, and visualize spending patterns.
from flask import Flask, render_template, request
import pandas as pd
import pickle
from datetime import datetime
import plotly.express as px 
import os

app = Flask(__name__)

# Load ML model and vectorizer
with open('model/expense_classifier.pkl', 'rb') as f:
    model, vectorizer = pickle.load(f)

# File to store expenses
CSV_PATH = 'data/expenses.csv'
os.makedirs('data', exist_ok=True)

# Initialize CSV if not present
if not os.path.exists(CSV_PATH):
    pd.DataFrame(columns=["Date", "Description", "Amount", "Category"]).to_csv(CSV_PATH, index=False)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        desc = request.form['description']
        amount = float(request.form['amount'])
        date = datetime.now().strftime("%Y-%m-%d")

        category = model.predict(vectorizer.transform([desc]))[0]

        df = pd.read_csv(CSV_PATH)
        df.loc[len(df)] = [date, desc, amount, category]
        df.to_csv(CSV_PATH, index=False)

    df = pd.read_csv(CSV_PATH)

    # Generate Pie Chart
    if not df.empty:
        fig = px.pie(df, names='Category', values='Amount', title='Spending by Category')
        chart = fig.to_html(full_html=False)
    else:
        chart = "<p>No data available to show chart.</p>"

    return render_template('base.html', table=df.tail(5).to_html(classes='table table-striped'), chart=chart)

if __name__ == '__main__':
    app.run(debug=True)
