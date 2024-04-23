from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        val1 = request.form['bedrooms']
        val2 = request.form['bathrooms']
        val3 = request.form['floors']
        
        arr = np.array([val1, val2, val3])
        arr = arr.astype(np.float64)
        
        pred = model.predict([arr])
        return render_template('index.html', data=int(pred))
    else:
        return "Please submit a form."

if __name__ == '__main__':
    app.run(debug=True)
