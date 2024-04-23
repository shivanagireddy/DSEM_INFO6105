import pandas as pd
from flask import Flask,render_template,request
import pickle
import numpy as np

app= Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl",'rb'))

@app.route('/')
def index():

    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    predicted_price = 100  # Example predicted price
    
    location = str(request.form.get('location'))
    bhk = str(request.form.get('bhk'))
    bath = str(request.form.get('bath'))
    sqft = str(request.form.get('total_sqft'))
    print(location, bhk, bath, sqft)

    """location ='1st Block Jayanagar'
    bhk = '2'
    bath = '2'
    sqft = '2000"""
    input = pd.DataFrame([[location,sqft,bath,bhk]],columns=['location', 'total_sqft', 'bath', 'bhk'])
    print(input)
    prediction = pipe.predict(input)[0] * 1e4
    return str(np.round(prediction,2))
    

if __name__== "__main__":
    app.run(debug=True, port=5001)
