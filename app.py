from flask import Flask, render_template, flash, redirect, url_for, session, request
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from keras.models import model_from_json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg'
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg'
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)


# Load JSON and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")
print("Loaded model from disk")

model.compile(loss='mean_squared_error', optimizer='adam')

# Load data
df = pd.read_pickle('./dfe.pkl')

@app.route('/home')
def home():
    return render_template('home.html')



@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            flash('Login successful!', 'success')
            session['user_id'] = user.id
            return redirect(url_for('index'))
        else:
            flash('Login failed. Check your email and password.', 'danger')
    return render_template('login.html')

@app.route('/register/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password == confirm_password:
            hashed_password = generate_password_hash(password)
            new_user = User(username=username, email=email, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Passwords do not match.', 'danger')

    return render_template('register.html')

@app.route('/logout/')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('home'))


@app.route("/index",methods=['GET', 'POST'])
def index():
    print(request.method)
    if request.method == 'POST':
        if request.form.get('Continue') == 'Continue':
           return render_template("test1.html")
    else:
        # pass # unknown
        return render_template("index.html")

import plotly.graph_objs as go

@app.route('/predict_bitcoin', methods=['POST', 'GET'])
def predict_bitcoin():
    if request.method == 'POST':
        date = request.form['date']
        n = int(request.form['n'])

        loc = df.index.get_loc(date)
        prev_data = df.iloc[loc - 15: loc].Price.astype(float)

        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        ds = min_max_scaler.fit_transform(prev_data.values.reshape(-1, 1))
        ds = ds.reshape(1, 15, 1)

        look_back = 15
        x_input = ds[len(ds) - look_back:].reshape(1, -1)

        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()

        lst_output = []
        i = 0

        while i < n:
            if len(temp_input) > look_back:
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, look_back, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                lst_output.extend(yhat.tolist())
                i = i + 1
            else:
                x_input = x_input.reshape((1, look_back, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i = i + 1

        res = min_max_scaler.inverse_transform(lst_output)

        # Create a Plotly graph
        historical_data = go.Scatter(x=np.arange(1, len(prev_data) + 1), y=prev_data,
                                     mode='lines+markers', name='Historical Data')
        predicted_data = go.Scatter(x=np.arange(len(prev_data) + 1, len(prev_data) + n + 1),
                                    y=res.flatten(), mode='lines+markers', name='Predicted Data')

        layout = go.Layout(title='Historical and Predicted Bitcoin Prices',
                           xaxis=dict(title='Days'),
                           yaxis=dict(title='Price'),
                           showlegend=True)

        graph_data = [historical_data, predicted_data]
        graph_layout = layout
        fig = go.Figure(data=graph_data, layout=graph_layout)

        # Save the plot to a file
        image_path = os.path.join(app.static_folder, 'predicted_line_graph.html')
        fig.write_html(image_path)

        return render_template('bitcoin_result.html',
                               date=date,
                               n=n,
                               prices=res.flatten().tolist(),
                               image='predicted_line_graph.html'
                               )



# Load JSON and create model  Etherum 


json_file = open('etha.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("etha.h5")
print("Loaded model from disk")

model.compile(loss='mean_squared_error', optimizer='adam')

# Load data
df = pd.read_pickle('./eth.pkl')


@app.route('/base')
def base():
    return render_template('base.html')


@app.route('/about')
def about():
    
    return render_template('about.html')


@app.route('/predicts_ethereum', methods=['POST', 'GET'])
def predicts_ethereum():
    if request.method == 'POST':
        date = request.form['date']
        n = int(request.form['n'])

        loc = df.index.get_loc(date)
        prev_data = df.iloc[loc - 15: loc].Price.astype(float)

        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        ds = min_max_scaler.fit_transform(prev_data.values.reshape(-1, 1))
        ds = ds.reshape(1, 15, 1)

        look_back = 15
        x_input = ds[len(ds) - look_back:].reshape(1, -1)

        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()

        lst_output = []
        i = 0

        while i < n:
            if len(temp_input) > look_back:
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, look_back, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                lst_output.extend(yhat.tolist())
                i = i + 1
            else:
                x_input = x_input.reshape((1, look_back, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i = i + 1

        res = min_max_scaler.inverse_transform(lst_output)

        # Create a Plotly graph
        historical_data = go.Scatter(x=np.arange(1, len(prev_data) + 1), y=prev_data,
                                     mode='lines+markers', name='Historical Data')
        predicted_data = go.Scatter(x=np.arange(len(prev_data) + 1, len(prev_data) + n + 1),
                                    y=res.flatten(), mode='lines+markers', name='Predicted Data')

        layout = go.Layout(title='Historical and Predicted Ethereum Prices',
                           xaxis=dict(title='Days'),
                           yaxis=dict(title='Price'),
                           showlegend=True)

        graph_data = [historical_data, predicted_data]
        graph_layout = layout
        fig = go.Figure(data=graph_data, layout=graph_layout)

        # Save the plot to a file
        image_path = os.path.join(app.static_folder, 'predicted_ethereum_graph.html')
        fig.write_html(image_path)

        return render_template('ethereum_output.html',
                               date=date,
                               n=n,
                               prices=res.flatten().tolist(),
                               image='predicted_ethereum_graph.html'
                               )

    return render_template('output.html')




#dogecoin predictions




json_file = open('dogecoin.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("dogecoin.h5")
print("Loaded model from disk")

model.compile(loss='mean_squared_error', optimizer='adam')

# Load data
df = pd.read_pickle('./coin.pkl')


@app.route('/dogecoin')
def dogecoin():
    return render_template('dogecoin.html')




@app.route('/predicts_dogecoin', methods=['POST', 'GET'])
def predicts_dogecoin():
    if request.method == 'POST':
        date = request.form['date']
        n = int(request.form['n'])

        loc = df.index.get_loc(date)
        prev_data = df.iloc[loc - 15: loc].Price.astype(float)

        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        ds = min_max_scaler.fit_transform(prev_data.values.reshape(-1, 1))
        ds = ds.reshape(1, 15, 1)

        look_back = 15
        x_input = ds[len(ds) - look_back:].reshape(1, -1)

        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()

        lst_output = []
        i = 0

        while i < n:
            if len(temp_input) > look_back:
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, look_back, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                lst_output.extend(yhat.tolist())
                i = i + 1
            else:
                x_input = x_input.reshape((1, look_back, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i = i + 1

        res = min_max_scaler.inverse_transform(lst_output)

        # Create a Plotly graph
        historical_data = go.Scatter(x=np.arange(1, len(prev_data) + 1), y=prev_data,
                                     mode='lines+markers', name='Historical Data')
        predicted_data = go.Scatter(x=np.arange(len(prev_data) + 1, len(prev_data) + n + 1),
                                    y=res.flatten(), mode='lines+markers', name='Predicted Data')

        layout = go.Layout(title='Historical and Predicted Dogecoin Prices',
                           xaxis=dict(title='Days'),
                           yaxis=dict(title='Price'),
                           showlegend=True)

        graph_data = [historical_data, predicted_data]
        graph_layout = layout
        fig = go.Figure(data=graph_data, layout=graph_layout)

        # Save the plot to a file
        image_path = os.path.join(app.static_folder, 'predicted_dogecoin_graph.html')
        fig.write_html(image_path)

        return render_template('dogecoin_output.html',
                               date=date,
                               n=n,
                               prices=res.flatten().tolist(),
                               image='predicted_dogecoin_graph.html'
                               )

    return render_template('output.html')





if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
