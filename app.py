from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)


with open('flight_ticket_model.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/', methods=['GET', 'POST'])
def index():

    prediction = None

    if request.method == 'POST':
        # Get input values from the form
        departure_time = float(request.form['departure_time'])
        flight_distance = float(request.form['flight_distance'])
        passenger_count = int(request.form['passenger_count'])

        # Make a prediction using the trained model
        input_data = np.array([[departure_time, flight_distance, passenger_count]])
        prediction = model.predict(input_data)[0]

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
