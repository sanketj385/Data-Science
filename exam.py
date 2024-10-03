import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Generating random data for demonstration
np.random.seed(42)
num_samples = 100
departure_times = np.random.uniform(0, 24, num_samples)
flight_distances = np.random.uniform(500, 5000, num_samples)
passenger_counts = np.random.randint(50, 300, num_samples)
ticket_prices = 100 + 10 * departure_times + 0.2 * flight_distances + 2 + np.random.normal(0, 100, num_samples)

# Creating a DataFrame with the generated data
data = pd.DataFrame({
    'Departure Time': departure_times,
    'Flight Distance': flight_distances,
    'Passenger Count': passenger_counts,
    'Ticket Price': ticket_prices
})

# Splitting the data into features (X) and target variable (y)
X = data[['Departure Time', 'Flight Distance', 'Passenger Count']]
y = data['Ticket Price']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluating the model on the test set
accuracy = model.score(X_test, y_test)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Saving the trained model to a pickle file
with open('flight_ticket_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print('Model saved as flight_ticket_model.pkl')


