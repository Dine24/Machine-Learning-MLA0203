import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.io as io
import plotly.express as px

# Set the default renderer for Plotly to 'browser'
io.renderers.default = 'browser'

# Load the dataset
data = pd.read_csv("futuresale prediction.csv")

# Display the first 5 rows of the dataset
print(data.head())

# Display a random sample of 5 rows from the dataset
print(data.sample(5))

# Check for missing values in each column
print(data.isnull().sum())

# Visualize the relationship between Sales and TV advertising
figure_tv = px.scatter(data_frame=data, x="Sales", y="TV", size="TV", trendline="ols")
figure_tv.show()

# Visualize the relationship between Sales and Newspaper advertising
figure_newspaper = px.scatter(data_frame=data, x="Sales", y="Newspaper", size="Newspaper", trendline="ols")
figure_newspaper.show()

# Visualize the relationship between Sales and Radio advertising
figure_radio = px.scatter(data_frame=data, x="Sales", y="Radio", size="Radio", trendline="ols")
figure_radio.show()

# Calculate correlations and sort them in descending order
correlation = data.corr()
print(correlation["Sales"].sort_values(ascending=False))

# Prepare feature matrix and target vector
x = np.array(data.drop(["Sales"], axis=1))
y = np.array(data["Sales"])

# Split the dataset into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Create and train a Linear Regression model
model = LinearRegression()
model.fit(xtrain, ytrain)

# Evaluate the model on the test data
score = model.score(xtest, ytest)
print("R^2 Score:", score)

# Make a prediction using a sample feature vector
sample_features = np.array([[230.1, 37.8, 69.2]])  # Example values for TV, Radio, Newspaper
prediction = model.predict(sample_features)
print("Predicted Sales:", prediction)
