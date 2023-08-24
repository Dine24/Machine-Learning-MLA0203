import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'PlotArea': [6900000, 3500000, 5700000, 22500000, 46000000],
    'Year': [2010, 2016, 2012, 2018, 2015],
    'Exterior': ['vinylSd', 'Metalsd', 'HdBoard', 'Metalsd', 'HdBoard'],
    'Price': [6000, 12000, 8000, 15000, 10000]
}

df = pd.DataFrame(data)
df = pd.get_dummies(df, columns=['Exterior'])
X = df.drop('Price', axis=1)
y = df['Price']
print(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\n")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

new_house_features = {
    'PlotArea': [40000],
    'Year': [2017],
    'Exterior_HdBoard': [0],
    'Exterior_Metalsd': [1],
    'Exterior_vinylSd': [0]
}

new_house_df = pd.DataFrame(new_house_features)
predicted_price = model.predict(new_house_df)

print(f"Predicted Price for the new House: {predicted_price[0]}")

