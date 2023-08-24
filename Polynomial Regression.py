import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Create some sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5]).reshape(-1, 1)

# Create a linear regression object and fit the data
reg = LinearRegression().fit(X, y)

# Predict new values
X_new = np.array([6]).reshape(-1, 1)
y_pred = reg.predict(X_new)

# Plot the data and the linear regression line
plt.scatter(X, y)
plt.plot(X, reg.predict(X), color='red')
plt.show()

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Create some sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5]).reshape(-1, 1)

# Transform the data to include another axis
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Create a polynomial regression object and fit the data
reg = LinearRegression().fit(X_poly, y)

# Predict new values
X_new = np.array([6]).reshape(-1, 1)
X_new_poly = poly.transform(X_new)
y_pred = reg.predict(X_new_poly)

# Plot the data and the polynomial regression curve
plt.scatter(X, y)
plt.plot(X, reg.predict(X_poly), color='red')
plt.show()
