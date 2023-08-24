import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate sample data
np.random.seed(0)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X + 1 + np.random.randn(100, 1)

# Create linear regression object
lr_model = LinearRegression()

# Train the model using the training sets
lr_model.fit(X, y)

# Print the coefficients
print('Coefficients: ', lr_model.coef_)
print('Intercept: ', lr_model.intercept_)

# Plot the data and the linear regression line
plt.scatter(X, y, color='blue')
plt.plot(X, lr_model.predict(X), color='red', linewidth=3)
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
