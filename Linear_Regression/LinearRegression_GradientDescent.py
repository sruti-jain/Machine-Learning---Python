import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading the database
data = pd.read_csv('ex1data1.txt', names=['population', 'profit'])

X_df = pd.DataFrame(data.population)
y_df = pd.DataFrame(data.profit)
m = len(y_df)

# Initializing some initial values
iterations = 1500
alpha = 0.01
X_df['intercept'] = 1
X = np.array(X_df)
y = np.array(y_df).flatten()
theta = np.array([0, 0])

# Computing the Cost function
def costFunction(X, y, theta):
    # cost_function(X, y, theta) computes the cost of using theta as the parameter for linear regression
    m = len(y)      # number of training examples

    J = np.sum((X.dot(theta) - y) ** 2) / 2 / m
    return (J)

# Gradient Descent Algorithm
def gradientDescent(X, y, theta, alpha, iterations):
    costHistory = [0] * iterations

    for iteration in range(iterations):
        hypothesis = X.dot(theta)
        loss = hypothesis - y
        gradient = X.T.dot(loss) / m
        theta = theta - alpha * gradient
        cost = costFunction(X, y, theta)
        costHistory[iteration] = cost
    return theta, costHistory

(t, c) = gradientDescent(X,y,theta,alpha, iterations)

bestFit_x = np.linspace(0, 25, 20)
bestFit_y = [t[1] + t[0]*xx for xx in bestFit_x]

# Plotting the dataset and the best-fit- regression line
plt.figure(figsize=(10,6))
plt.plot(X_df.population, y_df, '.')
plt.plot(bestFit_x, bestFit_y, '-')
plt.axis([0,25,-5,25])
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.title('Profit vs. Population with Linear Regression Line')
plt.show()