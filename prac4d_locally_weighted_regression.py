import numpy as np 
import matplotlib.pyplot as plt 

# Generate random dataset 
np.random.seed(0) 
X = np.sort(5 * np.random.rand(80, 1), axis=0) 
y = np.sin(X).ravel() 
y[::5] += 3 * (0.5 - np.random.rand(16)) 

# Locally Weighted Regression function 
def locally_weighted_regression(query_point, X, y, tau=0.1): 
    m = X.shape[0] 
    weights = np.exp(-((X - query_point) * 2).sum(axis=1) / (2 * tau * 2)) 
    W = np.diag(weights) 
    X_bias = np.c_[np.ones((m, 1)), X] 
    theta = np.linalg.inv(X_bias.T.dot(W).dot(X_bias)).dot(X_bias.T).dot(W).dot(y) 
    x_query = np.array([1, query_point]) 
    prediction = x_query.dot(theta) 
    return prediction 

# Generate test points 
X_test = np.linspace(0, 5, 100) 

# Predict using locally weighted regression 
predictions = [locally_weighted_regression(query_point, X, y, tau=0.1) for query_point in X_test] 

# Plot results 
plt.scatter(X, y, color='black', s=30, marker='o', label='Data Points') 
plt.plot(X_test, predictions, color='blue', linewidth=2, label='LWR Fit') 
plt.xlabel('X') 
plt.ylabel('y') 
plt.title('Locally Weighted Regression') 
plt.legend() 
plt.show() 

