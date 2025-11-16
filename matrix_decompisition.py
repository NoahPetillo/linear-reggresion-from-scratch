import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def matrix_decompisition(X_train, y_train):
    #Construct Matrices
    
    X = np.column_stack([np.ones_like(X_train), X_train])
    Y = y_train.to_numpy().reshape(-1, 1)

    #Calculate A = [b\\m]
    A = np.linalg.inv(X.T @ X) @ (X.T @ Y)  #Follows matrix decompisition formula of A = (A^T X)^-1 X^T Y
    
    # Find error matrix, E
    #First, create a nx1 matrix of the predicted value by our line Y = AX
    Y_predicted = X@A
    
    #Consturct E, a nx1 matrix of the squared errors of each point in our training data
    E = np.array([(Y[i] - Y_predicted[i])**2 for i in range(len(y_train))])    
    
    return Y,X,A,E

df = pd.read_csv("Car_weight.csv")
X_values = df["weight"]
y_values = df["mpg"]
X_train, X_test, y_train, y_test = train_test_split(X_values, y_values, random_state=42, test_size = 0.2)


Y, X, A, E = matrix_decompisition(X_train, y_train)

y_predicted = []
b,m = A[0], A[1]
#Create y_predicted using m and b derived from matrix decompisition
for i in range(len(X_test)):
    y_predicted.append((X_test.to_numpy()[i]*m) + b)
y_predicted = np.array(y_predicted)
print("Mine:")
print("MSE:", mean_squared_error(y_test, y_predicted))
print("R²:", r2_score(y_test, y_predicted))
print(" Intercept (b):", b)
print(" Slope (m):", m)

#Also, we know in E^T @ E = SSE, so let's check that:
print("SSE calculated from matrix E: ", E.T @ E)
print("Mean: ", (E.T @ E) / len(X_train))
# # Plot regression line

# # Scatter original data
# plt.scatter(X_values, y_values, label="Data")

# # Create line values
# x_line = np.linspace(X_values.min(), X_values.max(), 100)
# y_line = m * x_line + b

# # Plot regression line
# plt.plot(x_line, y_line, color ="orange", label="Regression Line")

# plt.xlabel("Weight")
# plt.ylabel("MPG")
# plt.title("Linear Regression Fit (from scratch)")
# plt.legend()
# plt.show()


#Compare to SkLearn
model = LinearRegression()
model.fit(X_train.values.reshape(-1,1), y_train)

sk_y_pred = model.predict(X_test.values.reshape(-1,1))

print("\n Sklearn Comparison")
print(" MSE:", mean_squared_error(y_test, sk_y_pred))
print(" R²:", r2_score(y_test, sk_y_pred))
print(" Intercept (b):", model.intercept_)
print(" Slope (m):", model.coef_[0])

