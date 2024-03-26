import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

from code.tools import show_plot, svr_preprocess_data, polynomial_preprocess_data

X_train, X_test, y_train, y_test = svr_preprocess_data('../../data_sets/Data.csv')
scaler_x = StandardScaler()
scaler_y = StandardScaler()
X = scaler_x.fit_transform(X_train)
y = scaler_y.fit_transform(y_train.reshape(len(y_train), 1))

regressor = SVR(kernel='rbf')
regressor.fit(X, y)

y_pred = scaler_y.inverse_transform(regressor.predict(scaler_x.transform(X_test)).reshape(-1, 1))
# print(np.concatenate(y_pred.reshape(len(y_pred), 1)), y_test.reshape(len(y_test), 1))
print(r2_score(y_test, y_pred))

# # Visualizing Support Vector regression plot
# show_plot(x_axis=scaler_x.inverse_transform(X),
#           y_axis=scaler_y.inverse_transform(y),
#           x_label='Position Level',
#           y_label='Salary',
#           title='Salary based on Level(SVR)',
#           x_plot=scaler_x.inverse_transform(X),
#           y_plot=scaler_y.inverse_transform(regressor.predict(X).reshape(-1, 1)))