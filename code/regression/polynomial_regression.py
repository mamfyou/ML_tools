import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from code.tools import polynomial_preprocess_data, show_plot

X, y = polynomial_preprocess_data('../../data_sets/Position_Salaries.csv')

np.set_printoptions(precision=2)

# linear regression
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

# polynomial regression
poly_features = PolynomialFeatures(degree=5)
X_poly = poly_features.fit_transform(X)

poly_linear_regressor = LinearRegression()
poly_linear_regressor.fit(X_poly, y)

# Visualizing linear regression plot
show_plot(x_axis=X,
          y_axis=y,
          x_label='Position Level',
          y_label='Salary',
          title='Salary based on Level(Linear)',
          x_plot=X,
          y_plot=linear_regressor.predict(X))

# Visualizing Polynomial regression plot
show_plot(x_axis=X,
          y_axis=y,
          x_label='Position Level',
          y_label='Salary',
          title='Salary based on Level(Polynomial)',
          x_plot=X,
          y_plot=poly_linear_regressor.predict(X_poly))

y_pred = linear_regressor.predict([[6.5]])

print(y_pred, 'vs', poly_linear_regressor.predict(poly_features.fit_transform([[6.5]])))
