import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

from code.tools import polynomial_preprocess_data, show_plot

X_train, X_test, y_train, y_test = polynomial_preprocess_data('../../data_sets/Data.csv')

np.set_printoptions(precision=2)

# polynomial regression
poly_features = PolynomialFeatures(degree=5)
X_poly = poly_features.fit_transform(X_train)

poly_linear_regressor = LinearRegression()
poly_linear_regressor.fit(X_poly, y_train)

# # Visualizing Polynomial regression plot
# show_plot(x_axis=X_test,
#           y_axis=y_test,
#           x_label='Position Level',
#           y_label='Salary',
#           title='Salary based on Level(Polynomial)',
#           x_plot=X_test,
#           y_plot=poly_linear_regressor.predict(X_poly))

y_pred = poly_linear_regressor.predict(poly_features.fit_transform(X_test))

print(r2_score(y_test, y_pred))
