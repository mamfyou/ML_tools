from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

from code.tools import show_plot, svr_preprocess_data, polynomial_preprocess_data

X, y = svr_preprocess_data('../../data_sets/Position_Salaries.csv')
scaler_x = StandardScaler()
scaler_y = StandardScaler()
X = scaler_x.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(len(y), 1))

regressor = SVR(kernel='rbf')
regressor.fit(X, y)

y_pred = scaler_y.inverse_transform(regressor.predict(scaler_x.transform([[6.5]])).reshape(-1, 1))
print(y_pred)

# Visualizing Support Vector regression plot
show_plot(x_axis=scaler_x.inverse_transform(X),
          y_axis=scaler_y.inverse_transform(y),
          x_label='Position Level',
          y_label='Salary',
          title='Salary based on Level(SVR)',
          x_plot=scaler_x.inverse_transform(X),
          y_plot=scaler_y.inverse_transform(regressor.predict(X).reshape(-1, 1)))