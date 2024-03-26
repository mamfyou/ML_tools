from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from code.tools import polynomial_preprocess_data, show_plot

file_path = '../../data_sets/Data.csv'
X_train, X_test, y_train, y_test = polynomial_preprocess_data(file_path, delimiter=',', fill=False, encoding=False)

regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print(r2_score(y_test, y_pred))

# # Visualizing plot of decision tree
# show_plot(x_axis=X,
#           y_axis=y,
#           x_label='Position Level',
#           y_label='Salary',
#           title='Salary based on Level(Decision Tree)',
#           x_plot=X,
#           y_plot=regressor.predict(X))
