from code.tools import polynomial_preprocess_data, show_plot
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np
file_path = '../../data_sets/Position_Salaries.csv'
X, y = polynomial_preprocess_data(file_path, delimiter=',', fill=False, encoding=False)

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

y_pred = regressor.predict([[6.8]])
print(y_pred)

# Visualizing plot of decision tree
show_plot(x_axis=X,
          y_axis=y,
          x_label='Position Level',
          y_label='Salary',
          title='Salary based on Level(Decision Tree)',
          x_plot=X,
          y_plot=regressor.predict(X))

