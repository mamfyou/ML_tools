from code.tools import preprocess_data, predict_with_linear_regression, show_plot

X_train, X_test, y_train, y_test = preprocess_data('../../data_sets/Purchase.csv')

y_predict, regressor = predict_with_linear_regression(X_train, y_train, X_test)

# training sets plot
show_plot(X_train,
          y_train,
          title='Salary based on Years of Experience (Training Sets)',
          x_label='Years of Experience',
          y_label='Salary',
          x_plot=X_train,
          y_plot=regressor.predict(X_train))

# test sets plot
show_plot(X_test,
          y_test,
          title='Salary based on Years of Experience (Test Sets)',
          x_label='Years of Experience',
          y_label='Salary',
          x_plot=X_test,
          y_plot=y_predict)

print(regressor.predict([[12]]))

# y = coef + YearsOfExperience * intercept_
print(regressor.coef_)
print(regressor.intercept_)
