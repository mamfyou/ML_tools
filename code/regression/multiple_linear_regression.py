import numpy as np
from sklearn.linear_model import LinearRegression

from code.tools import preprocess_data

X_train, X_test, y_train, y_test = preprocess_data('../../data_sets/wine.csv', encoding=False,
                                                   apply_feature_scaling=False)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)

print(np.concatenate(
    (
        y_pred.reshape(len(y_pred), 1),
        y_test.reshape(len(y_test), 1)
    ), 1))
print(regressor.score(X_test, y_test))
print(regressor.intercept_, 'intercept')
print(regressor.coef_)
