import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from code.tools import preprocess_data

X_train, X_test, y_train, y_test = preprocess_data('../../data_sets/Data.csv', encoding=False,
                                                   apply_feature_scaling=False)
regressor = LinearRegression()
print(X_train, y_train)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)

print(np.concatenate(
    (
        y_pred.reshape(len(y_pred), 1),
        y_test.reshape(len(y_test), 1)
    ), 1))
print(r2_score(y_test, y_pred))

