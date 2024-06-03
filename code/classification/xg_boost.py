from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from code.tools import preprocess_data

X_train, X_test, y_train, y_test = preprocess_data('../../data_sets/breast_cancer.csv', encoding=True)
print(X_train)
print(y_train)

classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# y_pred = classifier.predict(X_test)

# cross-val score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print(accuracies.std())
print(accuracies.mean())
print("Accuracy: {:.2f} %".format(accuracies.mean() * 100))
