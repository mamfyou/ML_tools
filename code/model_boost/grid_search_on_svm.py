import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from code.tools import preprocess_data

X_train, X_test, y_train, y_test = preprocess_data('../../data_sets/breast_cancer.csv')

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = SVC(kernel='rbf', C=0.5, gamma=0.1)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# print(classifier.predict(scaler.transform([[30, 87000]])))

# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))/

matrix = confusion_matrix(y_test, y_pred)
accuracy_score = accuracy_score(y_test, y_pred)
print(matrix)
print(accuracy_score)

# grid search
parameters = [
    {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
    {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
]
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, cv=10, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_, 'best parameters')
print(grid_search.best_score_, 'best score')

# cross-val score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Accuracy: {:.2f} %".format(accuracies.mean() * 100))
