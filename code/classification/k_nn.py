import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from code.tools import preprocess_data

X_train, X_test, y_train, y_test = preprocess_data('../../data_sets/breast_cancer.csv')

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# print(classifier.predict(scaler.transform([[30, 87000]])))

# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))/

matrix = confusion_matrix(y_test, y_pred)
accuracy_score = accuracy_score(y_test, y_pred)
print(matrix)
print(accuracy_score)

# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Accuracy: {:.2f} %".format(accuracies.mean() * 100))


# # # displaying plot for Train. Set
# X_set, y_set = scaler.inverse_transform(X_test), y_test
# X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.5),
#                      np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.5))
# plt.contourf(X1, X2, classifier.predict(scaler.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
#              alpha=0.75, cmap=ListedColormap(('salmon', 'dodgerblue')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# #
# #
# # displaying plot for Test Set
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('salmon', 'dodgerblue'))(i), label=j)
# plt.title('K Nearest Neighbors (Test set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()
