from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from code.utils import extract_feature_and_dependent_variable, fill_missing_data, hot_encoder, \
    split_test_and_train_vectors, apply_feature_scaling, svr_apply_feature_scaling


def preprocess_data(file_path, delimiter=';', fill=False, encoding=False, apply_feature_scaling=True):
    # extracting feature vector and independent variable vector
    feature_vector, independent_variable_vector = extract_feature_and_dependent_variable(file_path, delimiter=delimiter)

    if fill:
        # taking care of missing data
        feature_vector = fill_missing_data(feature_vector, 1, 3, strategy='mean')

    if encoding:
        # encoding country column data with hot encoder
        feature_vector = hot_encoder(feature_vector, column_index=3)

        # encoding independent variable with label encoder
        # independent_variable_vector = label_encoder(independent_variable_vector)

    # splitting dataset to test and train sets
    (feature_vector_train,
     feature_vector_test,
     independent_variable_vector_train,
     independent_variable_vector_test) = split_test_and_train_vectors(feature_vector,
                                                                      independent_variable_vector,
                                                                      test_size=0.2)

    if apply_feature_scaling:
        # applying feature scaling
        feature_vector_train, feature_vector_test = apply_feature_scaling(feature_vector_train, feature_vector_test,
                                                                          first_applicable_column=3)
    # print(feature_vector_train)
    # print(feature_vector_test)
    # print(independent_variable_vector_train)
    # print(independent_variable_vector_test)
    return feature_vector_train, feature_vector_test, independent_variable_vector_train, independent_variable_vector_test


def predict_with_linear_regression(feature_vector_train, dependent_variable_train, feature_vector_test):
    regressor = LinearRegression()
    regressor.fit(feature_vector_train, dependent_variable_train)
    dependent_variable_predict = regressor.predict(feature_vector_test)
    return dependent_variable_predict, regressor


def show_plot(x_axis, y_axis, title, x_label, y_label, x_plot, y_plot):
    plt.scatter(x_axis, y_axis, color='purple')
    plt.plot(x_plot, y_plot, color='orange')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def polynomial_preprocess_data(file_path, delimiter=',', fill=False, encoding=False):
    # extracting feature vector and independent variable vector
    feature_vector, independent_variable_vector = extract_feature_and_dependent_variable(file_path, delimiter=delimiter)

    if fill:
        # taking care of missing data
        feature_vector = fill_missing_data(feature_vector, 1, 3, strategy='mean')

    if encoding:
        # encoding country column data with hot encoder
        feature_vector = hot_encoder(feature_vector, column_index=3)

        # encoding independent variable with label encoder
        # independent_variable_vector = label_encoder(independent_variable_vector)

    return feature_vector, independent_variable_vector


def svr_preprocess_data(file_path, delimiter=',', fill=False, encoding=False, apply_feature_scaling=True):
    # extracting feature vector and independent variable vector
    feature_vector, independent_variable_vector = extract_feature_and_dependent_variable(file_path, delimiter=delimiter)

    if fill:
        # taking care of missing data
        feature_vector = fill_missing_data(feature_vector, 1, 3, strategy='mean')

    if encoding:
        # encoding country column data with hot encoder
        feature_vector = hot_encoder(feature_vector, column_index=3)

        # encoding independent variable with label encoder
        # independent_variable_vector = label_encoder(independent_variable_vector)

    return feature_vector, independent_variable_vector
