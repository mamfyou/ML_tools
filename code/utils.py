import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


def extract_feature_and_dependent_variable(file_path, dataset=None, delimiter=','):
    if dataset is not None:
        feature_vector = dataset.iloc[:, :-1].values
        dependent_variable_vector = dataset.iloc[:, -1].values
        return feature_vector, dependent_variable_vector
    dataset = pd.read_csv(file_path, delimiter=delimiter)
    feature_vector = dataset.iloc[:, :-1].values
    dependent_variable_vector = dataset.iloc[:, -1].values
    return feature_vector, dependent_variable_vector


def fill_missing_data(vector, start_index_column, end_index_column, strategy='mean'):
    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    imputer.fit(vector[:, start_index_column:end_index_column])
    vector[:, start_index_column:end_index_column] = imputer.transform(vector[:, start_index_column:end_index_column])
    return vector


def hot_encoder(vector, column_index=None, remainder='passthrough'):
    if column_index is not None:
        print(vector)
        vector = vector.copy()
        transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [column_index])],
                                        remainder=remainder)
        vector = np.array(transformer.fit_transform(vector))
        return vector


def label_encoder(vector, column_index=None, remainder='passthrough'):
    if not column_index:
        encoder = LabelEncoder()
        vector = np.array(encoder.fit_transform(vector))
        return vector


def split_test_and_train_vectors(X_vector, y_vector, test_size, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(X_vector,
                                                        y_vector,
                                                        test_size=test_size,
                                                        random_state=random_state)
    return X_train, X_test, y_train, y_test


def apply_feature_scaling(X_train, X_test, first_applicable_column, start_applicable_column=None,
                          end_applicable_column=None):
    scaler = StandardScaler()
    if first_applicable_column is not None:
        X_train[:, first_applicable_column:] = scaler.fit_transform(X_train[:, first_applicable_column:])
        X_test[:, first_applicable_column:] = scaler.transform(X_test[:, first_applicable_column:])
        return X_train, X_test
    X_train[:, first_applicable_column:] = scaler.fit_transform(X_train[:, first_applicable_column:])
    X_test[:, first_applicable_column:] = scaler.transform(X_test[:, first_applicable_column:])
    return X_train, X_test


def svr_apply_feature_scaling(X, first_applicable_column):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X
