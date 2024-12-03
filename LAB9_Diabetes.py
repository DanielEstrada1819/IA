import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler

# Clasificador Euclidiano
class EuclideanClassifier:
    def __init__(self):
        self.centroids = None

    def fit(self, X, y):
        # Calcular el centroide de cada clase
        self.centroids = {label: X[y == label].mean(axis=0) for label in np.unique(y)}

    def predict(self, X):
        predictions = []
        for sample in X:
            distances = {label: distance.euclidean(sample, centroid) for label, centroid in self.centroids.items()}
            predictions.append(min(distances, key=distances.get))
        return np.array(predictions)


# Validación Hold-Out 70/30
def hold_out_validation(X, y, classifier):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, cm


# Validación 10-Fold Cross-Validation
def k_fold_validation(X, y, classifier, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))
    return np.mean(scores)


# Validación Leave-One-Out
def leave_one_out_validation(X, y, classifier):
    loo = LeaveOneOut()
    scores = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))
    return np.mean(scores)


# Cargar dataset Pima Indians Diabetes desde archivo CSV
def load_pima_dataset():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
    data = pd.read_csv(url, header=None, names=column_names)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y


# Preprocesamiento (escalado de datos)
def preprocess_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)


# Main para ejecutar con el dataset Pima Indians Diabetes
X, y = load_pima_dataset()
X = preprocess_data(X)

# Clasificador Euclidiano
euclidean_classifier = EuclideanClassifier()
acc_hold_out, cm_hold_out = hold_out_validation(X, y, euclidean_classifier)
acc_k_fold = k_fold_validation(X, y, euclidean_classifier)
acc_loo = leave_one_out_validation(X, y, euclidean_classifier)

# Clasificador 1NN
knn_classifier = KNeighborsClassifier(n_neighbors=1)
acc_hold_out_knn, cm_hold_out_knn = hold_out_validation(X, y, knn_classifier)
acc_k_fold_knn = k_fold_validation(X, y, knn_classifier)
acc_loo_knn = leave_one_out_validation(X, y, knn_classifier)

# Resultados
print("Resultados del Clasificador Euclidiano (Pima Indians Diabetes):")
print("Hold-Out Accuracy:", acc_hold_out)
print("Hold-Out Matriz de Confusion:\n", cm_hold_out)
print("10-Fold Cross-Validation Accuracy:", acc_k_fold)
print("Leave-One-Out Accuracy:", acc_loo)

print("\nResultados del Clasificador 1NN (Pima Indians Diabetes):")
print("Hold-Out Accuracy:", acc_hold_out_knn)
print("Hold-Out Matriz de Confusion:\n", cm_hold_out_knn)
print("10-Fold Cross-Validation Accuracy:", acc_k_fold_knn)
print("Leave-One-Out Accuracy:", acc_loo_knn)
