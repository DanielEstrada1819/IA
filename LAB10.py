from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# Carga de datasets
datasets = {
    "Iris": load_iris(),
    "Breast Cancer": load_breast_cancer(),
    "Wine": load_wine()
}

# Función para mostrar los resultados
def evaluar_modelo(clf, X, y, metodo, X_train=None, X_test=None, y_train=None, y_test=None):
    print(f"\n{metodo} - {clf.__class__.__name__}")
    
    if metodo == "Hold-Out (70/30)":
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
    
    elif metodo == "10-Fold Cross-Validation":
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=kf, scoring='accuracy')
        print("Accuracy promedio:", np.mean(scores))
    
    elif metodo == "Leave-One-Out":
        loo = LeaveOneOut()
        scores = cross_val_score(clf, X, y, cv=loo, scoring='accuracy')
        print("Accuracy promedio:", np.mean(scores))

# Loop principal para probar los datasets
for dataset_name, dataset in datasets.items():
    print(f"\n--- Dataset: {dataset_name} ---")
    X, y = dataset.data, dataset.target

    # Dividir el dataset para Hold-Out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Clasificadores
    naive_bayes = GaussianNB()
    print("\nProbando Naive Bayes...")
    evaluar_modelo(naive_bayes, X, y, "Hold-Out (70/30)", X_train, X_test, y_train, y_test)
    evaluar_modelo(naive_bayes, X, y, "10-Fold Cross-Validation")
    evaluar_modelo(naive_bayes, X, y, "Leave-One-Out")

    print("\nProbando KNN con diferentes valores de k...")
    for k in [1, 3, 5, 7, 10]:
        knn = KNeighborsClassifier(n_neighbors=k)
        print(f"\nResultados para k={k}:")
        evaluar_modelo(knn, X, y, "Hold-Out (70/30)", X_train, X_test, y_train, y_test)
        evaluar_modelo(knn, X, y, "10-Fold Cross-Validation")
        evaluar_modelo(knn, X, y, "Leave-One-Out")
