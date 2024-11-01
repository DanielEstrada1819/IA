import pandas as pd

# Define el nombre del archivo
file_name = 'bezdekIris.data'

# Carga el dataset en un DataFrame
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_df = pd.read_csv(file_name, header=None, names=column_names)

# Muestra algunas filas de cada clase
print(iris_df.groupby('class').head(2))
