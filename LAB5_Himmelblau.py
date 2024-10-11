import numpy as np

def himmelblau(x, y):
    """
    Calcula el valor de la función de Himmelblau para un par de coordenadas (x, y).
    
    Parámetros:
    - x: Coordenada en el eje x.
    - y: Coordenada en el eje y.
    
    Retorna:
    - El valor de la función de Himmelblau en el punto (x, y).
    """
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def find_minima_random_search(iterations=10000):
    """
    Encuentra los mínimos de la función de Himmelblau mediante una búsqueda aleatoria.
    
    Parámetros:
    - iterations: Número de iteraciones o puntos aleatorios a evaluar (por defecto 10,000).
    
    Retorna:
    - Lista con las 4 mejores soluciones encontradas, ordenadas según el valor de f(x, y).
    """
    # Lista para almacenar los mejores puntos (x, y) junto con sus valores f(x, y)
    best_points = []
    
    # Generación y evaluación de puntos aleatorios en el espacio de búsqueda
    for _ in range(iterations):
        # Generar valores aleatorios para x y y en el rango [-5, 5]
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5)
        
        # Calcular el valor de la función de Himmelblau en el punto (x, y)
        f_val = himmelblau(x, y)
        
        # Guardar el punto (x, y) junto con su valor f(x, y)
        best_points.append((x, y, f_val))
    
    # Ordenar los puntos encontrados por el valor de la función f(x, y) de menor a mayor
    best_points.sort(key=lambda point: point[2])
    
    # Seleccionar los 4 mejores puntos con los menores valores de f(x, y)
    top_4 = best_points[:4]
    
    return top_4

def main():
    """
    Función principal que encuentra y muestra los mejores 4 mínimos de la función de Himmelblau
    utilizando búsqueda aleatoria.
    """
    # Encontrar los 4 mejores mínimos utilizando búsqueda aleatoria
    top_4_minima = find_minima_random_search()
    
    # Mostrar los resultados
    print("Los mejores 4 valores donde la función Himmelblau es mínima:")
    for i, (x, y, f_val) in enumerate(top_4_minima, 1):
        print(f"{i}: x = {x:.6f}, y = {y:.6f}, f(x,y) = {f_val:.6f}")

if __name__ == "__main__":
    # Ejecutar la función principal cuando se ejecuta el script
    main()
