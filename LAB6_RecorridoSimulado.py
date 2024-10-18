import numpy as np
import random
import math

# Definir la función de Himmelblau
def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# Algoritmo de Recorrido Simulado
def recocido(himmelblau, limites, temp, enfriamiento, iteraciones):
    # Inicializar la solución con un punto aleatorio dentro de los límites
    x = random.uniform(limites[0], limites[1])
    y = random.uniform(limites[0], limites[1])

    # Evaluar la función en la solución inicial
    current_energy = himmelblau(x, y)

    # Guardar la mejor solución encontrada
    best_x, best_y = x, y
    best_energy = current_energy

    for i in range(iteraciones):
        # Reducir la temperatura de acuerdo con la tasa de enfriamiento
        temp *= enfriamiento

        # Generar un nuevo punto vecino aleatorio
        new_x = x + random.uniform(-1, 1)
        new_y = y + random.uniform(-1, 1)

        # Asegurarse de que el nuevo punto esté dentro de los límites
        new_x = max(limites[0], min(new_x, limites[1]))
        new_y = max(limites[0], min(new_y, limites[1]))

        # Evaluar la función en el nuevo punto
        new_energy = himmelblau(new_x, new_y)

        # Calcular la diferencia de energía
        delta_energy = new_energy - current_energy

        # Decidir si aceptar la nueva solución
        if delta_energy < 0 or math.exp(-delta_energy / temp) > random.random():
            x, y = new_x, new_y
            current_energy = new_energy

        # Actualizar la mejor solución encontrada
        if current_energy < best_energy:
            best_x, best_y = x, y
            best_energy = current_energy

    return best_x, best_y, best_energy

# Parámetros iniciales
limites = (-5, 5)
temp_inicial = 10000
enfriamiento = 0.99
iteraciones = 10000

best_x, best_y, best_energy = recocido(himmelblau, limites, temp_inicial, enfriamiento, iteraciones)
print(f"Los valores (x, y) que minimizan la función son: x = {best_x}, y = {best_y}")
print(f"Valor mínimo de la función de Himmelblau: {best_energy}")
