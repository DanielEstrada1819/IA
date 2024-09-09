import numpy as np

# AAqui se difne la función
def f(x, y):
    return ((1.5 - x + x * y) ** 2 + 
            (2.25 - x + x * y ** 2) ** 2 + 
            (2.625 - x + x * y ** 3) ** 2)

# Limites del Dominio
x_min, x_max = -4.5, 4.5
y_min, y_max = -4.5, 4.5

# Inicialización de variables para valores minimos
best_x, best_y = None, None
min_value = float('inf')

# Busqueda Aleatoria con 10000 iteraciones
iterations = 10000
for _ in range(iterations):
    x = np.random.uniform(x_min, x_max)
    y = np.random.uniform(y_min, y_max)
    current_value = f(x, y)
    
    # Se actualiza el valor minimo si se encuentra uno nuevo
    if current_value < min_value:
        min_value = current_value
        best_x, best_y = x, y

# Impresion de los valores 
print(f"Valores mínimos encontrados:")
print(f"x = {best_x}, y = {best_y}, f(x, y) = {min_value}")
