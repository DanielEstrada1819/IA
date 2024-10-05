import heapq
import time

# Representación del laberinto (0 = camino, 1 = pared)
maze = [
    [1, 0, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1]
]

# Posición de inicio y salida
start = (0, 1)  # Coordenadas de inicio (fila, columna)
end = (3, 4)    # Coordenadas de salida (fila, columna)

# Movimientos posibles (arriba, abajo, izquierda, derecha)
directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

# Función para calcular la heurística (distancia de Manhattan)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Algoritmo A*
def astar(maze, start, end):
    rows, cols = len(maze), len(maze[0])
    open_list = []
    heapq.heappush(open_list, (0 + heuristic(start, end), 0, start))
    came_from = {}
    g_score = {start: 0}

    while open_list:
        _, current_cost, current = heapq.heappop(open_list)
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Devuelve el camino desde el inicio hasta el final

        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and maze[neighbor[0]][neighbor[1]] == 0:
                new_cost = current_cost + 1
                if neighbor not in g_score or new_cost < g_score[neighbor]:
                    g_score[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, end)
                    heapq.heappush(open_list, (priority, new_cost, neighbor))
                    came_from[neighbor] = current
    return None  # No hay camino

# Medir el tiempo de ejecución del primer escenario
start_time = time.time()
path = astar(maze, start, end)
end_time = time.time()

# Resultado y tiempo de ejecución
if path:
    print("Camino encontrado:", path)
else:
    print("No se encontró un camino.")
print(f"Tiempo de ejecución: {end_time - start_time:.6f} segundos")

# Escenario adicional: laberinto más grande y complejo
maze_large = [
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 1],
    [1, 1, 1, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0, 0, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

start_large = (0, 1)
end_large = (8, 9)

# Medir el tiempo de ejecución del segundo escenario
start_time = time.time()
path_large = astar(maze_large, start_large, end_large)
end_time = time.time()

# Resultado y tiempo de ejecución para el laberinto más grande
if path_large:
    print("Camino encontrado en el laberinto grande:", path_large)
else:
    print("No se encontró un camino en el laberinto grande.")
print(f"Tiempo de ejecución: {end_time - start_time:.6f} segundos")
