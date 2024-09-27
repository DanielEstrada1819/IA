from collections import deque





# Función para comprobar si una posición es válida dentro del laberinto
def is_valid_move(maze, visited, x, y):
    return 0 <= x < len(maze) and 0 <= y < len(maze[0]) and maze[x][y] == 0 and not visited[x][y]

# BFS para encontrar el camino más corto
def bfs(maze, start, end):
    queue = deque([start])
    visited = [[False for _ in range(len(maze[0]))] for _ in range(len(maze))]
    visited[start[0]][start[1]] = True
    parent = {start: None}  # Para rastrear el camino
    
    while queue:
        current = queue.popleft()
        if current == end:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1]  # Invertir el camino para tenerlo en orden correcto
        
        # Movimientos posibles: arriba, abajo, izquierda, derecha
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for move in moves:
            next_x, next_y = current[0] + move[0], current[1] + move[1]
            if is_valid_move(maze, visited, next_x, next_y):
                queue.append((next_x, next_y))
                visited[next_x][next_y] = True
                parent[(next_x, next_y)] = current

    return None  # No hay camino

# Representación del laberinto
# 0 = camino, 1 = pared
maze = [
    [1, 0, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1]
]

# Posición de inicio y salida
start = (0, 1)  # Coordenadas (fila, columna) de inicio
end = (3, 4)    # Coordenadas (fila, columna) de salida

# Resolver el laberinto
path = bfs(maze, start, end)

if path:
    print("Camino encontrado:", path)
else:
    print("No hay camino disponible.")
