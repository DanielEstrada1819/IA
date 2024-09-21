class Stack:
    """Implementación simple de una pila (stack)"""
    def __init__(self):
        self.stack = []
    
    def push(self, item):
        self.stack.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.stack.pop()
        return None
    
    def is_empty(self):
        return len(self.stack) == 0


# DFS para resolver el laberinto
def dfs_maze(maze, start, end):
    stack = Stack()
    visited = set()
    stack.push((start, [start]))  # (posición actual, ruta recorrida)

    while not stack.is_empty():
        (position, path) = stack.pop()
        x, y = position

        # Si llegamos a la salida, devolvemos la ruta seguida
        if position == end:
            return path

        # Si ya visitamos esta posición, la ignoramos
        if position in visited:
            continue

        visited.add(position)

        # Movimientos posibles
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for move in moves:
            new_x, new_y = x + move[0], y + move[1]

            # Verificar si no choca con pared y no se sale del laberinto
            if 0 <= new_x < len(maze) and 0 <= new_y < len(maze[0]) and maze[new_x][new_y] == 0:
                stack.push(((new_x, new_y), path + [(new_x, new_y)]))

    return None  # Si no se encuentra solución

# Representación del laberinto
maze = [
    [1, 0, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1]
]

start = (0, 1)
end = (3, 4)

solution_path = dfs_maze(maze, start, end)

if solution_path:
    print("Camino encontrado:", solution_path)
else:
    print("No hay solución para el laberinto.")
