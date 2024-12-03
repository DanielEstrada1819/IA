import math

def crear_tablero():
    return [[' ' for _ in range(4)] for _ in range(4)]

def imprimir_tablero(tablero):
    for fila in tablero:
        print('|'.join(fila))
        print('-' * 7)

def verificar_ganador(tablero, jugador):
    # Verificar filas, columnas y diagonales
    for i in range(4):
        # Filas y columnas
        if all([tablero[i][j] == jugador for j in range(4)]) or all([tablero[j][i] == jugador for j in range(4)]):
            return True
    # Diagonales
    if all([tablero[i][i] == jugador for i in range(4)]) or all([tablero[i][3-i] == jugador for i in range(4)]):
        return True
    return False

# 4. Función para verificar si el tablero está lleno
def tablero_lleno(tablero):
    return all([cell != ' ' for fila in tablero for cell in fila])

# 5. algoritmo minimax con poda alfa-beta
def minimax(tablero, profundidad, es_maximizador, alfa, beta, limite=5):
    # Verificar si hay un ganador o empate
    if verificar_ganador(tablero, 'X'):  # IA
        return 10 - profundidad
    elif verificar_ganador(tablero, 'O'):  # Humano
        return profundidad - 10
    elif tablero_lleno(tablero):
        return 0

    # Limitar la profundidad de la búsqueda para evitar bucles
    if profundidad >= limite:
        return 0

    if es_maximizador:
        mejor_valor = -math.inf
        for i in range(4):
            for j in range(4):
                if tablero[i][j] == ' ':
                    tablero[i][j] = 'X'
                    valor = minimax(tablero, profundidad + 1, False, alfa, beta, limite)
                    tablero[i][j] = ' '
                    mejor_valor = max(mejor_valor, valor)
                    alfa = max(alfa, mejor_valor)
                    if beta <= alfa:
                        # Poda alfa-beta
                        break
        return mejor_valor
    else:
        mejor_valor = math.inf
        for i in range(4):
            for j in range(4):
                if tablero[i][j] == ' ':
                    tablero[i][j] = 'O'
                    valor = minimax(tablero, profundidad + 1, True, alfa, beta, limite)
                    tablero[i][j] = ' '
                    mejor_valor = min(mejor_valor, valor)
                    beta = min(beta, mejor_valor)
                    if beta <= alfa:
                        # Poda alfa-beta
                        break
        return mejor_valor

# 6. Función para encontrar el mejor movimiento (IA)
def mejor_movimiento(tablero, jugador):
    mejor_valor = -math.inf if jugador == 'X' else math.inf
    mejor_mov = None
    for i in range(4):
        for j in range(4):
            if tablero[i][j] == ' ':
                tablero[i][j] = jugador
                valor_mov = minimax(tablero, 0, jugador == 'O', -math.inf, math.inf)
                tablero[i][j] = ' '
                
                # Actualizar mejor movimiento dependiendo del jugador
                if (jugador == 'X' and valor_mov > mejor_valor) or (jugador == 'O' and valor_mov < mejor_valor):
                    mejor_valor = valor_mov
                    mejor_mov = (i, j)
    return mejor_mov

# Modalidades de juego
# 7. Modo Humano vs Humano
def humano_vs_humano():
    tablero = crear_tablero()
    jugador = 'O'  # Comienza el jugador 'O'
    while not tablero_lleno(tablero):
        imprimir_tablero(tablero)
        fila, col = map(int, input(f"Turno del jugador {jugador}. Ingresa fila y columna (0-3): ").split())
        if tablero[fila][col] == ' ':
            tablero[fila][col] = jugador
            if verificar_ganador(tablero, jugador):
                imprimir_tablero(tablero)
                print(f"¡Jugador {jugador} gana!")
                return
            # Cambiar de turno entre 'O' y 'X'
            jugador = 'X' if jugador == 'O' else 'O'
        else:
            print("¡Esa posición ya está ocupada, intenta nuevamente!")
    imprimir_tablero(tablero)
    print("¡Es un empate!")

def humano_vs_ia():
    tablero = crear_tablero()
    jugador = 'O'  # Humano comienza
    while not tablero_lleno(tablero):
        imprimir_tablero(tablero)
        if jugador == 'O':
            fila, col = map(int, input(f"Turno del jugador {jugador}. Ingresa fila y columna (0-3): ").split())
            if tablero[fila][col] == ' ':
                tablero[fila][col] = jugador
                if verificar_ganador(tablero, jugador):
                    imprimir_tablero(tablero)
                    print(f"¡Jugador {jugador} gana!")
                    return
                jugador = 'X'
        else:
            print("Turno de la IA...")
            mov = mejor_movimiento(tablero, 'X')
            if mov:
                tablero[mov[0]][mov[1]] = 'X'
                if verificar_ganador(tablero, 'X'):
                    imprimir_tablero(tablero)
                    print("¡La IA gana!")
                    return
            jugador = 'O'
    imprimir_tablero(tablero)
    print("¡Es un empate!")

def ia_vs_ia():
    tablero = crear_tablero()
    jugador = 'O'  # Comienza la IA como O
    while not tablero_lleno(tablero):
        imprimir_tablero(tablero)
        if jugador == 'O':
            print("Turno de la IA (jugador O)...")
            mov = mejor_movimiento(tablero, 'O')
            if mov:
                tablero[mov[0]][mov[1]] = 'O'
                if verificar_ganador(tablero, 'O'):
                    imprimir_tablero(tablero)
                    print("¡La IA (O) gana!")
                    return
            jugador = 'X'
        else:
            print("Turno de la IA (jugador X)...")
            mov = mejor_movimiento(tablero, 'X')
            if mov:
                tablero[mov[0]][mov[1]] = 'X'
                if verificar_ganador(tablero, 'X'):
                    imprimir_tablero(tablero)
                    print("¡La IA (X) gana!")
                    return
            jugador = 'O'
    imprimir_tablero(tablero)
    print("¡Es un empate!")


# Menú principal
def menu():
    print("1. Humano vs Humano")
    print("2. Humano vs IA")
    print("3. IA vs IA")
    opcion = int(input("Selecciona una opción: "))
    if opcion == 1:
        humano_vs_humano()
    elif opcion == 2:
        humano_vs_ia()
    elif opcion == 3:
        ia_vs_ia()

# Ejecutar menú
menu()
