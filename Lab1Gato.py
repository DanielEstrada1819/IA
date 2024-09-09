import numpy as np
import random

# Inicializacion del tablero vacio
def inicializar_tablero():
    return [[" " for _ in range(4)] for _ in range(4)]

# Imprimir tablero en consola
def mostrar_tablero(tablero):
    print("   0   1   2   3")
    print("  -----------------")
    for i, fila in enumerate(tablero):
        print(f"{i} | " + " | ".join(fila) + " |")
        print("  -----------------")

# Verificacion si el jugador gano
def verificar_ganador(tablero, jugador):
    # Revisa filas y columnas
    for i in range(4):
        if all([tablero[i][j] == jugador for j in range(4)]) or all([tablero[j][i] == jugador for j in range(4)]):
            return True
    # Revisa las diagonales
    if all([tablero[i][i] == jugador for i in range(4)]) or all([tablero[i][3 - i] == jugador for i in range(4)]):
        return True
    return False

# Revisa si hay empate
def verificar_empate(tablero):
    return all([tablero[i][j] != " " for i in range(4) for j in range(4)])

# Movimiento aleatorio de la computadora
def movimiento_computadora(tablero):
    movimientos_posibles = [(i, j) for i in range(4) for j in range(4) if tablero[i][j] == " "]
    return random.choice(movimientos_posibles)

# Juego
def jugar():
    tablero = inicializar_tablero()
    mostrar_tablero(tablero)
    
    while True:
        # Turno del jugador
        print("Turno del jugador (X):")
        fila = int(input("Elige fila (0-3): "))
        columna = int(input("Elige columna (0-3): "))
        
        if tablero[fila][columna] != " ":
            print("Esa casilla ya está ocupada. Intenta de nuevo.")
            continue
        
        tablero[fila][columna] = "X"
        mostrar_tablero(tablero)
        
        if verificar_ganador(tablero, "X"):
            print("¡Felicidades, has ganado!")
            break
        
        if verificar_empate(tablero):
            print("¡Es un empate!")
            break
        
        # Turno de la computadora
        print("Turno de la computadora (O):")
        fila, columna = movimiento_computadora(tablero)
        tablero[fila][columna] = "O"
        mostrar_tablero(tablero)
        
        if verificar_ganador(tablero, "O"):
            print("La computadora ha ganado.")
            break
        
        if verificar_empate(tablero):
            print("¡Es un empate!")
            break

# Ejecuta el juego
jugar()
