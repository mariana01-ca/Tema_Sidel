import numpy as np
A = np.array([
    [3, -0.1, -0.2],
    [0.1, 7, -0.3],
    [0.3, -0.2, 10]
])

b = np.array([7.85, -19.3, 71.4])

x = np.zeros_like(b)  
tol = 0.001
max_iteraciones = 100
alfa = 0.1 
def gauss_seidel(A, b, tol, max_iteraciones):
    n = len(b)
    x = np.zeros_like(b)
    
    for k in range(max_iteraciones):
        x_nuevo = np.copy(x)
        for i in range(n):
            suma = sum(A[i][j] * x_nuevo[j] for j in range(n) if j != i)
            x_nuevo[i] = (b[i] - suma) / A[i][i]

        print(f"Iteración {k}: {x_nuevo}")
        
        if np.linalg.norm(x_nuevo - x, ord=np.inf) < tol:
            print(f"Convergencia alcanzada en la iteración {k}")
            break       
        x = x_nuevo
    
    return x

# Ejecutar el método Gauss-Seidel
solucion = gauss_seidel(A, b, tol, max_iteraciones)

print("\nSolución final:")
print(solucion)