import numpy as np
import scipy.optimize as spo

# Definimos la función objetivo
def f(x):
    return sum(x_i**4 for x_i in x)

# Definimos el gradiente de la función (derivada parcial de f respecto a cada x_i)
def grad_f(x):
    return np.array([4 * x_i**3 for x_i in x])

def steepest_descent(f, df, x_0, tol=1.e-8, max_iterations=50):
    xk = np.array(x_0)
    iters = 0
    while np.linalg.norm(df(xk)) > tol and iters < max_iterations:
        lambda_k = spo.golden(lambda l: f(xk - l * df(xk)))
        xk = xk - lambda_k * df(xk)
        iters += 1
    return xk

# Punto inicial
x_0 = [1, -1]

minimo = steepest_descent(f, grad_f, x_0)
print(f'El punto mínimo encontrado es: {minimo}')