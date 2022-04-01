"""
TRABAJO 1. 
Nombre Estudiante: Ricardo Ruiz Fernández de Alba
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# Función auxiliar para generar gráficos
'''
Esta función muestra una figura 3D con la función a optimizar junto con el 
óptimo encontrado y la ruta seguida durante la optimización. Esta función, al igual
que las otras incluidas en este documento, sirven solamente como referencia y
apoyo a los estudiantes. No es obligatorio emplearlas, y pueden ser modificadas
como se prefiera. 
    rng_val: rango de valores a muestrear en np.linspace()
    fun: función a optimizar y mostrar
    ws: conjunto de pesos (pares de valores [x,y] que va recorriendo el optimizador
                           en su búsqueda iterativa del óptimo)
    colormap: mapa de color empleado en la visualización
    title_fig: título superior de la figura
    
Ejemplo de uso: display_figure(2, E, ws, 'plasma','Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')
'''
def display_figure(rng_x, rng_y, fun, ws, var1="x", var2="y", fname="f",
                   colormap="plasma", alpha=0.6, title_fig=""):
    # https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
    from mpl_toolkits.mplot3d import Axes3D


    X, Y = np.meshgrid(rng_x, rng_y)
    Z = fun(X, Y) 
    fig = plt.figure()
    ax = Axes3D(fig,auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                            cstride=1, cmap=colormap, alpha=alpha)
    if len(ws)>0:
        ws = np.asarray(ws)
        min_point = np.array([ws[-1,0],ws[-1,1]])
        min_point_ = min_point[:, np.newaxis]
        ax.plot(ws[:-1,0], ws[:-1,1], fun(ws[:-1,0], ws[:-1,1]), 'r*', markersize=5)
        ax.plot(min_point_[0], min_point_[1], fun(min_point_[0], min_point_[1]), 'r*', markersize=10)
    if len(title_fig)>0:
        fig.suptitle(title_fig, fontsize=16)
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    ax.set_zlabel(f"{fname}({var1}, {var2})")
    plt.show()

#######

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')

def gradient_descent(w_ini, lr, grad_fun, fun, epsilon, max_iters,
                     stop_cond, hist=False):
    """
    Algoritmo de gradiente descendiente para minimizar funciones.
    Implementación para función de dos variables.
    
    :param w_ini: Punto inicial
    :param lr: Tasa de aprendizaje
    :param grad_fun: Gradiente de la función a minimizar
    :param fun: Función a minimizar

    :param stop_cond: Función de condición de parada. 
        stop_cond(it, w, epsilon, max_iters, grad_fun, fun)
        Cuando evalúa a True para cierta iteración it el algoritmo para
    
    :param hist: True si se quiere devolver el histórico de pesos.

    :return: 
        Si hist=False (w, it) donde
            it: nº de iteraciones utilizadas
            w: array de coordenadas (x,y) que minimiza fun
        Si hist=True (ws, it) donde
            it: nº de iteraciones utilizadas
            ws: lista de tuplas x,y. Histórico del gradiente descendente.
    """
    # Histórico de pesos opcional
    if hist: 
        ws = [tuple(w_ini)]

    w = w_ini
    it = 0

    # Cuando stop_cond evalúe a True, el algoritmo para.
    while not stop_cond(it, w, epsilon, max_iters, grad_fun, fun):
        w = w - lr * grad_fun(w[0], w[1])
        it += 1

        if hist: ws.append(tuple(w))

    return (ws, it) if hist else (w, it)

print('Ejercicio 2\n')

def E(u,v):
    """Función de error a minimizar"""
    return np.float64((u * v * np.exp(-u**2-v**2))**2)


def dEu(u,v):
    """Derivada parcial de E con respecto a u"""
    return np.float64(
        -2 * u * (2*u**2 - 1) * v**2 * np.exp(-2*(u**2+v**2))
    )

def dEv(u,v):
    """Derivada parcial de E con respecto a v"""
    return np.float64(
        -2 * v * (2*v**2 - 1) * u**2 * np.exp(-2*(u**2+v**2))
    )

def gradE(u,v):
    """Gradiente de E"""
    return np.array([dEu(u,v), dEv(u,v)])


eta = 0.1 
maxIter = 10_000_000_000
error2get = 1e-8
initial_point = np.array([0.5,-0.5])

def stop_cond_error(it, w, epsilon, max_iters, grad_fun, fun):
    """Evalúa a True si fun alcanza cierta cota de error"""

    return fun(w[0], w[1]) <= epsilon


w, it = gradient_descent(initial_point, eta, gradE, E, error2get, maxIter,
                         stop_cond_error)


print (f"2a) Numero de iteraciones: {it}")
print (f"2b) Coordenadas obtenidas: ({w[0]}, {w[1]})")

print("Gráfico para visualizar ejercicio 2")

ws, it = gradient_descent(initial_point, eta, gradE, E, error2get, maxIter,
                          stop_cond_error, hist=True)
                          
rng_val = 1.5
x = np.linspace(-rng_val, rng_val, 50)
y = np.linspace(-rng_val, rng_val, 50)
display_figure(x, y, E, ws, "u", "v", "E")

input("\n--- Pulsar tecla para continuar ---\n")
###############################################################################
print('Ejercicio 3\n')

def f(x, y):
    """Función a minimizar"""
    return np.float64(x**2 + 2*y**2 + 2*np.sin(2*np.pi*x)*np.sin(np.pi*y))

def dfx(x, y):
    """Derivada parcial de f con respecto a x"""
    return np.float64(2*x + 4*np.pi*np.sin(np.pi*y)*np.cos(2*np.pi*x))

def dfy(x,y):
    """Derivada parcial de f con respecto a y"""
    return np.float64(4*y + 2*np.pi*np.sin(2*np.pi*x)*np.cos(np.pi*y))

def gradf(x,y):
    """Gradiente de f"""
    return np.array([dfx(x,y), dfy(x,y)])

# η = 0.01

eta = 0.01 
maxIter = 50
initial_point = np.array([-1, 1])

def stop_cond_maxIter(it, w, epsilon, max_iters, grad_fun, fun):
    """Evalúa a True si se alcanza el nº maximo de iteraciones"""
    return it >= max_iters


ws, it = gradient_descent(initial_point, eta, gradf, f, None, maxIter,
                         stop_cond_maxIter, hist=True)

print (f"3. Numero de iteraciones: {it}")
print (f"3. Coordenadas obtenidas: ({ws[-1][0]}, {ws[-1][1]})")

print("3a) Gráfico descenso del gradiente para f. η = 0.01")

x = np.linspace(-1.5, -0.5, 50)
y = np.linspace(-0.5, 1.5, 50)
display_figure(x, y, f, ws)

# η = 0.1

eta = 0.1 
ws2, it = gradient_descent(initial_point, eta, gradf, f, None, maxIter,
                         stop_cond_maxIter, hist=True)

# print (f"3a.2) Numero de iteraciones: {it}")
# print (f"3a.2) Coordenadas obtenidas: ({ws2[-1][0]}, {ws2[-1][1]})")

print("3a.2) Gráfico descenso del gradiente para f. η = 0.1")

x = np.linspace(-3, 2, 50)
y = np.linspace(-1, 1.5, 50)
display_figure(x, y, f, ws2, alpha=0.4)

# η = 0.005

# eta = 0.005
# ws3, it = gradient_descent(initial_point, eta, gradf, f, None, maxIter,
                        #  stop_cond_maxIter, hist=True)

## Gráfica de comparación eta 0.1 y 0.01

eta001_f = np.array([f(x,y) for x,y in ws])
eta01_f = np.array([f(x,y) for x,y in ws2])

x = np.arange(51)

fig, ax = plt.subplots()
ax.set_title("Variación de $f(x,y)$ según $\eta$")

ax.set(xlabel="Iteraciones", ylabel="f(x,y)")
ax.plot(x, eta001_f, color = "red", label = "$\eta=0.01$")
ax.plot(x, eta01_f,  color = "blue", label = "$\eta=0.1$")
ax.set_yticks(np.arange(-1, 9))
ax.set_xticks(np.arange(0, 51, 5))
ax.legend(loc = "upper right")
plt.show()

## Extra: Gráfica de comparación eta 0.01 y 0.005

# eta005_f =  np.array([f(x,y) for x,y in ws3])

# fig, ax = plt.subplots()
# ax.set_title("Variación de $f(x,y)$ según $\eta$")

# ax.plot(x, eta001_f, color = "red", label = "$\eta=0.01$")
# ax.plot(x, eta005_f, color = "green", label = "$\eta=0.005$")
# ax.set_xticks(np.arange(0, 51, 5))
# ax.legend(loc = "upper right")

# plt.show()

print("3b) Tabla de mínimos dependiente de punto inicial y tasa de aprendizaje")

initial_points = [np.array([-0.5, -0.5]), np.array([1, 1]), 
                  np.array([2.1, -2.1]), np.array([-3, 3]),
                  np.array([-2, 2])]

# Imprimimos dos tablas para cada η
for eta in (0.1, 0.01):
    print(f"eta = {eta}")

    print("    w_0       |    (x, y)      |  min(f,x) ")
    print("--------------| ---------------| ---------")
    for initial_point in initial_points:
        x0, y0 = initial_point
        ws, it = gradient_descent(initial_point, eta, gradf, f, None, maxIter,
                                 stop_cond_maxIter, hist=True)
        f_ws = [(x,y, f(x,y)) for x,y in ws]
        x, y, fxy = min(f_ws, key=lambda x: x[2])

        print(f"({x0}, {y0}) | ({x:.4f},{y:.4f}) | {fxy:.4f}")
 
input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################
###############################################################################
print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 1\n')


label5 = 1
label1 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Funcion para calcular el error
def MSE(x,y,w):
    """Error cuadrático medio (MSE)
    :param x: Matriz de datos de entrada 
    :param y: Vector objetivo 
    :param w: Vector de pesos

    :return: Error cuadrático medio. (No negativo)
    """

    return np.linalg.norm(x @ w - y)**2 / len(x)

def dMSE(x, y, w):
    """Derivada del error cuadrático medio
    :param x: Matriz de datos de entrada
    :param y: Vector objetivo
    :param w: Vector de pesos

    :return: Derivada del error cuadrático medio.
    """
    return  2/len(x) * x.T @ (x @ w - y)


# Gradiente Descendente Estocastico para Regresión Lineal
def sgd(x, y, lr, epsilon, max_iters, stop_cond, batch_size=32, hist=False):
    """
    Algoritmo de gradiente descendiente estocástico (sgd) aplicado
    a Regresión Lineal.

    :param x: matriz de datos de entrada
    :param y: vector objetivo
    :param lr: Tasa de aprendizaje eta
    :param epsilon: error máximo (depende de stop_cond)
    :param max_iters: numero maximo de iteraciones (depende de stop_cond)
    :param stop_cond: Función de condición de parada. 
           stop_cond(it, w, w_err, epsilon, max_iters)
    :param hist: True si se quiere devolver el histórico de pesos.

    :return: 
        Si hist=False (w, it) donde
            it: nº de iteraciones utilizadas
            w: array de coordenadas (x,y) que minimiza E_in
        Si hist=True (ws, it) donde
            it: nº de iteraciones utilizadas
            ws: lista de tuplas x,y. Histórico del sgd.
    """
    w = np.zeros((x.shape[1]), )
    if hist: 
        ws = [tuple(w)]

    w_err = [MSE(x, y, w)]    # Vector de errores
    x_ids = np.arange(len(x))
    it = 0
    batch_start = 0

    while not stop_cond(it, w, w_err, epsilon, max_iters):
        # En cada epoch permutamos los índices
        if batch_start == 0:
            np.random.shuffle(x_ids)

        batch_ids = x_ids[batch_start : batch_start + batch_size]

        # Sólo un mini-batch participa en la adaptación
        w = w - lr*dMSE(x[batch_ids, :], y[batch_ids], w)
        err = MSE(x[batch_ids, :], y[batch_ids], w)
        if isinstance(err, list): print("UNA LISTA!")
        w_err.append(err)

        it += 1
        batch_start += batch_size

        # Nueva epoch
        if batch_start >= len(x): 
            batch_start = 0

        if hist: ws.append(tuple(w))
        
    return (ws, it) if hist else (w, it)

def sgd_maxIter(x, y, lr, max_iters, epsilon=None, batch_size=32, hist=False):
    """Gradiente descendiente estocástico con nº máximo de iteraciones"""

    stop_cond_maxIter = lambda it, _, __, ___, max_iters: it >= max_iters

    return sgd(x, y, lr, epsilon, max_iters, stop_cond_maxIter, batch_size=batch_size, hist=hist)

def sgd_error(x, y, lr, epsilon, max_iters=None, batch_size=32, hist=False):
    """Gradiente descendiente estocástico con condición de parada por error"""

    # stop_cond_error = lambda _, __, w_err, epsilon, ___: sum(w_err)/len(w_err) <= epsilon
    def stop_cond_error(it, __, w_err, epsilon, ___): 
        mean_w_err = sum(w_err)/len(w_err)
        print(f"it {it}: {mean_w_err}")
        return mean_w_err <= epsilon

    return sgd(x, y, lr, epsilon, max_iters, stop_cond_error, batch_size=batch_size, hist=hist)

def sgd_maxIter_error(x, y, lr, epsilon, max_iters=50_000, batch_size=32, hist=False):
    """Gradiente descendiente estocástico con condición de parada por error o nº maximo de iteraciones"""

    stop_cond_maxIter_error = lambda it, _, w_err, epsilon, max_iter: it >= max_iters or w_err <= epsilon

    return sgd(x, y, lr, epsilon, max_iters, stop_cond_maxIter_error, batch_size=batch_size, hist=hist)

# Pseudoinversa	
def pseudoinverse(x):
    """Pseudo-inversa (de Moore-Penrose).
    :param x: Matriz de datos de entrada.
    :nota: se descompone la matriz en valores singulares.
           Si X = U D V^T, entoncex X^\dagger = V D^\dagger U^T
    :return: Pseudoinversa de x         (x^t x)^-1 x^t
    """
    # Alternativamente
    # return np.linalg.pinv(x) que también emplea SVD

    U, d, VT = np.linalg.svd(x, full_matrices=True)
    D = np.zeros(x.shape)
    # Matriz diagonal rectangular. Resto ceros.
    D[:x.shape[1], :x.shape[1]] = np.diag(d)
    V = VT.T

    return V @ np.linalg.inv(D.T @ D) @ D.T @ U.T

def regresion_pinv(x, y):
    """Resuelve el problema de regresión mediante el algoritmo de la Pseudoinversa

    :param x: Matriz de datos de entrada
    :param y: Vector objetivo

    :return: w_lin = x^\dagger y
    """

    return pseudoinverse(x) @ y

def plot_regression(x, y, ws, xlabel, ylabel):
    """
    Grafica los puntos junto con las distintas rectas de regresión asociadas
    que definen los vectores de pesos

    :param x: Matriz de datos de entrada
    :param y: Vector objetivo
    :param ws: Vector de pesos 
    :param xlabel: Nombre eje X
    :param ylabel: Nombre eje Y
    """

    pass

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

eta = 0.01
epsilon = 1e-14
maxit1, maxit2 = 500, 200_000_000
ws_sgd, it = sgd_maxIter(x, y, eta, max_iters=maxit1, hist=True)
ws_sgd2, it2 = sgd_maxIter(x, y, eta, max_iters=maxit2, hist=True)
#ws_sgd2, it2 = sgd_error(x, y, eta, epsilon, hist=True)
w_pinv = regresion_pinv(x, y)

ein_sgd = MSE(x, y, ws_sgd[-1])
ein_sgd2 = MSE(x, y, ws_sgd2[-1])
ein_pinv = MSE(x, y, w_pinv)

eout_sgd = MSE(x_test, y_test, ws_sgd[-1])
eout_sgd2 = MSE(x_test, y_test, ws_sgd2[-1])
eout_pinv = MSE(x_test, y_test, w_pinv)

print ('Bondad del resultado para grad. descendente estocastico:\n')

print(f"Para {maxit1} iteraciones: ")
print(f"\tEin: {ein_sgd}")
print(f"\tEout: {eout_sgd}")

print(f"Para {maxit2} iteraciones: ")
print(f"\t Ein: {ein_sgd2}")
print(f"\t Eout: {eout_sgd2}")

print("Pseudoinversa")
print(f"\t Ein: {ein_pinv}")
print(f"\t Eout: {eout_pinv}")

fig, ax = plt.subplots()

# Map entre etiquetas y número correspondientes
label_number_map = {-1: "Número 1", 1 : "Número 5"}
color_cycle = itertools.cycle(["blue", "orange", "brown", "red", "green"])

for label, number in label_number_map.items():
    # Filtramos los elementos de x de la clase label
    x_cls = x[y == label]
    ax.scatter(x_cls[:, 1], x_cls[:, 2], s=5, alpha=0.5, color=next(color_cycle),
               label=number)

# Fijamos los límites de los ejes para cuadrar con el mínimo y el máximo
minmax = lambda arr: (np.min(arr), np.max(arr))
x1_min, x1_max = minmax(x[:, 1])
x2_min, x2_max = minmax(x[:, 2])
ax.set_xlim(x1_min, x1_max)
ax.set_ylim(x2_min, x2_max)

pesos = {f"SGD con {maxit1} iteraciones" : ws_sgd[-1],
         f"SGD con {maxit2} iteraciones" : ws_sgd2[-1],
          "Pseudoinversa" : w_pinv }

# Pintar la línea (hiperplano) que define cada array de pesos
for w_label, w in pesos.items():
    x1_ar = np.array([x1_min, x1_max])
    # Si w0 + w1x1 + w2x2 = 0 => x2 = -1*(w1*x1 + w0)/w2
    x2_line = -1*(w[1]*x1_ar + w[0]) / w[2]
    ax.plot(x1_ar, x2_line, label=w_label)

ax.legend()
plt.xlabel("Intensidad media")
plt.ylabel("Simetría")
print("2-1) Gráfica que compara intensidad media y simetría ")
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#Seguir haciendo el ejercicio...

print('Ejercicio 2\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

def sign(x):
	if x >= 0:
		return 1
	return -1

def f(x1, x2):
	return sign((x1-0.2)**2+x2**2-0.6) 

#Seguir haciendo el ejercicio...



