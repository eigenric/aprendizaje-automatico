# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Ricardo Ruiz Fernandez de Alba
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import itertools

figures = "memoria/chap1/images"

# Fijamos la semilla
np.random.seed(1)

def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gauss(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out

def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b


###############################################################################

# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida
# correspondiente

print("Ejercicio 1.1a \n")
print("Figura 1.1. Gráfica de nube de puntos uniformemente distribuidos.")

x = simula_unif(50, 2, [-50,50])

fig, ax = plt.subplots()
ax.scatter(x[:, 0], x[:, 1], s=10, color="b")
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)

plt.savefig(f"{figures}/Figure_1.png", bbox_inches='tight',pad_inches = 0, dpi=600)
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# ###############################################################################

# EJERCICIO 1.1a: Dibujar una gráfica con la nube de puntos de salida
# correspondiente

print("Ejercicio 1.1b \n")
print("Figura 1.2. Gráfica de nube de puntos distribuición gaussiana")

fig, ax = plt.subplots()
x = simula_gauss(50, 2, np.array([5, 7]))
ax.scatter(x[:, 0], x[:, 1], s=10, color="b")
plt.savefig(f"{figures}/Figure_2.png", bbox_inches='tight',pad_inches = 0, dpi=600)
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

################################################################################

# EJERCICIO 1.2a: Usar recta simulada para etiquetar puntos

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)

print("Ejercicio 1.2a \n")

def scatter_label_line(x, y, a, b, x1_lim=None, x2_lim=None,
                       xlabel="$x$ axis", ylabel="$y$ axis", 
                       ax=None, figname="Figure1",
                       legend_upper=False,):
    """
    Dibuja nube 2D de puntos con etiquetas y una recta.

    :param x: Array de puntos (x,y)
    :param y: Vector de etiquetas

    :param a: Pendiente de la recta
    :param b: Ordenada en el origen

    :param x1_lim: Tupla mínimo y máximo valor de x1 
    :param x2_lim: Tupla mínimo y máximo valor de x2

    :param xlabel: Etiqueta del Eje X
    :param ylabel: Etiqueta del Eje Y 

    :param legend_upper: la leyenda se situa arriba si es True
    """

    if ax is None:
        _, ax = plt.subplots()

    # Pintamos los puntos de cada etiqueta con color distinto
    colors = itertools.cycle("rbgmcyk")

    for label in np.unique(y):
        x_label = x[y == label]
        ax.scatter(x_label[:, 0], x_label[:, 1], s=10, color=next(colors), 
                   alpha=1, label=f"Etiqueta {label}")

    # Pintamos la linea
    step = (x1_lim[1] - x1_lim[0]) // 10
    xs = np.linspace(x1_lim[0], x1_lim[1], step)
    line = a*xs + b
    line_label = f"y = {a:.2f}x + {b:.2f}"

    ax.plot(xs, line, label=line_label, color="k")

    if legend_upper:
        ax.legend(bbox_to_anchor=(0,1.02,1,0.2), 
                  loc="lower left", mode="expand", 
                  borderaxespad=0, ncol=3)
    else:
        ax.legend()
    
    ax.set(xlabel=xlabel, ylabel=ylabel,
           xlim=x1_lim, ylim=x2_lim)

    plt.savefig(f"{figures}/{figname}.png", bbox_inches='tight',pad_inches = 0, dpi=600)
    plt.show()

# 1.2a Dibujar una gráfica donde los puntos muestren el resultado de su
# etiqueta, junto con la recta usada para ello 

print("Figura 1.3. Etiquetado de puntos uniformemente distribuidos según recta.")
points = simula_unif(100, 2, [-50, 50])
a, b = simula_recta([-50, 50])
y_labels = np.array([f(x, y, a, b) for x,y in points])
y_original = y_labels

scatter_label_line(points, y_labels, a, b, 
                   x1_lim=(-50, 50), x2_lim=(-50, 50),
                   figname="Figure_3", legend_upper=True)

input("\n--- Pulsar tecla para continuar ---\n")

# Introducimos 10% de ruido en las etiquetas positivas 
# y en las negativas.
print("Ejercicio 1.2b \n")

for label in (-1, 1):
    label_ids = np.where(y_labels == label)[0]
    N = len(label_ids)
    noise_label = np.random.choice(label_ids, size=round(0.1*N),
                                   replace=False)

    y_labels[noise_label] = -y_labels[noise_label]

# Dibujamos de nuevo la gráfica con el ruido

print("Figura 1.4. Muestra uniforme con 10% de ruido") 
scatter_label_line(points, y_labels, a, b, 
                   x1_lim=(-50, 50), x2_lim=(-50, 50),
                   figname="Figure_4", legend_upper=True)

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.2c: Supongamos ahora que las siguientes funciones definen la
# frontera de clasificación de los puntos de la muestra en lugar de una recta

def plot_datos_cuad(X, y, fz, title='Point cloud plot', 
                    xaxis='x axis', yaxis='y axis',
                    figname="Figure_1"):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.001
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white')
    
    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')
    
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)

    plt.savefig(f"{figures}/{figname}.png", bbox_inches='tight',pad_inches = 0, dpi=600)
    plt.show()
    
def fz(fn):
    """Decorador función de dos variables para aceptar matriz grid"""
    def fi(X):
        return fn(X[:, 0], X[:, 1])
    return fi

def f1(x,y):
    return (x - 10)**2 + (y - 20)**2 - 400

def f2(x,y):
    return 0.5 * (x + 10)**2 + (y - 20)**2 - 400

def f3(x,y):
    return 0.5 * (x - 10)**2 - (y + 20)**2 - 400

def f4(x,y):
    return y - 20*x**2 - 5*x + 3

# Reutilizamos el etiquetado generado en el apartado 2b
X = points
y = y_labels

print("Figura 1.5. Muestra clasificada por circunferencia f_1")
plot_datos_cuad(X, y, fz(f1), title="Circunferencia $f_1(x,y)$", figname="Figure_5")
f1_labels = np.array([signo(f1(x,y)) for x,y in X])
misc_rate1 = np.count_nonzero(f1_labels != y_original) / len(X) * 100
print(f"Misclassification rate (Circunferencia): {misc_rate1}%")

input("\n--- Pulsar tecla para continuar ---\n")

print("Figura 1.6. Muestra clasificada por elipse f_2")
plot_datos_cuad(X, y, fz(f2), title="Elipse", figname="Figure_6")
f2_labels = np.array([signo(f2(x,y)) for x,y in X])
misc_rate2 = np.count_nonzero(f2_labels != y_original) / len(X) * 100
print(f"Misclassification rate (Elipse): {misc_rate2}%")

input("\n--- Pulsar tecla para continuar ---\n")

print("Figura 1.7. Muestra clasificada por hipérbola f_3")
plot_datos_cuad(X, y, fz(f3), title="Hipérbola", figname="Figure_7")
f3_labels = np.array([signo(f3(x,y)) for x,y in X])
misc_rate3 = np.count_nonzero(f3_labels != y_original) / len(X) * 100
print(f"Misclassification rate (Hipérbola): {misc_rate3}%")

input("\n--- Pulsar tecla para continuar ---\n")

print("Figura 1.8. Muestra clasificada por parábola f_4")
plot_datos_cuad(X, y, fz(f4), title="Parábola", figname="Figure_8")
f4_labels = np.array([signo(f4(x,y)) for x,y in X])
misc_rate4 = np.count_nonzero(f4_labels != y_original) / len(X) * 100
print(f"Misclassification rate (Parábola): {misc_rate4}%")

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 2.1: ALGORITMO PERCEPTRON

def ajusta_PLA(datos, label, max_iter, vini):
    #CODIGO DEL ESTUDIANTE
    
    pass

#CODIGO DEL ESTUDIANTE

# Random initializations
iterations = []
for i in range(0,10):
    pass
    #CODIGO DEL ESTUDIANTE
    
# print('Valor medio de iteraciones necesario para converger:{}'
        # .format(np.mean(np.asarray(iterations))))

input("\n--- Pulsar tecla para continuar ---\n")

# Ahora con los datos del ejercicio 1.2.b

#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

def sgdRL():
    #CODIGO DEL ESTUDIANTE

    return w



#CODIGO DEL ESTUDIANTE

input("\n--- Pulsar tecla para continuar ---\n")
    


# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).


#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################
#BONUS: Clasificación de Dígitos


# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])


#mostramos los datos
# fig, ax = plt.subplots()
# ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 
#         '.', color='red', label='4', markersize=4)
# ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 
#         '.', color='blue', label='8', markersize=4)
# ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
# ax.set_xlim((0, 1))
# plt.legend()
# plt.show()

# fig, ax = plt.subplots()
# ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 
#         '.', color='red', label='4', markersize=4)
# ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 
#         '.', color='blue', label='8', markersize=4)
# ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
# ax.set_xlim((0, 1))
# plt.legend()
# plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#LINEAR REGRESSION FOR CLASSIFICATION 

#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")



#POCKET ALGORITHM
  
#CODIGO DEL ESTUDIANTE




input("\n--- Pulsar tecla para continuar ---\n")


#COTA SOBRE EL ERROR

#CODIGO DEL ESTUDIANTE
