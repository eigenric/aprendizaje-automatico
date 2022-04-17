# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Ricardo Ruiz Fernandez de Alba
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import itertools

figures = r"memoria\chap1\images"

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

def scatter_points(x, x1_lim=None, x2_lim=None, 
                   xlabel="$x$ axis", ylabel="$y$ axis",
                   figname="Figure_1"):
    """Dibuja una nube de puntos"""

    fig, ax = plt.subplots()
    ax.scatter(x[:, 0], x[:, 1], s=10, color="b")

    ax.set_xlim(x1_lim)
    ax.set_ylim(x2_lim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    plt.savefig(fr"{figures}\{figname}.png", dpi=600)
    plt.show(block=True)
    plt.close()


    
print("Ejercicio 1.1a \n")
print("Figura 1.1. Gráfica de nube de puntos uniformemente distribuidos.")

x = simula_unif(50, 2, [-50,50])

#scatter_points(x, x1_lim=(-50, 50), x2_lim=(-50, 50), figname="Figure_1")

#input("\n--- Pulsar tecla para continuar ---\n")

# ###############################################################################

# EJERCICIO 1.1a: Dibujar una gráfica con la nube de puntos de salida
# correspondiente

print("Ejercicio 1.1b \n")
print("Figura 1.2. Gráfica de nube de puntos distribuición gaussiana")

x = simula_gauss(50, 2, np.array([5, 7]))

#scatter_points(x, x1_lim=(-5, 6.5), x2_lim=(-7.5, 7.5), figname="Figure_2")

#input("\n--- Pulsar tecla para continuar ---\n")

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

def scatter_label_lines(x, y, ws_labels=None, ws_colors=None,
                       x1_lim=None, x2_lim=None,
                       xlabel="$x$ axis", ylabel="$y$ axis", 
                       ax=None, figname="Figure1",
                       legend_upper=False):
    """
    Dibuja nube 2D de puntos con etiquetas y recta(s).

    :param x: Array de puntos (x,y)
    :param y: Vector de etiquetas

    :param a: Pendiente de la recta
    :param b: Ordenada en el origen
    :param ws_labels: Diccionario:  w  -> w_label para regresión lineal

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

    # Colores para las rectas
    infinite_colors = itertools.cycle("kmycgbr")

    # Generamos etiquetas y colores por defecto
    if ws_labels is not None:
        if ws_colors is None:
            ws_colors = tuple(itertools.islice(infinite_colors, len(ws_labels)))

    for label in np.unique(y):
        x_label = x[y == label]
        ax.scatter(x_label[:, 0], x_label[:, 1], s=10, color=next(colors), 
                   alpha=1, label=f"Etiqueta {label}")

    if ws_labels is not None:
        x1_min, x1_max = np.array([np.min(x[:, 0]), np.max(x[:, 0])])
        x2_min, x2_max = np.array([np.min(x[:, 1]), np.max(x[:, 1])])

        # Pintar la línea (hiperplano) que define cada array de pesos
        for (w_label, w), w_color in zip(ws_labels.items(), ws_colors):
            x1_ar = np.array([x1_min, x1_max])
            # Si w0 + w1x1 + w2x2 = 0 => x2 = -1*(w1*x1 + w0)/w2
            x2_line = -1*(w[1]*x1_ar + w[0]) / w[2]
            ax.plot(x1_ar, x2_line, color=w_color, label=w_label)

    if legend_upper:
        ax.legend(bbox_to_anchor=(0,1.02,1,0.2), 
                  loc="lower left", mode="expand", 
                  borderaxespad=0, ncol=3)
    else:
        ax.legend()
    
    ax.set(xlabel=xlabel, ylabel=ylabel,
           xlim=x1_lim, ylim=x2_lim)

    plt.savefig(f"{figures}/{figname}.png", dpi=600)
    plt.show(block=True)

def scatter_label_line(x, y, a, b, x1_lim=None, x2_lim=None,
                       xlabel="$x$ axis", ylabel="$y$ axis", 
                       ax=None, figname="Figure1",
                       legend_upper=False):

    w = np.array([-b, -a, 1])
    ws_labels = {f"y={a:.2f}x + {b:.2f}": w}

    scatter_label_lines(x, y, ws_labels, ws_colors="kmy",
                        x1_lim=x1_lim, x2_lim=x2_lim,
                        xlabel=xlabel, ylabel=ylabel,
                        ax=ax, figname=figname, 
                        legend_upper=legend_upper)

# 1.2a Dibujar una gráfica donde los puntos muestren el resultado de su
# etiqueta, junto con la recta usada para ello 

print("Figura 1.3. Etiquetado de puntos uniformemente distribuidos según recta.")
points = simula_unif(100, 2, [-50, 50])
a, b = simula_recta([-50, 50])
y_labels = np.array([f(x, y, a, b) for x,y in points])
y_original = y_labels.copy()

w_original = np.array([-b, -a, 1])

#scatter_label_line(points, y_labels, a, b, 
#                   x1_lim=(-50, 50), x2_lim=(-50, 50),
#                   figname="Figure_3", legend_upper=True)

#input("\n--- Pulsar tecla para continuar ---\n")

# Introducimos 10% de ruido en las etiquetas positivas 
# y en las negativas.
print("Ejercicio 1.2b \n")

for label in (-1, 1):
    label_ids = np.where(y_labels == label)[0]
    N = len(label_ids)
    noise_label = np.random.choice(label_ids, size=round(0.1*N),
                                   replace=False)

    y_labels[noise_label] = -y_labels[noise_label]

y_noise = y_labels

# Dibujamos de nuevo la gráfica con el ruido

print("Figura 1.4. Muestra uniforme con 10% de ruido") 
#scatter_label_line(points, y_labels, a, b, 
#                   x1_lim=(-50, 50), x2_lim=(-50, 50),
#                   figname="Figure_4", legend_upper=True)

#input("\n--- Pulsar tecla para continuar ---\n")

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

    plt.savefig(f"{figures}/{figname}.png", dpi=600)
    plt.show(block=True)
    plt.close()
    
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
#plot_datos_cuad(X, y, fz(f1), title="Circunferencia $f_1(x,y)$", figname="Figure_5")
f1_labels = np.array([signo(f1(x,y)) for x,y in X])
misc_rate1 = np.count_nonzero(f1_labels != y_original) / len(X) * 100
print(f"Misclassification rate (Circunferencia): {misc_rate1}%")

#input("\n--- Pulsar tecla para continuar ---\n")

print("Figura 1.6. Muestra clasificada por elipse f_2")
#plot_datos_cuad(X, y, fz(f2), title="Elipse", figname="Figure_6")
f2_labels = np.array([signo(f2(x,y)) for x,y in X])
misc_rate2 = np.count_nonzero(f2_labels != y_original) / len(X) * 100
print(f"Misclassification rate (Elipse): {misc_rate2}%")

#input("\n--- Pulsar tecla para continuar ---\n")

print("Figura 1.7. Muestra clasificada por hipérbola f_3")
#plot_datos_cuad(X, y, fz(f3), title="Hipérbola", figname="Figure_7")
f3_labels = np.array([signo(f3(x,y)) for x,y in X])
misc_rate3 = np.count_nonzero(f3_labels != y_original) / len(X) * 100
print(f"Misclassification rate (Hipérbola): {misc_rate3}%")

#input("\n--- Pulsar tecla para continuar ---\n")

print("Figura 1.8. Muestra clasificada por parábola f_4")
#plot_datos_cuad(X, y, fz(f4), title="Parábola", figname="Figure_8")
f4_labels = np.array([signo(f4(x,y)) for x,y in X])
misc_rate4 = np.count_nonzero(f4_labels != y_original) / len(X) * 100
print(f"Misclassification rate (Parábola): {misc_rate4}%")

#input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 2.1: ALGORITMO PERCEPTRON

def ajusta_PLA(datos, label, max_iter, vini):
    """Algoritmo de aprendizaje del Perceptrón
    :param datos: matriz homogénea de caracteristicas
    :param label: vector de etiquetas
    :param max_iter: número máximo de iteraciones
    :param vini: valor inicial del vector

    :returns: lista coeficientes w, iteraciones, error de clasificacion
    """
    w = vini
    w_old = None
    it = 0
    err = np.Infinity
    errs = []
    ws = []

    while it < max_iter:
        w_old = w
        err = 0
        for x, y in zip(datos, label):
            if signo(w.T @ x) != y:
                err += 1
                w = w + y * x
        ws.append(w)
        errs.append(err)
        it += 1

        if np.allclose(w_old, w):
            break

    return ws, it, errs

class Animation:
    def __init__(self, X, y, interval=75, x1_lim=(-50, 50), x2_lim=(-50, 50), 
                 animname="Animacion", xlabel="$x$ axis", ylabel="$y$ axis"):
    
        self._init(x1_lim, x2_lim, xlabel, ylabel)

        self.X = X
        self.y = y

        self.colors = itertools.cycle("rbgmcyk")
        self.interval = interval
        self.animname = animname


    @classmethod
    def _init(cls, x1_lim, x2_lim, xlabel, ylabel):
        cls.fig, cls.ax = plt.subplots()

        cls.linean, = cls.ax.plot([], [], color="k")
        cls.y_label = cls.fig.text(0.40, 0.912, "")

        cls.xlabel = xlabel
        cls.ylabel = ylabel

        cls.x1_lim = x1_lim
        cls.x2_lim = x2_lim

        cls.ax.set(xlabel=cls.xlabel, ylabel=cls.ylabel, 
                   xlim=cls.x1_lim, ylim=cls.x2_lim)
        
    def scatter_label(self):
        """Pinta los puntos con sus clases (colores)"""
        for label in np.unique(self.y):
            x_label = self.X[:, 1:][y == label]
            self.ax.scatter(x_label[:, 0], x_label[:, 1], s=10, 
                       color=next(self.colors), alpha=1, 
                       label=f"Etiqueta {label}")
        self.ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", 
                      mode="expand", borderaxespad=0, ncol=3)

    @staticmethod
    def start():
        return Animation._start()

    @staticmethod
    def update(i, ws):
        return Animation._update(i, ws)

    @classmethod
    def _start(cls):
        """Antes del frame inicial"""

        cls.linean.set_data([], [])
        cls.y_label.set_text("")

        return cls.linean, cls.y_label

    @classmethod
    def _update(cls, i, ws):
        """Actualiza la animacion en el frame i-esimo"""

        if i < len(ws): 
            xs = np.array([cls.x1_lim[0], cls.x1_lim[1]])
            w = ws[i]
            # Pendiente y ordenada en el origen
            a, b = -w[1]/w[2], -w[0]/w[2]
            line = a*xs + b
            # Actualizamos la recta 
            cls.linean.set_data(xs, line)
            cls.y_label.set_text(f"It {i}: y={a:.2f}x + {b:.2f}")
        else: # Ultimo frame
            w = ws[-1]
            a, b = -w[1]/w[2], -w[0]/w[2]

            cls.linean.set(linewidth=2)
            cls.y_label.set(x=0.38, color="g", fontsize=12)
            cls.y_label.set_text(f"It {i}: y={a:.2f}x + {b:.2f}")
    
        return cls.linean, cls.y_label

    def render(self, ws):
        """Abre una ventana con la animacion"""

        self.scatter_label() 

        self.anim = FuncAnimation(self.fig, Animation.update, fargs=(ws, ),
                                  init_func=Animation.start, repeat=False,
                                  frames=len(ws)+1, interval=self.interval, 
                                  blit=False)
        
    @staticmethod
    def show():
        plt.show()
        #plt.close()

    def save(self):
        """BORRAR"""

        self.anim.save(fr"{figures}\{self.animname}.gif", writer="imagemagick")

def tabla_resultados(X, y, max_iter=10_000, figname1=None, figname2=None, 
                     animation=False, linewidth=1, interval=25):
    vini = np.array([0, 0, 0])
    ws, it, errs0 = ajusta_PLA(X, y, max_iter, vini)
    err = errs0[-1]
    w0 = ws[-1]
    err_percent = err / len(X) * 100
    err_ini = errs0[0] / len(X) * 100

    print("Vector inicial | Iteraciones | Coeficientes w | Error de clasificación | Error inicial")
    print("----- | ----- | ----- | ---- ")
    print(f"$[{vini[0]:.3f}, {vini[1]:.3f}, {vini[2]:.3f}]$ | ${it}$ | $[{w0[0]:.2f},{w0[1]:.2f},{w0[2]:.2f}]$ | ${err_percent}\%$ | {err_ini}")

    if animation:
        animation = Animation(X, y, interval=interval, animname="perceptron")
        animation.render(ws)
        animation.show()

    fig, ax = plt.subplots()
    xs = range(len(errs0))
    ax.set_xlabel("Número de iteración")
    ax.set_ylabel("% Error de clasificación")
    ax.plot(xs, errs0, linewidth=linewidth)
    plt.savefig(fr"{figures}\{figname1}.png", dpi=600)
    plt.show()

    # # Random initializations
    iterations = []
    errs = []
    vinis = []
    for i in range(10):
        vini = np.random.uniform(size=3)
        ws, it, errs_r = ajusta_PLA(X, y, max_iter, vini)
        w = ws[-1]
        err = errs_r[-1]
        err_percent = err / len(X) * 100
        err_ini = errs_r[0] / len(X) * 100

        if i == 7:
            w7 = ws[-1]
            errs7 = errs_r

        iterations.append(it)
        errs.append(err)
        vinis.append(vini)

        #print(f"$[{vini[0]:.3f}, {vini[1]:.3f}, {vini[2]:.3f}]$ | ${it}$ | $[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f}]$ | ${err_percent}\%$")
        print(f"$[{vini[0]:.3f}, {vini[1]:.3f}, {vini[2]:.3f}]$ | ${it}$ | $[{w0[0]:.2f},{w0[1]:.2f},{w0[2]:.2f}]$ | ${err_percent}\%$ | {err_ini}")

    print(f"Promedio iteraciones: {np.mean(np.asarray(iterations))}")
    print(f"Promedio error de clasificación: {np.mean(np.asarray(errs))}")

    ws_labels = {"Recta original" : w_original,
                "v_ini nulo": w0,
                "v_ini aleatorio (7ta it)": w7}

    scatter_label_lines(X[:, 1:], y, ws_labels=ws_labels, 
                        x1_lim=(-50, 50), x2_lim=(-50, 50),
                        figname=figname2, legend_upper=True)


# Reutilizamos apartado 2a del Ej 1. (Linealmente separables)
# Con datos en forma homogénea (1 | x1 | x2)
print("Ejercicio 2a \n")
columna_unos = np.ones((len(X), 1))
X = np.hstack((columna_unos, X))
y = y_original

y_s = y.copy()
y_s.shape = (len(y_s), 1)
np.savetxt("muestra.csv", np.hstack((X, y_s)), delimiter=",")

tabla_resultados(X, y, max_iter=10_000, figname1="Figure_9",
                 figname2="Figure_10", animation=False, interval=100)


print("Ejercicio 2b \n")
# Reutilizamos apartado 2b del Ej 1. (10% Ruido no l.s)
# Con datos en forma homogénea
y = y_noise

y_s = y.copy()
y_s.shape = (len(y_s), 1)
np.savetxt("muestra_noise.csv", np.hstack((X, y_s)), delimiter=",")

tabla_resultados(X, y, max_iter=5_000, figname1="Figure_11",
                 figname2="Figure_12", linewidth=0.3, 
                animation=False)


#input("\n--- Pulsar tecla para continuar ---\n")

# Ahora con los datos del ejercicio 1.2.b

#CODIGO DEL ESTUDIANTE


#input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

def sgdRL():
    #CODIGO DEL ESTUDIANTE

    return w



#CODIGO DEL ESTUDIANTE

#input("\n--- Pulsar tecla para continuar ---\n")
    


# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).


#CODIGO DEL ESTUDIANTE


#input("\n--- Pulsar tecla para continuar ---\n")


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

#input("\n--- Pulsar tecla para continuar ---\n")

#LINEAR REGRESSION FOR CLASSIFICATION 

#CODIGO DEL ESTUDIANTE


#input("\n--- Pulsar tecla para continuar ---\n")



#POCKET ALGORITHM
  
#CODIGO DEL ESTUDIANTE




#input("\n--- Pulsar tecla para continuar ---\n")


#COTA SOBRE EL ERROR

#CODIGO DEL ESTUDIANTE
