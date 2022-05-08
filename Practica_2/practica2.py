# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Ricardo Ruiz Fernandez de Alba
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import itertools
import warnings

from matplotlib.animation import FuncAnimation

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
                   title=None, figname="Figure_1"):
    """Dibuja una nube de puntos"""

    fig, ax = plt.subplots()
    ax.scatter(x[:, 0], x[:, 1], s=10, color="b")

    ax.set_xlim(x1_lim)
    ax.set_ylim(x2_lim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)
    
    plt.show(block=True)
    plt.close()


    
print("Ejercicio 1.1a \n")
print("Figura 1.1. Gráfica de nube de puntos uniformemente distribuidos.")

x = simula_unif(50, 2, [-50,50])

scatter_points(x, x1_lim=(-50, 50), x2_lim=(-50, 50), 
               title="Nube de puntos uniformemente distribuidos",
               figname="Figure_1")

input("\n--- Pulsar tecla para continuar ---\n")

# ###############################################################################

# EJERCICIO 1.1a: Dibujar una gráfica con la nube de puntos de salida
# correspondiente

print("Ejercicio 1.1b \n")
print("Figura 1.2. Gráfica de nube de puntos distribuición gaussiana")

x = simula_gauss(50, 2, np.array([5, 7]))

scatter_points(x, x1_lim=(-5, 6.5), x2_lim=(-7.5, 7.5), 
               title="Nube de puntos siguiendo distribuicion gaussiana",
               figname="Figure_2")

input("\n--- Pulsar tecla para continuar ---\n")

################################################################################

# EJERCICIO 1.2a: Usar recta simulada para etiquetar puntos

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1

def f_r(x, y, a, b):
    return y - a*x - b

def f(x, y, a, b):
	return signo(f_r(x, y, a, b))

print("Ejercicio 1.2a \n")

def scatter_label_lines(x, y, size=10, lw=2, ws_labels=None, ws_colors=None,
                       x1_lim=None, x2_lim=None,
                       xlabel="$x$ axis", ylabel="$y$ axis", 
                       ax=None, figname="Figure1", title=None,
                       legend_upper=False):
    """
    Dibuja nube 2D de puntos con etiquetas y recta(s).

    :param x: Array de puntos (x,y)
    :param y: Vector de etiquetas

    :param size: Tamaño de los puntos
    :param ws_labels: Diccionario:  w  -> w_label para regresión lineal
    :param ws_colors: Colores para las rectas

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
        ax.scatter(x_label[:, 0], x_label[:, 1], s=size, color=next(colors), 
                   alpha=1, label=f"Etiqueta {int(label)}")

    if ws_labels is not None:
        x1_min, x1_max = np.array([np.min(x[:, 0]), np.max(x[:, 0])])
        x2_min, x2_max = np.array([np.min(x[:, 1]), np.max(x[:, 1])])

        # Pintar la línea (hiperplano) que define cada array de pesos
        for (w_label, w), w_color in zip(ws_labels.items(), ws_colors):
            x1_ar = np.array([x1_min, x1_max])
            # Si w0 + w1x1 + w2x2 = 0 => x2 = -1*(w1*x1 + w0)/w2
            x2_line = -1*(w[1]*x1_ar + w[0]) / w[2]
            ax.plot(x1_ar, x2_line, color=w_color, label=w_label, linewidth=lw)

    if legend_upper:
        ax.legend(bbox_to_anchor=(0,1.02,1,0.2), 
                  loc="lower left", mode="expand", 
                  borderaxespad=0, ncol=3)
    else:
        ax.legend()
    
    ax.set(xlabel=xlabel, ylabel=ylabel,
           xlim=x1_lim, ylim=x2_lim)

    if title is not None:
        #ax.set_title(title, 0.40, 0.912)
        ax.set_title(title, y=1.08)
        
    plt.show(block=True)

def scatter_label_line(x, y, a, b, size=10, lw=2, x1_lim=None, x2_lim=None,
                       xlabel="$x$ axis", ylabel="$y$ axis", 
                       ax=None, figname="Figure1", title=None,
                       legend_upper=False):

    w = np.array([-b, -a, 1])
    ws_labels = {f"y={a:.2f}x + {b:.2f}": w}

    scatter_label_lines(x, y, size=size, lw=lw, ws_labels=ws_labels, ws_colors="kmy", 
                        x1_lim=x1_lim, x2_lim=x2_lim, xlabel=xlabel, ylabel=ylabel,
                        ax=ax, figname=figname, title=title, legend_upper=legend_upper)

# 1.2a Dibujar una gráfica donde los puntos muestren el resultado de su
# etiqueta, junto con la recta usada para ello 

print("Figura 1.3. Etiquetado de puntos uniformemente distribuidos según recta.")
points = simula_unif(100, 2, [-50, 50])
a, b = simula_recta([-50, 50])
y_labels = np.array([f(x, y, a, b) for x,y in points])
y_original = y_labels.copy()

w_original = np.array([-b, -a, 1])

scatter_label_line(points, y_labels, a, b, 
                   x1_lim=(-50, 50), x2_lim=(-50, 50),
                   title="Etiquetado de puntos uniformemente distribuidos según recta",
                   figname="Figure_3", legend_upper=True)

input("\n--- Pulsar tecla para continuar ---\n")

# Introducimos 10% de ruido en las etiquetas positivas 
# y en las negativas.
print("Ejercicio 1.2b \n")

for label in (-1, 1):
    label_ids = np.where(y_labels == label)[0]
    N = len(label_ids)
    noise_label = np.random.choice(label_ids, size=int(np.ceil(0.1*N)),
                                   replace=False)

    y_labels[noise_label] = -y_labels[noise_label]

y_noise = y_labels

# Dibujamos de nuevo la gráfica con el ruido

print("Figura 1.4. Muestra uniforme con 10% de ruido") 
scatter_label_line(points, y_labels, a, b, 
                   x1_lim=(-50, 50), x2_lim=(-50, 50),
                   title="Muestra uniforme con 10% de ruido",
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
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=1.5, 
                cmap="RdYlBu", edgecolor='white')
    
    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')
    
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)

    plt.show(block=True)
    plt.close()
    

print("Ejercicio 1.2c \n")


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


print("Figura 1.6. Muestra clasificada por elipse f_2")
plot_datos_cuad(X, y, fz(f2), title="Elipse", figname="Figure_6")
f2_labels = np.array([signo(f2(x,y)) for x,y in X])
misc_rate2 = np.count_nonzero(f2_labels != y_original) / len(X) * 100
print(f"Misclassification rate (Elipse): {misc_rate2}%")


print("Figura 1.7. Muestra clasificada por hipérbola f_3")
plot_datos_cuad(X, y, fz(f3), title="Hipérbola", figname="Figure_7")
f3_labels = np.array([signo(f3(x,y)) for x,y in X])
misc_rate3 = np.count_nonzero(f3_labels != y_original) / len(X) * 100
print(f"Misclassification rate (Hipérbola): {misc_rate3}%")


print("Figura 1.8. Muestra clasificada por parábola f_4")
plot_datos_cuad(X, y, fz(f4), title="Parábola", figname="Figure_8")
f4_labels = np.array([signo(f4(x,y)) for x,y in X])
misc_rate4 = np.count_nonzero(f4_labels != y_original) / len(X) * 100
print(f"Misclassification rate (Parábola): {misc_rate4}%")


# Etiquetado con circunferencia + ruido

print("Figura 1.9. Muestra etiquetada por circunferencia f_1")

# Etiquetamos con circunferencia y añadimos 10% de ruido

f1_labels = np.array([signo(f1(x,y)) for x,y in X])
f1_labels_original = f1_labels.copy()
for label in (-1, 1):
    label_ids = np.where(f1_labels == label)[0]
    N = len(label_ids)

    noise_label = np.random.choice(label_ids, size=int(np.ceil(0.1*N)),
                                   replace=False)

    f1_labels[noise_label] = -f1_labels[noise_label]


plot_datos_cuad(X, f1_labels, fz(f1), title="Circunferencia etiquetada $f_1(x,y)$", figname="Figure_29")

input("\n--- Pulsar tecla para continuar ---\n")

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
    def __init__(self, X, y, interval=75, figsize=(8, 6), size=10, x1_lim=(-50, 50), x2_lim=(-50, 50), 
                 animname="Animacion", xlabel="$x$ axis", ylabel="$y$ axis"):
    
        self._init(X, y, size, figsize, x1_lim, x2_lim, xlabel, ylabel)

        self.size = size
        self.interval = interval
        self.animname = animname

        self.start()


    @classmethod
    def _init(cls, X, y, size, figsize, x1_lim, x2_lim, xlabel, ylabel):

        cls.fig, cls.ax = plt.subplots(figsize=figsize)

        cls.X = X
        cls.y = y

        cls.size = size
        #cls.linean, = cls.ax.plot([], [], color="k")
        cls.y_label = cls.fig.text(0.35, 0.9093, "")

        cls.xlabel = xlabel
        cls.ylabel = ylabel

        cls.x1_lim = x1_lim
        cls.x2_lim = x2_lim

        cls.colors = "rbgmcyk"

        cls.ax.set(xlabel=cls.xlabel, ylabel=cls.ylabel, 
                   xlim=cls.x1_lim, ylim=cls.x2_lim)

    

    @staticmethod
    def start():
        return Animation._start()

    @staticmethod
    def update(i, ws):
        return Animation._update(i, ws)

    @classmethod
    def _start(cls):
        """Antes del frame inicial"""

        cls.y_label.set_text("")

        X = cls.X[:, 1:]
        cls.min_xy = X.min(axis=0)
        cls.max_xy = X.max(axis=0)
        cls.border_xy = (cls.max_xy-cls.min_xy)*0.01

        cls.xx, cls.yy = np.mgrid[cls.min_xy[0]-cls.border_xy[0]:cls.max_xy[0]+cls.border_xy[0]+0.001:cls.border_xy[0], 
                          cls.min_xy[1]-cls.border_xy[1]:cls.max_xy[1]+cls.border_xy[1]+0.001:cls.border_xy[1]]

        cls.grid = np.c_[cls.xx.ravel(), cls.yy.ravel(), np.ones_like(cls.xx).ravel()]
        cls.pred_y = np.vectorize(f_r)(cls.grid[:, 0], cls.grid[:, 1], 0, 0)
        cls.pred_y = np.clip(cls.pred_y, -1, 1).reshape(cls.xx.shape)
        cls.pred_y = np.array([np.vectorize(signo)(yi) for yi in cls.pred_y])

        cls.contourf = cls.ax.contourf(cls.xx, cls.yy, cls.pred_y, 1, cmap='RdBu',vmin=-1, vmax=1)
        cls.ax_c = cls.fig.colorbar(cls.contourf)
        ticks = np.linspace(y.min(axis=0), y.max(axis=0), 3)
        cls.ax_c.set_ticks(ticks)
        cls.ax_c.set_label('$f(x, y)$')

        colors_it = iter(cls.colors)
        for label in np.unique(cls.y):
            x_label = X[cls.y == label]
            cls.ax.scatter(x_label[:, 0], x_label[:, 1], s=50, linewidth=1.25,
                       color=next(colors_it), edgecolor="white", 
                       label=f"Etiqueta {int(label)}")

        xs = np.linspace(round(min(cls.min_xy)), round(max(cls.max_xy)), X.shape[0])
        ys = xs
        XX, YY = np.meshgrid(xs, ys)
        positions = np.vstack([XX.ravel(), YY.ravel()])
        FXY = f_r(positions.T[:, 0], positions.T[:, 1], 0, 0).reshape(X.shape[0], X.shape[0])
        
        cls.contour = cls.ax.contour(XX, YY, FXY, [0], colors="black")

        cls.ax.legend(bbox_to_anchor=(0,1.02,1,0.2), 
                  loc="lower left", mode="expand", 
                  borderaxespad=0, ncol=3)

        cls.ax.set(xlabel=cls.xlabel, ylabel=cls.ylabel, 
                   xlim=cls.x1_lim, ylim=cls.x2_lim)

        return cls.contourf, cls.contour, cls.y_label


    @classmethod
    def plot_frame(cls, a, b):
        """Ultimo frame de una animacion"""

        for tp, rp in zip(cls.contourf.collections, cls.contour.collections):
            tp.remove()
            rp.remove()

        X = cls.X[:, 1:]

        cls.pred_y = np.vectorize(f_r)(cls.grid[:, 0], cls.grid[:, 1], a, b)
        cls.pred_y = np.clip(cls.pred_y, -1, 1).reshape(cls.xx.shape)
        cls.pred_y = np.array([np.vectorize(signo)(yi) for yi in cls.pred_y])

        cls.ax.clear()
        cls.contourf = cls.ax.contourf(cls.xx, cls.yy, cls.pred_y, 1, cmap='RdBu',vmin=-1, vmax=1)

        colors_it = iter(cls.colors)
        for label in np.unique(cls.y):
            x_label = X[cls.y == label]
            cls.ax.scatter(x_label[:, 0], x_label[:, 1], s=50, linewidth=1.25,
                       color=next(colors_it), edgecolor="white", 
                       label=f"Etiqueta {int(label)}")


        xs = np.linspace(round(min(cls.min_xy)), round(max(cls.max_xy)), X.shape[0])
        ys = xs
        XX, YY = np.meshgrid(xs, ys)
        positions = np.vstack([XX.ravel(), YY.ravel()])
        FXY = f_r(positions.T[:, 0], positions.T[:, 1], a, b)

        # Nuevo contour
        cls.contour = cls.ax.contour(XX, YY, FXY.reshape(X.shape[0], X.shape[0]), [0], colors="black")

        cls.ax.legend(bbox_to_anchor=(0,1.02,1,0.2), 
                  loc="lower left", mode="expand", 
                  borderaxespad=0, ncol=3)

        cls.ax.set(xlabel=cls.xlabel, ylabel=cls.ylabel, 
                   xlim=cls.x1_lim, ylim=cls.x2_lim)

        return cls.contourf, cls.contour


    @classmethod
    def _update(cls, i, ws):
        """Actualiza la animacion en el frame i-esimo"""
        
        if i < len(ws): 
            it, w = ws[i]
            # Pendiente y ordenada en el origen
            a, b = -w[1]/w[2], -w[0]/w[2]

            cls.plot_frame(a, b)
            cls.y_label.set_text(f"It {it}: y={a:.2f}x + {b:.2f}")

        else: # Ultimo frame
            it, w = ws[-1]
            a, b = -w[1]/w[2], -w[0]/w[2]

            collections = cls.plot_frame(a, b)

            #cls.linean.set_data([], [])
            cls.y_label.set(x=0.33, color="g", fontsize=11)
            cls.y_label.set_text(f"It {it}: y={a:.2f}x + {b:.2f}")

    
        return cls.contourf, cls.contour, cls.y_label

    def render(self, ws, skip=None, first=None, last=None):
        """Abre una ventana con la animacion"""

        # Reducir longitud de la animacion
        its = len(ws)
        if first is None and last is not None:
            ids = list(range(its-last, its))
            ws_anim = list(zip(ids, ws[-last:]))
        elif first and last is not None:
            ids = list(range(first)) + list(range(its-last, its))
            ws_anim = list(zip(ids, ws[:first] + ws[-last:]))
        elif first is None and last is None:
            ws_anim = list(enumerate(ws))

        if skip is not None:
            ws_anim = ws_anim[skip:]

        self.anim = FuncAnimation(self.fig, Animation.update, fargs=(ws_anim, ),
                                  repeat=False,
                                  frames=len(ws_anim)+1, interval=self.interval, 
                                  blit=False)
        
    @staticmethod
    def show():
        plt.show()


def homogeneizar(X):
    """Añade una columna inicial de unos a una matriz"""

    columna_unos = np.ones((len(X), 1))
    return np.hstack((columna_unos, X))

def tabla_resultados(X, y, max_iter=10_000, figname1=None, figname2=None, 
                     animation=False, linewidth=1, interval=75):

    """Genera tabla de resultados para PLA"""
    
    vini = np.array([0, 0, 0])
    ws, it, errs0 = ajusta_PLA(X, y, max_iter, vini)
    err = errs0[-1]
    w0 = ws[-1]
    err_percent = err / len(X) * 100
    err_ini = errs0[0] / len(X) * 100

    print("Vector inicial | Iteraciones | Coeficientes w | Error de clasificación") 
    print("----- | ----- | ----- | ---- ")
    print(f"$[{vini[0]:.3f}, {vini[1]:.3f}, {vini[2]:.3f}]$ | ${it}$ | $[{w0[0]:.2f},{w0[1]:.2f},{w0[2]:.2f}]$ | ${err_percent}\%$")

    if animation:
        anim = Animation(X, y, interval=interval,
                        size=25, x1_lim=(-50, 50), x2_lim=(-50, 50), 
                        animname="perceptron")
        anim.render(ws)
        anim.show()

    fig, ax = plt.subplots()
    xs = range(len(errs0))
    ax.set_xlabel("Número de iteración")
    ax.set_ylabel("% Error de clasificación")
    ax.plot(xs, errs0, linewidth=linewidth)
    plt.show()
    plt.close()
    

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

        print(f"$[{vini[0]:.3f}, {vini[1]:.3f}, {vini[2]:.3f}]$ | ${it}$ | $[{w0[0]:.2f},{w0[1]:.2f},{w0[2]:.2f}]$ | ${err_percent}\%$")

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
print("Ejercicio 2.1a \n")

X = homogeneizar(X)
y = y_original

y_s = y.copy()
y_s.shape = (len(y_s), 1)
warnings.filterwarnings("ignore")

tabla_resultados(X, y, max_iter=10_000, figname1="Figure_9",
                 figname2="Figure_10", animation=True, interval=50)


input("\n--- Pulsar tecla para continuar ---\n")

print("Ejercicio 2.1b \n")
# Reutilizamos apartado 2b del Ej 1. (10% Ruido no l.s)
# Con datos en forma homogénea
y = y_noise

y_s = y.copy()
y_s.shape = (len(y_s), 1)
max_iter = 5_000

tabla_resultados(X, y, max_iter=max_iter, figname1="Figure_11",
                 figname2="Figure_12", linewidth=0.3, 
                animation=False)


input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 2.2: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

def error_clas(X, y, w):
    """
    Calcula el porcentaje de error de clasificacion

    Las entradas negativas son aquellas donde los signos iniciales eran distintos.
    (Elementos mal clasificados)
    """
    signos = y * (X @ w)
    return 100 * len(signos[signos < 0]) / len(signos)

def error_RL(X, y, w):
    """Calcula el error de entropía cruzada E_in para Regresión Logística"""

    return np.mean(np.log(1 + np.exp(-y * X.dot(w))))

def grad_ein_RL(xi, yi, w):
    """Calcula el gradiente de Ein para Regresion Logistica"""

    return -yi*xi / (1 + np.exp(yi*w.dot(xi)))

def sgdRL(X, y, lr, max_iter, vini=np.zeros(3), epsilon=0.01):
    """Algoritmo de regresión logística. Basado en SGD Batch 1 elemento

    :param X: matriz homogénea de datos
    :param y: vector de etiquetas
    :param lr: learning rate (tasa de aprendizaje eta)
    :param max_iter: máximo numero de iteraciones
    :param vini: vector inicial de pesos. (0, 0, 0) por defecto

    :returns: histórico de pesos ws, y número de iteraciones para converger it
        (ws, it)
    """

    it = 0
    X_ids = np.arange(len(X))

    # vini = (0, 0, 0) por defecto
    w = vini
    ws = []
    w_dif = np.Infinity

    while it < max_iter and w_dif >= epsilon:
        w_old = w.copy()
        X_ids = np.random.permutation(X_ids)

        # Batch de tamaño 1
        for batch_id in X_ids:
            grad = grad_ein_RL(X[batch_id], y[batch_id], w)
            w = w - lr*grad

        ws.append(w)
        w_dif = np.linalg.norm(w_old - w)

        it += 1  # Numero de epocas

    return ws, it

print("Ejercicio 2.2 \n")
print("Ejecución de experimento (100 repeticiones)")

E_ins = []
E_outs = []
E_clas_ins = []
E_clas_outs = []
epocas = []

print("Repetición | $E_{in}$ | $E_{out}$ | $E_{in}^{clas}$ (\%) | $E_{out}^{clas}$ (\%) |  Épocas ")
print("---------- | -------- | -------- | ------- |  ----------- |  ------")

def experimento_RL(rep=100, N=100, M=1000):
    """Realiza N repeticiones del siguiente experimento:
    1. Simular conjunto de datos uniformes de N puntos en [0, 2]
    2. Ejecutar RL para encontrar la función g y evaluar E_out usando una
    muestra nueva de M elementos

    Se verán los resultados obtenidos por RL en la primera iteración.
    
    Imprime tabla de resultados junto con los promedios. O
    """

    for i in range(rep):
        # Nuevo conjunto de datos χ = [0, 2] x [0, 2]
        # N = 100
        # Recta aleatoria para clasificar
    
        X_train = homogeneizar(simula_unif(N, 2, [0, 2]))
        X_test = homogeneizar(simula_unif(M, 2, [0, 2]))
    
        # Etiquetar los puntos train
        a, b = simula_recta([0, 2])
        y_train = np.fromiter((f(x1, x2, a, b) for _, x1, x2 in X_train), np.int64)
    
        # Etiquetar puntos test
        y_test = np.fromiter((f(x1, x2, a, b) for _, x1, x2 in X_test), np.int64)
        
        eta = 0.01
        max_iter = 1000
    
        ws, it = sgdRL(X_train, y_train, eta, max_iter)
    
        E_in = error_RL(X_train, y_train, ws[-1])
        E_clas_in = error_clas(X_train, y_train, ws[-1])
    
        E_out = error_RL(X_test, y_test, ws[-1])
        E_clas_out = error_clas(X_test, y_test, ws[-1])
    
        E_ins.append(E_in)
        E_clas_ins.append(E_clas_in)
        E_clas_outs.append(E_clas_out)
        E_outs.append(E_out)
        epocas.append(it)
    
        print(f"${i}$ | ${E_in:.3f}$ | ${E_out:.3f}$ | ${E_clas_in} \%$ | ${E_clas_out} \%$ | ${it}$")
    
        # Mostrar animacion y figuras del primero
        if i == 0: 
            anim = Animation(X_train, y_train, interval=100,
                            size=25, x1_lim=(0, 2), x2_lim=(0, 2), 
                            animname="RegresionLogistica")
            anim.render(ws, skip=45, first=90, last=10)
            anim.show()
    
            input("\n--- Pulsar tecla para continuar ---\n")
    
            # Mostramos clasificación para conjunto de entrenamiento y test
            w = ws[-1]
            a, b = -w[1]/w[2], -w[0]/w[2]
    
            scatter_label_line(X_train[:, 1:], y_train, a, b,
                            size=5, lw=1, x1_lim=(0, 2), x2_lim=(0, 2),
                            figname="Figure_13", legend_upper=True)
    
            scatter_label_line(X_test[:, 1:], y_test, a, b,
                                size=5, lw=1, x1_lim=(0, 2), x2_lim=(0, 2),
                                figname="Figure_14", legend_upper=True)
    

    print("--------------------------")
    print("Resultados del experimento")
    print("--------------------------")

    epocas_mean = np.mean(epocas)
    E_in_mean = np.mean(E_ins)
    E_out_mean = np.mean(E_outs)
    E_clas_in_mean = np.mean(E_clas_ins)
    E_clas_out_mean = np.mean(E_clas_outs)

    print("Repeticion | E_out | E_clas_out | Epocas")
    print("---------- | ----- | ---------- | ----- ")
    print(f"Promedio:  {E_out_mean:.3f} | {E_clas_out_mean:.3f} | {epocas_mean}")
    #print(f"Promedio | {E_in_mean} | {E_clas_in_mean}"")

    epocas_min, epocas_max = np.min(epocas), np.max(epocas)
    epocas_std = np.std(epocas)
    print(f"Extremos épocas: ({epocas_min}, {epocas_max})")
    print(f"Desviación típica de épocas: {epocas_std:.3f}")


experimento_RL(rep=100, N=100, M=1000)

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################
#BONUS: Clasificación de Dígitos

print("BONUS: Clasificación de Dígitos \n")

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

print("Ejercicio 3.2a y b \n")

print("Algoritmo | $E_{in}$ | $E_{out}$ | $E_{in}^{clas}$ | E_{out}^{clas} | It")
print(" -------- | -------- | --------- | --------------- | -------------- | ---")

# LINEAR REGRESSION FOR CLASSIFICATION

def MSE(x,y,w):
    """Error cuadrático medio (MSE)
    :param x: Matriz de datos de entrada 
    :param y: Vector objetivo 
    :param w: Vector de pesos

    :return: Error cuadrático medio. (No negativo)
    """
    return np.linalg.norm(x @ w - y)**2 / len(x)

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

w = regresion_pinv(x, y)
E_in_LinR = MSE(x, y, w)
E_out_LinR = MSE(x_test, y_test, w)
E_in_clas = error_clas(x, y, w) / 100
E_out_clas = error_clas(x_test, y_test, w) / 100
w_reg = w

print(f"LINEAR REG | ${E_in_LinR:.3f}$ | ${E_out_LinR:.3f}$ | ${E_in_clas:.3f}$ | ${E_out_clas:.3f}$ | -----")

scatter_label_line(x[:, 1:], y, -w[1]/w[2], -w[0]/w[2], 
                 x1_lim=[0, 1], x2_lim=[-7, -1], 
                 xlabel="Intensidad Promedio", ylabel="Simetría",
                 figname="Figure_15", title="Dígitos Manuscritos (TRAINING) Reg. Lineal",
                 legend_upper=True)

scatter_label_line(x_test[:, 1:], y_test, -w[1]/w[2], -w[0]/w[2], 
                 x1_lim=[0, 1], x2_lim=[-7, -1], 
                 xlabel="Intensidad Promedio", ylabel="Simetría",
                 figname="Figure_16", title="Dígitos Manuscritos (TEST) Reg. Lineal",
                 legend_upper=True)


# PERCEPTRON LEARNING ALGORITHM

v_ini = np.array([0, 0, 0])
max_iter = 1_000
ws, it, errs = ajusta_PLA(x, y, max_iter, v_ini)
w = ws[-1]
E_in_PLA = error_clas(x, y, w) / 100
E_out_PLA = error_clas(x_test, y_test, w) / 100

print(f"PLA | ${E_in_PLA:.3f}$ | ${E_out_PLA:.3f}$ | ${E_in_PLA:.3f}$ | ${E_out_PLA:.3f}$ | ${it}$ ")

scatter_label_line(x[:, 1:], y, -w[1]/w[2], -w[0]/w[2], 
                 x1_lim=[0, 1], x2_lim=[-7, -1], 
                 xlabel="Intensidad Promedio", ylabel="Simetría",
                 figname="Figure_17", title="Dígitos Manuscritos (TRAINING) PLA",
                 legend_upper=True)

scatter_label_line(x_test[:, 1:], y_test, -w[1]/w[2], -w[0]/w[2], 
                 x1_lim=[0, 1], x2_lim=[-7, -1], 
                 xlabel="Intensidad Promedio", ylabel="Simetría",
                 figname="Figure_18", title="Dígitos Manuscritos (TEST) PLA",
                 legend_upper=True)

# LOGISTIC REGRESSION FOR CLASSIFICATION

v_ini = np.array([0, 0, 0])
lr = 0.01
max_iter = 10_000
ws, it = sgdRL(x, y, lr, max_iter, v_ini)
w = ws[-1]
E_in_RL = error_RL(x, y, ws[-1])
E_out_RL = error_RL(x_test, y_test, ws[-1])

E_in_clas = error_clas(x, y, w) / 100
E_out_clas = error_clas(x_test, y_test, w) / 100

print(f"RL | ${E_in_RL:.3f}$ | ${E_out_RL:.3f}$ | ${E_in_clas:.3f}$ | ${E_out_clas:.3f}$ | ${it}$")

scatter_label_line(x[:, 1:], y, -w[1]/w[2], -w[0]/w[2], 
                 x1_lim=[0, 1], x2_lim=[-7, -1], 
                 xlabel="Intensidad Promedio", ylabel="Simetría",
                 figname="Figure_19", title="Dígitos Manuscritos (TRAINING) Reg. Logística",
                 legend_upper=True)

scatter_label_line(x_test[:, 1:], y_test, -w[1]/w[2], -w[0]/w[2], 
                 x1_lim=[0, 1], x2_lim=[-7, -1], 
                 xlabel="Intensidad Promedio", ylabel="Simetría",
                 figname="Figure_20", title="Dígitos Manuscritos (TEST) Reg. Logística",
                 legend_upper=True)


# PERCEPTRON LEARNING POCKET ALGORITHM (PLA-POCKET)

def ajusta_PLA_POCKET(datos, label, max_iter, vini):
    """Algoritmo de aprendizaje del Perceptrón versión POCKET
    :param datos: matriz homogénea de caracteristicas
    :param label: vector de etiquetas
    :param max_iter: número máximo de iteraciones
    :param vini: valor inicial del vector

    :returns: lista coeficientes w, iteraciones, error de clasificacion
    """
    w = vini
    w_old = None
    it = 0
    ws = []
    E_in = np.Infinity

    while it < max_iter:
        w_old = w
        E_in_old = E_in

        for x, y in zip(datos, label):
            if signo(w.T @ x) != y:
                w_new = w + y * x
                E_in = error_clas(datos, label, w_new)

                if E_in < E_in_old:
                    w = w_new

        ws.append(w)
        it += 1

        if np.allclose(w_old, w):
            break

    return ws, it


v_ini = np.array([0, 0, 0])
max_iter = 1_000
ws, it = ajusta_PLA_POCKET(x, y, max_iter, v_ini)
w = ws[-1]
E_in_PLA_POCKET = error_clas(x, y, ws[-1]) / 100
E_out_PLA_POCKET = error_clas(x_test, y_test, ws[-1]) / 100

print(f"PLA-POCKET | ${E_in_PLA_POCKET:.3f}$ | ${E_out_PLA_POCKET:.3f}$ | ${E_in_PLA_POCKET:.3f}$ | ${E_out_PLA_POCKET:.3f}$ | ${it}$ ")

scatter_label_line(x[:, 1:], y, -w[1]/w[2], -w[0]/w[2], 
                 x1_lim=[0, 1], x2_lim=[-7, -1], 
                 xlabel="Intensidad Promedio", ylabel="Simetría",
                 figname="Figure_21", title="Dígitos Manuscritos (TRAINING) PLA-POCKET",
                 legend_upper=True)

scatter_label_line(x_test[:, 1:], y_test, -w[1]/w[2], -w[0]/w[2], 
                 x1_lim=[0, 1], x2_lim=[-7, -1], 
                 xlabel="Intensidad Promedio", ylabel="Simetría",
                 figname="Figure_22", title="Dígitos Manuscritos (TEST) PLA-POCKET",
                 legend_upper=True)


input("\n--- Pulsar tecla para continuar ---\n")

print("Ejercicio 3.2c \n")

# Usando el vector de pesos obtenidos por regresión lineal para
# inicializar los algoritmos.

print("Usando pesos de regresión lineal como vector inicial")
print("Algoritmo | $E_{in}$ | $E_{test}$ | $E_{in}^{clas}$ | E_{test}^{clas} | It")
print(" -------- | -------- | ---------- | --------------- | --------------- | --")

# PERCEPTRON LEARNING ALGORITHM

v_ini = w_reg
max_iter = 1_000
ws, it, errs = ajusta_PLA(x, y, max_iter, v_ini)
w = ws[-1]
E_in_PLA = error_clas(x, y, w) / 100
E_out_PLA = error_clas(x_test, y_test, w) / 100

print(f"PLA | ${E_in_PLA:.3f}$ | ${E_out_PLA:.3f}$ | ${E_in_PLA:.3f}$ | ${E_out_PLA:.3f}$ | ${it}$ ")

scatter_label_line(x[:, 1:], y, -w[1]/w[2], -w[0]/w[2], 
                 x1_lim=[0, 1], x2_lim=[-7, -1], 
                 xlabel="Intensidad Promedio", ylabel="Simetría",
                 figname="Figure_23", title="Dígitos Manuscritos (TRAINING) PLA",
                 legend_upper=True)

scatter_label_line(x_test[:, 1:], y_test, -w[1]/w[2], -w[0]/w[2], 
                 x1_lim=[0, 1], x2_lim=[-7, -1], 
                 xlabel="Intensidad Promedio", ylabel="Simetría",
                 figname="Figure_24", title="Dígitos Manuscritos (TEST) PLA",
                 legend_upper=True)

lr = 0.01
max_iter = 10_000
v_ini = w_reg
ws, it = sgdRL(x, y, lr, max_iter, v_ini)
w = ws[-1]
E_in_RL = error_RL(x, y, ws[-1])
E_out_RL = error_RL(x_test, y_test, w)

E_in_clas = error_clas(x, y, w) / 100
E_out_clas = error_clas(x_test, y_test, w) / 100

# LOGISTIC REGRESSION FOR CLASSIFICATION

print(f"RL | ${E_in_RL:.3f}$ | ${E_out_RL:.3f}$ | ${E_in_clas:.3f}$ | ${E_out_clas:.3f}$ | ${it}$")

scatter_label_line(x[:, 1:], y, -w[1]/w[2], -w[0]/w[2], 
                 x1_lim=[0, 1], x2_lim=[-7, -1], 
                 xlabel="Intensidad Promedio", ylabel="Simetría",
                 figname="Figure_25", title="Dígitos Manuscritos (TRAINING) Reg. Logística",
                 legend_upper=True)

scatter_label_line(x_test[:, 1:], y_test, -w[1]/w[2], -w[0]/w[2], 
                 x1_lim=[0, 1], x2_lim=[-7, -1], 
                 xlabel="Intensidad Promedio", ylabel="Simetría",
                 figname="Figure_26", title="Dígitos Manuscritos (TEST) Reg. Logística",
                 legend_upper=True)


# PERCEPTRON LEARNING POCKET ALGORITHM (PLA-POCKET)

v_ini = w_reg
max_iter = 1_000
ws, it = ajusta_PLA_POCKET(x, y, max_iter, v_ini)
w = ws[-1]
E_in_PLA_POCKET = error_clas(x, y, w) / 100
E_out_PLA_POCKET = error_clas(x_test, y_test, w) / 100

print(f"PLA-POCKET | ${E_in_PLA_POCKET:.3f}$ | ${E_out_PLA_POCKET:.3f}$ | ${E_in_PLA_POCKET:.3f}$ | ${E_out_PLA_POCKET:.3f}$ | ${it}$ ")

scatter_label_line(x[:, 1:], y, -w[1]/w[2], -w[0]/w[2], 
                 x1_lim=[0, 1], x2_lim=[-7, -1], 
                 xlabel="Intensidad Promedio", ylabel="Simetría",
                 figname="Figure_27", title="Dígitos Manuscritos (TRAINING) PLA-POCKET",
                 legend_upper=True)

scatter_label_line(x_test[:, 1:], y_test, -w[1]/w[2], -w[0]/w[2], 
                 x1_lim=[0, 1], x2_lim=[-7, -1], 
                 xlabel="Intensidad Promedio", ylabel="Simetría",
                 figname="Figure_28", title="Dígitos Manuscritos (TEST) PLA-POCKET",
                 legend_upper=True)


# COTA SOBRE EL ERROR
# Vease Memoria Apartado 3d 