# -*- coding: utf-8 -*-
# Ricardo Ruiz

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data

## Ejercicio 1

# Leemos la base de datos de Iris
iris = load_iris(as_frame=True)
df = iris.frame

# Filtramos la primera y tercera columna
df = df.iloc[:, [0,2]]

# Representamos la gráfica de puntos
fig, ax = plt.subplots()

# Fijamos las etiquetas de los ejes
cols = df.columns.to_list() 
ax.set_xlabel(cols[0])
ax.set_ylabel(cols[1])

# Diccionario index-clase: color
colors = dict(enumerate(("red", "green", "blue")))

# Recorremos las diferentes clases y representamos
# cada una con su color correspondiente
for i, clsname in enumerate(iris.target_names):
    class_df = df[iris.target == i]
    x, y = class_df.iloc[:, 0], class_df.iloc[:, 1]
    ax.scatter(x, y, c=colors[i], label=clsname)

ax.legend()
plt.show()

## Ejercicio 2

# Obtenemos el número de elementos de cada clase
nsetosa = np.count_nonzero(iris.target == 0)
nversicolor = np.count_nonzero(iris.target == 1)
nvirginica = np.count_nonzero(iris.target == 2)

# Diccionario que mapea 0,1,2 con el número de clase.
ncls = dict(enumerate((nsetosa, nversicolor, nvirginica)))

# Obtenemos muestra aleatoria de 20% de índices de clase setosa
setosa_test = np.random.choice(range(nsetosa), size=int(0.2*nsetosa), 
                               replace=False)

# Obtenemos muestra aleatoria de 20% de índices de clase versicolor
versicolor_range = range(nsetosa, nsetosa + nversicolor)
versicolor_test = np.random.choice(versicolor_range, size=int(0.2*nversicolor), 
                                   replace=False)

# Obtenemos muestra aleatoria de 20% de índices de clase virginica
virginica_range = range(nsetosa + nversicolor, len(df))
virginica_test = np.random.choice(virginica_range, size=int(0.2*nvirginica), 
                                  replace=False)

# Concatenamos los índices para conformar el conjunto de test con las mismas
# proporciones de cada clase
test_indexes = np.concatenate((setosa_test, versicolor_test,  virginica_test))
np.random.shuffle(test_indexes)

# Obtenemos los índices de entrenamiento hallando la diferencia
indexes = set(range(len(df)))
training_indexes = list(indexes - set(test_indexes))
np.random.shuffle(training_indexes)

for i, clsname in enumerate(iris.target_names):
    print(f"--- Clase {clsname} ---")
    print(f"Ejemplos train: {0.8 * ncls[i]:.0f}")
    print(f"Ejemplos test: {0.2 * ncls[i]:.0f}")
    
# Imprimimos las clases de los elementos de entrenamiento
# Nota: si bien las clases son números enteros, lo imprimimos como
# tipo float para coincidir con la imagen del enunciado
print("Clase de los ejemplos de entrenamiento: ")
print(iris.target.filter(items=training_indexes, axis=0).astype(float).values)

print("Clase de los ejemplos test: ")
print(iris.target.filter(items=test_indexes, axis=0).astype(float).values)

## Ejercicio 3

x = np.linspace(0, 4*np.pi, num=100)

fig, ax = plt.subplots()

ax.plot(x, 10**(-5) * np.sinh(x), "g--", label="y=1e-5*sinh(x)")
ax.plot(x, np.cos(x), "k--", label="y=cos(x)")
ax.plot(x, np.tanh(2*np.sin(x) - 4*np.cos(x)), "r--",
        label="y=tanh(2*sin(x)-4*cos(x)")

ax.legend()
plt.show()

## Ejercicio 4

# Figura con doble de ancho que de largo
fig = plt.figure(figsize=plt.figaspect(0.45))

# -- Primer subplot --
ax = fig.add_subplot(1, 2, 1, projection='3d')


# Meshgrid y función 1
X = np.arange(-6, 6, 0.35)
Y = np.arange(-6, 6, 0.35)
X, Y = np.meshgrid(X, Y)
Z = 1 - np.abs(X+Y) - np.abs(Y-X)

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=False,
                       rstride=1, cstride=1,
                       linewidth=0.005, edgecolors='w')

ax.set_title("Pirámide")

# -- Segundo subplot --
ax = fig.add_subplot(1, 2, 2, projection='3d')


# Meshgrid y función 2
X = np.arange(-2, 2, 0.05)
Y = np.arange(-2, 2, 0.05)
X, Y = np.meshgrid(X, Y)
Z = X * Y * np.exp(-X**2 - Y**2)

surf = ax.plot_surface(X, Y, Z, cmap="viridis", antialiased=True,
                       rstride=1, cstride=1, 
                       linewidth=0.1, edgecolors='w')

ax.set_title(r"$x\cdot y \cdot e^{\left(-x^2 - y^2\right)}$")
