import numpy as np
import matplotlib.pyplot as plt
import time

#Preparar datos
X = np.random.uniform(size=(100, 2))

def fr(x, y, a, b):
    return y - a*x - b

def fz(grid, a, b):
    return grid[:, 1] - a*grid[:, 0] - b

def signo(x):
    if x >= 0:
        return 1
    else:
        return -1

min_xy = X.min(axis=0)
max_xy = X.max(axis=0)
border_xy = (max_xy-min_xy)*0.001

#Generar grid de predicciones
xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                    min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]

a,b = -3, 2

y = np.array([signo(fr(x,y, a, b)) for x,y in X])


pred_y = fz(grid, a, b)
# pred_y[(pred_y>-1) & (pred_y<1)]
#pred_y = np.array([np.vectorize(signo)(yi) for yi in pred_y]).reshape(xx.shape)
pred_y = np.vectorize(signo)(pred_y).reshape(xx.shape)

print(pred_y)

#Plot
f, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
ax_c = f.colorbar(contour, ax=ax)
ax_c.set_label('$f(x, y)$')
ax_c.set_ticks(np.linspace(y.min(axis=0), y.max(axis=0), 3))
ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
            cmap="RdYlBu", edgecolor='white')

XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
positions = np.vstack([XX.ravel(), YY.ravel()])
cts = ax.contour(XX,YY,fz(positions.T, a, b).reshape(X.shape[0],X.shape[0]),[0], colors='black')

ax.set(
    xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
    ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
    xlabel="$x$ axis", ylabel="$y$ axis")

for tp in cts.collections:
    tp.remove()

plt.show()