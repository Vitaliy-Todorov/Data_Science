import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z) :
    return 1/(1 + np.exp(-z))

z = np.arange(-7, 7, .1)
phi_z = sigmoid(z)
plt.plot(z, phi_z)
plt.axvline(0, color='k')
    #Вертикальная сплошная линия
plt.axhspan(0, 1, edgecolor = 'k', facecolor = '1', ls='--')
    #axhspan(x1, x2) - закрашивает область от x1 до x2 (можно сделать вертикально для оси y)
    #edgecolor - цвет краёв
    #facecolor - цвет внутренней части (в данном случаи - 1 прозрачный)
    #ls - вид краёв (в данном случаи пунктир)
plt.axhline(y = .5, ls = 'dotted', color = 'k')
    #Пунктирная горизонтальная линия
plt.yticks([0, .5, 1])
    #Показывает какаие координаты отображать на оси y
plt.ylim(-.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)')
plt.show()
