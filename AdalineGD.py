# %% codecell
#ADALINE - пакетный градиентный спуск

import numpy as np
import pandas as pd

class AdalineGD(object) :

    def __init__ (self, eta = 0.01, n_iter = 50) :
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y) :
        self.w_ = np.zeros(X.shape[1]+1)
        self.cost_ = []

        for i in range(self.n_iter) :
            output = self.net_input(X)                                  #ф(z)
            errors = y - output                                         #y-ф(z)
            self.w_[1:] += self.eta*X.T.dot(errors)                     # wj=xjСУМi(yi - ф(zi))=dJ
            self.w_[0] += self.eta*errors.sum()
            cost = (errors**2).sum() / 2.0                                #J=1/2СУМ(y - ф(z))^2
            self.cost_.append(cost)
        return self


    def net_input(self, X) :
        '''Рассчитать чистый вход'''
        return np.dot(X, self.w_[1:] + self.w_[0])                      #X*W - произведение матриц

    def activation(self, X) :
        """Рассчитать линейную активацию"""
        return self.net_input(X)

    def predict(self, X) :
        """В е рнуть метку класса посл е е дин ичного скачка"""
        return np.where(self.activation(X) >= 0.0, 1, -1)


# %% codecell
#Получаем данные
import pandas as pd

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

# %% codecell
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))                            #plt.subplots() - (помещает несколько графиков на одном холсте) Несколько Axes на одной Figure
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)                                     #Обучаем Adaline
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')         #plot(количество прогонов (Эпох, ось x), сумма квадратов ошибок (ось y), метка)
ax[0].set_xlabel('Эпохи')
ax[0].set_ylabel('log(Сумма квадратичных ошибок)')
ax[0].set_title('ADALINE (темп обучения 0.01)')

ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Эпохи')
ax[1].set_ylabel('Сумма квадратичных ошибок')
ax[1].set_title('ADALINE (темп обучения 0.0001)')

# %% codecell
#Стандартизация

X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

# %% codecell
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=.02) :
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])                       #ListedColormap() - создаёт палитру (объект в котором содержится набор цветов)

    #Вывести поверхность решения
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                          np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)                                                #Задаём поверхность. Только я так и не понял что именно сдесь происходит.
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)                         #contourf() - нарисуйте контурные линии и заполненные контуры
    plt.xlim(xx1.min(), xx1.max())                                          #xlim() - Получите или установите пределы x текущих осей
    plt.ylim(xx2.min(), xx2.max())

    # показать образцы классов
    for idx, cl in enumerate(np.unique(y)) :                   #enumerate() - (нумерует элементы коллекции) создает объект, который генерирует кортежи, состоящие из двух элементов - индекса элемента и самого элемента.
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],                       #X[y == cl, 0] - если y == cl, то берём элемент из 0 стобца
                    alpha=0.8, color=cmap(idx), marker=markers[idx], label=cl)
                    #scatter() - график разброса (добавляет точки на график)

ada = AdalineGD(n_iter=15, eta=0.01)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada)
plt.title('ADALINE (градиентный спуск)')
plt.xlabel('длина чашелистика [стандартизованная]')
plt.ylabel('длина лепестка [стандартизованная]')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Эпохи')
plt.ylabel('Cyммa квадратичных ошибок')
plt.show()
