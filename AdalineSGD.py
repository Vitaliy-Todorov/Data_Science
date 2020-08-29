from numpy.random import seed
import numpy as np

class AdalineSGD(object) :
    '''
    Классификатор на основе ADALINE (ADAptive Linear Neuron).
    Параметры
    eta : float
    Темп обучения (между 0.0 и 1.0)
    n iter : int
    Проходы по тренировочному набору данных.
    Атрибуты
    w : 1 - мерный массив
    Веса после подгонки .
    errors : list/cпиcoк
    Число случаев ошибочной классификации в каждой эпохе.
    shuffie : bool (по умолчанию: True)
    Перемешивает тренировочные данные в каждой эпохе, если True,
    для предотвращения зацикливания .
    random_state : int ( по умолчанию: None)
    Инициализирует генератор случайных чисел
    для перемешивания и инициализации весов.
    '''
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None) :
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state :
            seed(random_state)

    def fit(self, X, y) :
        """
        Выполнить подгонку под тренировочные данные.
        Параметры
        Х : {массивоподобный), форма= [n_samples, n_features ]
        Тренировочные векторы, где
        n_samples - число образцов и
        n_features - число признаков.
        у массивоподобный , форма [n_samples]
        Целевые значения.
        Возвращает
        self : объект
        """
        self.cost_ = []
        self._initialized_weights(X.shape[1])

        for i in range(self.n_iter) :
            if self.shuffle:
                X, y = self._shuffle(X, y)
                    #Перемешать тренировочные данные
            cost = []
            for  xi, target in zip(X, y) :
                #zip(X, y) - ставит в соответствие n-му элементу из X, n-й элемент из y (короче делает пару (x,y) )
                cost.append(self._update_weights(xi, target))
                    #_update_weights() - Применить обучающее правило ADALINE, чтобы обновить веса
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y) :
        if not self.w_initialized :
            self._initialized_weights(X.shape[1])
        if y.ravel().shape[0] > 1 :
            for xi, target in zip(X, y) :
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y) :
        """Перемешать тренировочные данные"""
        r = np.random.permutation(len(y))
            #random.permutation() -  возвращает случайную перестановку элементов массива или случайную последовательность заданной длинны из его элементов.
        return X[r], y[r]

    def _initialized_weights(self, m) :
        """Инициализировать веса нулями"""
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target) :
        """Применить обучающее правило ADALINE, чтобы обновить веса"""
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X) :
        """Рассчитать чистый вход"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X) :
        """Рассчитать линейную активацию"""
        return self.net_input(X)

    def predict(self, X) :
        """Вернуть метку класса после единичного скачка"""
        return np.where(self.activation(X) >= 0.0, -1, 1)

# %% codecell

import pandas as pd

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:100, [0, 2]].values
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

# %% codecell
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution = .02) :
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha = .4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)) :
        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1],
        alpha = .8, color = cmap(idx), marker = markers[idx],label = cl)


# %% codecell
ada = AdalineSGD(n_iter = 15, eta = .01, random_state = 1)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, ada)
plt.title('ADALINE (стохастический градиентный спуск)')
plt.xlabel('длина чашелистика [стандар тизованная ]')
plt.ylabel('длина лепестка [стандарти зованная]')
plt.legend(loc = 'upper left')
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker = 'o')
plt.xlabel('Эпохи')
plt.ylabel('Cyммa квадратичных ошибок')
plt.show()
