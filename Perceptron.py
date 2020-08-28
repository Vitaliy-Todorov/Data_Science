# %% codecell
#Сам неирон
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron(object):
    """классификатор на основе персептрона.
    Параметры
    ---------
    eta : float
        Темп обучения (между О.О и 1 .0)
    n_iter : int
        Проходы по тренировочному набору данных.

    Атрибуты
    ---------
    w_ : 1 - мерный массив
        Весовые коэффициенты после подгонки.
    errors_ : список
        Число случаев ошибочной классификации в к аждой эпохе.
    """
    def __init__(self, eta=0.01, n_iter=10) :
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Выполнить подгонку модели под тренировочные данные .

        Параметры
        ---------
        Х : {массивоподобный) , форма = [п_samples , п_ features]
            тренировочные векторы , где
            п_ samples - число об разцов и
            п_ features - число призн а ков .
        у массиво п о добный , фо рма [n_samples]
            Целевые значения.

        Возвращает
        ----------
        self : object
        """
        self.w_ = np.zeros(1 + X.shape[1])
            #shape для массивов numpy возвращает размеры массива. Если Y имеет n строки и столбцы m, то Y.shape - (n,m)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                #zip(X, y) - ставит в соответствие n-му элементу из X, n-й элемент из y (короче делает пару (x,y) )
                #xi - входной вектор
                update = self.eta * (target - self.predict(xi))
                    # шаг*(истинная - идентифицированная метки)
                    #target - идентифицированная метка класса
                self.w_[1:] += update * xi
                    #(0 или шаг)* вектор входных данных
                self.w_[0] += update
                errors += int(update != 0.0)
                    #количество ошибок в текущей итерации
            self.errors_.append(errors)
        return self

    def net_input(self, X) :
        """Рассчитать чистый вход (находит z)"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
            #dot() - вычисляет скалярное произведение двух массивов

    def predict(self, X):
        """Вернуть метку класса после единичного скачка (функцию активации)"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
            #where() - возвращает элементы, которые могут выбираться из двух массивов (в данном случаи не масивы а 1, -1) в зависимости от условия.



# %% codecell
#Получаем данные
import pandas as pd
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
df.tail()
    #tail(n) - Эта функция возвращает последние n строк из объекта




# %% codecell
#Визуализируем данные
import matplotlib.pyplot as plt
import numpy as np

y = df.iloc[0:100, 4].values
    #X.iloc - преобразует X в DataFrame
    #DataFrame.values - Возвращает Numpy-представление DataFrame
    #y - список в котором собержится информация о том к какому классу относиться цветок
y = np.where(y == 'Iris-setosa', -1, 1)
    #Переходим от Iris-setosa и Iris-versicolor к -1 и 1
X = df.iloc[0:100, [0, 2]].values
plt.scatter(X[:50, 0], X[:50, 1],
           color='red', marker='o', label='щетинистый')
    #scatter - График разброса, здесь мы добовляем на него точки
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='разноцветный')
plt.xlabel('длина чвшелистика')
plt.ylabel('длина лепистка')
    #Называем оси
plt.legend(loc='upper left')
plt.show()




# %% codecell
#Обучаем модель
ppn = Perceptron(eta=0.1, n_iter=10)
    #Создаём персептрон
ppn.fit(X, y)
    #Обучаем персептрон
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    #plot(Эпоха(ось x), число ошибок(ось y)) -
plt.xlabel('Эпоха')
#Число ошибочно классифицированных случаев во время обновлений
plt.ylabel('Число случаев ошибочной классифицированных')
plt.show()




# %% codecell
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=.02) :
    # настроить генератор маркеров и палитру
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
        #ListedColormap() - создаёт палитру (обект в котором содержится набор цветов)
        #unique() - находит уникальные элементы массива и возвращает их в отсортированном массиве

    #Вывести поверхность решения
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                          np.arange(x2_min, x2_max, resolution))
                          #meshgrid() создает список массивов координатных сеток N-мерного координатного пространства для указанных одномерных массивов
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        #ravel() - возвращает однострочный массив (составляем старого массива, одномерны массив)
    Z = Z.reshape(xx1.shape)
        #Z.reshape((m,n)) - заполняем матрицу замером m на n элемтами массива Z.
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        #contourf() - нарисуйте контурные линии и заполненные контуры
    plt.xlim(xx1.min(), xx1.max())
        #xlim() - Получите или установите пределы x текущих осей
    plt.ylim(xx2.min(), xx2.max())

    # показать образцы классов
    for idx, cl in enumerate(np.unique(y)):
        #enumerate() - (нумерует элементы колекции) создает объект, который генерирует кортежи, состоящие из двух элементов - индекса элемента и самого элемента.
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
            #X[y == cl, 0] - если y == cl, то берём элемент из 0 стобца
                    alpha=0.8, color=cmap(idx), marker=markers[idx], label=cl)
                        #scatter() - график разброса (добавляет точки на график)

plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('дпина чашелистика [см]')
plt.ylabel('длина лепестка [см]')
plt.legend(loc='upper left')
plt.show()
