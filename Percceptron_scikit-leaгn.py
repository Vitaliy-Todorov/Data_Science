# %% codecell
#Получаем данные

from sklearn import datasets
import  numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

#%% codecell
#Разделяем массив данных на тренировочный и тестовый

from sklearn.model_selection import train_test_split

#train_test_split - мы произвольным образом разделяем массивы Х и у на тестовые данные в размере 30% от об щего объема (45 образцов) и тренировочные данные в размере 70% (105 образцов).
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size = .3, random_state = 0)

# %% codecell
#Cтандартизация

# В приведенном ниже примера мы загрузили класс StandardScaler из модуля предобработки
# и инициализировали новый объект StandardScaler, который мы присвоили
# переменной sc. Затем мы воспользовались методом fit объекта StandardScaler,
# чтобы вычислить параметры μ (эмпирическое среднее) и cr (стандартное отклонение)
# для каждой размерности признаков из тренировочных данных. Вызвав метод
# transform, мы затем стандартизировали тренировочные данные, используя для этого
#расчетные параметры μ и cr.
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# %% codecell
from sklearn.linear_model import Perceptron

ppn = Perceptron(n_iter_no_change = 40, eta0 = .1, random_state = 0)
    #n_iter_no_change - Количество итераций без каких-либо улучшений, чтобы дождаться ранней остановки
ppn.fit(X_train_std, y_train)

# Проверка. Проверим насколько точно наша модель классифицирует
#у_test - это истинные метки классов,
y_pred = ppn.predict(X_test_std)
#ошибки классификации и метрика верности
print('Чиcлo ошибочно классифицированных образцов : %d' % (y_test != y_pred).sum())
n = (y_test != y_pred).sum()
p = n*100 / len(y_test)
print('Ошибки классификации : ', p, '%')
print('Метрика верности : ', (100-p), '%')

from sklearn.metrics import accuracy_score

print('Bepнocть: %.2f ' %accuracy_score(y_test, y_pred))

# %% codecell

from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, test_idx = None, esolution = .02) :
    markers = ('s', 'x', 'o', '^', 'v ')
    colors = ('red', 'Ьlue', 'lightgreeп' , 'gray', 'суап')
    cmap = ListedColormap(colors[:list(np.unique(y))])

    #Определяем поверхность
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()].T))
    Z = Z.reshape(xx1.shape)

    #Строим график
    plt.contourf(xx1, xx2, X, alpha = .4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    #Добавляем точки
    for idx, cl in enumerate(np.unique(y)) :
        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1], 
                alpha = .8, color = cmap(idx), marker = markers[idx], label = cl)

        # показать все образцы
        if test_idx :
            X_test, y_test = X[test_idx, :], y[test_idx]
            plt.scatter(X_test[:, 0], X_test[: 1], c = '', alpha = 1,
                    linewidths = 1, marker = 'o', s = 55, label = 'тестовый набор')
