from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# %% codecell
titanic_data = pd.read_csv(r'C:\programming\Data_Science\working\stepik_Data_Sciense\train.csv')
titanic_data.head()

#Считаем скольлько у нас есть nan в каждой из колонок
titanic_data.isnull()
titanic_data.isnull().sum()


X = titanic_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis = 1)
y = titanic_data.Survived
X.head()

#Переводим строковые значение в числовые
X = pd.get_dummies(X)
X.head()

#Заполняем nan
X_median = X.Age.median()
#X.fillna(2) - поставить вместа nan X_median во всей таблице
X = X.fillna({'Age': X_median})

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

# %% codecell
plt.figure(figsize=(100, 25))
tree.plot_tree(clf, fontsize=10, feature_names=list(X), filled=True)

# %% codecell

#Разделяем данные на обучающие и тестовые
from sklearn.model_selection import train_test_split
#test_size - показывает в какой пропорции разделить набор данных для включения в тестовое разбиеие
#random_state - Управляет перетасовкой, применяемой к данным перед применением разделения
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

#Проверка обучения
#обучаем
clf.fit(X_train, y_train)
clf.score(X_train, y_train)
#проверяем обучение на тестовой выборке
clf.score(X_test, y_test)

# %% codecell
#Ограничем количество ветвлений 3
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth = 3)

clf.fit(X_train, y_train)
clf.score(X_train, y_train)

clf.score(X_test, y_test)
