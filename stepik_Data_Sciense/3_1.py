import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score

# %% codecell
titanic_data = pd.read_csv(r'C:\programming\Data_Science\working\stepik_Data_Sciense\train.csv')
titanic_data.head()

X = titanic_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis = 1)
y = titanic_data.Survived

#Переводим строковые значение в числовые
X = pd.get_dummies(X)
X.head()

X_median = X.Age.median()
#X.fillna(2) - поставить вместа nan X_median во всей таблице
X = X.fillna({'Age': X_median})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# %% codecell
#max_depth - Максимальная глубина ветвления
#min_samples_split - Минимальное число образцов в узле, чтобы его можно было разделить на 2
#min_samples_leaf - Минимальное число образцов в листьях (при получившемся значении ниже разделение не будет произведено)
#min_impurity_decrease - ожидаемое минимальное уменьшение неопределенности (IG). С увеличением min_impurity_decrease уменьшается переобучаемость модели
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth = 3, min_samples_split = 100, min_samples_leaf=10)
clf.fit(X_train, y_train)

plt.figure(figsize=(125, 40))
tree.plot_tree(clf, fontsize=100, feature_names=list(X), filled=True)
