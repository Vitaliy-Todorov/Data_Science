import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.datasets import load_iris

# %% codecell
dt = DecisionTreeClassifier(max_depth=5, min_samples_split=5)

# %% codecell
train_data_tree = pd.read_csv(r'C:\programming\Data_Science\working\stepik_Data_Sciense\train_data_tree.csv')
train_data_tree.head()

train_data_tree.isnull().sum()

X = train_data_tree.drop(['num'], axis=1)
y = train_data_tree.num

clf = DecisionTreeClassifier(criterion='entropy')

clf.fit(X, y)

tree.plot_tree(clf, fontsize=10, feature_names=list(X), filled=True)

# %% codecell
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# %% codecell
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train, y_train)
dt.score(X_test, y_test)

predict = dt.predict(X_test)

# %% codecell
dt = DecisionTreeClassifier()
parametrs = {'max_depth': range(1, 10), 'min_samples_split': range(2, 10), 'min_samples_leaf': range(1, 10)}
search = GridSearchCV(dt, parametrs)
search.fit(X_train, y_train)

best_tree = search.best_estimator_
best_tree.score(X_test, y_test)

# %% codecell
#Более быстрый, но менее надёжный способ, проходит не по всем данным а лишь по случайным
dt = DecisionTreeClassifier()
parametrs = {'max_depth': range(1, 10), 'min_samples_split': range(2, 10), 'min_samples_leaf': range(1, 10)}
search = RandomizedSearchCV(dt, parametrs)
search.fit(X_train, y_train)

best_tree = search.best_estimator_
best_tree.score(X_test, y_test)

X_train = train.drop(['y'], axis = 1)
y_train = train.y

dt = DecisionTreeClassifier()
parametrs = {'max_depth': range(1, 10), 'min_samples_split': range(2, 10), 'min_samples_leaf': range(1, 10)}
search = GridSearchCV(td, parametrs)
search.fit(X_train, y_train)

best_tree = search.best_estimator_
predictions = best_tree.predict(tast)
