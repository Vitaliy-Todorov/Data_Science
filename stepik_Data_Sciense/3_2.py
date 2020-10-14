import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV



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
#Создание леса деревьев
clf_rf = RandomForestClassifier()
#n_estimators - число деревьев которое мы будем использовать
parametrs = {'n_estimators': [10, 20, 30], 'max_depth': [1, 5, 7, 10]}
grid_search_CV_clf = GridSearchCV(clf_rf, parametrs, cv = 5)
grid_search_CV_clf.fit(X_train, y_train)
grid_search_CV_clf.best_params_

best_clf = grid_search_CV_clf.best_estimator_

best_clf.score(X_test, y_test)
#feature_importances_ - возвращает вектор "важностей" (Information Gain) признаков
feature_importances = best_clf.feature_importances_
feature_importances_df = pd.DataFrame({'feature': list(X_train),
        'feature_importances': feature_importances})
feature_importances_df.sort_values('feature_importances', ascending = False)



# %% codecell
heart = pd.read_csv(r'C:\programming\Data_Science\working\stepik_Data_Sciense\heart.csv')
heart.head()

X = heart.drop(['target'], axis = 1)
y = heart.target

np.random.seed(0)

rf = RandomForestClassifier(10, max_depth=5)
rf.fit(X, y)

imp = pd.DataFrame(rf.feature_importances_, index=X.columns, columns=['importance'])
imp.sort_values('importance').plot(kind='barh', figsize=(12, 8))
