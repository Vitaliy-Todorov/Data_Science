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

clf = tree.DecisionTreeClassifier()
#parametrs - набор параметров для clf
parametrs = {'criterion': ['gini', 'entropy'], 'max_depth': range(1, 30)}
#GridSearchCV - лучшее решение найденное на крос ваидации
grid_search_CV_clf = GridSearchCV(clf, parametrs, cv = 5)
grid_search_CV_clf.get_params().keys()
grid_search_CV_clf.fit(X_train, y_train)

#Сохраняем лучшую модель обучения
best_clf = grid_search_CV_clf.best_estimator_
best_clf

best_clf.score(X_test, y_test)
y_pred = best_clf.predict(X_test)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)

y_predicted_prob = best_clf.predict_proba(X_test)
#Распределение вероятностей (выживания)
pd.Series(y_predicted_prob[:, 1])
pd.Series(y_predicted_prob[:, 1]).hist()

#Если значение больше 0.8, то 1, в обратном случаи 0
y_pred = np.where(y_predicted_prob[:, 1] > 0.8, 1, 0)
#Если мы увеличиваем порог, то precision растёт а recall уменьшается и наоборот
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)

# %% codecell
#True Positive Rate - сколько не выжевших пасажирова мы угадали (это recall)
#False Positive Rate - сколько не выжевших мы упустили
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_predicted_prob[:,1])
roc_auc= auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
