from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

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
max_depth_values = range(1, 100)
scores_data = pd.DataFrame()

#обучение дерева с разной глубиной. Глубина от 0 до 50
for max_depth in max_depth_values:
    clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = max_depth)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    mean_cross_val_score = cross_val_score(clf, X_train, y_train).mean()

    #Сохраняет глубину дерева и характеристику качества обучения в датафрейм
    temp_score_data = pd.DataFrame({'max_depth': [max_depth], 'train_score': [train_score],
            'test_score': [test_score], 'cross_val_score': mean_cross_val_score})
    scores_data = scores_data.append(temp_score_data)

scores_data.query("'set_type' == 'cross_val_score'").head(20)

#Перегрупперуем данные
#var_name - название колонки с переменными (ключ)
#value_name - название колонки со значениями (значение)
scores_data_long = pd.melt(scores_data, id_vars = ['max_depth'],
            value_vars = ['train_score', 'test_score', 'cross_val_score'], var_name='set_type', value_name='score')
scores_data_long.head()

sns.lineplot(x = 'max_depth', y = 'score', hue = 'set_type', data = scores_data_long)

#Переобучение может происходить из-за того, что мы используем ону и ту же обучающую и тестовую выборку при выборе глубины лбучения. Что видно при сравнении оранжевой и зелёной линии на графике, что для конкрктной выборке максимум у нас на 3 уровне ветвления, а при тестировании на большем числе выборок это уже не так (максимум при ветвлении = 10).

# %% codecell
clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 10)

#разбиваем нашу выборку на 5 фолдеров (fold). Потом наше дерево обучается на первых четырёх и предсказывает пятый (0.76666667), потому обучается на всех кроме втрого, ан на нём проверяется (0.82352941) и т д.
cross_val_score(clf, X_train, y_train, cv = 5)
cross_val_score(clf, X_train, y_train, cv = 5).mean()

# %% codecell
train_iris = pd.read_csv(r'C:\programming\Data_Science\working\stepik_Data_Sciense\train_iris.csv')
test_iris = pd.read_csv(r'C:\programming\Data_Science\working\stepik_Data_Sciense\test_iris.csv')
train_iris.head()

iris.isnull().sum()

X_train_iris = train_iris.drop(['Unnamed: 0', 'species'], axis=1)
y_train_iris = train_iris.species
X_test_iris = test_iris.drop(['Unnamed: 0', 'species'], axis=1)
y_test_iris = test_iris.species
#X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X, y, test_size = 0.3, random_state = 42)

scores_data_iris = pd.DataFrame()

for max_depth in range(1, 100):
    clf_iris = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = max_depth)
    clf_iris.fit(X_train_iris, y_train_iris)
    train_score = clf_iris.score(X_train_iris, y_train_iris)
    test_score = clf_iris.score(X_test_iris, y_test_iris)

    mean_cross_val_score = cross_val_score(clf_iris, X_train_iris, y_train_iris).mean()
    temp_score_data = pd.DataFrame({'max_depth': [max_depth],
            'train_score': [train_score],
            'test_score': [test_score],
            'mean_cross_val_score': [mean_cross_val_score]})
    scores_data_iris = scores_data_iris.append(temp_score_data)

scores_data_iris.query("'set_type' == 'cross_val_score'").head(20)

scores_scores_data_iris = pd.melt(scores_data_iris, id_vars = 'max_depth',
        value_vars = ['train_score', 'test_score', 'mean_cross_val_score'],
        var_name='set_type', value_name='score')
scores_data_long.head()

sns.lineplot(x = 'max_depth', y = 'score', hue = 'set_type', data = scores_scores_data_iris)

# %% codecell
X_test_DC = pd.read_json('https://vk.com/doc93782781_570424605')

dogs_n_cats = pd.read_csv('https://stepik.org/media/attachments/course/4852/dogs_n_cats.csv')
dogs_n_cats.shape
dogs_n_cats.head()


dogs_n_cats = pd.get_dummies(dogs_n_cats)
dogs_n_cats.head()

X_train_DC = dogs_n_cats.drop(['Вид_котик', 'Вид_собачка'], axis = 1)
y_train_DC = dogs_n_cats.Вид_собачка

clf_DC = tree.DecisionTreeClassifier(criterion='entropy', max_depth = max_depth)
clf_DC.fit(X_train_DC, y_train_DC)

prediction_DC = clf_DC.predict(X_test_DC)

dog = pd.Series(prediction_DC)[prediction_DC == 1].count()
print(dog)
