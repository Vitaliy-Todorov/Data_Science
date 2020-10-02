from sklearn import tree
from scipy.stats import entropy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# %% codecell
#data = pd.read_csv('example_data.csv')
data = pd.DataFrame({'X_1': [1, 1, 1, 0, 0, 0, 0, 1], 'X_2': [0, 0, 0, 1, 0, 0, 0, 1], 'Y': [1, 1, 1, 1, 0, 0, 0, 0]})
data

#Создаём дерево решений
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf

#Входные данные это обычно датафрейм, а выход, это вектор
X = data[['X_1', 'X_2']]
y = data.Y

clf.fit(X, y)

tree.plot_tree(clf, feature_names=list(X),
               class_names=['Negative', 'Positive'],
               filled=True);

# %% codecell
dogs_df = pd.read_csv(r'C:\programming\Data_Science\working\stepik_Data_Sciense\dogs.csv')
dogs_df = dogs_df.rename(columns = {'Unnamed: 0': 'Unnamed:0', 'Лазает по деревьям': 'Лазает_по_деревьям'})
dogs_df.head(10)

clf = tree.DecisionTreeClassifier(criterion = 'entropy')

# %% codecell
input = dogs_df[['Unnamed:0', 'Шерстист', 'Гавкает', 'Лазает_по_деревьям']]
output = dogs_df.Вид

clf.fit(input, output)

tree.plot_tree(clf, feature_names=list(input), class_names=['Negative', 'Positive'], filled = True)

# %% codecell
cats_df = pd.read_csv(r'C:\programming\Data_Science\working\stepik_Data_Sciense\cats.csv')
cats_df = cats_df.rename(columns = {'Unnamed: 0': 'Unnamed:0', 'Лазает по деревьям': 'Лазает_по_деревьям'})
cats_df.head(10)

clf = tree.DecisionTreeClassifier(criterion = 'entropy')

input = cats_df[['Unnamed:0', 'Шерстист', 'Гавкает', 'Лазает_по_деревьям']]
output = cats_df.Вид

clf.fit(input, output)

#feature_names - наименование вершин
tree.plot_tree(clf, feature_names=list(input), class_names=['Negative', 'Positive'], filled = True)

# %% codecell
cats_df.head(10)

#Энтропия
N = cats_df[cats_df.Шерстист == 1].Вид.count()
p_sh_dog = cats_df[(cats_df.Вид == 'собачка') & (cats_df.Шерстист == 1)].Вид.count() / N
p_sh_cat = cats_df[(cats_df.Вид == 'котик') & (cats_df.Шерстист == 1)].Вид.count() / N
sh_1 = - p_sh_dog * math.log2(p_sh_dog) - p_sh_cat * math.log2(p_sh_cat)
sh_1

df = cats_df
def ent(data):
  return entropy(data.Вид.value_counts() / len(data), base=2)

print('Шерстист на 0: ', ent(df[df.Шерстист == 0]))
print('Шерстист на 1: ', ent(df[df.Шерстист == 1]))

#IG
def IG(function):
    N = len(cats_df)
    p_0 = len(cats_df[function == 0]) / N
    p_1 = len(cats_df[function == 1]) / N

    E_Y_X = p_0 * ent(df[function == 0]) + p_1 * ent(df[function == 1])
    E = ent(df)
    return E - E_Y_X

IG(df.Шерстист)
IG(df.Гавкает)
IG(df.Лазает_по_деревьям)
