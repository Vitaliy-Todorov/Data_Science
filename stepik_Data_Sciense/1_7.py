# %% codecell
import  pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

students_performance = pd.read_csv(r'C:\programming\Data_Science\working\stepik_Data_Sciense\StudentsPerformance.csv')
students_performance = students_performance.rename(columns = {
        'parental level of education': 'parental_level_of_education',
        'test preparation course': 'test_preparation_course',
        'math score': 'math_score', 'reading score': 'reading_score',
        'writing score': 'writing_score'})
students_performance.head()

# %% codecell
students_performance.math_score.hist()

students_performance.plot.scatter(x = 'math_score', y = 'reading_score')

sns.lmplot(x = 'math_score', y = 'reading_score', data = students_performance)

sns.lmplot(x = 'math_score', y = 'reading_score', hue = 'gender', data = students_performance)

# %% codecell
ax = sns.lmplot(x = 'math_score', y = 'reading_score', hue = 'gender', data = students_performance, fit_reg = False)

ax.set_xlabels('Math score')
ax.set_ylabels('Reading score')

# %% codecell
df = pd.read_csv('https://stepik.org/media/attachments/course/4852/income.csv')
df

sns.lineplot(x=df.index, y=df.income)

df.plot(kind='line')

sns.lineplot(data=df)

df.income.plot()

df.plot()

df['income'].plot()

plt.plot(df.index, df.income)


# %% codecell

df = pd.read_csv(r'C:\programming\Data_Science\working\stepik_Data_Sciense\dataset_209770_6.txt', sep = ' ')
df.head()
sns.scatterplot(df.x, df.y)

# %% codecell
df = pd.read_csv(r'C:\programming\Data_Science\working\stepik_Data_Sciense\genome_matrix.csv', index_col=0)
df.head()

g = sns.heatmap(df, cmap="viridis")
g.xaxis.set_ticks_position('top')
g.xaxis.set_tick_params(rotation=90)
g

# %% codecell
mDota2 = pd.read_csv(r'C:\programming\Data_Science\working\stepik_Data_Sciense\dota_hero_stats.csv')
mDota2.head()

roles = mDota2.groupby('name').agg({'roles' : lambda x: (x[100].count(',') + 1)})
roles.head()
roles.hist()

mDota2.roles.str.split(',').apply(len)
#[x.count(',')+1 for x in mDota2.roles]

# %% codecell
iris_df = pd.read_csv('C:\programming\Data_Science\working\stepik_Data_Sciense\iris.csv', index_col=0)
iris_df = iris_df.rename(columns={'sepal length' : 'sepal_length',
        'sepal width' : 'sepal_width',
        'petal length' : 'petal_length',
        'petal width' : 'petal_width',})
iris_df.head()

for parameters in iris_df :
    sns.kdeplot(iris_df[parameters])

sns.displot(data=iris_df, kde=True)

# %% codecell
sns.violinplot(x=iris_df["petal_length"])

# %% codecell
sns.pairplot(iris_df, hue="species", markers=["o", "s", "D"], height=1.5)

iris_df.corr()

# %% codecell
type(list(students_performance))

X = ['1', '2', '8', '11']
type(X)
[i*2 for i in X if '1' in i]
