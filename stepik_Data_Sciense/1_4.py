import numpy as np
import pandas as pd

student_performance = pd.read_csv('https://stepik.org/media/attachments/course/4852/StudentsPerformance.csv')

student_performance.head(7)

student_performance.tail(7)

student_performance.describe()

student_performance.dtypes

student_performance.groupby('gender').aggregate({'writing score' : 'mean'})

student_performance.iloc[0:3, [0, 3, 5]]

student_performance_with_names = student_performance.iloc[[0, 3, 4, 7, 8]]
student_performance_with_names.index = ['Cersei', 'Tywin', 'Gregor', 'Joffrey', 'Ilyn Payne']       #переименовать строки
student_performance_with_names

student_performance_with_names.loc[{'Cersei', 'Joffrey'}]

type(student_performance_with_names)

x = student_performance_with_names.iloc[:, 0]
x
type(x)

pd.Series([1, 2, 3], index = ['Cersei', 'Tywin', 'Gregor'])                 #одномерный массив с именами. А DataFrame в свою очередь, это объединённые Series

my_series_1 = pd.Series([1, 2, 3], index = ['Cersei', 'Tywin', 'Gregor'])
my_series_2 = pd.Series([6, 7, 9], index = ['Cersei', 'Tywin', 'Gregor'])
pd.DataFrame({'col_name_1' : my_series_1, 'col_name_2' : my_series_2})


# %% codecell
request1 = student_performance_with_names['gender']
request1
type(request1)

request2 = student_performance_with_names[['gender']]
request2
type(request2)
#Таким образом в зависимости от того, как мы формируем запрос, на выходе у нас могут получиться разные типы данных.
#Это имеет значение т. к. допустим:
request1.shape
request2.shape

# %% codecell

student_performance.size
#df.select_dtypes(include=types_to_include, exclude=types_to_exclude) - возвращает часть датафрэйма, куда были включены колонки с типами, указанными в include, или исключены колонки с типами, указанными в exclude
student_performance.select_dtypes(include=int, exclude=None)
#df.index - возвращает коллекцию с индексом всех строк
student_performance.index
student_performance.columns
student_performance.dtypes.value_counts()
student_performance.dtypes

# %% codecell
titanic = pd.read_csv(r'C:\programming\Data_Science\working\stepik_Data_Sciense\titanic.csv')
titanic

titanic.shape
titanic.dtypes.value_counts()
