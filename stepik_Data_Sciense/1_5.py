import  pandas as pd
import numpy as np

students_performance = pd.read_csv(r'C:\programming\Data_Science\working\stepik_Data_Sciense\StudentsPerformance.csv')
students_performance

# %% codecell
#Запросы

#Показать в каких строках gender принимает значение 'female'
students_performance.gender == 'female'

type(students_performance.gender == 'female')

#вывести столбци со сторома, в которых gender принимает значение 'female'
students_performance.loc[students_performance.gender == 'female', ['gender', 'writing score']]

mean_writing_score = students_performance['writing score'].mean()
students_performance.loc[students_performance['writing score'] > mean_writing_score]

students_performance.loc[(students_performance['writing score'] > 100) & (students_performance.gender == 'female')]

#Доля студентов у которых lunch имеет значение free/reduced
(students_performance['lunch'] == 'free/reduced').mean()

# %% codecell
#Среднее, дисперсия, средне квадратичное

lunch_reduced = students_performance.loc[students_performance['lunch'] == 'free/reduced']
lunch_standard = students_performance.loc[students_performance['lunch'] == 'standard']
# lunch_reduced_score = lunch_reduced.loc[:, ['math score', 'reading score', 'writing score']]
#вывести средне по строкам
# lunch_reduced_score.mean(axis=1)
lunch_reduced.mean()
lunch_standard.mean()

lunch_reduced.var()
lunch_standard.var()

lunch_reduced.std()
lunch_standard.std()

# %% codecell
students_performance_ = students_performance.rename(columns = {
        'parental level of education': 'parental_level_of_education',
        'test preparation course': 'test_preparation_course',
        'math score': 'math_score', 'reading score': 'reading_score',
        'writing score': 'writing_score'})

students_performance_

students_performance_.query('writing_score > 74')

students_performance_.query("gender == 'female' & writing_score > 74")

writing_score_query = 78
students_performance_.query("writing_score > @writing_score_query")

#Верно! Обратите внимание, query не применяется к колонкам, название которых содержит недопустимые символы (типа пробел, слэша). Ещё пример query, аналогичный isin() -
variants = ["bachelor's degree", "master's degree"]
students_performance_.query("parental_level_of_education == @variants")

#DataFrame.isin([x, y]) - Возвращает Series, элементами которого являются true или false, в зависимости от того, присутствует ли в строке DataFrame одно из значений x или y. Содержится ли каждый элемент в фрейме данных в значениях.
students_performance[student_stats['parental level of education'].isin(["bachelor's degree", "master's degree"])]

#Показать название всех столбцов
list(students_performance_)

#DataFrame.students_performance_() - Выбираем столбци в которых есть слово'score'
subject_score = [i for i in list(students_performance_) if 'score' in i]
subject_score
students_performance_[subject_score]

#Выбираем столбци в которых есть слово'score'
students_performance_.filter(like = 'score')

student_performance_with_names = student_performance.iloc[[0, 3, 4, 7, 8]]
student_performance_with_names.index = ['Cersei', 'Tywin', 'Gregor', 'Joffrey', 'Ilyn Payne']       #переименовать строки
student_performance_with_names

#Выбираем строки в которых есть буква 'i'
student_performance_with_names.filter(like = 'i', axis=0)
