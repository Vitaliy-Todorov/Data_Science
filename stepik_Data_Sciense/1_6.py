import  pandas as pd
import numpy as np

students_performance = pd.read_csv(r'C:\programming\Data_Science\working\stepik_Data_Sciense\StudentsPerformance.csv')
students_performance = students_performance.rename(columns = {
        'parental level of education': 'parental_level_of_education',
        'test preparation course': 'test_preparation_course',
        'math score': 'math_score', 'reading score': 'reading_score',
        'writing score': 'writing_score'})
students_performance.head()

#группируем по переменной
students_performance.groupby('gender')

#находим среднее для групперующей переменной
students_performance.groupby('gender').mean()

#Выполнить несколько функций. выведем среднее для столбца 'math score', и среднеквадротичное для 'reading score'
students_performance.groupby('gender').aggregate({'math_score': 'mean', 'reading_score': 'std'})

#Изменяем внешний вид (убераем лишние строки) -  as_index = False. rename - переименовываем столбци
students_performance.groupby('gender', as_index = False) \
    .aggregate({'math_score': 'mean', 'reading_score': 'std'}) \
    .rename(columns={'math_score' : 'mean_math_score', 'reading_score' : 'std_reading_score'})

#Группируем по нескольким переменным
students_performance.groupby(['gender', 'race/ethnicity'], as_index = False) \
.aggregate({'math_score': 'mean', 'reading_score': 'std'})

#мульти индексы
mean_std_scores = students_performance.groupby(['gender', 'race/ethnicity']) \
.aggregate({'math_score': 'mean', 'reading_score': 'std'})

mean_std_scores
#Мы выдем, что теперь наши индексы состоят из нескольких уровней: 'gender' и 'race/ethnicity'
mean_std_scores.index

mean_std_scores.loc[[('female', 'group A'), ('female', 'group B')]]

#все уникальные значения в столбце 'math_score'
students_performance.math_score.nunique()

#Уникальные значения в каждой из групп
students_performance.groupby(['gender', 'race/ethnicity']).math_score.nunique()

#Упорядочиваем по столбцам 'gender', 'math_score'. ascending=False - Порядок сортировки, от большего к меньшему
students_performance.sort_values(['gender', 'math_score'], ascending=False)

#пять верхних столбцов каждой из групп
students_performance.sort_values(['gender', 'math_score'], ascending=False).groupby(['gender', 'math_score']).head(5)

#Сщздание новых столбцов
students_performance['total_score'] = students_performance.math_score \
        + students_performance.reading_score \
        + students_performance.writing_score
students_performance

students_performance = students_performance.assign(total_score_log = np.log(students_performance.total_score))
students_performance

students_performance.drop(['total_score', 'lunch'], axis = 1)

#упрожнения
mDota2 = pd.read_csv('https://stepik.org/media/attachments/course/4852/dota_hero_stats.csv')
mDota2.head()

mDota2.groupby('legs').size()

mDota2.groupby(['attack_type', 'primary_attr']).size()

loopa_poopa = pd.read_csv('https://stepik.org/media/attachments/course/4852/accountancy.csv')
loopa_poopa.head(10)

loopa_poopa.groupby(['Executor', 'Type']).Salary.mean()

concentrations = pd.read_csv('http://stepik.org/media/attachments/course/4852/algae.csv')
concentrations

# %% codecell
concentrations.groupby('genus').agg(['min', 'mean', 'max']).loc['Fucus', 'alanin']

concentrations.groupby('group', as_index=False) \
        .aggregate({"species":'nunique',
                'citrate':'var',
                'sucrose': np.ptp}) \
        .rename(columns={'species': 'glucose_count',
                'citrate': 'citrate_var',
                'sucrose': 'sucrose_scope'})
