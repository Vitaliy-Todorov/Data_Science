import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
#Меняет внешний вид графика
sns.set(rc = {'figure.figsize' : (9, 6)})



# %% codecell
event_data = pd.read_csv('C:\programming\Data_Science\working\stepik_Data_Sciense\event_data_train.csv')
submissions_data = pd.read_csv('C:\programming\Data_Science\working\stepik_Data_Sciense\submissions_data_train.csv')
submissions_data.head()

event_data['date'] = pd.to_datetime(event_data.timestamp, unit = 's')
event_data['day'] = event_data.date.dt.date
event_data.head()

submissions_data['date'] = pd.to_datetime(submissions_data.timestamp, unit = 's')
submissions_data['day'] = submissions_data.date.dt.date
submissions_data.head()

#pivot_table() - Группируем данные и выполняем для групп указанную функцию (в данном случай count)
#index - задаёт строки, columns - задаёт столбци.
#values - значения (параметр) по которому будем проводить анализ.
#aggfunc - функция которую мы применяем к values.
#fill_value - Значение для замены отсутствующих значений
user_scores = submissions_data.pivot_table(index='user_id', columns='submission_status',
        values='step_id', aggfunc='count', fill_value=0).reset_index()

users_data = event_data.groupby('user_id', as_index = False).agg({'timestamp': 'max'}) \
        .rename(columns={'timestamp': 'last_timestamp'})

now = 1526772811
#После этого времени считаем что ученик дропнулся
drop_out_threshold = 30 * 24 * 60 * 60
#Разность между текущим днём (1526772811) и последним посещением
users_data['is_gone_user'] = (now - users_data.last_timestamp) > drop_out_threshold

#merge() - соединяет два детафрейма users_data и user_scores
users_data = users_data.merge(user_scores, on = 'user_id', how = 'outer')
#Заполним пропущенные данные нулём
users_data = users_data.fillna(0)

#Число удачно пройденных степиков
user_event_data = event_data.pivot_table(index = 'user_id', columns = 'action',
            values = 'step_id', aggfunc = 'count', fill_value = 0)

users_data = users_data.merge(user_event_data, on = 'user_id', how = 'outer')

#число уникальных дней
users_days = event_data.groupby('user_id').day.nunique() \
        .to_frame().reset_index()
#.to_frame().reset_index() - Перейти от серии к датафрейму. Обновить индексы
users_data = users_data.merge(users_days, how = 'outer')

#Успешно прошедшие курс (прошедшие более 170 степиков)
users_data['passed_corse'] = users_data.passed > 170
users_data.shape



 # %% codecell
 users_data[users_data.passed_corse].day.hist()

 #Находим когда пользователь впервые зашёл на курс
 user_min_time = event_data.groupby('user_id', as_index=False) \
        .agg({'timestamp': 'min'}) \
        .rename({'timestamp': 'min_timestamp'}, axis = 1)
user_min_time.head()

users_data = users_data.merge(user_min_time, how = 'outer')
users_data.head()

event_data_train = pd.DataFrame()

#отбираем действия совершённые в течение первых трёх дней

# не правильный метод, работает не определённо долго
# for user_id in users_data.user_id:
#     min_user_time = users_data[users_data.user_id == user_id].min_timestamp.item()
#     time_threshold = min_user_time + 3*24*60*60
#
#     users_event_data = event_data[(event_data.user_id == user_id) & (event_data.timestamp < time_threshold)]
#     event_data_train = event_data_train.append(users_event_data)

#--------------------

# event_data_train = event_data.merge(users_data[['user_id', 'min_timestamp']], on='user_id', how='left') \
#         .query("(timestamp - min_timestamp) < (3 * 24 * 60 * 60)")
# event_data_train.shape

#event_data.merge(user_min_time, on='user_id', how='left') - дополняем event_data стобцоми из user_min_time
#how='left' - объединяем таким образом, что если в event_data нет строки из user_min_time, то мы вставляем её из user_min_time с пустыми значениями

event_data_train = event_data.merge(user_min_time, on='user_id', how='left')
event_data_train['time_from_beginning'] = event_data_train['timestamp'] - event_data_train['min_timestamp']
event_data_train = event_data_train[event_data_train.time_from_beginning <= 3*24*60*60]

#Более короткий вариант
# event_data_train = event_data.merge(user_min_time, on='user_id', how='left')
# event_data_train = event_data[event_data.timestamp <= event_data_train.min_timestamp + 3*24*60*60]

#(1014985, 8)
event_data_train.shape



# %% codecell
#Определяем после какого шага пользователи застревают и уходят с курса
submissions_data.head()

#Сортируем по времени
complex_task = submissions_data.sort_values(['timestamp'])
#Группируем по id и берём последний элеммент в группе (тоесть последнеее действие пользователя)
complex_task = complex_task.groupby(['user_id'], group_keys=False).nth([-1])
#осталяем только те строки, где последнее дестиве было неправильнм
complex_task = complex_task[complex_task.submission_status == "wrong"]
#Групперуем по шагам, считаем сколько раз этот шаг был последним для пользователя на этом урсе, сортируем по убыванию
complex_task.groupby(['step_id'], group_keys=False).nunique().sort_values(['timestamp'], ascending = False)



# %% codecell
# Возвращаемся к анализу действий пользователя за первые три дня
# Убедимся, что мы отобрали только те действия, что произведены в первые 3 дня (ответ 4)
event_data_train.groupby('user_id').day.nunique().max()

#Добавляем в submissions_data время первого степа
submissions_data = submissions_data.merge(user_min_time, on='user_id', how='left')
# Время от первого степа до последнего
submissions_data['users_time'] = submissions_data['timestamp'] - submissions_data['min_timestamp']
#Выбираем те степы, первых трёх дней
submissions_data_train = submissions_data[submissions_data.users_time <= 3*24*60*60]
submissions_data_train.groupby('user_id').day.nunique().max()

submissions_data_train.head()
submissions_data_train.shape



# %% codecell

X = submissions_data_train.groupby('user_id').day.nunique() \
        .to_frame().reset_index().rename({'day': 'days'})
X.head()
X.shape

steps_tried = submissions_data_train.groupby('user_id').step_id.nunique() \
        .to_frame().reset_index().rename({'step_id': 'step_tried'})
steps_tried.head()
X.shape

X = X.merge(steps_tried, on = 'user_id', how = 'outer')
X.head()
X.shape

#добавляем correct, wrong
X = X.merge(submissions_data_train.pivot_table(index='user_id',
                                columns='submission_status',
                                values='step_id',
                                aggfunc='count',
                                fill_value=0).reset_index())
X.head()
X.shape

X['correct_ratio'] = X.correct / (X.correct + X.wrong)
X.head()
X.shape

#Добавляем discovered, passed, started_attempt, viewed
X = X.merge(event_data_train.pivot_table(index='user_id',
                                columns='action',
                                values='step_id',
                                aggfunc='count',
                                fill_value=0).reset_index(), how = 'outer')
X.head()
X.shape

#Вмета Nan ставим 0
X = X.fillna(0)

X = X.merge(users_data[['user_id', 'passed_corse', 'is_gone_user']], how = 'outer')
X.head()
X.shape

#Выберем тех пользователей, что ещё не бросили, но ещё и не закончили курс
#X = X[(X.is_gone_user == False) & (X.passed_corse == False)]
X = X[X.is_gone_user | X.passed_corse]
X.head()
X.shape

X.groupby(['passed_corse', 'is_gone_user']).user_id.count()

y = X.passed_corse
X = X.drop(['passed_corse', 'is_gone_user'], axis = 1)
