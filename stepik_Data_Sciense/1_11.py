import numpy as np
import pandas as pd

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
#Меняет внешний вид графика
sns.set(rc = {'figure.figsize' : (9, 6)})

event_data = pd.read_csv('C:\programming\Data_Science\working\stepik_Data_Sciense\event_data_train.csv')
submissions_data = pd.read_csv('C:\programming\Data_Science\working\stepik_Data_Sciense\submissions_data_train.csv')

event_data['date'] = pd.to_datetime(event_data.timestamp, unit = 's')
event_data['day'] = event_data.date.dt.date
event_data.head(10)

submissions_data['date'] = pd.to_datetime(submissions_data.timestamp, unit = 's')
submissions_data['day'] = submissions_data.date.dt.date
submissions_data.head(10)

# %% codecell
#что-то там считаем
#pivot_table() - Группируем данные и выполняем для групп указанную функцию (в данном случай count)
#index - задаёт строки, columns - задаёт столбци.
#values - значения (параметр) по которому будем проводить анализ.
#aggfunc - функция которую мы применяем к values.
#fill_value - Значение для замены отсутствующих значений
user_scores = submissions_data.pivot_table(index='user_id', columns='submission_status',
        values='step_id', aggfunc='count', fill_value=0).reset_index()
user_scores.head(10)

user_scores.correct.hist()
event_data[['user_id', 'day', 'timestamp']].head()
#drop_duplicates - убераем строки с похожими значениями (тоесть в 'user_id' не будет повторений)
#drop_duplicates(subset=['user_id', 'day']) - уникальные пары из пользователей и дней
#drop_duplicates(subset=['day']) - таким запросом мы оставим не по одному действию для каждого пользователя за день, а ровно одно действие за день среди всех пользователей (а не каждого).
event_data_drop_duplicates = event_data[['user_id', 'day', 'timestamp']] \
        .drop_duplicates(subset=['user_id', 'day'])

event_data_drop_duplicates.head()
#apply(list)- Помещаем все значения в список
#apply(np.diff) - Разница между ближайшими наблюдениями
event_data_drop_duplicates_timestamp = event_data_drop_duplicates.groupby('user_id')['timestamp'].apply(list)
event_data_drop_duplicates_timestamp.head()
gap_data = event_data_drop_duplicates_timestamp.apply(np.diff).values

gap_data = pd.Series(np.concatenate(gap_data, axis = 0))
gap_data = gap_data / (24 * 60 * 60)
gap_data.head()

gap_data[gap_data < 200].hist()

gap_data.quantile(0.95)

# %% codecell
#Считаем сколько правильных решений у каждого пльзователя
submissions_data_submission = submissions_data.groupby(['user_id', 'submission_status']).size()
#.to_frame().reset_index() - Перейти от серии к датафрейму. Обновить индексы
submissions_data_submission = submissions_data_submission.to_frame().reset_index()

#.sort_values() - Упорядочиваем по столбцу 0. ascending=False - Порядок сортировки, от большего к меньшему
submissions_data_submission.loc[submissions_data_submission.submission_status == 'correct'] \
        .sort_values(0, ascending=False)

# %% codecell

#tail() - возвращает последние строки
event_data.tail()

users_data = event_data.groupby('user_id', as_index = False).agg({'timestamp': 'max'}) \
        .rename(columns={'timestamp': 'last_timestamp'})
users_data.head(7)

now = 1526772811
#После этого времени считаем что ученик дропнулся
drop_out_threshold = 30 * 24 * 60 * 60
#Разность между текущим днём (1526772811) и последним посещением
users_data['is_gone_user'] = (now - users_data.last_timestamp) > drop_out_threshold
users_data.head(7)

# %% codecell
user_scores.head()
#merge() - соединяет два детафрейма users_data и user_scores
users_data.merge(user_scores).head(7)
#При таком варианте у нас теряются данные, одной из таблиц, при условии, что им нет соответствия данным в другой таблице.
#Что бы не происходило подобного, вместо отсутствующих данных будем ставить Nan
users_data.merge(user_scores, how = 'outer').head(7)
#Для того, что бы сохранить значени определённого столбца, нужно указать on = ''
users_data = users_data.merge(user_scores, on = 'user_id', how = 'outer')
#Заполним пропущенные данные нулём
users_data = users_data.fillna(0)
users_data.head(7)

#Число удачно пройденных степиков
user_event_data = event_data.pivot_table(index = 'user_id', columns = 'action',
            values = 'step_id', aggfunc = 'count', fill_value = 0)
user_event_data.head()

users_data = users_data.merge(user_event_data, on = 'user_id', how = 'outer')

#число уникальных дней
users_days = event_data.groupby('user_id').day.nunique() \
        .to_frame().reset_index()
#.to_frame().reset_index() - Перейти от серии к датафрейму. Обновить индексы
users_days.head(7)

users_data = users_data.merge(users_days, how = 'outer')

#Успешно прошедшие курс (прошедшие более 170 степиков)
users_data['passed_corse'] = users_data.passed > 170
users_data.head()

users_data.groupby('passed_corse').count()

100 * 1425 / 19234

#Проверяем, не потеряли ли мы данные
users_data.user_id.nunique()
event_data.user_id.nunique()
