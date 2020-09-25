import numpy as np
import pandas as pd

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
#Меняет внешний вид графика
sns.set(rc = {'figure.figsize' : (9, 6)})

event_data = pd.read_csv('C:\programming\Data_Science\working\stepik_Data_Sciense\event_data_train.csv')
event_data.head(10)

#Изучаем данные

# %% codecell
#Смотрим какие значения может прнимать столбец action
event_data.action.unique()

#Представляем timestamp в види читаемого формата данных
event_data['date'] = pd.to_datetime(event_data.timestamp, unit = 's')
event_data.head()

event_data.dtypes

event_data.date.min()
event_data.date.max()

# %% codecell
event_data['day'] = event_data.date.dt.date
event_data.head()

event_data_dey_user = event_data.groupby('day').user_id.nunique()
event_data_dey_user.head()

event_data_dey_user.plot()

# %% codecell
#Неправильное решение, мы не учитываем тех пользователе , что не решили ни одного step
step_passed_user = event_data[event_data.action == 'passed'] \
        .groupby('user_id', as_index=False).agg({'step_id' : 'count'}) \
        .rename(columns={'step_id' : 'step_passed'})
step_passed_user.head()
step_id_passed_user.step_passed.hist()

#Правильное решение
#index - задаёт строки, columns - задаёт столбци.
#values - значения (параметр) по которому будем проводить анализ.
#aggfunc - функция которую мы применяем к values.
#fill_value - Значение для замены отсутствующих значений
step_id_passed_user = event_data.pivot_table(index = 'user_id', columns = 'action',
            values = 'step_id', aggfunc = 'count', fill_value = 0) \
            .reset_index()
step_id_passed_user.head()

step_id_passed_user.discovered.hist()
