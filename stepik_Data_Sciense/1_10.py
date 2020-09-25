import numpy as np
import pandas as pd

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

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
