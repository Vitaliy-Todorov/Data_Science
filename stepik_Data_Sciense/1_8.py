import pandas as pd
import numpy as np

my_data = pd.DataFrame()
my_data['type'] = ['A', 'A', 'B', 'B']
my_data['value'] = [10, 14, 12, 23]
my_data

# %% codecell
my_stat = pd.read_csv('https://stepik.org/media/attachments/course/4852/my_stat.csv')
my_stat.head(7)

subset_1 = my_stat.loc[:9, {'V1', 'V3'}]
subset_1

subset_2 = my_stat.loc[:, {'V2', 'V4'}].drop([0, 4])
subset_2

subset_1 = my_stat.loc[(my_stat['V1'] > 0) & (my_stat.V3 == 'A')]
subset_1

subset_2 = my_stat.loc[(my_stat['V2'] != 10) | (my_stat.V4 >= 1)]
subset_2

my_stat['V5'] = my_stat['V1'] + my_stat['V4']
my_stat

my_stat['V6'] = np.log(my_stat['V2'])
my_stat

my_stat.drop(columns = ['V5', 'V6'])

my_stat = my_stat.rename(columns = {'V1' : 'session_value',
        'V2' : 'group',
        'V3' : 'time',
        'V4' : 'n_users'})
my_stat

# %% codecell

my_stat = pd.read_csv('https://stepik.org/media/attachments/course/4852/my_stat_1.csv')
my_stat.head(7)

my_stat = my_stat.fillna(0)
my_stat

my_stat_mean = my_stat.n_users.loc[my_stat.n_users >= 0].median()
my_stat.loc[my_stat.n_users < 0, 'n_users'] = my_stat_mean
my_stat

# %% codecell
my_stat = pd.read_csv('https://stepik.org/media/attachments/course/4852/my_stat_1.csv')

mean_session_value_data = my_stat.groupby('group', as_index=False).agg({'session_value' : 'mean'})
mean_session_value_data
mean_session_value_data = mean_session_value_data.rename(columns={'session_value' : 'mean_session_value'})
mean_session_value_data
