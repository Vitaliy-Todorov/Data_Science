import  pandas as pd
import numpy as np

student_performance = pd.read_csv(r'C:\programming\Data_Science\working\stepik_Data_Sciense\StudentsPerformance.csv')
student_performance

#Показать в каких строках gender принимает значение 'female'
student_performance.gender == 'female'

type(student_performance.gender == 'female')

#вывести стодбци со сторома, в которых gender принимает значение 'female'
student_performance.loc[student_performance.gender == 'female', ['gender', 'writing score']]
