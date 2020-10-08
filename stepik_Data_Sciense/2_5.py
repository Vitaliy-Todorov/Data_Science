from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score

# %% codecell
songs = pd.read_csv(r'C:\programming\Data_Science\working\stepik_Data_Sciense\songs.csv')
songs.head()

songs.isnull().sum()
len(songs)

X = songs.drop(['artist', 'lyrics'], axis = 1)
X = pd.get_dummies(X)
y = pd.get_dummies(songs.artist).gangstarr
X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.3)

# score_data = pd.DataFrame()

# for max_deph in range(1, 100):
#     clf = tree.DecisionTreeClassifier(max_depth = max_deph, criterion = 'entropy')
#     clf.fit(X_train, y_train)
#     train_score = clf.score(X_train, y_train)
#     test_score = clf.score(X_test, y_test)
#     medan_cross_val_score = cross_val_score(clf, X_train, y_train).mean()
#
#     temp_score_data = pd.DataFrame({'train_score': [train_score],
#             'test_score': [test_score], 'medan_cross_val_score': [medan_cross_val_score]})
#     score_data = score_data.append(temp_score_data)
#
#  score_data.head()
# tree.plot_tree(clf, fontsize=10, feature_names=list(X), filled=True)

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
precision = precision_score(predictions, y_test, average='micro')
