import pandas as pd, numpy as np
from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from scipy import stats
df_train = pd.read_csv('train_knn_norm.csv')
df_test = pd.read_csv('test_knn_norm.csv')
len(df_train.columns) == len(df_test.columns)

X_train = df_train.iloc[:,:-1]
y_train = df_train.iloc[:,-1]
X_test = df_test.iloc[:,:-1]
y_test = df_test.iloc[:,-1]

def rforest(X_train, y_train, X_test):

    rforest = BaggingClassifier(base_estimator = RandomForestClassifier())
    rforest.fit(X_train, y_train)
    y_pred = rforest.predict(X_test)
    
    return y_pred
def knn(X_train, y_train, X_test):
    
    knn = BaggingClassifier(base_estimator = KNeighborsClassifier())
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    return y_pred
def nb(X_train, y_train, X_test):
    
    nb = BaggingClassifier(base_estimator = GaussianNB())
    nb.fit(X_train,y_train)
    y_pred = nb.predict(X_test)
    
    return y_pred

def ensemble(X_train, y_train, X_test):
    
    r_y = rforest(X_train, y_train, X_test)
    k_y = knn(X_train, y_train, X_test)
    n_y = nb(X_train, y_train, X_test)
    
    final_y = []
    
    for i in range(len(X_test)):
        final_y.append(stats.mode([r_y[i], k_y[i], n_y[i]])[0][0])
        
    return final_y

trial_run = ensemble(X_train, y_train, X_test)
for i in range(4):
    print(['tn', 'fp', 'fn', 'tp'][i],confusion_matrix(trial_run, y_test).ravel()[i])

print(accuracy_score(trial_run, y_test), precision_score(trial_run, y_test), recall_score(trial_run, y_test), f1_score(trial_run, y_test))