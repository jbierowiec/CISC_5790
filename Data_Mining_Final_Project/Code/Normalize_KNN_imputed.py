import pandas as pd, numpy as np
df_train = pd.read_csv('train_knn_imputed_encoded.csv')
df_test = pd.read_csv('test_knn_imputed_encoded.csv')
def z_normalize(df):
    print(df.mean(), df.std())
    res = (df - df.mean())/df.std()
    return res
df_train['Age'] = z_normalize(df_train['Age'])
df_train['fnlwgt'] = z_normalize(df_train['fnlwgt'])
df_train['capital-gain'] = z_normalize(df_train['capital-gain'])
df_train['capital-loss'] = z_normalize(df_train['capital-loss'])
df_train['hours-per-week'] = z_normalize(df_train['hours-per-week'])
df_test['Age'] = (df_test['Age'] - 38.58164675532078)/13.640432553581146 
df_test['fnlwgt'] = (df_test['fnlwgt'] - 189778.36651208502)/105549.97769702233
df_test['capital-gain'] = (df_test['capital-gain'] - 1077.6488437087312)/7385.292084839299
df_test['capital-loss'] = (df_test['capital-loss'] - 87.303829734959)/402.960218649059
df_test['hours-per-week'] = (df_test['hours-per-week'] - 40.437455852092995)/12.34742868173081
df_train.to_csv('train_knn_norm.csv')
df_test.to_csv('test_knn_norm.csv')