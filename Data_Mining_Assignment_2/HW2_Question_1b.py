import numpy as np
import pandas as pd
from pandas import DataFrame

#pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# This method is to fo Z score normalization for any DataFrame.
def normalize_training_dataframe(data_frame):
    normalized_df = data_frame.copy()
    for col in data_frame.columns:
        col_mean = data_frame[col].mean()
        col_std = data_frame[col].std()
        normalized_df[col] = (data_frame[col] - col_mean) / col_std
    
    return normalized_df

# This method is to fo Z score normalization for Test DataFrame using TrainingDF.
def normalize_test_dataframe(test_df, train_df):
    normalized_df = test_df.copy()
    for col in test_df.columns:
        col_mean = train_df[col].mean()
        col_std = train_df[col].std()
        normalized_df[col] = (test_df[col] - col_mean) / col_std
    
    return normalized_df
                
def calculate_distances_dataframe(train_df , test_df):
    appended_data = []
    for index, row in enumerate(test_df.itertuples(index=False, name='Pandas')):
        distance_series = calculate_euclidean_distance(train_df, row)
        test_row_indices = [index] * len(distance_series.values)
        
        data_for_final_df = {
            'train_row_index': distance_series.index,
            'distance': distance_series.values,
            'test_row_index': test_row_indices
        }
        df_each_test_row_distance = pd.DataFrame(data_for_final_df)
        appended_data.append(df_each_test_row_distance)
    
    distance_df = pd.concat(appended_data, axis=0)
    
    return distance_df
    
def calculate_euclidean_distance(train_df, test_row):
    distances = (((train_df.sub(test_row, axis=1))**2).sum(axis=1))**0.5
    distances.sort_values(ascending=True, inplace=True)
    
    return distances

train_df = pd.read_csv("spam_train.csv")
test_df = pd.read_csv("spam_test.csv")

# Preprocessing: Dropping identifier and label columns
test_df.drop(test_df.columns[0], axis=1, inplace=True)
train_labels = train_df[['class']].copy()
test_labels = test_df[['Label']].copy()
train_df.drop('class', axis=1, inplace=True)
test_df.drop('Label', axis=1, inplace=True)

train_dfNormalized = normalize_training_dataframe(train_df)  
test_dfNormalized = normalize_test_dataframe(test_df,train_df)

# Calculate distances between train and test rows
distance_df = calculate_distances_dataframe(train_dfNormalized , test_dfNormalized)

# Iterate over different values of k to find the best k for KNN
k_values = [1, 5, 11, 21, 41, 61, 81, 101, 201, 401]
accuracy_list = []

for k in range(len(k_values)) :
    indexCounter = 0
    predictedLabel = []
    for row in test_dfNormalized.itertuples(index=False, name='Pandas'):
        distance_dfForRow = distance_df[ distance_df['test_row_index'] == indexCounter]
        nnIndex = distance_dfForRow.loc[:(k_values[k] - 1) ,'train_row_index']
        tmp = train_labels.iloc[nnIndex]['class'].value_counts()
    
        predictedLabel.append(tmp.idxmax())
        indexCounter += 1
    
    tmpList = {'Label' : predictedLabel}
    predictedTestLabel = pd.DataFrame(tmpList)

    differenceLabel = test_labels.sub(predictedTestLabel , axis=1)
    accurateClassCount = len(differenceLabel[ differenceLabel['Label'] ==0 ])
    accuracyPercent = accurateClassCount/test_labels['Label'].count()*100
    print(f'Accuracy for k = {k_values[k]}: {(accurateClassCount/test_labels["Label"].count())*100} %')
    
    accuracy_list.append(accuracyPercent)

tempDict = {'k' : k_values, 'Accuray %' : accuracy_list}
accuracyDF = pd.DataFrame(tempDict)
