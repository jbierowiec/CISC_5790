import numpy as np
import pandas as pd
from pandas import DataFrame

# Setting display options for DataFrames (uncomment if needed for debugging)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def calculate_distances_dataframe(train_df, test_df):
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

# Load training and testing datasets
train_df = pd.read_csv("spam_train.csv")
test_df = pd.read_csv("spam_test.csv")

# Preprocessing: Dropping identifier and label columns
test_df.drop(test_df.columns[0], axis=1, inplace=True)
train_labels = train_df[['class']].copy()
test_labels = test_df[['Label']].copy()
train_df.drop('class', axis=1, inplace=True)
test_df.drop('Label', axis=1, inplace=True)

# Calculate distances between train and test rows
distance_df = calculate_distances_dataframe(train_df, test_df)

# Iterate over different values of k to find the best k for KNN
k_values = [1, 5, 11, 21, 41, 61, 81, 101, 201, 401]
accuracy_list = []

for k in k_values:
    predicted_labels = []
    for index in range(len(test_df)):
        distance_df_for_row = distance_df[distance_df['test_row_index'] == index]
        nearest_neighbors_indices = distance_df_for_row.iloc[:k]['train_row_index']
        
        # Majority vote for prediction
        most_common_label = train_labels.iloc[nearest_neighbors_indices]['class'].value_counts().idxmax()
        predicted_labels.append(most_common_label)
    
    # Calculate accuracy
    predictions_df = pd.DataFrame({'Label': predicted_labels})
    correct_predictions = (test_labels['Label'] == predictions_df['Label']).sum()
    accuracy = correct_predictions / len(test_labels) * 100
    print(f'Accuracy for k = {k}: {accuracy} %')
    accuracy_list.append(accuracy)

# Optionally, save or further process the accuracy results
accuracy_df = pd.DataFrame({'KValue': k_values, 'Accuracy %': accuracy_list})
print(accuracy_df)
