from scipy.io import arff
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import time

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def normalize_df(dataFrame):
    df_normalized = dataFrame.copy()
    column_list = list(dataFrame.columns)
    
    for col in range(len(column_list)):
        column_mean = dataFrame[column_list[col]].mean()
        column_std = dataFrame[column_list[col]].std()
        df_normalized[column_list[col]] = (dataFrame[column_list[col]] - column_mean)/column_std
    
    return df_normalized

def get_predicted_class_using_KNN(train_df, test_row, train_labels):
    kValue = 7

    #This DF will have the distance sorted (ascending)
    distance_df = calculate_euclidean_distance(train_df , test_row)
    kRows = distance_df.iloc[:kValue]
    
    distances = train_labels.iloc[kRows.index.tolist()]['CLASS'].value_counts()
    
    return distances.idxmax()

def calculate_euclidean_distance(train_df, test_row):
    distances = (((train_df.sub( test_row, axis=1))**2).sum(axis=1))**0.5
    distances.sort_values(axis=0, ascending=True, inplace=True)
    return distances

start_time = time.process_time()

data = arff.loadarff("veh-prime.arff")
train_df = pd.DataFrame(data[0])
train_labels = train_df[['CLASS']].copy()
train_df.drop('CLASS' , axis=1, inplace=True)

# Updating noncar as 0 and car as 1.
train_labels['CLASS'] = np.where(train_labels['CLASS'] == b'noncar', 0, 1)
print(train_labels)

# Z score normalization
train_df_normalized = normalize_df(train_df) 

feature_list = train_df.columns.tolist()
remaining_features_list = train_df.columns.tolist()
print(feature_list)

selected_features_list = []
attain_accuracy = 0
iteration = 1
print("********* Starting feature selection using wrapper method (with empty set of feature) **********")
while (len(remaining_features_list) > 0):  
    print("Iteration = ", iteration)
    iteration += 1
    distances_accuracy_list = []
    for counter in range(len(remaining_features_list)):
        distnaces_feature_list = selected_features_list + [remaining_features_list[counter]]
        distancestrain_df = train_df_normalized[distnaces_feature_list]
        index = 0
        accuracy_count = 0
        predicted_class_list = []
        for row in distancestrain_df.itertuples(index=False):
            distnaces_df = distancestrain_df.drop(index)
            predicted_class = get_predicted_class_using_KNN(distnaces_df, row, train_labels) 
            predicted_class_list.append(predicted_class)
        
        predicted_test_label_df = pd.DataFrame({"CLASS" : predicted_class_list})
        differenceLabel = train_labels.sub(predicted_test_label_df , axis=1)
        accuracy_count = len(differenceLabel[ differenceLabel['CLASS'] ==0 ])
        
        distances_accuracy_list.append(round(((accuracy_count/len(train_df_normalized))*100),2))
    
    print("Features    = ", remaining_features_list)
    print("Accuracies  = ", distances_accuracy_list)  
    
    maximum_accuracy = max(distances_accuracy_list)
    maximum_accuracy_index = distances_accuracy_list.index(max(distances_accuracy_list))
    maximum_accuracy_feature = remaining_features_list[maximum_accuracy_index]
        
    print("Maximum Accuracy achieved is ", maximum_accuracy, "%, with feature ",maximum_accuracy_feature)
    if(maximum_accuracy >= attain_accuracy):
        selected_features_list.append(maximum_accuracy_feature)
        remaining_features_list.remove(maximum_accuracy_feature)
        attain_accuracy = maximum_accuracy
        print("New Selected feature subset is ", selected_features_list)
    else:
        print("Accuracy is not increased from the previous feature set, Breaking the iteration")
        break
    
print("Final Selected Feature set is ,", selected_features_list)
print("Final Accuracy with above feature set is ", attain_accuracy)

print('Total Time taken is ', (time.process_time() - start_time))
