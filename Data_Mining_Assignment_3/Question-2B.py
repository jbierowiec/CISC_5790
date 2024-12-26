from scipy.io import arff
import numpy as np
import pandas as pd
from pandas import DataFrame as df

def normalize_df(dataFrame):
    df_normalized = dataFrame.copy()
    column_list = list(dataFrame.columns)
    
    for col in range(len(column_list)):
        column_mean = dataFrame[column_list[col]].mean()
        column_std = dataFrame[column_list[col]].std()
        df_normalized[column_list[col]] = (dataFrame[column_list[col]] - column_mean)/column_std
    
    return df_normalized

def calculate_PCC(x_df, y_df, y_sum_squared, y_mean):
    x_sum_squared = np.sum(np.square(x_df))
    sum_of_products = np.sum( x_df * y_df )
    x_mean = np.mean(x_df)
    
    x_pop_sd = np.sqrt( (x_sum_squared / float(len(x_df))) - (x_mean**2) )
    y_pop_sd = np.sqrt( (y_sum_squared / float(len(y_df))) - (y_mean**2) )
    xy_cov = ( (sum_of_products / len(y_df)) - (x_mean * y_mean) )
    
    correlation = ( xy_cov / (x_pop_sd * y_pop_sd) )
    
    return correlation

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

data = arff.loadarff("veh-prime.arff")
train_df = pd.DataFrame(data[0])
train_labels = train_df[['CLASS']].copy()
train_df.drop('CLASS' , axis=1, inplace=True)

# Updating noncar as 0 and car as 1.
train_labels['CLASS'] = np.where(train_labels['CLASS'] == b'noncar', 0, 1)

# Z score normalization
train_df_normalized = normalize_df(train_df) 

# Calculating Sum Squared Y (For class lebel)
y_sum_squared = np.sum(np.square(train_labels['CLASS']))
y_mean = np.mean(train_labels['CLASS'])

pcc_list = []
abs_pcc_list = []
feature_list = []
for counter in range(len(train_df.columns)):
    feature_list.append(train_df.columns[counter])
    
    pcc = calculate_PCC(train_df[train_df.columns[counter]], train_labels['CLASS'], y_sum_squared, y_mean)
    pcc_list.append( pcc )
    abs_pcc_list.append( np.abs(pcc) )
    
temp_dict = {'feature' : feature_list , 'pcc' : pcc_list, 'abspcc' : abs_pcc_list}
pcc_df = pd.DataFrame(temp_dict)
pcc_df.sort_values(['abspcc'] , ascending=0 , inplace=True)

ranked_feature_list = pcc_df['feature'].tolist()

for counter in range(len(ranked_feature_list)):
    print("Selected Feature set -- ", ranked_feature_list[:counter+1])
    temp_train_df = train_df_normalized[ranked_feature_list[:counter+1]]
    index = 0
    accuracy_count = 0
    for row in temp_train_df.itertuples(index=False):
        temp_df = temp_train_df.drop(index)
        predictedClass = get_predicted_class_using_KNN(temp_df, row, train_labels) 
        if(predictedClass == train_labels.iloc[index]['CLASS']):
            accuracy_count += 1        
        index += 1
     
    print("Accuracy Count       = ", accuracy_count)
    print("Accuracy Percentage  = ", round((accuracy_count / len(train_df_normalized))*100, 2))
    print("\n")
