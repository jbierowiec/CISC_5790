from scipy.io import arff
import numpy as np
import pandas as pd
from pandas import DataFrame as df

def calculate_PCC(x_df, y_df, y_sum_squared, y_mean):
    x_sum_squared = np.sum(np.square(x_df))
    sum_of_products = np.sum( x_df * y_df )
    x_mean = np.mean(x_df)
    
    x_pop_sd = np.sqrt( (x_sum_squared / float(len(x_df))) - (x_mean**2) )
    y_pop_sd = np.sqrt( (y_sum_squared / float(len(y_df))) - (y_mean**2) )
    xy_cov = ( (sum_of_products / len(y_df)) - (x_mean * y_mean) )
    
    correlation = ( xy_cov / (x_pop_sd * y_pop_sd) )
    
    return correlation

data = arff.loadarff("veh-prime.arff")
train_df = pd.DataFrame(data[0])

# Updating noncar as 0 and car as 1.
train_df['CLASS'] = np.where(train_df['CLASS'] == b'noncar', 0, 1)

# Calculating Sum Squared Y (For class lebel)
y_sum_squared = np.sum(np.square(train_df['CLASS']))
y_mean = np.mean(train_df['CLASS'])

pcc_list = []
abspcc_list = []
feature_list = []
for counter in range(len(train_df.columns) - 1):
    feature_list.append(train_df.columns[counter])
    
    pcc = calculate_PCC(train_df[train_df.columns[counter]], train_df['CLASS'], y_sum_squared, y_mean)
    pcc_list.append( pcc )
    abspcc_list.append( np.abs(pcc) )
    
temp_dict = {'feature' : feature_list , 'pcc' : pcc_list, '|pcc|' : abspcc_list}
pcc_df = pd.DataFrame(temp_dict)
print(pcc_df.sort_values(['|pcc|'] , ascending=0))
