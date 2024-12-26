import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
from pylab import rcParams

rcParams['figure.figsize'] = 12, 10
font = {'color':  'black',
        'size': "100",
        }

df = pd.read_csv("Encoded-census-income.data_WithImputation_UsingRandomForest.csv")
df1 = pd.read_csv("census-income.data_WithImputation_UsingRandomForest.csv") 
ncount = len(df1)

print(len(df1[df1["Class"] == " <=50K"]))
print(len(df1[df1["Class"] == " >50K"]))
print(len(df1))
plt.figure(figsize=(12,8))
ax = sb.countplot(x='Class',data=df1, palette='hls')
plt.title('Data Imbalance')
plt.xlabel('Class')

for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom') # set the alignment of the text


plt.show()