import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
        
# Create dictionary
data = {
        'Instance':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'True_Class_Label': ['P','N','P','P','N','P','N','N','N','P'],
        'Predicted_Probability_Positive_Class': [0.95, 0.85, 0.78, 0.66, 0.60, 0.55, 0.43, 0.42, 0.41, 0.4]
       }

# Convert data dictionary to dataframe
df = pd.DataFrame(data) # df

df['Pred_Class_Label'] = np.where(df['Predicted_Probability_Positive_Class'] >= 0.5, 'P', 'N')

from sklearn.metrics import confusion_matrix
true = df['True_Class_Label']
pred = df['Pred_Class_Label']
confmat = confusion_matrix(y_true=true, y_pred=pred)
print(confmat)

# Plotting confusion matrix using Matplotlib's matshow function
fig, ax = plt.subplots(figsize=(10,5)) 
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3) 
for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
                ax.text(x=j, y=i, s=confmat[i,j], va='center',ha='center')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Dynamically calculating the confusion matrix metrics
TN = confmat[0, 0]
FP = confmat[0, 1]
FN = confmat[1, 0]
TP = confmat[1, 1]

metrics_text = f"True Negative: {TN}\nFalse Positve: {FP}\nFalse Negative: {FN}\nTrue Positive: {TP}"
fig.text(0.8, 0.5, metrics_text, fontsize=12, verticalalignment='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='black'))
plt.savefig('Diagram.png')

# Outputs based on diagram 
tp, tn, fn, fp = 4, 3, 1, 2

#  Accuracy (ACC)
ACC = (tp + tn) / (fp + fn + tp + tn)
print("Accuracy: %.2f" % ACC)

# Precision (PRE)
PRE = tp / (tp + fp)
print("Precision: %.2f" % PRE)

# Recall (REC) 
REC = tp / (fn + tp)
print("Recall: %.2f" % REC)

# F1 Score (F1)
F1 = 2 * ((PRE * REC) / (PRE + REC))
print("F1-score: %.2f" % F1)

# Specificity (SPE)
SPE = tn / (tn + fp)
print("Specificity: %.2f" % SPE)
