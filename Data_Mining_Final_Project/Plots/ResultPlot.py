import numpy as np
import matplotlib.pyplot as plt

# Data organized in dictionaries for loop processing
datasets = {
    'BeforeNormalization': {
        'Accuracy_score': (81.97, 81.86),
        'Precision_score': (32.45, 32.48),
        'Recall_score': (78.69, 77.82),
        'F1_score': (45.95, 45.83)
    },
    'AfterNormalization': {
        'Accuracy_score': (85.51, 85.41),
        'Precision_score': (64.12, 61.67),
        'Recall_score': (71.58, 61.15),
        'F1_score': (67.65, 61.41)
    }
}

N = 2 # Number of models
ind = np.arange(N) # the x locations for the groups
width = 0.15 # the width of the bars

# Loop through each dataset and generate a plot
for key, data in datasets.items():
    fig, ax = plt.subplots(figsize=(12,12))

    # Creating the bars for each score
    rects1 = ax.bar(ind, data['Accuracy_score'], width, color='red')
    rects2 = ax.bar(ind + width, data['Precision_score'], width, color='blue')
    rects3 = ax.bar(ind + 2*width, data['Recall_score'], width, color='green')
    rects4 = ax.bar(ind + 3*width, data['F1_score'], width, color='yellow')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Score (In %)')
    ax.set_title(f'Model Performance by Imputation Techniques: {key}')
    ax.set_xticks(ind + width * 1.5)
    ax.set_xticklabels(('KNN', 'RandomForest'))

    ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), ('Accuracy', 'Precision', 'Recall', 'F1-Score'))

    # Function to add labels to the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                    '%.2f' % float(height),
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    plt.savefig(f'{key}Results.png')
    plt.close(fig)
