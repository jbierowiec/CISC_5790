import pandas as pd

# Define the data
data = {
    'Method': ['KNN', 'Random Forest', 'Normalized KNN', 'Normalized Random Forest'],
    'Accuracy': [81.95, 81.97, 85.41, 85.51],
    'Precision': [32.71, 32.45, 59.75, 64.12],
    'Recall': [78.23, 78.69, 73.51, 71.58],
    'F1-Score': [46.13, 45.95, 65.92, 67.65]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to an Excel file
#df.to_excel('Model_Performance_Summary.xlsx', index=False)

# Alternatively, save to a CSV file
df.to_csv('Model_Performance_Summary.csv', index=False)
