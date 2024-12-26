The data mining/machine learning algorithm uses primarily python coding packages, 
along with a few R codes to impute data to be used for KNN or Random Forest imputation.

The python codes use the packages as follows
- numpy as np 
- pandas as pd 
- sklearn
- scipy

For the R codes, a filepath is called in directly to where my files are located in my computer,
so that will need to be manipulated in order run the code. 

I have four separate python codes for each of the test cases I performed,
either with the KNN or Random Forest imputation method,
and then either with or without Z-Score normalization.

For my ensemble algorithms, I call in the functions directly for the KNN, Naive Bayes, and Random Forest algorithms. 
They are then appended to an ensemble function to be tested for accuracy measures.
The ensemble function is put through a confusion matrix to find tp, tn, fp, and fn measures,
which are then used to find accuracy, precision, recall, and F-1 score measures. 