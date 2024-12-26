# KNN_Implementation
Supervised machine learning algorithm, KNN (K-nearest neighbour) Implementation 

Our implementation should accept two data files as input (both are posted with the assignment): a spam train.csv file and a
spam test.csv file. Both files contain examples of e-mail messages, with each example having a class label of either \1" (spam) or \0" (no-spam). 

Each example has 57 (numeric) features that characterize the message. Our classifier should examine each example in the spam test set and classify it as one of the two classes. The classification will be based on an unweighted vote of its k nearest examples in the spam train set. We will measure all distances using regular Euclidean distance. <br />

1. 

(a) Report test accuracies when k = 1; 5; 11; 21; 41; 61; 81; 101; 201; 401 without nor-malizing the features.<br />
(b) Report test accuracies when k = 1; 5; 11; 21; 41; 61; 81; 101; 201; 401 with z-score normalization applied to the features.<br />
(c) In the (b) case, generate an output of KNN predicted labels for the first 50 instances (i.e. t1 - t50) when k = 1; 5; 11; 21; 41; 61; 81; 101; 201; 401 (in this order).<br />
For example, if t5 is classified as class "spam" when k = 1; 5; 11; 21; 41; 61 and classified as class "no-spam" when k = 81; 101; 201; 401, then your output line for t5 should be:<br />
t5 spam, spam, spam, spam, spam, spam, no, no, no, no

(d) I can conclude from the perfomance of HW2_Question_1a.py from (a) and HW2_Question_1b.py from (b) that (b) is more accurate with its calculations, as the numbers range in the range of 83 to 88 for (b), whereas for (a) the numbers range from 71 to 75. 

2. 

For this part, a table was provided for us with a small training set. Each line includes an individual’s education, occupation choice, years of experience, and an indication of salary. My task here was to create a complete decision tree including the number of low’s & high’s , entropy at each step and the information gain for each feature examined at each node in the tree. This tree can be found in HW2_Question2_&_Question3.pdf.

After that, we were supposed to prune the tree obtained using the validation data given in a new Table 2. This is also located in HW2_Question2_&_Question3.pdf.

3. 

Lastly, for this part a Naive Bayes classifier was constrcuted for the given training data in Table 1 with the "add 1 smoothing" technique covered in the lecture slides. I used that model to classify the following new instances provided in HW2.pdf. That can also be found in the HW2_Question2_&_Question3.pdf.