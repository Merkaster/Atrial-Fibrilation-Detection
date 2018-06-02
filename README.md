# Atrial-Fibrilation-Detection
Detect if a patient has Atrial Fibrilation using ECG signals using different Machine Learning classifiers

Using dataset from the  Physionet Challenge 2017 (https://www.physionet.org/challenge/2017/) we classify an ECG signal to one of 4 categories:
  -  normal sinus rhythm
  - atrial fibrillation (AF)
  - an alternative rhythm
  - or is too noisy to be classified
 
 At first we preprocess an ECG signal to extract the required features for the classification process, we accomplish this by using a python package(neurokit).
 After extracting the required features we find the importance of each feature using Random Forest and we use classification methods from the sklearn package to classify a signal.
 The classifiers we used are Random Forest,SVM, KNN, Naive Bayes using Grid Search for parameters tuning.
 We estimated the number of features we need for better accuracy in the classification.
