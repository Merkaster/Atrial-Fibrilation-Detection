import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import neurokit as nk
import csv
import glob
import os
import pandas as pd
from matplotlib.pyplot import axis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.metrics.classification import accuracy_score
from sklearn.ensemble.forest import RandomForestClassifier
from random import shuffle

def classify(name,model,features,classes):
    
    #Using 10-KFold cross validation
    kf = KFold(10,shuffle=True)
    score = []
    for trainIndex,testIndex in kf.split(features):
        xTrain, xTest = features[trainIndex],features[testIndex]
        yTrain, yTest = classes[trainIndex],classes[testIndex]
        
        model.fit(xTrain,yTrain)
        prediction = model.predict(xTest)
        score.append( accuracy_score(yTest,prediction))
        
        
    print("The average score for ",name," is ",sum(score)/len(score))


def preprocessClass(classes,name):
    
    for idx,val in enumerate(classes):
        
        if val ==name:
            classes[idx] = '0'
        else:
            classes[idx] = '1'
    
    return classes
    
def classificationProcess(features,classes):
    
    #Calculate the Feature Importance using Forest Tree
    model = ExtraTreesClassifier(100)
    model.fit(features,classes)
    featureImportance = model.feature_importances_
    
    specificClass = 'A'
    
    if specificClass != 'all':
        classes = preprocessClass(classes,specificClass)
    
    #Choose the 5 features with the highest importance
    bestFeatures= []
    for i in range(6):
        max = featureImportance.argmax()
        bestFeatures.append(max)
        featureImportance[max] = 0
     
     #Adjust the features array  to contain only the columns of the best features   
    features = features[:,[bestFeatures[0],bestFeatures[1],bestFeatures[2],bestFeatures[3],bestFeatures[4]]]
    
    ######KNN###########
    parameters = []
    knn = KNeighborsClassifier(4)
    classify("KNN",knn, features, classes)
    
    #Using exhaustive GridSearchCV for parameter tuning 
    
    #####SVM############
    parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]
    svm = GridSearchCV(SVC(),parameters, cv=5)
    classify("SVM",model,features,classes)
    
    #####Random Forest #####
    parameters = [{'n_estimators': [100, 500, 1000,1500]}]
    forest = GridSearchCV(RandomForestClassifier(),parameters, cv=5)
    classify("Random Forest",forest,features,classes)
    
    #####Naive Bayes ######
    gauss = GaussianNB()
    classify("Gaussian", gauss, features, classes)
    
    
    
def loadFromFile():
     
     file =  pd.read_csv('data.csv')
     #Features are the first 9 columns of the file and classes are the last column of the file
     features = file.take([0,1,2,3,4,5,6,7,8],axis=1)
     classes = file['Class']
     
     return features.values,classes.values
    

def writeToCsv(features,classes):
    
    with open('data.csv', 'w') as csvfile:
        fieldnames = ['mean', 'std','sdNN','meanNN','madNN','pNN50','RMSSD','shannon','sampleEntropy','Class']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        counter = 0 # iterator for the class list
        for f in features:
            writer.writerow({'mean':f[0], 'std':f[1],'sdNN':f[2],'meanNN':f[3], 'madNN':f[4],'pNN50':f[5],'RMSSD':f[6],
                             'shannon':f[7],'sampleEntropy':f[8],'Class':classes[counter]})
            counter = counter +1
        


def SignalPreprocessing():
    
    #Create a dictionary with key the filename and value the class that it belongs
    referenceFileData ={}
    with open('training2017/REFERENCE.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i in reader:
            referenceFileData[i[0]] = i[1]
    
    #Choose every training data  from the folder
    for file in glob.glob(os.path.join('training2017/','*.mat')):
        
        #parse the filename eg from training2017/A00001.mat to A00001
        fileName = file.split('/')
        fileName = fileName[1].split('.')
        fileName = fileName[0]
        print(fileName)
        
        #Assign the class for the specific data
        classes.append(referenceFileData[fileName])
        
        mat = scipy.io.loadmat(file)
        signal = np.array(mat['val']).flatten()
        
        #Find the R peaks of the ECG signal
        peaks = nk.ecg_find_peaks(signal, 300)
        
        #Calculate the Heart Rate Variability and extract its measures
        hrv = nk.ecg_hrv(peaks, sampling_rate=300)
        
        #Calculate the features
        mean = signal.mean() # Mean of the ECG signal
        std = signal.std() # Standard deviation of the ECG signal
        sdNN = hrv['sdNN'] # Standard deviation of RR interval
        meanNN = hrv['meanNN'] # Mean of RR interval 
        madNN = hrv['madNN']  # Median Absolute Deviation (MAD) of the RR intervals.
        pNN50 = hrv['pNN50'] # The number of interval differences of successive RR intervals greater than 50 ms
        RMSSD = hrv['RMSSD'] # the root mean square of the RR intervals
        shannon = hrv['Shannon'] # Shannon entropy
        sample_entropy = hrv['Sample_Entropy'] # Sample Entropy
        
        trainingFeatures = [mean, std, sdNN, meanNN, madNN, pNN50, RMSSD, shannon, sample_entropy]
        
        #Create an array containing all the features of the training set
        features.append(trainingFeatures)
        

    return features,classes


       
if __name__ == "__main__":
    
    features = []
    classes = []
    
    #features,classes = SignalPreprocessing()
    
    #writeToCsv(features,classes)
    
    features, classes = loadFromFile()
    classificationProcess(features,classes)
   












