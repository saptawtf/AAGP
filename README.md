# AAGP
AAGP:A machine learning framework for prediction of anti-aging peptides using diverse physicochemical and compositional features
# Description
This is the source code of AAGP, a machine learning predictor for prediction of anti-aging peptides (manuscript under review). The first stage is the optimization of feature vector, and the second stage is the hyperparameter tuning of the machine learning models. The trained models are also included in this package, and can help in prediction on a given peptide data set.

# Installation
Requiremenets:
* Python = 3.8, pycaret[full] = 2.3.10

Packages
* Install required packages using `pip install -r requirements.txt`

# Usage
Modify main_predict.py for your data set in fasta format
* Input file
  * One or more files in Fasta format (described below)
  
* output file
  * binary_vector.csv -- The prediction output in binary format (1 for positive and 0 for negative)
    
    ![image](https://github.com/YnnJ456/ENCAP/assets/95170485/89e9b8ac-c49a-465d-8119-069b7852807a)

  * probability.csv -- The prediction probability estimate
    
    ![image](https://github.com/YnnJ456/ENCAP/assets/95170485/c03deada-58cc-4c1f-814f-301f9362fa21)

When dataset = 'DS1', the program will use models trained on DS1, corresponding features and their normalization scaler to process data and perform prediction.

When dataset = 'DS2' , the program will use models trained on DS2, corresponding features and their normalization scaler to process data and perform prediction.
```py
# If you want to use different model, you can change dataset
dataset = 'DS1'
if dataset == 'DS1':
    model_use = '1'
elif dataset == 'DS2':
    model_use = '2'
```

```py
# Path setting
pathDict = {'paramPath': f'../data/param/{dataset}/',  # The path by default consists of featureTypeDict.pkl and robust.pkl
            'saveCsvPath': '../data/mlData/new_data/',  # Your encoded data will be automatically saved in this path
            'modelPath': f'../data/finalModel/{dataset}/',  # This path by default consists of ML models of catboost, et, and gbc
            'outputPath': '../data/output/'}  # Prediction result files will be saved in the path
```


Specify one or more fasta files in the 'inputPathList' parameter. Sequences from these fasta files will be concatenated for prediction, and prediction results will be written to the default output files, binary_vector.csv and probability.csv.

```py
# Input your FASTA file, the example file can be found in data/mlData/DS1/test_neg.FASTA
inputPathList = ['../data/mlData/DS1/AAPNegTest_DS1.FASTA', '../data/mlData/DS1/AAPPosTest_DS1.FASTA']
```



Here is the code snippet in main_predict.py. We already set the parameters and the program is ready to be excecuted.

```py
AAGP_1 = AAGP_Predict(model_use=model_use, pathDict=pathDict)
AAGP_1.loadData(inputDataDict=inputDataDict)
AAGP_1.featureEncode()
AAGP_1.doPredict()
```
