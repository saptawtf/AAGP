from userPackage.Package_Encode import EncodeAllFeatures
from userPackage.LoadDataset import LoadDataset
import pandas as pd
from MLProcess.PycaretWrapper import PycaretWrapper
from MLProcess.Predict import Predict

class AAGP_Predict:

    def __init__(self, model_use, pathDict):
        self.model_use = model_use
        self.pathDict = pathDict
        self.modelNameList = ['lightgbm', 'gbc', 'lda', 'knn', 'et', 'rf', 'mlp', 'qda', 'catboost']
        self.dataList = []
        self.predVectorDf = None
        self.probVectorDf = None
        self.predVectorListIndp = None
        self.probVectorListIndp = None
        if self.model_use == '1':
            self.featureNum = 50
            self.featureTypeDictjson = 'DS1_featureTypeDict.json'
            self.nmlzPkl = 'DS1_robust.pkl'
            self.featureRankCsv = '../data/mlData/DS1/featureRank_DS1.csv'
        elif self.model_use == '2':
            self.featureNum = 50
            self.featureTypeDictjson = 'DS2_featureTypeDict.json'
            self.nmlzPkl = 'DS2_robust.pkl'
            self.featureRankCsv = '../data/mlData/DS2/featureRank_DS2.csv'
        else:
            raise NameError('model_use should input 1 or 2, 1 = DS1, 2 = DS2')

    def loadData(self, inputDataList):
        ldObj = LoadDataset()
        for inputData in inputDataList:
            testSeqDict = ldObj.readFasta(inputData, minSeqLength=5)
            self.dataList.append(testSeqDict)

    def featureEncode(self):
        encodeObj = EncodeAllFeatures()
        encodeObj.dataEncodeSetup(loadJSONPath=f'{self.pathDict["paramPath"]}/{self.featureTypeDictjson}')
        encodeObj.dataEncodeOutput(dataList=self.dataList)
        testDf = encodeObj.dataNormalization(loadNmlzScalerPklPath=f'{self.pathDict["paramPath"]}/{self.nmlzPkl}')
        featureDf = pd.read_csv(self.featureRankCsv)
        featureList = featureDf['feature'].to_list()
        testDf1 = testDf[featureList]
        testDf1.to_csv(f'{self.pathDict["saveCsvPath"]}/test_F{self.featureNum}.csv')

    def doPredict(self):
        pycObj = PycaretWrapper()
        modelList = pycObj.doLoadModel(path=self.pathDict['modelPath'], fileNameList=self.modelNameList)
        dataTestDf = pd.read_csv(f'{self.pathDict["saveCsvPath"]}/test_F{self.featureNum}.csv', index_col=[0])
        predObjIndp = Predict(dataX=dataTestDf, modelList=modelList)
        self.predVectorListIndp, self.probVectorListIndp = predObjIndp.doPredict()
        self.predVectorDf = pd.DataFrame(self.predVectorListIndp, index=self.modelNameList, columns=dataTestDf.index).T
        self.probVectorDf = pd.DataFrame(self.probVectorListIndp, index=self.modelNameList, columns=dataTestDf.index).T
        self.predVectorDf.to_csv(f'{self.pathDict["outputPath"]}/binary_vector.csv')
        self.probVectorDf.to_csv(f'{self.pathDict["outputPath"]}/probability_vector.csv')

