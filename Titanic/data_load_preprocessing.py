import pandas as pd
import numpy as np

class DataSummary():

    def __init__(self, trainfile, testfile):
        self.trainfile = trainfile
        self.testfile = testfile

    def summary(self):
        train = pd.read_csv(self.trainfile)
        print(pd.DataFrame(train.isnull().sum()).T)
        print(train.describe())

# ob = DataSummary(trainfile='train.csv', testfile='test.csv')
# ob.summary()

class FeatureEngineering():


    def __init__(self):
        pass

    def _encode_(self):

        return

