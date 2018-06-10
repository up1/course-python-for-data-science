from pyspark import SparkContext, SparkConf
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import UserDefinedFunction
from pyspark.mllib.linalg import Vectors
import os
import pyspark_csv as pycsv

def createSparkContext():
    sc = SparkContext('local[*]')
    return sc

def loadData(sc, sqlContext, path):
    plain = sc.textFile(path)
    df = pycsv.csvToDataFrame(sqlContext, plain, sep=',')
    return df

sc = createSparkContext()
sc.addPyFile('/home/jovyan/work/pyspark_csv.py')
sqlContext = SQLContext(sc)

training_data = loadData(sc, sqlContext,'/home/jovyan/work/train.csv')
print(training_data.show(3))
