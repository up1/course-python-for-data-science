from pyspark import SparkContext, SparkConf
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import UserDefinedFunction
from pyspark.mllib.linalg import Vectors
import os
import pyspark_csv as pycsv

def createSparkContext():
    conf = (SparkConf()
            .setMaster("spark://35.198.225.50:7077")
            .setAppName("Titanic_Data")
            .set("spark.cores.max", "1")
            .set("spark.executor.memory", "512m")
            .set("spark.shuffle.service.enabled", "false")
            .set("spark.dynamicAllocation.enabled", "false")
            .set("spark.io.compression.codec", "snappy")
            .set("spark.rdd.compress", "true"))

    sc = SparkContext(conf=conf)
    return sc

def loadData(sc, sqlContext, path):
    plain = sc.textFile(path)
    df = pycsv.csvToDataFrame(sqlContext, plain, sep=',')
    return df

sc = createSparkContext()
sc.addPyFile('/src/pyspark_csv.py')
sqlContext = SQLContext(sc)
training_data = loadData(sc, sqlContext,'/src/train.csv')
print(training_data)
