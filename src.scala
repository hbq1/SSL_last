/**
  * To show interface && test classes
  */

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.log4j.{Level, Logger}

import SSVTester._
import SelfTrainingClassifier._
import CoTrainingClassifier._
import EMSSVClassifier._
import SSVData._

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql._
import org.apache.spark.ml.classification.RandomForestClassifier


object src {
  def main(args: Array[String]) {
    //my settings
    System.setProperty("hadoop.home.dir", "c:\\winutil\\")
    val conf = new SparkConf().setAppName("App for testing").setMaster("local")
                   .set("spark.hadoop.validateOutputSpecs", "false")
                   .set("spark.default.parallelism", "1")
                   .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                   .set("spark.executor.memory", "1g")
                   .set("spark.io.compression.codec", "lzf")
                   .set("spark.speculation", "true")
    conf.registerKryoClasses(Array(classOf[SSVData], classOf[SSVTester], classOf[DataFrame], classOf[RandomForestClassifier]))
    val sparkContext = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.ERROR)
    val sqlContext = new SQLContext(sparkContext)
    import sqlContext.implicits._

    //load rawData from hdd
    val rawData = sparkContext.textFile("C:\\Users\\Yuri\\Documents\\data\\Digit1.txt")

    //prepare data structure
    val data: DataFrame = rawData.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(x => x.toDouble)))
    }.toDF
    data.cache()

    //create tester-class and assign it with data
    //set labeled data quantity = 5%
    //set folds count = 3
    val SSVTester = new SSVTester().setData(data).setLabeledPart(0.05).setFolds(2)

    //create SelfTraining classifier and set parameters
    val RFforEM = new RandomForestClassifier().setNumTrees(15)
    val EMClassifier = new EMSSVClassifier(RFforEM)
    EMClassifier.setVerbose(false).setResidualPercent(3.0).setCountIterations(7).setFeaturesCol("features").setLabelCol("indexedLabel")

    //create SelfTraining classifier and set parameters
    val RFforST = new RandomForestClassifier().setNumTrees(15)
    val STClassifier = new SelfTrainingClassifier(RFforST)
    STClassifier.setVerbose(false).setThreshold(0.95).setCountIterations(3).setFeaturesCol("features").setLabelCol("indexedLabel")

    //create CoTraining classifier and set parameters
    val RFforCT1 = new RandomForestClassifier().setNumTrees(15)
    val RFforCT2 = new RandomForestClassifier().setNumTrees(15)
    val RFforCTFinal = new RandomForestClassifier().setNumTrees(15)
    val CTClassifier = new CoTrainingClassifier(RFforCT1, RFforCT2, RFforCTFinal)
    CTClassifier.setVerbose(false).setThreshold(0.90).setCountIterations(3).setFeaturesCol("features").setLabelCol("indexedLabel")

    //create RandomForestClassifier and set parameters
    val RFClassifier = new RandomForestClassifier().setNumTrees(15).setFeaturesCol("features").setLabelCol("indexedLabel")

    //create list of algorithms to test
    //format : (transformer, name)
    val algorithmsToTest = List((RFClassifier, "RandomForest"), (EMClassifier, "EM"), (STClassifier, "SelfTraining"), (CTClassifier, "CoTraining"))

    //launch testing procedure
    val testResult = SSVTester.evaluateWithAllMetrics(algorithmsToTest, data)

    //show results
    testResult.show()
  }
}


