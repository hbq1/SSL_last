package SSVTester

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, StringIndexerModel}
import SSVClassifier._

/*
 * Class for binary classification algorithms testing
 */
class SSVTester {
/*
 * Common indexers and data
 */
  var labelIndexer: StringIndexerModel = _
  var labelConverter: IndexToString = _
  var dataPositiveClass: DataFrame = _
  var dataNegativeClass: DataFrame = _

/*
 * Parameters and setters
 */
  var labeledPart: Double = 0.1
  var folds: Long = 5

  def setLabeledPart(value: Double) = {
    this.labeledPart = value
    this
  }

  def setFolds(value: Long) = {
    this.folds = value
    this
  }

 /*
  * Data setter
  * Separates classes and makes caching
  */
  def setData(data: DataFrame) = {
    labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
    labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
    dataPositiveClass = data.filter("label = 1.0").cache()
    dataNegativeClass = data.filter("label = -1.0").cache()
    this
  }

  def evaluateROC[
    FeatureType,
    E <: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],
    M <: org.apache.spark.ml.classification.ProbabilisticClassificationModel[FeatureType, M]
  ](
    classifier: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],
    isSSV: Boolean,
    ROCPath: String
  ): Unit = {
    val Array(positiveLabeledData, positiveUnlabeledData) = dataPositiveClass.randomSplit(Array(labeledPart,1-labeledPart))
    val Array(negativeLabeledData, negativeUnlabeledData) = dataNegativeClass.randomSplit(Array(labeledPart,1-labeledPart))
    val labeledData = positiveLabeledData.unionAll(negativeLabeledData)
    val unlabeledData = positiveUnlabeledData.unionAll(negativeUnlabeledData)
    var predictions : DataFrame = dataPositiveClass

    val pipeline = new Pipeline().setStages(Array(labelIndexer, classifier, labelConverter))

    if (isSSV) {
      val allDirtyData = labeledData.unionAll(unlabeledData.drop("label").withColumn("label", expr("0.0")).select("label","features"))
      val model = pipeline.fit(allDirtyData)
      predictions = model.transform(unlabeledData)
    } else {
      val model = pipeline.fit(labeledData)
      predictions = model.transform(unlabeledData)
    }

    predictions = predictions.withColumn("label", unlabeledData("label"))
    val predictionsAndLabelsB = predictions.map { x => (x(4).asInstanceOf[Vector](1), x(0).asInstanceOf[Double])}
    val metricsB = new BinaryClassificationMetrics(predictionsAndLabelsB)
    metricsB.roc.repartition(1).saveAsTextFile(ROCPath)
  }

 /*
  * Compute mean value of every metric on passed algorithms and data
  * Supported metrics: weightedPrecision, weightedRecall, weightedF1score, AUC, learning time
  */
  def evaluateWithAllMetrics[
    FeatureType,
    E <: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],
    M <: org.apache.spark.ml.classification.ProbabilisticClassificationModel[FeatureType, M]
  ] (
    algorithms : List[(org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M], String)],
    data : DataFrame
  ): DataFrame = {
    val countAlgorithms = algorithms.length
    val precision = new Array[Double](countAlgorithms)
    val recall = new Array[Double](countAlgorithms)
    val auc = new Array[Double](countAlgorithms)
    val f1 = new Array[Double](countAlgorithms)
    val time = new Array[Double](countAlgorithms)

    var foldNum = 1
    while (foldNum <= folds) {
      val Array(positiveLabeledData, positiveUnlabeledData) = dataPositiveClass.randomSplit(Array(labeledPart, 1 - labeledPart))
      val Array(negativeLabeledData, negativeUnlabeledData) = dataNegativeClass.randomSplit(Array(labeledPart, 1 - labeledPart))
      val labeledData = positiveLabeledData.unionAll(negativeLabeledData)
      val unlabeledData = positiveUnlabeledData.unionAll(negativeUnlabeledData)
      for (algoNum <- 0 until countAlgorithms) {
        if (algorithms(algoNum)._1.isInstanceOf[SSVClassifier])
          algorithms(algoNum)._1.asInstanceOf[SSVClassifier].setUnlabeledData(unlabeledData)
        val pipeline = new Pipeline().setStages(Array(labelIndexer, algorithms(algoNum)._1, labelConverter))
        val startTime = System.currentTimeMillis

        val model = pipeline.fit(labeledData)
        val predictions = model.transform(unlabeledData).withColumn("label", unlabeledData("label"))
        predictions.count()
        time(algoNum) += System.currentTimeMillis - startTime

        val predictionsAndLabelsB = predictions.map { x => (x(x.fieldIndex("probability")).asInstanceOf[Vector](1), x(0).asInstanceOf[Double]) }
        val predictionsAndLabelsM = predictions.map { x => (x(x.fieldIndex("predictedLabel")).asInstanceOf[String].toDouble, x(0).asInstanceOf[Double]) }

        val metricsB = new BinaryClassificationMetrics(predictionsAndLabelsB)
        val metricsM = new MulticlassMetrics(predictionsAndLabelsM)
        precision(algoNum) += metricsM.weightedPrecision
        recall(algoNum) += metricsM.weightedRecall
        f1(algoNum) += metricsM.weightedFMeasure
        auc(algoNum) += metricsB.areaUnderROC
      }
      foldNum += 1
    }
    val k = for (i <- 0 until countAlgorithms) yield (algorithms(i)._2, labeledPart, precision(i)/folds, recall(i)/folds, f1(i)/folds, auc(i)/folds, time(i)/(1000*folds))
    data.sqlContext.createDataFrame(k).toDF("name", "fraction", "weightedPrecision", "weightedRecall", "weightedF1score", "AUC", "learning time")
  }

}
