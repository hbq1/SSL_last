package SSVData

import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.sql.functions.{udf, col}
import org.apache.spark.sql.DataFrame


/*
 * Pipeline for semi-supervised learning
 * Acually, it is a wrapper for ml.Pipeline, extended with special methods for ssv-learning
 * It gets classifier and threshold of acceptance.
 * Used for Classifiers optimisation - once constructed, many times used
 */
class PipelineForSSVLearning [
  FeatureType,
  E <: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],
  M <: org.apache.spark.ml.classification.ProbabilisticClassificationModel[FeatureType, M]
](
   val classifier : org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],
   val thresholdTrust : Double = 0.95
 ) extends Serializable {
  var pipelineModel: org.apache.spark.ml.PipelineModel = _
  var pipeline:   org.apache.spark.ml.Pipeline = _

  var labelIndexer: org.apache.spark.ml.feature.StringIndexerModel = _
  var labelConverter: org.apache.spark.ml.feature.IndexToString = _

  /*
   * Pipeline construction and label's indexing on stages
   * Associated with base learner
   */
  def construct(data : SSVData): org.apache.spark.ml.Pipeline = {
    labelIndexer = new StringIndexer().setInputCol(data.labelCol).setOutputCol("indexedLabelSSV").fit(data.labeledData)
    labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("indexedLabelPredictionSSV").setLabels(labelIndexer.labels)
    pipeline = new Pipeline()
    classifier.setLabelCol("indexedLabelSSV").setFeaturesCol(data.featuresCol)
    pipeline.setStages(Array(labelIndexer, classifier, labelConverter))
    pipeline
  }

 /*
  * Construct base learner's model on passed data
  * It should manually pass throught previous pipeline stages (here the stage is labelIndexer)
  */
  def constructModel(data : SSVData): M = {
    classifier.setLabelCol(data.labelCol).setFeaturesCol(data.featuresCol)
    classifier.fit(labelIndexer.transform(data.labeledData))
  }

  /*
   * Fit the pipeline model
   */
  def fit(data : SSVData): org.apache.spark.ml.PipelineModel = {
    this.fit(data.labeledData)
  }

  def fit(labeledData : DataFrame): org.apache.spark.ml.PipelineModel = {
    pipelineModel = pipeline.fit(labeledData)
    pipelineModel
  }

  /*
   * Predict labels (eq. transform data)
   */
  def transform(data : SSVData): DataFrame = {
    pipelineModel.transform(data.unlabeledData)
  }

  def transform(unlabeledData : DataFrame): DataFrame = {
    pipelineModel.transform(unlabeledData)
  }

  /*
   * Special method for ssv-learning.
   * It predicts labels and returns samples which classifier defined as reliable (their predicted probability > thresholdTrust)
   * If featureTransformer passed, it used only before transforming
   * Returns DataFrame of reliable samples with source format (no matter if featureTransformer passed)
   */
  def getNewReliableData(data: SSVData, featuresTransformer: (DenseVector => DenseVector) = null): DataFrame = {
    val sqlContext = data.labeledData.sqlContext
    val sparkContext = sqlContext.sparkContext
    import sqlContext.implicits._

    val predictions = if (featuresTransformer != null) {
      val sqlfunc = udf(featuresTransformer)
      val unlData = data.rawUnlabeledData.withColumnRenamed(data.featuresCol, data.featuresCol + "_reserved")
        .withColumn(data.featuresCol, sqlfunc(col(data.featuresCol + "_reserved")))
      pipelineModel.transform(unlData).drop(data.featuresCol).withColumnRenamed(data.featuresCol+"_reserved", data.featuresCol)
    } else {
      pipelineModel.transform(data.unlabeledData)
    }
    val rdd = predictions.rdd.filter(
      x=> x(x.fieldIndex("probability")).asInstanceOf[DenseVector].toArray.max > thresholdTrust
    )
    sparkContext.parallelize(rdd.map(x => SchemaTrainingSample(x(x.fieldIndex("indexedLabelPredictionSSV")).asInstanceOf[String].toDouble, x(x.fieldIndex(data.featuresCol)).asInstanceOf[DenseVector])
    ).collect()).toDF.withColumnRenamed("label", data.labelCol).withColumnRenamed("features", data.featuresCol)
  }

 /*
 * Special methods for ssv-learning.
 */

  def getPreparedPredictedData(predictions: DataFrame, fCol: String, lCol: String): DataFrame = {
    val sqlContext = predictions.sqlContext
    val sparkContext = sqlContext.sparkContext
    import sqlContext.implicits._
    predictions.select("indexedLabelPredictionSSV", fCol)
      .map(x => SchemaTrainingSample(x(0).asInstanceOf[String].toDouble, x(1).asInstanceOf[DenseVector])).toDF
      .withColumnRenamed("label", lCol).withColumnRenamed("features", fCol)
  }

  def mixLabeledAndPredictedData(data: SSVData, predictions: DataFrame): DataFrame = {
    data.labeledData.unionAll(getPreparedPredictedData(predictions, data.featuresCol, data.labelCol))
  }

 /*
  * Copy method
  */
  def copy(): PipelineForSSVLearning[FeatureType, E, M] = {
    val result = new PipelineForSSVLearning[FeatureType,E,M](classifier, thresholdTrust)
    result.pipelineModel = pipelineModel
    result.pipeline = pipeline
    result.labelIndexer = labelIndexer
    result.labelConverter = labelConverter
    result
  }
}
