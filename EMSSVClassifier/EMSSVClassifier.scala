package EMSSVClassifier

import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.DataFrame
import SSVData._
import SSVClassifier._
import org.apache.spark.mllib.linalg.DenseVector

/*
 * Self-Training model realization
 * Works with dataframes
 * Constructor takes uid (uniq ID) and classifier (base learner)
 */

final class EMSSVClassifier [
  FeatureType,
  E <: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],
  M <: org.apache.spark.ml.classification.ProbabilisticClassificationModel[FeatureType, M]
] (
    val uid: String,
    val baseClassifier: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M]
  ) extends org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M] with Serializable with SSVClassifier {

  var pipelineSSV : PipelineForSSVLearning[FeatureType, E, M] = _

  def this(classifier: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M]) =
    this(Identifiable.randomUID("selftrainingclassifier"), classifier)

  /*
   * Classifier parameters with setters
   */
  var countIterations: Long = 10
  var minResidualPercent: Double = 10.0
  var verbose: Boolean = false

  def setCountIterations(value: Long) = {
    countIterations = value
    this
  }

  def setResidualPercent(value: Double) = {
    minResidualPercent = value
    this
  }

  def setVerbose(value: Boolean) = {
    verbose = value
    this
  }

  override def setFeaturesCol(value: String) = {
    data.featuresCol = value
    this.asInstanceOf[E]
  }

  override def setLabelCol(value: String) = {
    data.labelCol = value
    this.asInstanceOf[E]
  }

  /*
   * Set the pipeline for bootstrapping depending on the last version of data
   */
  def syncPipeline() = {
    pipelineSSV = new PipelineForSSVLearning(baseClassifier.asInstanceOf[E])
    pipelineSSV.construct(data)
  }


  def train(dataset: DataFrame): M = {
    data.labeledData = dataset.select(data.labelCol, data.featuresCol).cache()
    this.syncPipeline()
    this.train()
  }

  /*
  * Gets new labeled data on every iteration and re-trains pipelineForSSV
  * Returns base learner's model trained on extended during bootstrapping training dataset
  */
  def train(): M = {
    if (verbose) {
      println("\nEMClassifier: Start training")
      println("Unlabeled count: " + data.unlabeledData.count())
      println("Labeled count: " + data.labeledData.count())
    }
    if (data.unlabeledData.count > 0) {
      var numberOfIteration: Int = 0
      var residualPercent: Double = 100.0

      var time = System.currentTimeMillis()
      pipelineSSV.fit(data)

      if (verbose)
        println("Initial model train done in " + (System.currentTimeMillis()-time)/1000 + " sec")

      val prevPredictedLabels = new Array[Double](data.unlabeledData.count().toInt)
      data.labeledData.cache()
      data.unlabeledData.cache()
      while (numberOfIteration < countIterations && data.unlabeledData.count > 0 && residualPercent > minResidualPercent) {
        if (verbose)
          println("\nIteration " + (numberOfIteration + 1))
        time = System.currentTimeMillis
        val predictions = pipelineSSV.transform(data).cache()
        if (verbose)
          println("E-step done in " + (System.currentTimeMillis - time)/1000 + " sec")

        time = System.currentTimeMillis
        pipelineSSV.fit(pipelineSSV.mixLabeledAndPredictedData(data, predictions))
        if (verbose)
          println("M-step done in " + (System.currentTimeMillis - time)/1000 + " sec")

        val curPredictedLabels = predictions.select("prediction").map(x=>x(0).asInstanceOf[Double]).collect()
        var countMissedLabels = 0
        for(i <- prevPredictedLabels.indices) {
          if (Math.abs(prevPredictedLabels(i) - curPredictedLabels(i)) > 0.01) countMissedLabels += 1
          prevPredictedLabels(i) = curPredictedLabels(i)
        }
        if (numberOfIteration == 0)
          countMissedLabels = data.unlabeledData.count().toInt
        residualPercent = countMissedLabels.toDouble / data.unlabeledData.count() * 100
        if (verbose) {
          println("Residual percent: " + residualPercent)
        }
        numberOfIteration += 1
      }
    }
    if (verbose)
      println("\nEMClassifier: Training has been done")
    val predictions = pipelineSSV.transform(data).cache()
    data.labeledData =  data.labeledData.unionAll(pipelineSSV.getPreparedPredictedData(predictions, data.featuresCol, data.labelCol))
    pipelineSSV.constructModel(data)
  }

  override def copy(extra: org.apache.spark.ml.param.ParamMap):E = defaultCopy(extra)
}



