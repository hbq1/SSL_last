package SelfTrainingClassifier

import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.DataFrame

import SSVData._
import SSVClassifier._

/*
 * Self-Training model realization
 * Works with dataframes
 * Constructor takes uid (uniq ID) and classifier (base learner)
 */

final class SelfTrainingClassifier [
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
  var thresholdTrust: Double = 0.95
  var verbose: Boolean = false

  def setCountIterations(value: Long) = {
    countIterations = value
    this
  }

  def setThreshold(value: Double) = {
    thresholdTrust = value
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
    pipelineSSV = new PipelineForSSVLearning(baseClassifier.asInstanceOf[E], thresholdTrust)
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
      println("\nSeltTrainingClassifier: Start training")
      println("Unlabeled count: " + data.unlabeledData.count())
      println("Labeled count: " + data.labeledData.count())
    }
    if (data.unlabeledData.count > 0) {
      var numberOfIteration: Int = 0
      var countNewLabeled: Long = 1
      while (numberOfIteration < countIterations && data.unlabeledData.count > 0 && countNewLabeled > 0) {
        if (verbose) {
          print("\nIteration " + (numberOfIteration + 1) + "\n")
        }
        var time = System.currentTimeMillis
        pipelineSSV.fit(data)

        if (verbose)
          println("Stage 1 done in " + (System.currentTimeMillis - time)/1000 + " sec")

        time = System.currentTimeMillis
        val newData = pipelineSSV.getNewReliableData(data).repartition(4).cache()
        countNewLabeled = newData.count

        if (verbose)
          println("Stage 2 done in " + (System.currentTimeMillis - time)/1000 + " sec")

        time = System.currentTimeMillis
        data.unlabeledData = data.unlabeledData.except(newData.select(data.featuresCol)).repartition(4).cache()
        data.labeledData = data.labeledData.unionAll(newData).repartition(4).cache()

        if (verbose)
          println("Stage 3 done in " + (System.currentTimeMillis - time)/1000 + " sec")

        countNewLabeled = newData.count
        if (verbose) {
          print("New labeled count: " + countNewLabeled + "\n")
          print("Current Labeled Data count: " + data.labeledData.count() + "\n")
        }
        numberOfIteration += 1
      }
    }
    if (verbose)
      println("\nSeltTrainingClassifier: Training has been done")
    pipelineSSV.constructModel(data)
  }

  override def copy(extra: org.apache.spark.ml.param.ParamMap):E = defaultCopy(extra)
}



