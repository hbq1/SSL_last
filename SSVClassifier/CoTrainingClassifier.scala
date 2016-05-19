package SSVClassifier

import SSVData._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.sql.DataFrame


/*
 * Co-Training model realization
 * Works with dataframes
 * Constructor takes uid (uniq ID) and 3 classifiers: 2 for base, 1 for resulting model
 */

final class CoTrainingClassifier [
  FeatureType,
  E1 <: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E1, M1],
  M1 <: org.apache.spark.ml.classification.ProbabilisticClassificationModel[FeatureType, M1],
  E2 <: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E2, M2],
  M2 <: org.apache.spark.ml.classification.ProbabilisticClassificationModel[FeatureType, M2],
  E3 <: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E3, M3],
  M3 <: org.apache.spark.ml.classification.ProbabilisticClassificationModel[FeatureType, M3]
] (
    val uid: String,
    val baseClassifier1: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E1, M1],
    val baseClassifier2: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E2, M2],
    val finalClassifier: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E3, M3]
  ) extends org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E3, M3] with Serializable with SSVClassifier {


  var pipelineSSV1 : PipelineForSSVLearning[FeatureType, E1, M1] = _
  var pipelineSSV2 : PipelineForSSVLearning[FeatureType, E2, M2] = _
  var pipelineSSVFinal : PipelineForSSVLearning[FeatureType, E3, M3] = _

  def this(classifier1: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E1, M1],
           classifier2: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E2, M2],
           classifierFinal: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E3, M3]) =
    this(Identifiable.randomUID("cotrainingclassifier"), classifier1, classifier2, classifierFinal)



 /*
 * Classifier parameters with setters
 */
  var countIterations: Long = 10
  var thresholdTrust: Double = 0.95
  var verbose: Boolean = false

  var featureIndices1: List[Int] = _
  var featureIndices2: List[Int] = _

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

  def setFeaturesIndices(newFeaturesIndices1: List[Int], newFeaturesIndices2: List[Int]) = {
    featureIndices1 = newFeaturesIndices1
    featureIndices2 = newFeaturesIndices2
    this
  }

  def setFeaturesIndices1(newFeaturesIndices: List[Int]) = {
    featureIndices1 = newFeaturesIndices
    this
  }

  def setFeaturesIndices2(newFeaturesIndices: List[Int]) = {
    featureIndices2 = newFeaturesIndices
    this
  }

  override def setFeaturesCol(value: String) = {
    data.featuresCol = value
    this.asInstanceOf[E3]
  }

  override def setLabelCol(value: String) = {
    data.labelCol = value
    this.asInstanceOf[E3]
  }

 /*
 * Set pipelines for bootstrapping depending on the last version of data
 */
  def syncPipelines() = {
    pipelineSSV1 = new PipelineForSSVLearning[FeatureType, E1, M1](baseClassifier1, thresholdTrust)
    pipelineSSV2 = new PipelineForSSVLearning[FeatureType, E2, M2](baseClassifier2, thresholdTrust)
    pipelineSSVFinal = new PipelineForSSVLearning[FeatureType, E3, M3](finalClassifier, thresholdTrust)
    pipelineSSV1.construct(data)
    pipelineSSV2.construct(data)
    pipelineSSVFinal.construct(data)
  }

 /*
 * Returns function which extracts passed indices from DenseVector
 */
  def indicesExtractor(indices: List[Int]): (DenseVector => DenseVector) = {
    (x: DenseVector) => {
      val extractedFeatures = new Array[Double](indices.size)
      for (i <- indices.indices) extractedFeatures(i) = x(indices(i))
      new DenseVector(extractedFeatures)
    }
  }

 /*
 * Sets labeled data, saves raw data before training and sets pipelines
 * Raw data saved to get full features when this will be required
 */

  def train(dataset: DataFrame): M3 = {
    data.labeledData = dataset.select(data.labelCol, data.featuresCol).cache()
    data.saveDataAsRaw()
    this.syncPipelines()
    this.train()
  }

 /*
 * Gets new labeled data with learners on every iteration and re-trains them on re-mixed new datasets
 * Returns final learner's model trained on intersected during bootstrapping extended training datasets
 */

  def train(): M3 = {
    if (verbose) {
      println("\nCoTrainingClassifier: Start training")
      println("Unlabeled count: " + data.unlabeledData.count())
      println("Labeled count: " + data.labeledData.count())
    }

    var numberOfIteration: Int = 1
    var countNewLabeled: Long = 1

    val data1 = data.copy()
    val data2 = data.copy()

    if (featureIndices1 == null || featureIndices2 == null) {
      val featuresCount: Int = data.featuresCount()
      featureIndices1 = (0 until featuresCount/2).toList
      featureIndices2 = (featuresCount/2 until featuresCount).toList
    }

    val indicesExtractor1 = indicesExtractor(featureIndices1)
    val indicesExtractor2 = indicesExtractor(featureIndices2)

    data1.extractFeaturesIndices(featureIndices1)
    data2.extractFeaturesIndices(featureIndices2)
    var time: Long = 0
    while (numberOfIteration <= countIterations && data1.unlabeledData.count() > 0 &&
           data2.unlabeledData.count() > 0 && countNewLabeled > 0)
    {
      if (verbose)
        print("\nIteration " + numberOfIteration + "\n")
      time = System.currentTimeMillis
      pipelineSSV1.fit(data1)
      pipelineSSV2.fit(data2)

      if (verbose)
        println("Stage 1 done in " + (System.currentTimeMillis - time)/1000 + " sec")
      time = System.currentTimeMillis

      val newData1 = pipelineSSV1.getNewReliableData(data1, indicesExtractor1)
      val newData2 = pipelineSSV2.getNewReliableData(data2, indicesExtractor2)

      if (verbose)
        println("Stage 2 done in " + (System.currentTimeMillis - time)/1000 + " sec")
      time = System.currentTimeMillis

      data1.rawLabeledData = data1.rawLabeledData.unionAll(newData2).repartition(4).cache()
      data2.rawLabeledData = data2.rawLabeledData.unionAll(newData1).repartition(4).cache()
      data1.rawUnlabeledData = data1.rawUnlabeledData.except(newData1.select("features")).repartition(4).cache()
      data2.rawUnlabeledData = data2.rawUnlabeledData.except(newData2.select("features")).repartition(4).cache()

      data1.extractFeaturesIndices(featureIndices1)
      data2.extractFeaturesIndices(featureIndices2)

      if (verbose)
        println("Stage 3 done in " + (System.currentTimeMillis - time)/1000 + " sec")

      countNewLabeled = newData1.count() + newData2.count()
      if (verbose) {
        print("New labeled count in first: " + newData1.count() + "\n")
        print("Labeled Data count in first: " + data1.labeledData.count() + "\n")
        print("New labeled count in second: " + newData2.count() + "\n")
        print("Labeled Data count in second: " + data2.labeledData.count() + "\n")
      }
      numberOfIteration += 1
    }
    data1.labeledData = data1.rawLabeledData.intersect(data2.rawLabeledData).repartition(4).cache()
    if (verbose)
      println("\nCoTrainingClassifier: Training has been done")
    pipelineSSVFinal.constructModel(data1)
  }

  override def copy(extra: org.apache.spark.ml.param.ParamMap): E3 = defaultCopy(extra)
}
