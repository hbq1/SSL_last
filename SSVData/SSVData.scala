package SSVData

import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.sql.DataFrame


/*
 * Schema for ML pipeline
 */
case class SchemaTrainingSample(label: Double, features: DenseVector)


/*
 * Class for SSVlearning to prepare and operate with data
 * All algorithms use labeledData and UnlabeledData for training
 * raw versions are necessary for additional data-transforming (i.e. in cCoTraining)
 */
class SSVData extends Serializable {

  var rawLabeledData : DataFrame  = _
  var rawUnlabeledData : DataFrame  = _

  var labeledData : DataFrame  = _
  var unlabeledData : DataFrame = _

  var labelCol: String = "indexedLabel"
  var featuresCol: String = "features"

  var useRaw: Boolean = false

  def saveDataAsRaw() = {
    rawLabeledData = labeledData
    rawUnlabeledData = unlabeledData
    useRaw = true
  }

  /*
   * Get features space size
   */
  def featuresCount(): Int = {
    val t = labeledData.take(1)(0)
    t(t.fieldIndex(featuresCol)).asInstanceOf[DenseVector].size
  }

  def fullFeaturesCount(): Int = {
    val t = rawLabeledData.take(1)(0)
    t(t.fieldIndex(featuresCol)).asInstanceOf[DenseVector].size
  }

 /*
  * Extract features by passed indices (for CoTraining)
  */
  def extractFeaturesIndices(indices: List[Int]): SSVData = {
    val sqlContext = this.rawLabeledData.sqlContext
    import sqlContext.implicits._

    this.labeledData = this.rawLabeledData.rdd.map( sample => {
      val featuresVector = sample(sample.fieldIndex(featuresCol)).asInstanceOf[DenseVector]
      val extractedFeatures = new Array[Double](indices.size)
      for (i <- indices.indices) extractedFeatures(i) = featuresVector(indices(i))

      SchemaTrainingSample(sample(sample.fieldIndex(labelCol)).asInstanceOf[Double],
                           new DenseVector(extractedFeatures))
    }).toDF.withColumnRenamed("label", labelCol).withColumnRenamed("features", featuresCol).repartition(4).cache()
    this.unlabeledData = this.rawUnlabeledData.rdd.map( sample => {
      val featuresVector = sample(sample.fieldIndex(featuresCol)).asInstanceOf[DenseVector]
      val extractedFeatures = new Array[Double](indices.size)
      for (i <- indices.indices) extractedFeatures(i) = featuresVector(indices(i))

      SchemaTrainingSample(0.0, new DenseVector(extractedFeatures))
    }).toDF.drop(labelCol).withColumnRenamed("features", featuresCol).repartition(4).cache()
    this
  }

 /*
  * Mix labeled and unlabeled data
  */
  def getMixedData: DataFrame = {
    labeledData.drop(labelCol).unionAll(unlabeledData)
  }

  def copy(): SSVData = {
    val res = new SSVData()
    res.unlabeledData = unlabeledData
    res.labeledData = labeledData
    res.rawUnlabeledData = rawUnlabeledData
    res.rawLabeledData = rawLabeledData
    res
  }

}


