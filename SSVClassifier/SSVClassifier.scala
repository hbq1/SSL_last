package SSVClassifier

import org.apache.spark.sql.DataFrame
import SSVData._

/*
 * Trait to separate SSVClassifiers from other
 * Used in tester i.e.
 */
trait SSVClassifier {
  val data: SSVData = new SSVData()

  /*
 * Set unlabeled data to be used in training
 */
  def setUnlabeledData(unlabeledData: DataFrame) = {
    data.unlabeledData = unlabeledData.select("features").cache()
    this
  }
}
