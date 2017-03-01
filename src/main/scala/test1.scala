import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.util.MLUtils

// decision tree imports
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel

// importing CSV data into the expected format
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

import org.apache.spark.sql.Row

// Create Spark session
val sparkSession = SparkSession.builder
    .master("local[1]")
    .appName("Decision Tree example")
    .getOrCreate()

// Load the data
val text = sparkSession.sparkContext.textFile("files/spambase.data")

// Separate into array
val data = text.map(line => line.split(',').map(_.toDouble)).map(t => LabeledPoint(t(57), Vectors.dense(t.take(57))))

// Split the data into training and test sets (30% held out for testing)
val splits = data.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))

// Train a DecisionTree model.
//  Empty categoricalFeaturesInfo indicates all features are continuous.
val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val impurity = "gini"
val maxDepth = 5
val maxBins = 32

val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
  impurity, maxDepth, maxBins)

// Evaluate model on test instances and compute test error
val labelAndPreds = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}

val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testData.count()
println("Test Error = " + testErr)
println("Learned classification tree model:\n" + model.toDebugString)