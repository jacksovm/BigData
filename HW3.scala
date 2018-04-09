//import sparkSession and Logistic Regression

import org.apache.spark.sql.SparkSession
import Utilities.setupLogging
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.functions.hour

//import vectorAssembler and vectors
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors



object HW3 {
  def main(args: Array[String]): Unit = {

    //////////////////////////////////
    //  LOGISTIC REGRESSION PROJECT //
    //////////////////////////////////

    /* In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user slicked on an advertisement.
    we will try to create a model that will predict whether or not they will click on an add based off the features of that user.

    this data set contains the following features:
    ----------------------------------------------
    Daily Time Spent on Site: consumer time on site in minutes
    Age: customers age in years
    Area IncomeL Avg income of geographical area of consumer
    Daily Internet Usage: Ave minutes a day consumer is on the internet
    Ad Topic Line: headline of the advertisement
    City: City of consumer
    Male: whether or not consumer was male
    Country: Country of consumer
    Timestamp: Time at which consumer clicked on Ad or closed window
    Clicked on Ad: 0 or 1 indicated clicking on Ad
     */
      ///////////////////
     /// GET THE DATA //
    ///////////////////

    //create a spark session

    val spark = SparkSession
      .builder()
      .master("local[*]")
      .getOrCreate()

    setupLogging()

    //use spark to read in the advertising cvs file
    val data = spark.read.option("header","true")
      .option("inferSchema", "true")
      .format("csv")
      .load("C:\\Users\\Zenith\\Documents\\BigData\\data\\advertising.csv")

    //Print Scheme of the DataFrame
    data.printSchema()

    //////////////////////
    // display the data //
    //////////////////////

    //print out sample row
    val colnames = data.columns
    val firstrow = data.head(1)(0)

    println("Print out first row")
    for(ind<-Range(0, colnames.length)){
      println(s"${colnames(ind)}: ${firstrow(ind)}")
    }

    ///////////////////////////////////////////////
    // SET UP THE DATAFRAME FOR MACHINE LEARNING //
    ///////////////////////////////////////////////

    // rename the Clicked on Ad Column to label
    // grab columns
    import spark.implicits.StringToColumn
    val logregdatall = data.select(data("Clicked on Ad").as("label"), $"Daily Time Spent on Site", $"Age", $"Area Income", $"Daily Internet Usage", $"Timestamp", $"Male")
    val logregdata = logregdatall.na.drop()

    // create a new column called hour from the timestamp contains the hour of the clock
    val newColumn = logregdata.withColumn("Hour", hour(logregdata("Timestamp")))

    //create a new VectorAssembler object called assembler for the feature columns as the input.
    //set the output column to be called features
    val assembler = new VectorAssembler()
      .setInputCols(Array("Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage", "Hour", "Male"))
      .setOutputCol("features")
    val output = assembler.transform(newColumn).select($"label", $"features")

    //use randomsplit to create a train split of 70/30
    val Array(training, test) = newColumn.randomSplit(Array(0.7, 0.3))

    /////////////////////////
    // Set up the pipeline //
    /////////////////////////

    //import pipeline
    import org.apache.spark.ml.Pipeline

    //create a new pipeling with the stages assembler, lr
    val lr = new LogisticRegression()
    val pipeline = new Pipeline().setStages(Array(assembler, lr))

    // fit the pipeline to the training set
    val model = pipeline.fit(training)

    //get results on test set with transform
    val results = model.transform(test)

    ///////////////////////
    // MODEL EVALUTATION //
    ///////////////////////

    // FOR METRICS AND EVALUTION IMPORT MULTICLASS METRICS
    import org.apache.spark.mllib.evaluation.MulticlassMetrics
    import spark.implicits._

    // Convert the test results to an rdd using .as .rdd
    val predictionAndLables = results.select($"prediction", $"label").as[(Double,Double)].rdd

    //instantiate a new multiclass metrics object
    val metrics = new MulticlassMetrics(predictionAndLables)

    // print out the confusion matrix
    println("\nConfussion matrix\n" + metrics.confusionMatrix)
  }
}
