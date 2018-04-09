import org.apache.spark.sql.SparkSession
import Utilities._
import org.apache.spark.ml.regression.LinearRegression
//Import LinearRegression

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
//import Vector Assembler and vectors

object HW2 {
  def main(args: Array[String]): Unit = {
    //Start a simple Spark session
    val spark = SparkSession
      .builder
      .master("local[*]") //access all local cores
      .getOrCreate()

    setupLogging()

    //Use spark to read in the ecommerce customers file
    val data = spark.read.option("header", "true")
      .option("inferSchema", "true")
      .format("csv")
      .load("C:\\Users\\Zenith\\Documents\\BigData\\data\\Ecommerce Customers")

    //print the schema of the dataframe
    println("\n Show schema")
    data.printSchema()
    println()

    //Print out an example Row
    val colnames=data.columns
    val firstrow = data.head(1)(0)

    println("\nEaxmple data row \n_______________________")
    for(ind <- Range(0,colnames.length)){
      println(firstrow(ind))
    }
    println()

    ///////////////////////////////////////////////
    // Setting up DataFrame for Machine Learning //
    ///////////////////////////////////////////////

    // Create Label and features column
    // Import Vector Assembler and vectors at the top
    //Rename the yearly amount spent column as lable
    //also only grab numerical columns from the data
    //set as new dataframe named df

    import spark.implicits.StringToColumn

    val df = data.select(data("Yearly Amount Spent").as("label"),
      $"Avg Session Length",
      $"Time on App",
      $"Time on Website",
      $"Length of Membership")

    //an assembler converts the input values to a vector
    //a vector is what the ML algorithm reads to train a model
    //use VectorAssembler to convert the input columns of df
    //to a single output column of an array called "features"
    // set the input columns from which we are supposed to read the values
    //call this new object assembler

    val assembler = new VectorAssembler().setInputCols(
      Array("Avg Session Length",
        "Time on App",
        "Time on Website",
        "Length of Membership")
    ).setOutputCol("features")


    //use the assember to transform out dataframe to the two columns: label and features
    val output = assembler.transform(df).select($"label", $"features")

    // create a linear regression model object
    val lr = new LinearRegression()

    //fit the model to the data and call this model IrModel
    val lrModel = lr.fit(output)

    //print the coefficients and intercept for linear regression
    println(s"Coefficients: ${lrModel.coefficients} \nIntercept: ${lrModel.intercept}")
    println()

    //summarize the model over the training set and print out some metrics
    //use the .summary method of you model to create an object called training summary
    val trainingSummary = lrModel.summary

    //Show the residuals, the rmse, the mse, and the r^2 values
    trainingSummary.residuals.show(5)
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}\n")
    println(s"MSE: ${trainingSummary.meanSquaredError}\n")
    println(s"r2: ${trainingSummary.r2}")

  }
}
