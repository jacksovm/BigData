import org.apache.spark.sql.SparkSession

import Utilities._

object HW1 {
  def main(args: Array[String]): Unit = {

    //Start Spark session
    val spark = SparkSession
      .builder
      .master("local[*]") //access all local cores
      .getOrCreate()

    setupLogging()

    val df = spark.read.option("header", "true")
      .option("inferSchema", "true") // infer data types
      .csv("C:\\Users\\Zenith\\Documents\\Big Data\\data\\Netflix_2011_2016.csv")
    // load netflix stock csv file

    import spark.implicits.StringToColumn
    import org.apache.spark.sql.functions._

    println("\n Show schema") //what does schema look like
    df.printSchema()
    // column names:
    println()

    println("First 5")
    df.show(5)
    println()

    println("Describe") //learn about data frame
    df.describe().show()
    println()

    println("create new dataframe with a column called hv ratio with ratio of high price to stock traded for a day")
    val df1 = df.withColumn("HV Ratio", df("High") / df("Volume"))
    df1.show(5)

    println("Highest Peek Price")
    df1.orderBy(desc("High")).show(1)
    println()

    println("Mean of close column")
    df1.groupBy("Close").mean().select("avg(Close)").show()
    println()


    println("max and min of the Volume column")
    println("max:")
    df1.orderBy(desc("Volume")).show(1)
    println("min:")
    df1.orderBy("Volume").show(1)
    println()


    println("$ Syntax")
    println("max: ")
    df1.orderBy($"Volume".desc).show(1)
    println("min")
    df1.orderBy($"Volume").show(1)


    println()
    val df2 = df1.filter($"Close" < 600)
    println("Days Close lower than $600: "+df2.count())
    println()

    val df3 = df1.filter($"High" > 500)
    println("Days High greater than $500: "+df3.count())
    println()

    import org.apache.spark.sql.functions.corr
    println("Pearson correlation between High and Volume")
    df1.select(corr("High", "Volume")).show()
    println()

    println("Max High")
    df1.orderBy($"High".desc).show(1)
    println()

    println("Average Close each month")
    val df4 = df1.withColumn("month", month(df1("Date")))
    df4.groupBy("month").mean().select("month","avg(Close)").show()


  }
}
