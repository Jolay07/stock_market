package com.github.jolay_07.stockMarket

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.sql.functions.{avg, col, desc, expr, stddev}

object stockAnalysis extends App{

  val spark = utilities.createSpark("stockAnalysis")
  val filePath = "./src/resources/csv/stock_prices.csv"

  val df = spark.read
    .format("csv")
    .option("header", true)
    .option("inferSchema", true)
    .option("path", filePath)
    .load

  df.printSchema()
  df.describe().show(false)
  df.show(5)

  val dailyReturn = df
    .withColumn("daily_return", (col("close") - col("open"))/col("open")*100)
  dailyReturn.show(10, false)

//Daily return grouped by date
  dailyReturn
    .groupBy("date")
    .agg(avg("daily_return"))
    .orderBy(desc("avg(daily_return)"))
    .show(10, false)

//saving daily return increase in parquet file
  dailyReturn.write
    .format("parquet")
    .mode("overwrite")
    .save("./src/resources/parquet/dailyReturn")

  dailyReturn.select(stddev("daily_return")).show(false)

  val frequently = df
    .withColumn("frequently", col("close") * col("volume"))
  frequently.show(10,false)

//the most sold ticker in 2015/2016
  frequently
    .groupBy("ticker")
    .agg(avg("frequently"))
    .orderBy(desc("avg(frequently)"))
    .show(10, false)


  val dfDayBefore = df.withColumn("prevOpen", expr("" +
    "LAG (open,1,0) " +
    "OVER (PARTITION BY ticker " +
    "ORDER BY date )"))
    .withColumn("prevClose", expr("" +
      "LAG (close,1,0) " +
      "OVER (PARTITION BY ticker " +
      "ORDER BY date )"))
  dfDayBefore.show(10, false)

  import org.apache.spark.ml.feature.RFormula
  val supervised = new RFormula()
    .setFormula("open ~ prevOpen + prevClose ")

  val ndf = supervised
    .fit(dfDayBefore) //prepares the formula
    .transform(dfDayBefore) //generally transform will create the new data

  ndf.show(10, false)

  val endDf = ndf.where("prevOpen != 0.0") // null isn't needed
  endDf.show(10, false)

  val featureIndexer = new VectorIndexer()
    .setInputCol("features")
    .setOutputCol("indexedFeatures")
    .setMaxCategories(4)
    .fit(endDf)

  // Split the data into training and test sets (30% held out for testing).
  val Array(trainingData, testData) = endDf.randomSplit(Array(0.7, 0.3))

  // Train a DecisionTree model.
  val dt = new DecisionTreeRegressor()
    .setLabelCol("label")
    .setFeaturesCol("indexedFeatures")
  // Chain indexer and tree in a Pipeline.
  val pipeline = new Pipeline()
    .setStages(Array(featureIndexer, dt))

  // Train model. This also runs the indexer.
  val model = pipeline.fit(trainingData)

  // Make predictions.
  val predictions = model.transform(testData)
  predictions.show(10, false)

// In utilities we have function
  utilities.showAccuracy(predictions)


}
