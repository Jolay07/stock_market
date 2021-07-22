package com.github.jolay_07.stockMarket

import org.apache.spark.ml.classification.{DecisionTreeClassifier, RandomForestClassifier}
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


  val fittedRF = supervised.fit(endDf)
  val preparedDF = fittedRF.transform(endDf)
  preparedDF.show(false)
  preparedDF.sample(0.1).show(false)

  val Array(train, test) = preparedDF.randomSplit(Array(0.75, 0.25))
  val randomForest = new RandomForestClassifier()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setNumTrees(10) //this is so called hyperparameter for this particular algorithm

  val fittedModel = randomForest.fit(train) //create a model
  val testDF = fittedModel.transform(test)  //use this model to make predictions and save them in a DataFrame
  testDF.show(30,false)
}
