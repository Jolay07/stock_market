package com.github.jolay_07.stockMarket

import org.apache.spark.sql.functions.{avg, col, desc}

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

  val frequently = df
    .withColumn("frequently", (col("close") * (col("volume"))))
  frequently.show(10,false)

//the most sold ticker in 2015/2016
  frequently
    .groupBy("ticker")
    .agg(avg("frequently"))
    .orderBy(desc("avg(frequently)"))
    .show(10, false)



}
