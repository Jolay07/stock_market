package com.github.jolay_07.stockMarket

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
}
