package com.github.jolay_07.stockMarket

object ReadParquet extends App{

  val spark = utilities.createSpark("readParquet")
  val pFilePath = "./src/resources/parquet/dailyReturn"

  val dfParquet = spark.read
    .format("parquet")
    .load(pFilePath)

  dfParquet.printSchema()
  dfParquet.show(false)

}
