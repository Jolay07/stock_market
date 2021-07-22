package com.github.jolay_07.stockMarket

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{DataFrame, SparkSession}

object utilities {
  /**
   * @param appName Name of app
   * @param verbose spark version
   * @param master local
   * @return start spark session
   */
  def createSpark(appName:String, verbose:Boolean = true, master: String= "local"): SparkSession = {
    if (verbose) println(s"$appName with Scala version: ${util.Properties.versionNumberString}")

    val spark = SparkSession.builder().appName(appName).master(master).getOrCreate()
    spark.conf.set("spark.sql.shuffle.partitions", "5") //recommended for local, default is 200?
    if (verbose) println(s"Session started on Spark version ${spark.version}")
    spark
  }

  /**
   * Evaluate accuracy of algorithms for Classification and Regression
   * @param df Data Frame
   */
  def showAccuracy(df: DataFrame): Unit = {
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(df)
    println(s"DF size: ${df.count()} Accuracy $accuracy - Test Error = ${1.0 - accuracy}")
  }
}
