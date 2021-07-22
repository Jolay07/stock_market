package com.github.jolay_07.stockMarket

import org.apache.spark.sql.SparkSession

object utilities {
  /**
   * @param appName
   * @param verbose
   * @param master
   * @return
   */
  def createSpark(appName:String, verbose:Boolean = true, master: String= "local"): SparkSession = {
    if (verbose) println(s"$appName with Scala version: ${util.Properties.versionNumberString}")

    val spark = SparkSession.builder().appName(appName).master(master).getOrCreate()
    spark.conf.set("spark.sql.shuffle.partitions", "5") //recommended for local, default is 200?
    if (verbose) println(s"Session started on Spark version ${spark.version}")
    spark
  }

}
