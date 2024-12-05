from pyspark.sql.functions import lag, avg, stddev, col, minute, dayofmonth, month, to_timestamp, concat, lit
from pyspark.sql.window import Window
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
import random
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType
from functools import reduce
from pyspark.sql.functions import when
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, col, udf, count, when
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import length
from pyspark.sql.functions import udf, col, size, lit, when, create_map
from pyspark.sql.types import MapType, StringType, TimestampType, ArrayType
from pyspark.ml.feature import StopWordsRemover, NGram
from datetime import datetime
from pyspark.ml.feature import HashingTF, IDF, StringIndexer, VectorAssembler
from pyspark.sql.functions import col, udf
from pyspark.sql.types import MapType
from pyspark.sql.types import ArrayType, StringType, FloatType, DoubleType
from pyspark.sql.functions import array, log
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import re
import pandas as pd
import numpy as np
import sys
import pandas as pd

spark = SparkSession.builder.appName("AnomalyDetection").getOrCreate()

#Data ingestion and preprocessing
pattern = r"(\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2}\.\d{3})\s+(\d+)\s+(\d+)\s+(\w)\s+(\w+):\s+(.*)"

def process_log(file_path, pattern):
    log_df = spark.read.text(file_path)
    return log_df.select(
        regexp_extract(col("value"), pattern, 1).alias("date"),
        regexp_extract(col("value"), pattern, 2).alias("time"),
        regexp_extract(col("value"), pattern, 3).cast("int").alias("pid"),
        regexp_extract(col("value"), pattern, 4).cast("int").alias("tid"),
        regexp_extract(col("value"), pattern, 5).alias("level"),
        regexp_extract(col("value"), pattern, 6).alias("component"),
        regexp_extract(col("value"), pattern, 7).alias("content")
    )

log_files = sys.argv[1:]
processed_logs = [process_log(file, pattern) for file in log_files]
all_logs_df = reduce(DataFrame.unionAll, processed_logs)
row_count = all_logs_df.count()
print(f"Number of rows in all_logs_df: {row_count}")
filtered_logs_df = all_logs_df.filter(all_logs_df.level.isin("D", "I", "E", "W", "V"))
log_level_count = filtered_logs_df.groupBy("level").count().sort("count", ascending=False)
print("Counts per level in dataset:")
print(log_level_count.show())

#Pipeline and feature creation for Anomalies
logs_with_datetime = filtered_logs_df.withColumn("datetime", concat(col("date"), lit(" "), col("time")))
logs_with_datetime = logs_with_datetime.withColumn("datetime", to_timestamp(col("datetime"), "MM-dd HH:mm:ss.SSS"))

logs_with_datetime_features = logs_with_datetime.withColumn("minute", minute(col("datetime"))) \
    .withColumn("day", dayofmonth(col("datetime"))) \
    .withColumn("month", month(col("datetime")))

logs_aggregated = logs_with_datetime_features.groupBy("datetime", "minute", "day", "month", "level").count()

logs_with_lags = logs_aggregated.withColumn("prev_count", lag("count", 1).over(Window.orderBy("datetime"))) #Lag features for the data
logs_with_lags = logs_with_lags.withColumn("prev_5_count", lag("count", 5).over(Window.orderBy("datetime")))
logs_with_lags = logs_with_lags.withColumn("rolling_mean", avg("count").over(Window.orderBy("datetime").rowsBetween(-5, 0))) #Calculate rolling statistics (mean, stddev) over a window of 5 time intervals
logs_with_lags = logs_with_lags.withColumn("rolling_stddev", stddev("count").over(Window.orderBy("datetime").rowsBetween(-5, 0)))

logs_with_lags = logs_with_lags.withColumn("anomaly", #Anomaly is defined as a significant deviation in count (e.g., count > rolling_mean + 3 * rolling_stddev)
    (col("count") > col("rolling_mean") + 0.5 * col("rolling_stddev")).cast("int"))

logs_with_lags = logs_with_lags.filter(col("anomaly").isNotNull())  #Remove rows with null anomaly

assembler = VectorAssembler(
    inputCols=["prev_count", "prev_5_count", "rolling_mean", "rolling_stddev", "minute", "day", "month"],
    outputCol="features"
)
logs_with_features = assembler.transform(logs_with_lags)
logs_with_lags_filled = logs_with_lags.withColumn(
    "prev_5_count",
    when(col("prev_5_count").isNull(), 0).otherwise(col("prev_5_count"))
)

print("Features:")
print(logs_with_lags_filled.show(5, truncate=False))

#Pipeline to detect Anomalies
assembler = VectorAssembler(
    inputCols=["prev_count", "prev_5_count", "rolling_mean", "rolling_stddev", "minute", "day", "month"],
    outputCol="features"
)

logs_with_features = assembler.transform(logs_with_lags_filled)
print(logs_with_features.printSchema())
level_indexer = StringIndexer(inputCol="level", outputCol="label")
logs_with_labels = level_indexer.fit(logs_with_features).transform(logs_with_features)
print(logs_with_labels.select("datetime", "level", "label").show(5, truncate=False))

train_data, test_data = logs_with_labels.randomSplit([0.8, 0.2], seed=42)
rf_classifier = RandomForestClassifier(labelCol="anomaly", featuresCol="features", numTrees=10)
rf_model = rf_classifier.fit(train_data)
predictions = rf_model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(labelCol="anomaly", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")

conf_matrix_df = predictions.groupBy("prediction", "anomaly").count()
conf_matrix_pivot = conf_matrix_df.groupBy("prediction").pivot("anomaly").sum("count").fillna(0)
print(conf_matrix_pivot.show())