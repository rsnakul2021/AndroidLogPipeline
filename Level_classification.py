from pyspark.sql import SparkSession
import random
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType
from functools import reduce
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

spark = SparkSession.builder.appName("LogParsing").getOrCreate()

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

#Statistics of the content
filtered_logs_df = filtered_logs_df.withColumn("content_length", length(col("content")))
content_length_stats = filtered_logs_df.select("content_length").describe()
print("Statistics of content in our Data:")
print(filtered_logs_df.select("content_length").describe().show())

#Data preprocessing
def extract_key_value_pairs(text):
    if text:
        pattern = r'(\w+)\s*[:=]\s*([^\s,]+)'  
        pairs = re.findall(pattern, text)
        key_value_map = {key: value for key, value in pairs}

        for key, value in key_value_map.items():
            if value.lower() == 'true':
                key_value_map[key] = True
            elif value.lower() == 'false':
                key_value_map[key] = False
            try:
                key_value_map[key] = float(value)
            except ValueError:
                pass
        return key_value_map
    return {}

extract_kv_udf = udf(extract_key_value_pairs, MapType(StringType(), StringType())) #UDF for Key-Value extraction
logs_with_kv = filtered_logs_df.withColumn("key_value_map", extract_kv_udf(col("content")))
logs_with_kv = logs_with_kv.filter(col("key_value_map").isNotNull() & (size(col("key_value_map")) > 0)) #Remove NULL maps
logs_with_kv = logs_with_kv.withColumn("datetime", F.to_timestamp(F.concat(col("date"), lit(" "), col("time")), "MM-dd HH:mm:ss.SSS")) 
logs_with_kv = logs_with_kv.withColumn( #Replacing NULL values with custom values
    "key_value_map",
    when(col("key_value_map").isNull(), create_map(lit("default_key"), lit("N/A"))) 
    .otherwise(col("key_value_map"))
)

def tokenize_content(text):
    if text:
        return [token for token in text.split() if token.isalpha()] 
    return []

class stopwordsRemover:
    def __init__(self, input_col, output_col, stopwords=None):
        self.input_col = input_col
        self.output_col = output_col
        self.stopwords = stopwords or {"a", "an", "the", "and", "or", "but", "if", "to", "of", "in", "on", "for", "with"}

    def transform(self, df):
        def remove_stopwords(tokens):
            return [word for word in tokens if word not in self.stopwords]
        remove_stopwords_udf = udf(remove_stopwords, ArrayType(StringType()))
        return df.withColumn(self.output_col, remove_stopwords_udf(col(self.input_col)))
    
tokenize_udf = udf(tokenize_content, ArrayType(StringType())) #UDF for tokenization
tokenized_df = logs_with_kv.withColumn("content_tokens", tokenize_udf(col("content")))
stop_words_remover = StopWordsRemover(inputCol="content_tokens", outputCol="content_cleaned") #Stop words remover
cleaned_df = stop_words_remover.transform(tokenized_df)
ngram = NGram(n=2, inputCol="content_cleaned", outputCol="content_ngrams") #Adding 2-gram combination to our feature set 
ngram_df = ngram.transform(cleaned_df)

final_df = ngram_df.select("datetime", "component", "content", "content_tokens", "content_cleaned", "content_ngrams", "key_value_map")
print("Data after preprocessing for key-value map and N-gram:")
print(ngram_df.select("datetime", "component", "content", "content_tokens", "content_cleaned", "content_ngrams", "key_value_map").show(20, truncate=False))

#RegexTokenizer UDF
class RegexTokenizerM:
    def __init__(self, input_col, output_col, pattern=r'\W+'):
        self.input_col = input_col
        self.output_col = output_col
        self.pattern = pattern

    def transform(self, df):
        def tokenize(text):
            return [token.lower() for token in re.split(self.pattern, text) if token]
        tokenize_udf = udf(tokenize, ArrayType(StringType()))
        return df.withColumn(self.output_col, tokenize_udf(col(self.input_col)))

#TF-IDF UDF
class HashingTFM:
    def __init__(self, input_col, output_col, num_features=1000):
        self.input_col = input_col
        self.output_col = output_col
        self.num_features = num_features

    def transform(self, df):
        def compute_tf(tokens):
            term_freq = [0] * self.num_features
            for token in tokens:
                index = hash(token) % self.num_features
                term_freq[index] += 1
            return term_freq
        compute_tf_udf = udf(compute_tf, ArrayType(DoubleType()))
        return df.withColumn(self.output_col, compute_tf_udf(col(self.input_col)))

class IDFManual:
    def __init__(self, input_col, output_col):
        self.input_col = input_col
        self.output_col = output_col

    def fit(self, df):
        num_docs = df.count()
        def compute_idf(tf_column):
            return [log((1 + num_docs) / (1 + freq)) + 1 for freq in tf_column]
        tf_vectors = df.select(self.input_col).rdd.flatMap(lambda row: row[0])
        idf_values = np.zeros(len(tf_vectors.collect()[0]))
        for tf in tf_vectors.collect():
            idf_values += (np.array(tf) > 0).astype(int)
        idf_weights = compute_idf(idf_values)
        def apply_idf(tf_vector):
            return [tf * idf for tf, idf in zip(tf_vector, idf_weights)]
        apply_idf_udf = udf(apply_idf, ArrayType(DoubleType()))
        return df.withColumn(self.output_col, apply_idf_udf(col(self.input_col)))

#String Indexer UDF
class StringIndexerManual:
    def __init__(self, input_col, output_col):
        self.input_col = input_col
        self.output_col = output_col

    def fit(self, df):
        labels = df.select(self.input_col).distinct().rdd.flatMap(lambda x: x).collect()
        self.label_map = {label: idx for idx, label in enumerate(labels)}
        return self

    def transform(self, df):
        def map_label(label):
            return float(self.label_map[label])
        map_label_udf = udf(map_label, FloatType())
        return df.withColumn(self.output_col, map_label_udf(col(self.input_col)))
    
#Pipeline
hashing_tf_content = HashingTF(inputCol="content_cleaned", outputCol="content_tfidf_raw", numFeatures=1000)
idf_content = IDF(inputCol="content_tfidf_raw", outputCol="content_tfidf")
content_tfidf_df = hashing_tf_content.transform(ngram_df)
content_tfidf_df = idf_content.fit(content_tfidf_df).transform(content_tfidf_df)
hashing_tf_ngrams = HashingTF(inputCol="content_ngrams", outputCol="ngrams_tfidf_raw", numFeatures=1000)
idf_ngrams = IDF(inputCol="ngrams_tfidf_raw", outputCol="ngrams_tfidf")
ngrams_tfidf_df = hashing_tf_ngrams.transform(content_tfidf_df)
ngrams_tfidf_df = idf_ngrams.fit(ngrams_tfidf_df).transform(ngrams_tfidf_df)

def flatten_key_value_map(key_value_map): #Flatten Key-Value Pairs to Numeric Features
    flattened = {}
    if key_value_map:
        for key, value in key_value_map.items():
            if isinstance(value, bool):
                flattened[key] = float(value)
            elif isinstance(value, (int, float)):
                flattened[key] = float(value)
    return flattened

flatten_key_value_map_udf = udf(flatten_key_value_map, MapType(StringType(), FloatType()))
logs_with_kv_flattened = ngrams_tfidf_df.withColumn("key_value_flattened", flatten_key_value_map_udf(col("key_value_map")))

flattened_columns = [] #Convert the flattened map into separate columns for each key
for key in logs_with_kv_flattened.select("key_value_flattened").rdd.flatMap(lambda row: row.key_value_flattened.keys()).distinct().collect():
    flattened_column = f"key_value_flattened_{key}"
    logs_with_kv_flattened = logs_with_kv_flattened.withColumn(flattened_column, col("key_value_flattened").getItem(key))
    flattened_columns.append(flattened_column)

component_indexer = StringIndexer(inputCol="component", outputCol="component_indexed") #Indexing components
logs_with_component_indexed = component_indexer.fit(logs_with_kv_flattened).transform(logs_with_kv_flattened)

level_indexer = StringIndexer(inputCol="level", outputCol="level_indexed")
logs_with_labeled = level_indexer.fit(logs_with_component_indexed).transform(logs_with_component_indexed)

assembler = VectorAssembler(
    inputCols=["content_tfidf", "ngrams_tfidf"] + flattened_columns + ["component_indexed"],
    outputCol="features"
)

final_df = assembler.transform(logs_with_labeled)
print("Preprocessed Data:")
print(final_df.select("features", "level_indexed").show(20, truncate=False))

#Final classification stage of the Pipeline
lr = LogisticRegression(featuresCol="features", labelCol="level_indexed", maxIter=10, regParam=0.1, elasticNetParam=0.0)
pipeline = Pipeline(stages=[lr])
train_df, test_df = final_df.randomSplit([0.8, 0.2], seed=1234)
model = pipeline.fit(train_df)
predictions = model.transform(test_df)
evaluator = MulticlassClassificationEvaluator(labelCol="level_indexed", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy of classification model: {accuracy}")
print("Few Predictions:")
print(predictions.select("level_indexed", "prediction").show(20, truncate=False))

#Conf Matrix and Metrics
confusion_matrix = predictions.groupBy("level_indexed", "prediction").count()
confusion_matrix_pivot = confusion_matrix.groupBy("level_indexed") \
    .pivot("prediction") \
    .agg(F.first("count")) \
    .na.fill(0)  

print(confusion_matrix_pivot.show(truncate=False))

data = {
    'label': [0.0, 1.0, 2.0, 3.0, 4.0],
    0.0: [266829, 48685, 151, 158, 149],
    1.0: [23298, 292169, 203, 203, 1208],
    2.0: [11579, 11982, 28508, 145, 91],
    3.0: [547, 15744, 23, 32446, 43],
    4.0: [4286, 5185, 43, 13, 23438]
}

confusion_df = pd.DataFrame(data)
confusion_df.set_index('label', inplace=True)
precision = {}
recall = {}
f1_score = {}
total_sum = confusion_df.sum().sum()

for label in confusion_df.index:
    tp = confusion_df.at[label, label]  
    fp = confusion_df[label].sum() - tp
    fn = confusion_df.loc[label].sum() - tp
    tn = total_sum - tp - fp - fn

    precision[label] = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall[label] = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score[label] = 2 * (precision[label] * recall[label]) / (precision[label] + recall[label]) if (precision[label] + recall[label]) != 0 else 0

print("Precision, Recall, and F1 Score:")
for label in confusion_df.index:
    print(f"Label {label}: Precision = {precision[label]:.4f}, Recall = {recall[label]:.4f}, F1 Score = {f1_score[label]:.4f}")
    
#Saving the model
model_save_path = "gs://term_project_cs777/model"
model.save(model_save_path)