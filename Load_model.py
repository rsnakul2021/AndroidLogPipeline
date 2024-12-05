from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml import PipelineModel

spark = SparkSession.builder \
    .appName("LogisticRegressionClassification") \
    .getOrCreate()

model_path = "gs://term_project_cs777/model/"
pipeline_model = PipelineModel.load(model_path)
new_data_path = "gs://term_project_cs777/Book1.csv"
new_data = spark.read.csv(new_data_path, header=True, inferSchema=True)

feature_columns = ["col1"]  
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
new_data_transformed = assembler.transform(new_data)

predictions = pipeline_model.transform(new_data_transformed)
print(predictions.select("features", "prediction", "probability").show())