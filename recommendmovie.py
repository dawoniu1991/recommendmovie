from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql import SparkSession
spark = SparkSession.builder.master('spark://hadoop-server-00:7077').appName('apply').getOrCreate()
sc = spark.sparkContext
rdd = sc.textFile('/datas/movielen_data.txt')

rdd1 = rdd.map(lambda x:x.split('::'))
print(rdd1.collect()[:2])

rdd2 = rdd1.map(lambda x:Row(userId=int(x[0]),movieId=int(x[1]),rating=float(x[2]),timestamp=long(x[3])))

df = spark.createDataFrame(rdd2)
df.show(10)

train,test = df.randomSplit([0.8,0.2])

als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
coldStartStrategy="drop")
model = als.fit(train)
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# 为每个用户推荐10部电影
userRecs = model.recommendForAllUsers(10)
# 为每部电影推荐10个用户
movieRecs = model.recommendForAllItems(10)
#指定用户
users = [Row(userId=26), Row(userId=29), Row(userId=19)]
users_data_set = spark.createDataFrame(users)
userSubsetRecs = model.recommendForUserSubset(users_data_set , 10)
# 未指定电影推荐10个用户
movies = [Row(movieId=26), Row(movieId=29), Row(movieId=65)]
movies_data_set = spark.createDataFrame(movies)
movieSubSetRecs = model.recommendForItemSubset(movies_data_set , 10)
movieSubSetRecs.show(truncate=False)

#保存和加载训练好的模型
from pyspark.ml.recommendation import ALSModel
model.write().overwrite().save('/models/als/')
model1 = ALSModel.load('/models/als/')