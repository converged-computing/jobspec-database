import h2o
from h2o.estimators import H2ORandomForestEstimator
# Initialize H2O cluster
h2o.init(ip="localhost", port=54321)
# Load the airlines dataset
data = h2o.import_file(path="http://h2o-public-test-data.s3.amazonaws.com/smalldata/airlines/allyears2k_headers.zip")
# Define features and target
features = ["Year", "Month", "DayOfWeek", "DepTime", "UniqueCarrier", "Origin", "Dest", "Distance"]
target = "IsArrDelayed"
# Train a model
model = H2ORandomForestEstimator()
model.train(x=features, y=target, training_frame=data)
# Train a model
model = H2ORandomForestEstimator(ntrees=100, max_depth=50)
model.train(x=features, y=target, training_frame=data)
# Display model performance
print(model)
# Shutdown H2O cluster
h2o.cluster().shutdown()
