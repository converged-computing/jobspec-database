import numpy as np
import xgboost as xgb
import pandas as pd

diffimg_train = pd.read_csv('./data/eit/diffs_imgs_train.csv', header=None)
X_train = diffimg_train.iloc[:, 0:256].values
y_train = diffimg_train.iloc[:, 256:576].values

diffimg_test = pd.read_csv('./data/eit/diffs_imgs_test.csv', header=None)
X_test = diffimg_test.iloc[:, 0:256].values
y_test = diffimg_test.iloc[:, 256:576].values

# 3. Model Training
model = xgb.XGBRegressor(objective="reg:squarederror")
model.fit(X_train, y_train)

# 4. Model Evaluation
predictions = model.predict(X_test)

# 5. Compute mseloss
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predictions)
print("MSE: %.2f" % mse)