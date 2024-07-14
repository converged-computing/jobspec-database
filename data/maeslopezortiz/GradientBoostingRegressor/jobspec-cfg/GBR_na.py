#!/usr/bin/env python3
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor  # Import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error  # Import mean_squared_error
import numpy as np  # Import numpy for array operations
import csv  # Import CSV library for file handling
import random  # Import random for random sampling
import sys
import pandas as pd
from math import sqrt

SPECIES = sys.argv[1]
TRAIT = int(sys.argv[2])
CV = int(sys.argv[3])
FOLD = int(sys.argv[4])

#Display input parameters
print("Species: ", SPECIES)
print('Trait: ', TRAIT)
print('Cross fold: ', CV)
print('Fold: ', FOLD)


# Load data using an external script (LoadEasyGeSe.py)

with open("~/GenoPredict/EasyGeSe/LoadEasyGeSe.py") as f:
    exec(f.read())# Execute the script to load data

X, Y = LoadEasyGeSeData(SPECIES)
Ydf = pd.DataFrame(Y)
Xdf = pd.DataFrame(X)
# Handle missing values in both DataFrames
Ydf.replace("NA", np.nan, inplace=True)
Ydf = Ydf.dropna()
print('removing NAs values')
Xdf = Xdf.loc[Ydf.index]

# Convert to NumPy arrays
Y = Ydf.to_numpy()
X = Xdf.to_numpy()
print(Y.shape)
print(X.shape)

output_directory = "/~/GenoPredict/GBR_results/"
filename = f"{output_directory}GenPred_estimators_{SPECIES}{TRAIT}{CV}{FOLD}.csv"

resfile = open(filename,"w")


#csv_writer = csv.writer(fh)
csv_writer = csv.writer(resfile)

# write one row with headers (using `writerow` without `s` at the end)
csv_writer.writerow(["SPECIES","TRAIT","CV", "FOLD", "cor", "rmse", "markers", "usefulmarkers", "estimators"])

# List to store all results
all_results = [ ] # <-- list for all results

random.seed(CV)
print('setting up masks')
MASK = random.choices([1,2,3,4,5],k=X.shape[0])
print(MASK)
#%%
mm = np.array(MASK)
print('masking')
x_train = X[mm != FOLD,1:].astype('float64')
x_test = X[mm == FOLD,1:].astype('float64')
y_train = Y[mm != FOLD,TRAIT].astype('float64')
y_test = Y[mm == FOLD,TRAIT].astype('float64')
#%%
print('finished masking')

# the CODE HERE
print(x_train.shape)
#(259, 23590)
print(x_test.shape)
#(65, 23590)
print(X.shape)
#(324, 23591)

reg = GradientBoostingRegressor(random_state=0, n_estimators=1000)
reg.fit(x_train,y_train)

print(y_test)
#%%
y_pred = reg.predict(x_test)
cor1 = np.corrcoef(y_test,y_pred)[0,1] #correlation coefficient
rmse1 = sqrt(mean_squared_error(y_test,y_pred)) #root mean squared error
markers1 = x_train.shape[1]
print('correlation coefficient:', cor1)
print('root squared error:', rmse1)
print('total number of markers:', markers1)
feature_importance = reg.feature_importances_
sorted_idx = np.argsort(feature_importance)
onlyimportantlogic = feature_importance > 0
onlyimportant = feature_importance[onlyimportantlogic]
print(onlyimportant.shape)
p0 = sorted_idx[onlyimportantlogic]
usefulmarkers = p0.shape[0]
result1= [SPECIES,TRAIT,CV, FOLD,cor1, rmse1, markers1, usefulmarkers, 1000]
all_results.append( result1 )
markers = [50,100, 200, 300, 400, 500, 1000, 2000]

for i in markers:
    new_markers = p0[-i:]
    new_markers = new_markers.astype(int)  # Convert new_markers to an array of integers
    x_train_new = x_train[:, new_markers].astype('float64')
    x_test_new = x_test[:, new_markers].astype('float64')
    reg_new = GradientBoostingRegressor(random_state=0, n_estimators=int(i/2))  # Convert i/2 to an integer
    reg_new.fit(x_train_new, y_train)
    y_pred_new = reg_new.predict(x_test_new)
    cor = np.corrcoef(y_test, y_pred_new)[0, 1]
    rmse = mean_squared_error(y_test, y_pred_new)
    feature_importance_new = reg_new.feature_importances_
    sorted_idx_new = np.argsort(feature_importance_new)
    onlyimportantlogic_new = feature_importance_new > 0
    onlyimportant_new = feature_importance_new[onlyimportantlogic_new]
    usefulmarkers_new = onlyimportant_new.shape[0]
    result = [SPECIES, TRAIT, CV, FOLD, cor, rmse, i, usefulmarkers_new, int(i/2)]  # Convert i/2 to an integer
    all_results.append(result)



# write many rows with results (using `writerows` with `s` at the end)
csv_writer.writerows(all_results)
resfile.close()
