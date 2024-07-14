import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pandas as pd


# PARAMETERS:
# SMOTE ratio: sampling_strategy= <0.x>
# SMOTE n_neighbors
# Network Hidden Nodes
# Training Epochs

data = pd.read_csv('./Crash_Data.csv')

# fill in the columns where crash is not work zone related
data['WORK_ZONE_LOCATION'].fillna('Not Applicable', inplace=True)
data['WORK_ZONE_TYPE'].fillna('Not Applicable', inplace=True)

# For binary
data['CRASH_SEVERITY'] = data['CRASH_SEVERITY'].replace(['B', 'C', 'O'], 'NOT FATAL')
data['CRASH_SEVERITY'] = data['CRASH_SEVERITY'].replace(['A', 'K'], 'FATAL')

# Drop unnecessary columns with missing data
columns_to_drop = ['SPEED_DIFF_MAX', 'FUN', 'FAC', 'MPO_NAME', 'NODE', 'OFFSET', 'RNS_MP', 'OBJECTID', 'DOCUMENT_NBR', 'PHYSICAL_JURIS', 'RTE_NM', 'VSP', 'JURIS_CODE', 'K_PEOPLE', 'A_PEOPLE', 'B_PEOPLE', 'C_PEOPLE', 'PERSONS_INJURED']

data.drop(columns=columns_to_drop, inplace=True)

data.dropna(subset=['X', 'Y'], inplace=True)
data = data.reset_index(drop=True)

# encode date
from datetime import datetime

# 2015/10/16 03:59:59+00 -> number
def date_string_convert(date_string):
    length = len(date_string)
    date_string = date_string[0:length-3]
    date_obj = datetime.strptime(date_string, "%Y/%m/%d %H:%M:%S")
    timestamp = date_obj.timestamp()
    return timestamp

data["CRASH_DT"] = data["CRASH_DT"].apply(date_string_convert)

X = data.drop(columns=['CRASH_SEVERITY'])
y = data['CRASH_SEVERITY'].copy()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y_encoded, test_size=0.20, stratify=y, random_state=42)



import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

one_hot_features = ["COLLISION_TYPE","WEATHER_CONDITION","LIGHT_CONDITION","ROADWAY_SURFACE_COND","RELATION_TO_ROADWAY","ROADWAY_ALIGNMENT","ROADWAY_SURFACE_TYPE",
                    "ROADWAY_DEFECT","ROADWAY_DESCRIPTION","INTERSECTION_TYPE","TRAFFIC_CONTROL_TYPE","TRFC_CTRL_STATUS_TYPE","WORK_ZONE_RELATED","WORK_ZONE_LOCATION",
                    "WORK_ZONE_TYPE","SCHOOL_ZONE","FIRST_HARMFUL_EVENT","FIRST_HARMFUL_EVENT_LOC","ALCOHOL_NOTALCOHOL","ANIMAL","BELTED_UNBELTED","BIKE_NONBIKE",
                    "DISTRACTED_NOTDISTRACTED","DROWSY_NOTDROWSY","DRUG_NODRUG","GR_NOGR","HITRUN_NOT_HITRUN","LGTRUCK_NONLGTRUCK","MOTOR_NONMOTOR","PED_NONPED","SPEED_NOTSPEED",
                    "RD_TYPE","INTERSECTION_ANALYSIS","SENIOR_NOTSENIOR","YOUNG_NOTYOUNG","MAINLINE_YN","NIGHT","VDOT_DISTRICT","AREA_TYPE","SYSTEM","OWNERSHIP","PLAN_DISTRICT"]

num_features = ["X", "Y", "CRASH_YEAR", "CRASH_DT", "CRASH_MILITARY_TM", "PEDESTRIANS_KILLED", "VEH_COUNT", "PEDESTRIANS_INJURED"]

categorical_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_pipeline, one_hot_features),
        ('num', numerical_pipeline, num_features)
    ])


X_cleaned_train = preprocessor.fit_transform(X_train)
X_cleaned_test = preprocessor.transform(X_test)

# Apply SMOTE to resample data
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE # Oversampling technique to add synthetic data for unbalanced classes
from cuml.neighbors import NearestNeighbors # using cuML for GPU optimization for SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

nn = NearestNeighbors(n_neighbors=8)
enn = EditedNearestNeighbours(n_neighbors=nn)
smoteObj = SMOTE(sampling_strategy=0.6, k_neighbors=nn)
X_resampled, y_resampled = SMOTEENN(sampling_strategy=0.5, smote=smoteObj, enn=enn).fit_resample(X_cleaned_train, y_train)

mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():

    # Binary ANN classifier
    model = keras.Sequential([
        keras.layers.Dense(X_cleaned_train.shape[1], activation='relu', input_shape=(X_cleaned_train.shape[1],)),
        keras.layers.Dense(40, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 'accuracy'])
    
    history = model.fit(X_resampled.toarray(), y_resampled, epochs=100, validation_split=0.2, batch_size=256)
    
# Testing and Eval
test_loss, test_precision, test_recall, test_accuracy = model.evaluate(X_cleaned_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
print(f'Test Precision: {test_precision * 100:.2f}%')
print(f'Test Recall: {test_recall * 100:.2f}%')

# Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt
y_pred = model.predict(X_cleaned_test)
y_pred_binary = (y_pred > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred_binary)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['FATAL', 'NOT FATAL'])
disp.plot(cmap='Blues', values_format='d')
plt.show()