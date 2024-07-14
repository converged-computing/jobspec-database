# import unimodal_model as um
# import pandas as pd
import tensorflow as tf
import numpy as np
print("\n\n---GPU STATUS---")
print("----------------")
print(tf.config.list_physical_devices('GPU'))
print("----------------")

# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = np.random.random((1000, 784))
y_train = np.random.random((1000, 10))

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(tf.convert_to_tensor(x_train), tf.convert_to_tensor(y_train), epochs=5)



"""




# Define a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Generate some dummy data for testing
import numpy as np
x_train = np.random.random((1000, 784))
y_train = np.random.random((1000, 10))

# Train the model
model.fit(tf.convert_to_tensor(x_train, tf.float32), tf.convert_to_tensor(y_train, tf.float32), epochs=5, batch_size=4)

# Evaluate the model on some test data
x_test = np.random.random((100, 784))
y_test = np.random.random((100, 10))
loss, acc = model.evaluate(tf.convert_to_tensor(x_test), tf.convert_to_tensor(y_test))

# Print the test accuracy
print('Test accuracy:', acc)

# CLINICAL = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_clinical_data.csv"
# CNV = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_cnv_data.csv"
# training_cases = list(pd.read_csv("/users/anair27/data/TCGA_Data/project_LUAD/data_processed/training_cases.csv")["case_id"])
# testing_cases = list(pd.read_csv("/users/anair27/data/TCGA_Data/project_LUAD/data_processed/testing_cases.csv")["case_id"])
# # for CNV data
# clinical_df = pd.read_csv(CLINICAL)
# cnv_df = pd.read_csv(CNV)
# diagnosis = clinical_df[["vital_status_Dead", "case_id"]]
# data = cnv_df.merge(diagnosis, on = "case_id").drop_duplicates()
# data_training = data[data["case_id"].isin(training_cases)]
# data_testing = data[data["case_id"].isin(testing_cases)]
# x_train_raw = data_training.loc[:, data.columns != "vital_status_dead"].iloc[:,1:]
# x_test = data_testing.loc[:, data.columns != "vital_status_dead"].iloc[:,1:]
# y_train_raw= data_training["vital_status_Dead"].astype(int)
# y_test = data_testing["vital_status_Dead"].astype(int)
# um.create_model(x_train=x_train_raw, y_train=y_train_raw)


"""