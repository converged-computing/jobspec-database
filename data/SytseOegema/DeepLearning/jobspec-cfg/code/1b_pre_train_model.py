from datasets import load_dataset
from transformers import DefaultDataCollator, create_optimizer, TFAutoModelForQuestionAnswering, DistilBertConfig
import tensorflow as tf
from helpers.preprocessing import preprocess_function
from helpers.load_data import get_squad_data, get_squad_data_small
from helpers.store_results import store_dictionary, create_specific_folder

batch_size = 16
num_epochs = 2
data_base_path = "/data/s3173267/pre_BERT/"

pre_trained_path = ""
# pre_trained_path = "../data/pretrained_model/"

data = get_squad_data(data_base_path + "squad.dat")

# add folder for this specific run
# data_base_path = create_specific_folder(data_base_path)
print("output folder: " + data_base_path + "\n\n\n")



tokenized_data = data.map(preprocess_function, batched=True, remove_columns=data["train"].column_names)
data_collator = DefaultDataCollator(return_tensors="tf")

tf_train_set = tokenized_data["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "start_positions", "end_positions"],
    dummy_labels=True,
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)

tf_validation_set = tokenized_data["validation"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "start_positions", "end_positions"],
    dummy_labels=True,
    shuffle=False,
    batch_size=batch_size,
    collate_fn=data_collator,
)


total_train_steps = (len(tokenized_data["train"]) // batch_size) * num_epochs
print(total_train_steps)

optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=0,
    num_train_steps=total_train_steps,
)


model = TFAutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")

model.compile(optimizer=optimizer)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=data_base_path + "checkpoints/cp-{epoch:04d}.ckpt",
    verbose=1,
    save_weights_only=True)

history = model.fit(
    x=tf_train_set,
    validation_data=tf_validation_set,
    validation_freq=1,
    epochs=num_epochs)

store_dictionary(history.history, data_base_path)

model.save_pretrained(data_base_path + "pretrained_model")
