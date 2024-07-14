import os
import torch
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from huggingface_hub import login
from src.eval import get_predictions
from src.utils import seed_everything
from src.preprocess import create_test_prompt, get_disease_name
import src.config as config
from transformers import LlamaForCausalLM, LlamaTokenizer

load_dotenv()

login(token=os.getenv('HF_TOKEN'))


# Confirm that the GPU is detected
assert torch.cuda.is_available()

# Get the GPU device name.
device_name = torch.cuda.get_device_name()
n_gpu = torch.cuda.device_count()
print(f"Found device: {device_name}, n_gpu: {n_gpu}")
device = torch.device("cuda")
# Empty the cache
torch.cuda.empty_cache()

# Set seed for reproducibility
seed_everything()

# Set variables to be used for saving and loading results
text_column = "Text"
label_column = "Text_label"
prompt = "prompt2"
data_path = "./data/test_diabetes_filtered.csv"
output = "MIMIC_inference_small_"
chkpt="/checkpoint-2000"
save_name = output+"_"+config.d_code+"_"+prompt+'_yes_no_'+chkpt[1:]

# Load the data
df = pd.read_csv(data_path)

# Load the model and tokenizer
model =  LlamaForCausalLM.from_pretrained(config.l2_model_local_path)
tokenizer = LlamaTokenizer.from_pretrained(config.l2_model_id)

model.to(device)

# Get full disease name
code_text = get_disease_name(config.d_code)

# Convert each row to a prompt
list_examples =[create_test_prompt(row, code_text, text_column, label_column) for i, row in df.iterrows()]
labels = list(df["Label"].values)

# Get predictions (between 0 and 1)
pred = get_predictions(list_examples, model, tokenizer, config.max_length, config.batch_size, device)

# Store the results into a new df
result_df = pd.DataFrame(columns =["PatientID","Predictions","Label"])
result_df["PatientID"] = df["PatientId"].values
result_df["Predictions"] = pred
result_df["Label"] = labels

# Show results
print(result_df)

# Save results to file
result_df.to_csv("Predictions_" +save_name+".csv", index=False)

# Calculate and show metrics
roc = roc_auc_score(labels, pred)
pr_auc = average_precision_score(labels, pred)
acc = accuracy_score(labels, [1 if i>=0.5 else 0 for i in pred])

print("ROC Area under curve: ", roc)
print("Avg Precision Score: ", pr_auc)
print("Accuracy: ", acc)
