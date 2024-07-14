# %%
from datasets import load_dataset
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from tqdm import tqdm
from torch.optim import AdamW
import pickle
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from torch.utils.tensorboard import SummaryWriter 
import math

from sklearn.naive_bayes import MultinomialNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline


nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


# %%
def process_text(text):
    """Process the text: lowercasing, lemmatization, stopwords removal,
    and punctuation removal
    Input: text: the text to be processed"""
    text = text.lower()
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace("\r", " ")
    text = text.replace("  ", " ")

    # Word tokenization
    tokens = word_tokenize(text)

    # Normalization (lowercasing and lemmatization)
    #lemmatizer = WordNetLemmatizer()
    #tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Token filtering (stopwords removal)
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]

    # Remove punctuation tokens
    tokens = [token for token in tokens if token.isalpha()]

    # Join the tokens back into a string
    text = " ".join(tokens)

    return text


# %%
def plot_classes_distribution(labels, num_samples_per_class):
    plt.rcParams["figure.figsize"] = [10, 10]
    plt.barh(labels, num_samples_per_class)
    plt.xlabel("Number of samples")
    plt.ylabel("Class")
    plt.title("Distribution of the classes")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


# %%
# Load the dataset
dataset = load_dataset("argilla/medical-domain")

def basic_statistics(dataset):
    num_samples = len(dataset['train']) 
    
    # Number of classes
    labels = []
    for sample in dataset['train']['prediction']:
        label = sample[0]['label']
        if label not in labels:
            labels.append(label)
    
    num_classes = len(labels)
    num_samples_per_class=[]
    for label in labels:
        num_samples_per_class.append(len(dataset['train'].filter(lambda example: example['prediction'][0]['label'] == label)))

    return num_samples, labels, num_classes, num_samples_per_class

num_samples, labels, num_classes, num_samples_per_class = basic_statistics(dataset)
# %% [markdown]
# as a recap from exericse one, let's have a look again at the classes

# %%
# Calculate basic statistics 
print("Number of samples: ", num_samples) 
print("Number of classes: ", num_classes) 
plot_classes_distribution(labels, num_samples_per_class)

#Filter classes which are too small
#min_num_samples=30
#filtered_classes = np.array(labels)[np.array(num_samples_per_class)>min_num_samples]
#dataset['train'] = dataset['train'].filter(lambda sample: sample['prediction'][0]['label'] in filtered_classes)

num_samples, labels, num_classes, num_samples_per_class = basic_statistics(dataset)
print("Number of samples: ", num_samples) 
print("Number of classes: ", num_classes) 
plot_classes_distribution(labels, num_samples_per_class)

# %% [markdown]
# Next we are going to set the environment that we created in Task 1 for the preprocessing of the text using the pipeline provided in our Task 1 notebook.

# %%
# preprocess the data
dataset_p = pd.DataFrame(columns=["tokens", "label"])
for i in range(len(dataset["train"])):
    text = dataset["train"][i]["text"]
    label = dataset["train"][i]["prediction"][0]["label"] 
    text_p = process_text(text)
    dataset_p.loc[i] = [text_p, label]

# %%
dataset_p.head()

# %%
# how many labels are there
print(len(dataset_p["label"].unique()))
dataset_p["label"].value_counts().sort_index()

# %%
# convert labels to numbers
label2idx = {}
idx2label = {}
for i, label in enumerate(dataset_p["label"].unique()):
    label2idx[label] = i
    idx2label[i] = label

dataset_p["label"] = dataset_p["label"].map(label2idx)
dataset_p.head()


# %%
# wnat to use clinicalBERT for classification -> use its tokenizer
tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")

# %%
# get max length of the tokens
dataset_p["tokens"].max()
# Index where B is longest
idx = dataset_p["tokens"].apply(len).idxmax()

# Get that row
len(dataset_p.iloc[idx]["tokens"])


# %%
# Function to return the length of tokenized text
def count_tokens(text):
    return len(tokenizer.encode(text))


# Apply the function to each row and find the maximum length
token_lengths = dataset_p["tokens"].apply(count_tokens)
max_token_length = max(token_lengths)

print("Maximum number of tokens in a single row:", max_token_length)


# %% [markdown]
# the max nb of tokens that BERT can take is 512, so this entry will be truncated to 512 tokens



# %%
def count_tokens(text):
    return len(tokenizer.encode(text))


token_lengths = dataset_p["tokens"].apply(count_tokens)

# %%
# how many tokens have lenght > 512
token_lengths[token_lengths > 512].count() / len(token_lengths)

# %%
average_token_length = token_lengths.mean()
median_token_length = token_lengths.median()

# Plotting
plt.figure(figsize=(12, 6))
plt.hist(token_lengths, bins=30, color="skyblue", edgecolor="black")
plt.axvline(average_token_length, color="red", linestyle="dashed", linewidth=1)
plt.axvline(median_token_length, color="green", linestyle="dashed", linewidth=1)
plt.title("Distribution of Token Lengths")
plt.xlabel("Token Length")
plt.ylabel("Frequency")
plt.text(
    average_token_length + 5,
    plt.ylim()[1] * 0.9,
    f"Average: {average_token_length:.2f}",
    color="red",
)
plt.text(
    median_token_length - 300,
    plt.ylim()[1] * 0.96,
    f"Median: {median_token_length}",
    color="green",
)

plt.show()

# %% [markdown]
# the average length is 428 tokens and the median is 369, so truncating to 512 should be fine. however, in this case 30% of the entries will be truncated

# split rows that have more than 512 tokens into multiple rows to not lose data
def split_text(text, max_length):
    # Tokenize the text into words (not BERT tokens)
    words = text.split()

    #num of chunks
    num_of_chunks = math.ceil(len(words)/max_length)
    chunk_length = len(words) // num_of_chunks
    
    # Split words into chunks of chunk_length
    chunks = [
        " ".join(words[i : i + max_length]) for i in range(0, len(words), chunk_length)
    ]
    return chunks

train_df, eval_df = train_test_split(dataset_p, test_size=0.2, stratify=torch.tensor(dataset_p["label"].values), random_state=42)

max_length = 512 - 2  # accounting for [CLS] and [SEP]

new_rows_train = []
for _, row in train_df.iterrows():
    text_chunks = split_text(row["tokens"], max_length)
    for chunk in text_chunks:
        new_rows_train.append({"tokens": chunk, "label": row["label"]})
train_df = pd.DataFrame(new_rows_train)

new_rows_eval = []
for _, row in eval_df.iterrows():
    text_chunks = split_text(row["tokens"], max_length)
    for chunk in text_chunks:
        new_rows_eval.append({"tokens": chunk, "label": row["label"]})
eval_df = pd.DataFrame(new_rows_eval)

def tokenize(sent):
    encoded = tokenizer.encode_plus(
        text=sent,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }
    
train_labels = torch.tensor(train_df["label"].values)
eval_labels = torch.tensor(eval_df["label"].values)
#get a weight per sample based on the label: label_weight = 1/ (num_classes * num_samples_per_class[class_label])
weight_samples=1/(torch.index_select(torch.tensor(num_samples_per_class),0,train_labels)*num_classes) 


def evaluate(eval_labels, predictions): 
    accuracy = accuracy_score(eval_labels, predictions)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        eval_labels, predictions, average="macro"
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        eval_labels, predictions, average="micro"
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        eval_labels, predictions, average="weighted"
    )
    
    return {"accuracy": accuracy, "precision_macro": precision_macro, "recall_macro": recall_macro, "f1_macro": f1_macro, "precision_micro": precision_micro, "recall_micro": recall_micro, "f1_micro": f1_micro, "precision_weighted": precision_weighted, "recall_weighted": recall_weighted, "f1_weighted": f1_weighted}

train_inputs=train_df["tokens"].values
eval_inputs=eval_df["tokens"].values

# Pipelining
tfidf = TfidfVectorizer(ngram_range=(1,2))
clf = MultinomialNB()
nb_tfidf = make_pipeline(tfidf, clf) 

parameters = { 
        'multinomialnb__alpha': [0.001],
        'tfidfvectorizer__min_df': [35]
    }
grid_search = GridSearchCV(nb_tfidf, parameters, scoring="f1_macro", n_jobs=-1)

#No weighting possible
grid_search.fit(train_inputs, train_labels)
print("best params: " + str(grid_search.best_params_)) 
predictions = grid_search.predict(eval_inputs)
print("The Naive Bayes model has following scores: ")
print(evaluate(eval_labels, predictions)) 

with open('params_naive_bayes.txt', 'w') as f:
    f.write(str(grid_search.best_params_))
with open('scores_naive_bayes.txt', 'w') as f:
    f.write(str(evaluate(eval_labels, predictions)))

# Pipelining
tfidf = TfidfVectorizer(ngram_range=(1,2))
clf = RandomForestClassifier(class_weight="balanced")
nb_tfidf = make_pipeline(tfidf, clf) 

parameters = { 
        'randomforestclassifier__n_estimators': [150],
        'randomforestclassifier__max_depth': [20],
        'randomforestclassifier__min_samples_split': [10],
        'randomforestclassifier__max_features': ["sqrt"],
        'tfidfvectorizer__min_df': [15]
    }
grid_search = GridSearchCV(nb_tfidf, parameters, scoring="f1_macro", n_jobs=-1)  

grid_search.fit(train_inputs, train_labels)
print("best params: " + str(grid_search.best_params_)) 
predictions = grid_search.predict(eval_inputs)
print("The Random Forest model has following scores: ")
print(evaluate(eval_labels, predictions)) 

with open('params_random_forest.txt', 'w') as f:
    f.write(str(grid_search.best_params_))
with open('scores_random_forest.txt', 'w') as f:
    f.write(str(evaluate(eval_labels, predictions)))

# Pipelining
tfidf = TfidfVectorizer(ngram_range=(1,2))
clf = SVC(class_weight="balanced")
nb_tfidf = make_pipeline(tfidf, clf) 

#parameters = { 
#        'svc__gamma': ["scale", "auto", 1, 0.1, 0.01],
#        'svc__C': [1.0, 0.1, 0.01, 0.001], 
#        'svc__kernel': ["linear", "poly", "rbf"], 
#        'tfidfvectorizer__min_df': [15,20,35,50]
#    }
parameters = { 
        'svc__gamma': ["scale", "auto"],
        'svc__C': [10.0, 1.0, 0.1, 0.01], 
        'svc__kernel': ["poly", "rbf"], 
        'tfidfvectorizer__min_df': [15,35]
    }
grid_search = GridSearchCV(nb_tfidf, parameters, scoring="f1_macro", n_jobs=-1) 

grid_search.fit(train_inputs, train_labels)
print("best params: " + str(grid_search.best_params_)) 
predictions = grid_search.predict(eval_inputs)
print("The SVC model has following scores: ")
print(evaluate(eval_labels, predictions)) 


with open('params_SVC.txt', 'w') as f:
    f.write(str(grid_search.best_params_))
with open('scores_SVC.txt', 'w') as f:
    f.write(str(evaluate(eval_labels, predictions)))
  
train_encodings = train_df["tokens"].apply(lambda x: tokenize(x))
eval_encodings = eval_df["tokens"].apply(lambda x: tokenize(x))

train_inputs = torch.cat(
    train_encodings.apply(lambda x: x["input_ids"]).tolist(), dim=0
)
train_masks = torch.cat(
    train_encodings.apply(lambda x: x["attention_mask"]).tolist(), dim=0
)

eval_inputs = torch.cat(
    eval_encodings.apply(lambda x: x["input_ids"]).tolist(), dim=0
)
eval_masks = torch.cat(
    eval_encodings.apply(lambda x: x["attention_mask"]).tolist(), dim=0
)

train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
eval_dataset = TensorDataset(eval_inputs, eval_masks, eval_labels)

train_sampler = WeightedRandomSampler(weight_samples, num_samples=train_labels.shape[0], replacement=True)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=32)

eval_sampler = SequentialSampler(eval_dataset)
eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=32)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
def evaluate(model, dataloader): 

    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            batch_inputs, batch_masks, batch_labels = batch
            batch_inputs = batch_inputs.to(device)
            batch_masks = batch_masks.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_inputs, attention_mask=batch_masks)
            logits = outputs.logits

            logits = logits.detach().cpu().numpy()
            label_ids = batch_labels.to("cpu").numpy()

            predictions.append(logits)
            true_labels.append(label_ids)

    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    predictions = np.argmax(predictions, axis=1)

    data_to_save = {"true_labels": true_labels, "predictions": predictions}

    with open("model_evaluation_data_project_split_rows_with_pp.pkl", "wb") as file:
        pickle.dump(data_to_save, file)

    accuracy = accuracy_score(true_labels, predictions)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predictions, average="macro"
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        true_labels, predictions, average="micro"
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, predictions, average="weighted"
    )

    return {"accuracy": accuracy, "precision_macro": precision_macro, "recall_macro": recall_macro, "f1_macro": f1_macro, "precision_micro": precision_micro, "recall_micro": recall_micro, "f1_micro": f1_micro, "precision_weighted": precision_weighted, "recall_weighted": recall_weighted, "f1_weighted": f1_weighted} 


# %%
model = AutoModelForSequenceClassification.from_pretrained(
    "medicalai/ClinicalBERT",
    num_labels=num_classes,
)
model.to(device)

# %%
num_epochs = 100
max_patience=10
patience = max_patience
best_f1 = 0
optimizer = AdamW(model.parameters(), lr=5e-6)
it = 0
# Create tensorboard
summary = SummaryWriter("./", purge_step=0)

epoch_length = len(train_dataloader) 

from tqdm import tqdm

for epoch in tqdm(range(num_epochs)):
    print(f"Starting epoch {epoch + 1}/{num_epochs}...")
    model.train()

    for batch in train_dataloader:
        it += 1
        batch_inputs, batch_masks, batch_labels = batch
        batch_inputs = batch_inputs.to(device)
        batch_masks = batch_masks.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_inputs, attention_mask=batch_masks, labels=batch_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        if (it % (epoch_length/8)) == 0:
            summary.add_scalar("training loss", loss.cpu().item(), it)  
    
    model.eval()
    eval_metrics = evaluate(model, eval_dataloader)
    for key in eval_metrics:
        summary.add_scalar(key, eval_metrics[key], epoch +1)
        
    f1 = eval_metrics['f1_micro']            
    
    if best_f1 < f1:
        patience = max_patience
        best_f1=f1
        torch.save(model.state_dict(), "best_model.pt")

    else:
        patience -= 1
    
    print(f"My remaining patience is {patience}.") 
    print(f"Current f1 score is {f1}")
    print(f"Current loss is {loss.cpu().item()}")
    if patience <= 0:
        print("My patience run out.") 
        break
        
print("Training finished with " + str(num_epochs) + " epochs")   
            

# %%
model.load_state_dict(torch.load("best_model.pt", map_location=torch.device('cpu')))
eval_metrics = evaluate(model, eval_dataloader)
print(
    f"Validation Results - Accuracy: {eval_metrics['accuracy']:.3f}, Precision: {eval_metrics['precision_weighted']:.3f}, Recall: {eval_metrics['recall_weighted']:.3f}, F1: {eval_metrics['f1_weighted']:.3f}"
)
print("Finish Training of model with split rows and pp")
