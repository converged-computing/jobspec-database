from datasets import load_dataset, load_from_disk
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn import metrics
import os, argparse, sys 
import shlex
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer
from torch.nn.functional import softmax

## Note: expects directories called 'tokenized_data' and 'models'
# to exist and be in the same directory as this script


# MODEL_NAME = 'bert-base-uncased'  #./models/bert_epochs_2_lr_1e-05_batch_4' # something like ./models/bert_epochs_1_lr_1e-05_batch_16 when loading trained model
MODEL_NAME = "./models/bert_epochs_3_lr_1e-05_batch_4"
DATASET_NAME = 'ccdv/arxiv-classification'
TRAIN_MODEL = False
PREPROCESS_DATA = False
ORIGINAL_LABELS = ['math.AC', 'cs.CV', 'cs.AI', 'cs.SY', 'math.GR', 'cs.DS', 'cs.CE', 'cs.PL', 'cs.IT', 'cs.NE', 'math.ST']
USE_ORIGINAL_LABELS = True
DISPLAY_ERRORS = True

def predict_proba(texts):
    tokenized = tokenizer(texts, max_length=512, truncation=True, return_tensors='pt', padding='max_length')
    with torch.no_grad():
        inputs = {name: tensor.to(device) for name, tensor in tokenized.items()}
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=-1)
    return probs.squeeze().detach().cpu().numpy()

def fix_labels(instance):
    classConversion = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    instance['labels'] = classConversion[instance['label']]
    return instance

def tokenize_batch(batch):
    return tokenizer(batch['text'], max_length=512, truncation=True, return_tensors='pt')

def load_process_data_from_hub():
    subset = 'no_ref'
    test_data = load_dataset(DATASET_NAME, subset, split='test')
    train_data = load_dataset(DATASET_NAME, subset, split='train')
    val_data = load_dataset(DATASET_NAME, subset, split='validation')

    train_data = train_data.map(fix_labels)
    test_data = test_data.map(fix_labels)
    val_data = val_data.map(fix_labels)

    # Preprocessing
    train_data = train_data.map(tokenize_batch, batched=False)
    test_data = test_data.map(tokenize_batch)
    val_data = val_data.map(tokenize_batch)

    ## Save the tokenized datasets
    train_data.save_to_disk('./tokenized_data/train')
    test_data.save_to_disk('./tokenized_data/test')
    val_data.save_to_disk('./tokenized_data/val')

def train(train_loader, test_loader, val_loader, model, device):

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = NUM_EPOCHS * len(train_loader)
    lr_scheduler = get_scheduler('linear', optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    p_bar = tqdm(range(num_training_steps))
    model.train()

    for epoch in range(NUM_EPOCHS):
        for batch in train_loader:
            batch = {key: batch[key].to(device).squeeze() for key in batch}
            optimizer.zero_grad()
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            p_bar.update(1)
            p_bar.set_postfix({'loss': loss})
    model.save_pretrained(f'./models/bert_epochs_{NUM_EPOCHS}_lr_{LEARNING_RATE}_batch_{BATCH_SIZE}')
    return model

def analyze_errors(val_preds, val_labels):
    '''
    Generate plot of original labels for incorrect predictions
    '''

    errors = []
    for i, pred in enumerate(val_preds):
        if pred != val_labels[i]:
            errors.append(i)
    original_val_ds = load_dataset(DATASET_NAME, 'no_ref', split='validation')
    
    misclassifications = np.zeros(11)
    for i in errors:
        misclassifications[original_val_ds[i]['label']] += 1
    
    # plot histogram of misclassifications
    # X-axis: original label text
    # Y-axis: number of misclassifications
    plt.figure(figsize=(10, 5))
    plt.bar(ORIGINAL_LABELS, misclassifications)
    plt.xlabel('Original Label')
    plt.ylabel('Number of Misclassifications')
    plt.title('Misclassifications by Original Label')
    # save plot
    plt.savefig(f'bert_graphs/{LEARNING_RATE}_{NUM_EPOCHS}_{BATCH_SIZE}_misclassifications.png')


def display_errors(val_preds, val_labels):
    '''
    Print out original labels and text for incorrect predictions, one at a time
    Ask for user input to continue to next error
    '''
    errors = []
    correct = []
    for i, pred in enumerate(val_preds):
        if pred != val_labels[i]:
            errors.append(i)
        else:
            correct.append(i)
    original_val_ds = load_dataset(DATASET_NAME, 'no_ref', split='validation')
    tokenized_val_ds = load_from_disk('./tokenized_data/val')

    explainer = LimeTextExplainer(class_names=ORIGINAL_LABELS)

    for i in errors:
        print("Original label: ", ORIGINAL_LABELS[original_val_ds[i]['label']])
        print("Predicted label: ", val_preds[i])
        print("Correct label: ", val_labels[i])
        print("Text: ", tokenizer.decode(tokenized_val_ds[i]['input_ids'][0]))
        print("Text from DS: ", original_val_ds[i]['text'][:1000])

        explanation = explainer.explain_instance(tokenizer.decode(tokenized_val_ds[i]['input_ids'][0]), predict_proba, num_features=10, num_samples=1000)

        fig = explanation.as_pyplot_figure()
        plt.savefig(f'bert_graphs/{LEARNING_RATE}_{NUM_EPOCHS}_{BATCH_SIZE}_explanation_{i}.png')

if __name__ == '__main__':


    NUM_LABELS = 11 if USE_ORIGINAL_LABELS else 2

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, help='Input file name')
    file_args = parser.parse_args()
    file = file_args.input

    def parse_file(filename):
        with open(filename, 'r') as f:
            args = shlex.split(f.read())
        return args

    parser = argparse.ArgumentParser()
    ## parse the arguments for ep, weighted, atoms, sparsity, seed, epochs
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs for training')

    args = parser.parse_args(parse_file(file))
    global NUM_EPOCHS
    global BATCH_SIZE
    global LEARNING_RATE
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch
    LEARNING_RATE = args.lr
    global model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    global tokenizer 
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #change to torch.device('mps') if running on mac
    model = model.to(device)

    if PREPROCESS_DATA:
        load_process_data_from_hub()

    ## Load the tokenized datasets from disk
    train_data = load_from_disk('./tokenized_data/train')
    test_data = load_from_disk('./tokenized_data/test')
    val_data = load_from_disk('./tokenized_data/val')

    train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_data.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_data.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    if TRAIN_MODEL:
        model = train(train_loader, test_loader, val_loader, model, device)
    
    # evaluate on validation set
    model.eval()
    val_labels = []
    val_preds = []

    for batch in tqdm(val_loader):
        batch = {key: batch[key].to(device).squeeze() for key in batch}
        with torch.no_grad():
            outputs = model(**batch)
        val_labels.extend(batch['labels'].cpu().numpy().tolist())
        val_preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy().tolist())

    accuracy = metrics.accuracy_score(val_labels, val_preds)
    precision = metrics.precision_score(val_labels, val_preds, average='macro')
    recall = metrics.recall_score(val_labels, val_preds, average='macro')
    f1 = metrics.f1_score(val_labels, val_preds, average='macro')

    # Confusion matrix
    cm = metrics.confusion_matrix(val_labels, val_preds)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ORIGINAL_LABELS)
    disp.savefig(f'bert_graphs/{LEARNING_RATE}_{NUM_EPOCHS}_{BATCH_SIZE}_confusion_matrix.png')
    # plt.savefig(f'bert_graphs/{LEARNING_RATE}_{NUM_EPOCHS}_{BATCH_SIZE}_confusion_matrix.png')

    file = open(f'bert_results/{LEARNING_RATE}_{NUM_EPOCHS}_{BATCH_SIZE}_results.txt', 'w')
    
    with file as f:
        print("Num samples", len(val_loader) * BATCH_SIZE, file=f)
        print("Num samples", len(val_labels), file=f)
        print("Accuracy: ", accuracy, file=f)
        print("Precision: ", precision, file=f)
        print("Recall: ", recall, file=f)
        print("F1: ", f1, file=f)
        print("Params: ", file=f)
        print("Epochs: ", NUM_EPOCHS, file=f)
        print("Batch size: ", BATCH_SIZE, file=f)
        print("Learning rate: ", LEARNING_RATE, file=f)

    analyze_errors(val_preds, val_labels) if DISPLAY_ERRORS else None
    display_errors(val_preds, val_labels) if DISPLAY_ERRORS else None