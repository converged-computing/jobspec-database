import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split


column_names = ['query', 'positive_passage', 'negative_passage']
# Load and preprocess data
train_data = pd.read_csv('datasets/triples.train.small.tsv', sep='\t', header=None, names=column_names)

print(train_data.columns)
print(dev_data.columns)
# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize and prepare data
class RerankingDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query = str(self.data.iloc[idx]['query'])
        pos_passage = str(self.data.iloc[idx]['positive_passage'])
        neg_passage = str(self.data.iloc[idx]['negative_passage'])

        inputs = self.tokenizer(
            query,
            pos_passage,
            neg_passage,
            return_tensors='pt',
            max_length=self.max_len,
            truncation=True,
            padding='max_length'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(1)  # Binary label (1 for positive pair, 0 for negative pair)
        }

# Split data into training and validation sets
train_set, val_set = train_test_split(train_data, test_size=0.2, random_state=42)

# Create DataLoader
train_dataset = RerankingDataset(train_set, tokenizer)
val_dataset = RerankingDataset(val_set, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
num_epochs = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluation loop
model.eval()
val_losses = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc='Validation'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        val_losses.append(outputs.loss.item())

average_val_loss = sum(val_losses) / len(val_losses)
print(f'Average Validation Loss: {average_val_loss}')

# Save the model
model.save_pretrained('reranking_model.pth')
