# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from model import MidiSentimentModel
from dataset import MidiDataset
import numpy as np
import re
from sklearn.model_selection import train_test_split

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for midi_features, labels in dataloader:
            for i, midi_feature in enumerate(midi_features):
                inputs = model.tokenizer.tokenize(midi_feature)
                input_ids = torch.tensor(inputs).unsqueeze(0).to(device)
                attention_mask = torch.ones_like(input_ids).to(device)

                valence_label = labels[i, 0].unsqueeze(0).unsqueeze(-1).to(device)
                arousal_label = labels[i, 1].unsqueeze(0).unsqueeze(-1).to(device)

                valence_output, arousal_output = model(input_ids, attention_mask)

                valence_loss = criterion(valence_output, valence_label)
                arousal_loss = criterion(arousal_output, arousal_label)
                loss = valence_loss + arousal_loss
                total_loss += loss.item()

    return total_loss / len(dataloader)


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0
    for midi_features, labels in train_loader:
        for i, midi_feature in enumerate(midi_features):
            inputs = model.tokenizer.tokenize(midi_feature)
            input_ids = torch.tensor(inputs).unsqueeze(0).to(device)

            valence_label = labels[i, 0].unsqueeze(0).unsqueeze(-1).to(device)
            arousal_label = labels[i, 1].unsqueeze(0).unsqueeze(-1).to(device)

            optimizer.zero_grad()
            attention_mask = torch.ones_like(input_ids).to(device)
            valence_output, arousal_output = model(input_ids, attention_mask)

            valence_loss = criterion(valence_output, valence_label)
            arousal_loss = criterion(arousal_output, arousal_label)
            loss = valence_loss + arousal_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

    return train_loss / len(train_loader)




def predict(model, midi_content, device):
    model.eval()
    inputs = model.tokenizer.tokenize(midi_content)
    input_ids = torch.tensor(inputs).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    with torch.no_grad():
        valence_output, arousal_output = model(input_ids, attention_mask)
        return valence_output.item(), arousal_output.item()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess data
    full_data = MidiDataset('data/prompts.json')
    train_indices, val_indices = train_test_split(np.arange(len(full_data)), test_size=0.2, random_state=42)
    train_dataset = Subset(full_data, train_indices)
    val_dataset = Subset(full_data, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = MidiSentimentModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.MSELoss()

    num_epochs = 30
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    torch.save(model.state_dict(), 'trained_model.pt')

    # Load and predict with a sample ABC content
    sample_abc_content = "X: 1\nM: 4/4\nL: 1/8\nQ:1/4=120\nK:C % 0 sharps\nV:1\n%%clef treble\nzA,- [E-A,]3/2[B-E-]/2 [c-B-E-][e-c-BE-] [g-ec-E-]/2[g-e-cE-]/2[g-e-E-]/2[g-e-c-E-]/2| \\\n[g-e-c-B-E]2 [g-e-c-B-E-]/2[g-e-c-B-EA,-][g-ec-BE-A,-]3/2[g-cB-E-A,-]/2[g-B-E-A,-]/2 [g-c-B-E-A,-]/2[ge-c-B-E-A,-][g-ec-B-E-A,-]/2| \\\n[g-e-c-B-E-A,]/2[g-e-cB-E-]/2[g-e-c-BE-] [g-e-c-B-E]3/2[g-ecBE-][g-E-]/2[gE-C-] [E-C-]/2[G-EC-][A-G-C-]/2| \\\n[c-A-G-C-][e-cA-G-C-C,,-]/2[e-A-G-C-C,,-]/2 [e-c-AG-C-C,,-]/2[e-c-AGC-C,,][e-c-A-G-C]/2"
    valence, arousal = predict(model, sample_abc_content, device)
    print(f"Predicted Valence: {valence}, Predicted Arousal: {arousal}")


if __name__ == '__main__':
    main()
