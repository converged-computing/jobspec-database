#!/usr/bin/env python

from datasets import load_dataset
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import random
import torch
from torch.utils.data import Subset
from datasets import load_from_disk
from transformers import BlipProcessor, BlipForConditionalGeneration
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
import spacy
from torch.cuda.amp import GradScaler, autocast
import gc

nlp = spacy.load("en_core_web_lg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = load_dataset("generative-newsai/news-unmasked")

train_set = dataset["train"]
test_set = dataset["test"]

train_set = load_from_disk("./processed_data")
train_train_set, train_val_set = torch.utils.data.random_split(train_set,
                                                               [int(0.8 * len(train_set)),
                                                                len(train_set) - int(0.8 * len(train_set))])

device_id = 0 if torch.cuda.is_available() else -1

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def unmask(model, text, image, device):
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    mask_token_index = (inputs.input_ids == processor.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

    outputs = model(**inputs)
    logits = outputs.decoder_logits
    predicted_token_id = logits[0, mask_token_index-1].argmax(axis=-1)
    return processor.decode(predicted_token_id)

def custom_collate(original_batch):
    filtered_data = []

    for item in original_batch:
        image, text = item['image'], item['masked_headline']
        original = item['headline']
        inputs = processor(images=image, text=text, return_tensors="pt", padding='max_length')
        inputs["labels"] = processor(text=original, return_tensors="pt", padding='max_length').input_ids
        filtered_data.append(inputs)

    return default_collate(filtered_data)

def validation(model, data):
    model.eval()
    all_masked_words = []

    for each_dict in tqdm(data, desc="validation"):
        original = each_dict['headline'].split(' ')
        sentence = each_dict['masked_headline']
        image_id = each_dict['image_id']
        image = each_dict['image']
        if "[MASK]" in sentence:
            result = unmask(model, sentence, image, device).split(' ')
            indices = [i for i, x in enumerate(sentence.split()) if x == "[MASK]"]
            if len(indices) > 1:
                for i, each_result in enumerate(result):
                    all_masked_words.append([image_id, indices[i], each_result, original[indices[i]]])
            else:
                all_masked_words.append([image_id, indices[0], result[0], original[indices[0]]])
    cosine_sim_threshold = 0.5
    num_correct = 0
    similarity_list = []
    for each_masked_word in all_masked_words:
        top_word = each_masked_word[2]
        original_word = each_masked_word[3]
        similarity = nlp(top_word).similarity(nlp(original_word))
        similarity_list.append(similarity)
        if similarity >= cosine_sim_threshold:
            num_correct += 1

    accuracy = num_correct / len(all_masked_words) * 100
    return accuracy, np.mean(similarity_list), np.std(similarity_list)

# Training
model.load_state_dict(torch.load('best_model_0.pt'))
model = model.to(device)
gc.collect()
torch.cuda.empty_cache()

size, batch_size = len(train_train_set), 16
temp = Subset(train_train_set, random.sample(range(len(train_train_set)), 100))
train_loader = DataLoader(train_train_set, batch_size=batch_size, shuffle=False, collate_fn=custom_collate,
                         pin_memory=True, num_workers=4)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

losses = []
epochs = 10
best_accuracy = 0

scaler = GradScaler()
model.train()
for epoch in trange(epochs, desc="Epoch"):
    for i, inputs in enumerate(tqdm(train_loader, desc="Iteration")):
        for k, v in inputs.items():
            inputs[k] = inputs[k].squeeze(1)
        inputs = inputs.to(device)
        with autocast():
            outputs = model(**inputs)
            loss = outputs.loss
        losses.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if (i+1) % 1000 == 0:
            val_accuracy, mean_similarity, std_similarity = validation(model, temp)
            print(f"Loss: {np.mean(losses):.2f}, Train Accuracy: {val_accuracy:.2f}, " +
                  f"Mean Sim: {mean_similarity:.2f}, Std Sim: {std_similarity:.2f}")

            # val_temp = Subset(train_val_set, random.sample(range(len(train_val_set)), 1000))
            val_accuracy, mean_similarity, std_similarity = validation(model, train_val_set)
            print(f"Val Accuracy: {val_accuracy:.2f}, Mean Sim: {mean_similarity:.2f}, Std Sim: {std_similarity:.2f}")
            losses = []
            if best_accuracy < val_accuracy:
                print(f"Saving model with val accuracy: {val_accuracy}")
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), f"best_model_{epoch}.pt")
print(f"Saving final model with val accuracy: {val_accuracy}")
torch.save(model.state_dict(), f"best_model_{epoch}.pt")
