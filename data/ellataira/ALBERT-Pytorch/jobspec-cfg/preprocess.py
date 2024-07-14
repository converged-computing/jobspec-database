# from transformers import AlbertTokenizer
# from tqdm import tqdm
# import os
#
# # Set the directory to save the tokenizer vocabulary
# vocab_dir = "/scratch/taira.e/"
#
# # Specify the tokenizer model identifier
# model_identifier = "albert-base-v2"
#
# # Initialize the tokenizer
# try:
#     tokenizer = AlbertTokenizer.from_pretrained(model_identifier)
# except Exception as e:
#     print(f"Error: Unable to initialize tokenizer from {model_identifier}.")
#     print(f"Exception: {e}")
#     exit()
#
# # Read sentences from the input file
# input_file_path = "/scratch/taira.e/c4_10_dataset_distill.txt"
# if not os.path.exists(input_file_path):
#     print(f"Error: Input file {input_file_path} not found.")
#     exit()
#
# sentences = []
# try:
#     with open(input_file_path, "r", encoding="utf-8") as f:
#         for line in tqdm(f, desc="Reading sentences"):
#             sentences.append(line.strip())  # Assuming you want to strip newline characters
# except Exception as e:
#     print(f"Error: Unable to read sentences from {input_file_path}.")
#     print(f"Exception: {e}")
#     exit()
#
# # Tokenize sentences to build vocabulary
# print("now tokenizing... ")
# try:
#     tokenized_inputs = tokenizer(sentences, return_tensors="pt", truncation=True, padding=True)
# except Exception as e:
#     print("Error: Failed to tokenize sentences.")
#     print(f"Exception: {e}")
#     exit()
#
# print("done tokenizing")
#
# print("getting vocab from tokenizer")
# # Get the vocabulary from the tokenizer
# vocab = tokenizer.get_vocab()
#
# # Save vocabulary
# output_vocab_file = os.path.join(vocab_dir, "albert_vocab.txt")
# try:
#     with open(output_vocab_file, "w", encoding="utf-8") as f:
#         for token, index in tqdm(vocab.items(), desc="Saving vocabulary"):
#             f.write(f"{token}\t{index}\n")
#     print(f"Vocabulary saved successfully to {output_vocab_file}.")
# except Exception as e:
#     print("Error: Failed to save vocabulary.")
#     print(f"Exception: {e}")

from transformers import AlbertTokenizer
from tqdm import tqdm
import os

vocab_dir = "/scratch/taira.e/"
model_identifier = "albert-base-v2"

try:
    tokenizer = AlbertTokenizer.from_pretrained(model_identifier)
except Exception as e:
    print(f"Error initializing tokenizer from {model_identifier}: {e}")
    exit()

input_file_path = "/scratch/taira.e/c4_10_dataset_distill.txt"
if not os.path.exists(input_file_path):
    print(f"Input file {input_file_path} not found.")
    exit()

try:
    print("now tokenizing...")
    with open(input_file_path, "r", encoding="utf-8") as file:
        tokenized_inputs = []
        for line in tqdm(file, desc="Tokenizing lines", mininterval=10):
            tokens = tokenizer(line.strip(), return_tensors="pt", truncation=True, padding=True)
            tokenized_inputs.append(tokens)
    print("done tokenizing")
except Exception as e:
    print(f"Error tokenizing sentences: {e}")
    exit()

vocab = tokenizer.get_vocab()
output_vocab_file = os.path.join(vocab_dir, "albert_vocab.txt")

try:
    with open(output_vocab_file, "w", encoding="utf-8") as f:
        for token, index in tqdm(vocab.items(), desc="Saving vocabulary"):
            f.write(f"{token}\t{index}\n")
    print(f"Vocabulary saved successfully to {output_vocab_file}.")
except Exception as e:
    print(f"Error saving vocabulary: {e}")
