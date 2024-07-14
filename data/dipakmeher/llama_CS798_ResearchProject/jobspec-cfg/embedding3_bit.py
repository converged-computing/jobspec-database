import argparse
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from transformers import pipeline, BertTokenizer, BertModel
from scipy.spatial.distance import cosine
import re
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# Start timing
start_time = time.time()

print("Starting Code")

# Command line arguments setup
parser = argparse.ArgumentParser(description="Process datasets and calculate embeddings with Llama Chat and BERT.")
parser.add_argument('--amazon_dir', type=str, required=True, help="Directory path for original CSV files.")
parser.add_argument('--overlapping_dir', type=str, required=True, help="Directory path for storing overlapping CSV files.")
parser.add_argument('--csv_file_1', type=str, required=True, help="Filename of the first original CSV file.")
parser.add_argument('--csv_file_2', type=str, required=True, help="Filename of the second original CSV file.")
parser.add_argument('--pair_name', type=str, required=True, help="Pair name for naming the overlapping CSV files.")
args = parser.parse_args()

# Printing all arguments
print("Command Line Arguments:")
print(f"Amazon directory: {args.amazon_dir}")
print(f"Overlapping directory: {args.overlapping_dir}")
print(f"CSV file 1: {args.csv_file_1}")
print(f"CSV file 2: {args.csv_file_2}")
print(f"Pair name: {args.pair_name}")

# Initialize the Llama Chat model
# llama2chat = pipeline(model="meta-llama/Llama-2-7b-chat-hf")

print("Loading Llama Pretrained bit model")
base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-13b-chat-hf",
            return_dict = True,
            load_in_4bit = True,
            device_map = "auto"
        )

base_model.config.use_cache = False
print("Loading Tokenizer for pretrained bit model")
base_tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-13b-chat-hf"
        )

print("Initializing BERT model")
# Initialize BERT tokenizer and model for embedding generation
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def get_user_preference(reviewText, summary):
    """Generate user preference using Llama 2 Chat based on review text and summary."""
    print("Get User Preference stated...")
    system_prompt = "<s> [INST] <<SYS>> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t know the answer to a question, please don’t share false information.<</SYS>>"
    
    prompt = f"""{system_prompt} The user left the following review for a product:\n\n'{reviewText}'.\n\n
    The product summary is: \n\n'{summary}'.\n\n
    Based on the review and summary of a product, can you analyze the user's preferences (be as specific as you can)? [\INST] """
    
    # response = llama2chat(prompt)
    
    # generated_text = response[0]['generated_text']
    # if "Answer:" in generated_text:
        # actual_response = generated_text.split("Answer:")[1]
    # else:
        # actual_response = generated_text
    # actual_response = re.sub(r'\n+', '\n', actual_response).strip()

    input_ids = base_tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    outputs = base_model.generate(input_ids=input_ids, max_new_tokens=4096)
    output = base_tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
    
    #print("llama response: ", output)
    return output

def get_bert_embedding(text):
    print("Bert Embedding in Creation...")
    """Function to generate BERT embedding for a given text."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = bert_model(**inputs)
    embedding = outputs.pooler_output.detach().numpy()

    #print("text: ", text)
    #print("embedding: ", embedding)
    # Check if the embedding is empty or has NaN values
    if np.isnan(embedding).any():
        print(f"NaN embedding for text: {text}")
   
    if np.all(embedding == 0):
        print("Warning: Zero embedding generated.")
    return embedding

def test1_calculate_cosine_similarity(embedding1, embedding2):
    """Calculate the cosine similarity between two embeddings."""
    
    if isinstance(embedding1, pd.Series):
        embedding1 = embedding1.iloc[0]
    if isinstance(embedding2, pd.Series):
        embedding2 = embedding2.iloc[0]

    if embedding1.ndim > 1:
        embedding1 = embedding1.squeeze()
    if embedding2.ndim > 1:
        embedding2 = embedding2.squeeze()

    return 1 - cosine(embedding1, embedding2)

def calculate_cosine_similarity(embedding1, embedding2):
    print("Cosine Similarity Calcuation Started")
    """Calculate the cosine similarity between two embeddings."""
    try:
        # Ensure embeddings are numpy arrays and squeeze any extra dimensions
        embedding1 = np.squeeze(np.array(embedding1))
        embedding2 = np.squeeze(np.array(embedding2))

        # Check for zero vectors which could cause division by zero
        if np.linalg.norm(embedding1) == 0 or np.linalg.norm(embedding2) == 0:
            print("Warning: One of the embeddings is a zero vector.")
            return np.nan

        # Compute cosine similarity
        return 1 - cosine(embedding1, embedding2)

    except Exception as e:
        # Log any errors encountered during computation
        print(f"Error calculating cosine similarity: {str(e)}")
        return np.nan

def average_embeddings(df):
    print("average_embeddings started")
    # Assuming each 'embedding' is stored as a list or array-like structure
    # Convert each embedding list into a numpy array if it's not already
    df['embedding'] = df['embedding'].apply(lambda x: np.array(x))
    # Average embeddings by taking the mean across the index level
    return df.groupby('reviewerID')['embedding'].apply(lambda x: np.mean(np.vstack(x), axis=0)).reset_index()


def process_dataset(df, tokenizer, bert_model):
    print(f"Processing dataset from DF")
    #df = pd.read_csv(csv_path)
    tqdm.pandas(desc="Processing Rows")


    #selected_columns = ['reviewerID', 'asin', 'reviewText', 'summary', 'overall']
    #df_selected = df[selected_columns]
    df_sorted = df.sort_values(by='reviewerID')
    print("Sorting done")
    # Initialize empty lists to store results
    user_preferences = []
    embeddings = []

    """
    # Total number of rows for progress calculation
    total_rows = len(df_sorted)
    
    # Loop through each row in the DataFrame
    for idx, row in enumerate(df_sorted.itertuples(index=False), 1):
        # Get user preference
        user_preference = get_user_preference(row.reviewText, row.summary)
        user_preferences.append(user_preference)
        
        # Print progress every 100 rows or as needed
        if idx % 10 == 0 or idx == total_rows:
            print(f"Processed {idx} of {total_rows} rows.")
    
    # Get BERT embeddings for each user preference
    for idx, preference in enumerate(user_preferences, 1):
        embedding = get_bert_embedding(preference)
        embeddings.append(embedding)
        
        # Print progress every 100 rows or as needed
        if idx % 10 == 0 or idx == total_rows:
            print(f"Computed embeddings for {idx} of {total_rows} rows.")

    # Assign the results back to the DataFrame
    df_sorted['user_preference'] = user_preferences
    df_sorted['embedding'] = embeddings

    print(df_sorted.head())
    print("Dataset processing complete.")
    
    """ 

    df_sorted['user_preference'] = df_sorted.progress_apply(lambda row: get_user_preference(row['reviewText'], row['summary']), axis=1)
    df_sorted['embedding'] = df_sorted['user_preference'].progress_apply(get_bert_embedding)
    
    #print(df_sorted['user_preference'].isnull().sum())
    #print(df_sorted['embedding'].apply(lambda x: np.isnan(x).any() or np.isinf(x).any()).sum())
    print(df_sorted.head())

    print("Dataset processing complete.")
    
    return df_sorted

def main():
    print("Main Started")
    """
    #To start Amazon 5 core dataset from scratch"
    csv_path_1 = os.path.join(args.amazon_dir, args.csv_file_1)
    csv_path_2 = os.path.join(args.amazon_dir, args.csv_file_2)

    # Load datasets
    print(f"Loading dataset from {csv_path_1}")
    df1 = pd.read_csv(csv_path_1)
    print(f"Loading dataset from {csv_path_2}")
    df2 = pd.read_csv(csv_path_2)

    # Convert 'reviewerID' to string to ensure consistent data type
    df1['reviewerID'] = df1['reviewerID'].astype(str)
    df2['reviewerID'] = df2['reviewerID'].astype(str)

    # Find overlapping users without setting as index
    common_uids = set(df1['reviewerID']).intersection(set(df2['reviewerID']))
    print("Number of common UIDs:", len(common_uids))

    # Filter dataframes to only include common UIDs directly with the 'reviewerID' column
    df1_common = df1[df1['reviewerID'].isin(common_uids)]
    df2_common = df2[df2['reviewerID'].isin(common_uids)]

    print(f"Filtered df1 entries: {len(df1_common)}")
    print(f"Filtered df2 entries: {len(df2_common)}")
    
    """

    csv_path_1 = os.path.join(args.amazon_dir, args.csv_file_1)
    csv_path_2 = os.path.join(args.amazon_dir, args.csv_file_2)

    # Load datasets
    print(f"Loading dataset from {csv_path_1}")
    df1 = pd.read_csv(csv_path_1)
    print(f"Loading dataset from {csv_path_2}")
    df2 = pd.read_csv(csv_path_2)

    # Convert 'reviewerID' to string to ensure consistent data type
    df1['reviewerID'] = df1['reviewerID'].astype(str)
    df2['reviewerID'] = df2['reviewerID'].astype(str)

    # Store the lengths of df1 and df2
    len_df1 = len(df1)
    len_df2 = len(df2)

    print(f"Filtered df1 entries: {len_df1}")
    print(f"Filtered df2 entries: {len_df2}")

    """
    # Define output paths
    output_file_1 = os.path.join(args.output_dir, 'common_' + args.csv_file_1)
    output_file_2 = os.path.join(args.output_dir, 'common_' + args.csv_file_2)

    # Save filtered dataframes to CSV
    df1_common.to_csv(output_file_1)
    print(f"Saved filtered DataFrame to {output_file_1}")
    df2_common.to_csv(output_file_2)
    print(f"Saved filtered DataFrame to {output_file_2}")
    """


    if len(df1) == 0 or len(df2) == 0:
        print("One or both datasets are empty. Check the data source or filtering criteria.")
    else:
        # Initialize tokenizer and modeli
        #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #bert_model = BertModel.from_pretrained('bert-base-uncased')

        df_sorted_1 = process_dataset(df1, tokenizer, bert_model)
        df_sorted_2 = process_dataset(df2, tokenizer, bert_model)

        print("Checking indices after processing...")
        print("df_sorted_1 entries sample:", df_sorted_1.head())
        print("df_sorted_2 entries sample:", df_sorted_2.head())
        
        # Average embeddings for each reviewerID
        df_sorted_1 = average_embeddings(df_sorted_1)
        df_sorted_2 = average_embeddings(df_sorted_2)

       # Checking the results
        print("Unique entries in df_sorted_1:", df_sorted_1['reviewerID'].nunique())
        print("Total entries in df_sorted_1:", len(df_sorted_1))
        print("Unique entries in df_sorted_2:", df_sorted_2['reviewerID'].nunique())
        print("Total entries in df_sorted_2:", len(df_sorted_2))
        #print("Unique Entries 2: " + str( df_sorted_2['reviewerID'].values))
        #print("Unique Entries 1: " + str( df_sorted_1['reviewerID'].values))

        cosine_similarities = []
        #for uid in common_uids:
        for uid in tqdm(df_sorted_1['reviewerID'].unique(), desc="Calculating Cosine Similarities"):
            if uid in df_sorted_1['reviewerID'].values and uid in df_sorted_2['reviewerID'].values:
                emb1 = df_sorted_1[df_sorted_1['reviewerID'] == uid]['embedding'].iloc[0]
                emb2 = df_sorted_2[df_sorted_2['reviewerID'] == uid]['embedding'].iloc[0]
                similarity = calculate_cosine_similarity(emb1, emb2)
                cosine_similarities.append(similarity)
                print(f"Cosine similarity for {uid}: {similarity}")
            else:
                print(f"UID {uid} not found in both datasets")

        valid_similarities = [s for s in cosine_similarities if s is not None and not np.isnan(s)]
        average_cosine_similarity = np.mean(valid_similarities) if valid_similarities else float('nan')
        print(f"Average Cosine Similarity between datasets '{args.pair_name}': {average_cosine_similarity}")

        # End timing
        end_time = time.time()
        
        # Calculate the duration
        duration = end_time - start_time
        print(f"The code took {duration} seconds to execute.")
        
        """
        # Calculate cosine similarities
        cosine_similarities = []
        for uid in common_uids:
            if uid in df_sorted_1.index and uid in df_sorted_2.index:
                emb1 = df_sorted_1.loc[uid, 'embedding']
                emb2 = df_sorted_2.loc[uid, 'embedding']
                if isinstance(emb1, np.ndarray) and isinstance(emb2, np.ndarray):
                    similarity = calculate_cosine_similarity(emb1, emb2)
                    cosine_similarities.append(similarity)
                    print(f"Cosine similarity for {uid}: {similarity}")
                else:
                    print(f"Skipping {uid} due to incorrect data type")
            else:
                print(f"UID {uid} not found in both datasets")

        # Compute the average of the valid cosine similarities
        valid_similarities = [s for s in cosine_similarities if s is not None and not np.isnan(s)]
        average_cosine_similarity = np.mean(valid_similarities) if valid_similarities else float('nan')
        print(f"Average Cosine Similarity between datasets: {average_cosine_similarity}")
        """
        # Calculate cosine similarities
        #cosine_similarities = [calculate_cosine_similarity(df_sorted_1.loc[uid, 'embedding'], df_sorted_2.loc[uid, 'embedding']) for uid in common_uids if uid in df_sorted_1.index and uid in df_sorted_2.index]
        #average_cosine_similarity = np.mean(cosine_similarities) if cosine_similarities else float('nan')

        #print(f"Average Cosine Similarity between datasets: {average_cosine_similarity}")

if __name__ == "__main__":
    main()




"""
# Main execution logic
def main():
    csv_path_1 = os.path.join(args.amazon_dir, args.csv_file_1)
    csv_path_2 = os.path.join(args.amazon_dir, args.csv_file_2)
    overlapping_csv_1 = os.path.join(args.overlapping_dir, f"{args.csv_file_1.split('.')[0]}_{args.pair_name}.csv")
    overlapping_csv_2 = os.path.join(args.overlapping_dir, f"{args.csv_file_2.split('.')[0]}_{args.pair_name}.csv")

    # Load datasets and find overlapping users
    df1 = pd.read_csv(csv_path_1)
    df2 = pd.read_csv(csv_path_2)
    overlapping_users = pd.Series(list(set(df1['reviewerID']).intersection(set(df2['reviewerID']))))
    df1_overlapping = df1[df1['reviewerID'].isin(overlapping_users)]
    df2_overlapping = df2[df2['reviewerID'].isin(overlapping_users)]

    # Print lengths and unique user counts
    print(f"Length of df1_overlapping: {len(df1_overlapping)}")
    print(f"Number of unique users in df1_overlapping: {df1_overlapping['reviewerID'].nunique()}")
    print(f"Length of df2_overlapping: {len(df2_overlapping)}")
    print(f"Number of unique users in df2_overlapping: {df2_overlapping['reviewerID'].nunique()}")

    df_sorted_1 = process_dataset(csv_path_1, tokenizer, bert_model)
    df_sorted_2 = process_dataset(csv_path_2, tokenizer, bert_model)

    # Save the new dataframes to CSV files
    df1_overlapping.to_csv(overlapping_csv_1, index=False)
    df2_overlapping.to_csv(overlapping_csv_2, index=False)


    # Ensure 'reviewerID' is of the same data type
    df1['reviewerID'] = df1['reviewerID'].astype(str)
    df2['reviewerID'] = df2['reviewerID'].astype(str)


    print(df1['reviewerID'].dtype, df2['reviewerID'].dtype)
    print(set(df1['reviewerID']) == set(df2['reviewerID']))
    print(df1['reviewerID'].head(), df2['reviewerID'].head())


    # Calculate cosine similarities and average cosine similarity
    df_sorted_1.set_index('reviewerID', inplace=True)
    df_sorted_2.set_index('reviewerID', inplace=True)
    
    
    cosine_similarities = []
    #common_uids = set(df_sorted_1.index).intersection(set(df_sorted_2.index))
    #print("Unique IDs in df1:", set(df1['reviewerID']))
    #print("Unique IDs in df2:", set(df2['reviewerID']))
    common_uids = set(df1['reviewerID']).intersection(set(df2['reviewerID']))
    print("Common UIDs:", common_uids)
    print("Number of common UIDs:", len(common_uids))
    
    if not common_uids:
        print("No common UIDs found.")
    else:
        cosine_similarities = [calculate_cosine_similarity(df_sorted_1.loc[uid, 'embedding'], df_sorted_2.loc[uid, 'embedding']) for uid in common_uids if not np.isnan(df_sorted_1.loc[uid, 'embedding']).any() and not np.isnan(df_sorted_2.loc[uid, 'embedding']).any()]
    #cosine_similarities = [calculate_cosine_similarity(df_sorted_1.loc[uid, 'embedding'], df_sorted_2.loc[uid, 'embedding']) for uid in common_uids]
    average_cosine_similarity = np.mean(cosine_similarities)

    print(f"Average Cosine Similarity between {args.csv_file_1.split('.')[0]}_{args.pair_name} and {args.csv_file_2.split('.')[0]}_{args.pair_name}: {average_cosine_similarity}")


if __name__ == "__main__":
    main()
"""
