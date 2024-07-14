import pandas as pd
from transformers import pipeline, BertTokenizer, BertModel
import re

# Load your dataset
# Replace 'your_dataset.csv' with the path to your dataset file
df = pd.read_csv('./data/electronic_overlapping.csv')

# Select the specified columns
selected_columns = ['reviewerID', 'asin', 'reviewText', 'summary', 'overall']
df_selected = df[selected_columns]

# Sort df_selected based on 'reviewerID'
df_sorted = df_selected.sort_values(by='reviewerID').head(2)

# df_sorted now contains your sorted DataFrame
# You can now proceed with your further implementation on df_sorted

# Initialize the Llama 2 Chat model
llama2chat = pipeline(model="meta-llama/Llama-2-7b-chat-hf")

def test_get_user_preference(reviewText, summary):
    """Generate user preference using Llama 2 Chat based on review text and summary."""
    #prompt = f"Here is the text that a user gave as a review to the product: {reviewText} and here is the summary of that item: {summary}. Can you explain what kind of items the user prefers (be as specific as possible about their interests)?"
    #prompt = f"Here is the text that a user gave as a review to the product: {reviewText} and here is the summary of that item: {summary}. what specific types of features does the user prefer (be as specific as possible about their interests)?"
    #prompt = f"Based on the user's review: '{reviewText}' and the item summary: '{summary}', identify specific types of features the user shows preference for. Provide a brief, bullet-point list of these features without repeating the review or summary text."
    prompt = f"""The user left the following review for a product:'{reviewText}'.
    The product summary describes it as a '{summary}'.
    Based on the review and summary of a product, can you analyze the user's preferences (be as specific as you can)?"""
    
    print("Prompt: ", prompt)
    
    response = llama2chat(prompt)
    print("Response: ", response)
    
    # Assuming the response is a dict with a 'generated_text' key
    return response[0]['generated_text']


def get_user_preference(reviewText, summary):
    """Generate user preference using Llama 2 Chat based on review text and summary."""
    prompt = f"""The user left the following review for a product:'{reviewText}'.
    The product summary describes it as a '{summary}'.
    Based on the review and summary of a product, can you analyze the user's preferences (be as specific as you can)?"""

    print("Prompt: ", prompt)

    response = llama2chat(prompt)
    print("Full Response: ", response)

    # Assuming the response is a list with a dict that has a 'generated_text' key
    generated_text = response[0]['generated_text']

    # Splitting the generated text to extract the actual response after "Answer:"
    if "Answer:" in generated_text:
        actual_response = generated_text.split("Answer:")[1]
    else:
        # If "Answer:" is not found, use the full generated text as the response
        actual_response = generated_text

    # Filter out excess newline characters and consolidate multiple newlines into a single newline
    actual_response = re.sub(r'\n+', '\n', actual_response).strip()

    print("Actual Response: ", actual_response)
    return actual_response




# Update the DataFrame with user preferences
df_sorted['user_preference'] = df_sorted.apply(lambda row: get_user_preference(row['reviewText'], row['summary']), axis=1)

# Proceed with BERT embedding generation as described previously

# Initialize BERT tokenizer and model for embedding generation
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    """Function to generate BERT embedding for a given text."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.pooler_output.detach().numpy()

# Apply the embedding function to the 'user_preference' column
df_sorted['embedding'] = df_sorted['user_preference'].apply(get_bert_embedding)
# Assuming df_sorted['embedding'] contains the embeddings
#for i, embedding in enumerate(df_sorted['embedding'].head(5)):
 #   print(f"Embedding {i+1}: {embedding}\n")
# Print the first five entries of df_sorted with their embeddings
for i, row in df_sorted.iterrows():
    print(f"Entry {i+1}")
    print(f"ReviewerID: {row['reviewerID']}")
    print(f"ASIN: {row['asin']}")
    print(f"Review Text: {row['reviewText']}")
    print(f"Summary: {row['summary']}")
    print(f"Overall Rating: {row['overall']}")
    print(f"User Preference: {row['user_preference']}")
    print(f"Embedding: {row['embedding']}\n")

    if i == 4:  # Stop after printing the first 5 entries
        break
