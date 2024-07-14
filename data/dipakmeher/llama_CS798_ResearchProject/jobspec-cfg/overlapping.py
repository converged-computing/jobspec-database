import pandas as pd
import argparse

#cmd arguments
parser = argparse.ArgumentParser(description="Process Amazon dataset files and find overlapping users.")
parser.add_argument("--amazon_dataset_dir", required=True, help="Directory to save the output files.")
parser.add_argument("--dataset1", required=True, help="Path to the first dataset file (e.g., reviews for books).")
parser.add_argument("--dataset2", required=True, help="Path to the second dataset file (e.g., reviews for CDs and vinyl).")
parser.add_argument("--overlapping_output_dir", required=True, help="Directory to save the output files.")
parser.add_argument("--output_file1", required=True, help="Name of the first output file for overlapping data.")
parser.add_argument("--output_file2", required=True, help="Name of the second output file for overlapping data.")


args = parser.parse_args()

# Save the new dataframes to new CSV files
input_path1 = f"{args.amazon_dataset_dir}/{args.dataset1}"
input_path2 = f"{args.amazon_dataset_dir}/{args.dataset2}"

# Load the datasets
df1 = pd.read_csv(input_path1)
df2 = pd.read_csv(input_path2)
"""
# Load the datasets
df1 = pd.read_csv('/scratch/dmeher/datasets_recguru/reviews_Books_5.csv')
df2 = pd.read_csv('/scratch/dmeher/datasets_recguru/reviews_CDs_and_Vinyl_5.csv')
"""
# Find the overlapping users based on 'reviewerID'
overlapping_users = pd.Series(list(set(df1['reviewerID']).intersection(set(df2['reviewerID']))))

# Select rows from both dataframes where 'reviewerID' is in the list of overlapping users
df1_overlapping = df1[df1['reviewerID'].isin(overlapping_users)]
df2_overlapping = df2[df2['reviewerID'].isin(overlapping_users)]

# Save the new dataframes to new CSV files
output_path1 = f"{args.overlapping_output_dir}/{args.output_file1}"
output_path2 = f"{args.overlapping_output_dir}/{args.output_file2}"

# Save the new dataframes to new CSV files
df1_overlapping.to_csv(output_path1, index=False)
df2_overlapping.to_csv(output_path2, index=False)

# Print the length and number of unique users for both dataframes
print(f"Length of the first new dataframe: {len(df1_overlapping)}")
print(f"Number of unique users in the first new dataframe: {df1_overlapping['reviewerID'].nunique()}")
print(f"Length of the second new dataframe: {len(df2_overlapping)}")
print(f"Number of unique users in the second new dataframe: {df2_overlapping['reviewerID'].nunique()}")

# Assuming df1_overlapping and df2_overlapping are already loaded and available
# Task 1: Creating simplified dataframes and saving them
df1_simple = df1_overlapping[['reviewerID', 'asin', 'overall']].rename(columns={'reviewerID': 'uid', 'asin': 'iid', 'overall': 'y'})
df2_simple = df2_overlapping[['reviewerID', 'asin', 'overall']].rename(columns={'reviewerID': 'uid', 'asin': 'iid', 'overall': 'y'})

df1_simple_file1 = 'ptu_'+args.output_file1
df2_simple_file2 = 'ptu_'+args.output_file2

df1_simple_output_path1 = f"{args.overlapping_output_dir}/{df1_simple_file1}"
df2_simple_output_path2 = f"{args.overlapping_output_dir}/{df2_simple_file2}"

df1_simple.to_csv(df1_simple_output_path1, index=False)
df2_simple.to_csv(df2_simple_output_path2, index=False)

print("PTU Dataset----------------")
print("Dataset1: ", len(df1_simple))
print("Dataset2: ", len(df2_simple))

# Task 2: Filtering for 2500 data points and saving
df1_filtered = df1_overlapping[['reviewerID', 'reviewText', 'summary']]
df2_filtered = df2_overlapping[['reviewerID', 'reviewText', 'summary']]

# Ensure reviewerID is string
df1_filtered['reviewerID'] = df1_filtered['reviewerID'].astype(str)
df2_filtered['reviewerID'] = df2_filtered['reviewerID'].astype(str)

# Find overlapping reviewerIDs
common_reviewer_ids = pd.merge(df1_filtered[['reviewerID']], df2_filtered[['reviewerID']], on='reviewerID', how='inner')
print("Number of overlapping reviewerIDs:", len(common_reviewer_ids))
print("Some overlapping reviewerIDs:", common_reviewer_ids['reviewerID'].unique()[:10])


# Initialize empty dataframes to hold the data
temp_df1 = pd.DataFrame()
temp_df2 = pd.DataFrame()

# Initialize sets to track IDs in both dataframes
ids_in_temp_df1 = set()
ids_in_temp_df2 = set()

# Iterate through each unique reviewerID in the overlap
for reviewer_id in common_reviewer_ids['reviewerID'].unique():
    if len(temp_df1) >= 2500 or len(temp_df2) >= 2500:
        break

    # Filter the data for the current reviewerID
    df1_filtered = df1[df1['reviewerID'] == reviewer_id]
    df2_filtered = df2[df2['reviewerID'] == reviewer_id]

    # Determine the number of records to append based on the smaller dataset
    min_records = min(len(df1_filtered), len(df2_filtered))

    # Add IDs from filtered data to the tracking sets
    ids_in_temp_df1.update(df1_filtered['reviewerID'].unique())
    ids_in_temp_df2.update(df2_filtered['reviewerID'].unique())

    # Check if the sets of IDs still have an intersection
    if not (ids_in_temp_df1 & ids_in_temp_df2):
        print(f"Mismatch found: No common reviewerID between temp_df1 and temp_df2 at reviewerID {reviewer_id}.")
        break  # Stop if no overlapping IDs

    # Append the data to the temporary dataframes
    temp_df1 = pd.concat([temp_df1, df1_filtered.head(min_records)])
    temp_df2 = pd.concat([temp_df2, df2_filtered.head(min_records)])
    # Append the filtered data to the temporary dataframes
    #temp_df1 = pd.concat([temp_df1, df1_filtered])
    #temp_df2 = pd.concat([temp_df2, df2_filtered])

"""
# Saving the dataframes to files if the conditions are met
if len(temp_df1) >= 2500 or len(temp_df2) >= 2500:
    new_filename1 = '2500_llama_' + args.output_file1
    new_filename2 = '2500_llama_' + args.output_file2

    temp_df1.to_csv(new_filename1, index=False)
    temp_df2.to_csv(new_filename2, index=False)

    print("Llama Dataset--------------")
    print("Llama Dataset 1 length: ", len(temp_df1))
    print("Llama Dataset 2 length: ", len(temp_df2))
"""

tempdf1_file1 = '2500_llama_'+args.output_file1
tempdf2_file2 = '2500_llama_'+args.output_file2

new_filename1 = f"{args.overlapping_output_dir}/{tempdf1_file1}"
new_filename2  = f"{args.overlapping_output_dir}/{tempdf2_file2}"
#new_filename1 = '2500_llama_' + args.output_file1
#new_filename2 = '2500_llama_' + args.output_file2

temp_df1.to_csv(new_filename1, index=False)
temp_df2.to_csv(new_filename2, index=False)

print("Llama Dataset--------------")
print("Llama Dataset 1 length: ", len(temp_df1))
print("Llama Dataset 2 length: ", len(temp_df2))

print("Data Processing Done...")
