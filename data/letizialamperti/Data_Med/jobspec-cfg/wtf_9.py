import re
import sys
import logging
import warnings
import os
import pandas as pd
import gzip
from Bio import SeqIO
from pathlib import Path
from tqdm.notebook import tqdm

def setup_logging():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def exit_with_error(message):
    logging.error(message)
    sys.exit(1)

def locate_excel_file(directory):
    for entry in os.listdir(directory):
        if entry.lower().startswith("corr_tag") and entry.lower().endswith(".xlsx"):
            return Path(directory) / entry
    return None

def read_filename_list(filename_list_path):
    try:
        with open(filename_list_path, 'r') as file:
            filenames = [line.strip() for line in file]
        return filenames
    except Exception as e:
        exit_with_error(f"Error reading filename list: {str(e)}")


def load_metadata(excel_file):
    try:
        reference_df = pd.read_excel(excel_file)
        sample_column_name = reference_df.columns[reference_df.columns.str.lower() == 'sample'].tolist()
        tag_column_name = reference_df.columns[reference_df.columns.str.lower() == 'tag'].tolist()
        if not sample_column_name or not tag_column_name:
            exit_with_error("Column 'Sample' or 'TAG' not found in the Excel file. Exiting.")
        
        # Rename columns to 'TAG' and 'SAMPLE'
        reference_df.rename(columns={tag_column_name[0]: 'TAG', sample_column_name[0]: 'SAMPLE'}, inplace=True)

        # Convert all column names to uppercase
        reference_df.columns = reference_df.columns.str.upper()

        return reference_df
    except Exception as e:
        exit_with_error(f"Error loading metadata: {str(e)}")

def get_tag(seq: str) -> str:
    f_primer = "acaccgcccgtcactct"
    r_primer = "cttccggtacacttaccatg"

    f_primer_len = len(f_primer)
    r_primer_len = len(r_primer)

    tag = None
    pos_f = seq[:35].find(f_primer)
    pos_r = seq[:35].find(r_primer)

    if pos_f >= 8:  # Forward primer found
        tag = seq[pos_f - 8: pos_f]

    elif pos_r >= 8:  # Reverse primer found
        tag = seq[pos_r - 8: pos_r]

    return tag


def extract_base_sample_name(sample_name):
    return re.sub(r'_\d+$', '', sample_name)
    

def process_file(directory, filename, reference_df, unique_sample_dataframes, forward_file, reverse_file):
    logging.info("Processing: %s", filename)

    # Extract the RUN name from the filename using a regular expression
    run_name_match = re.search(r'_(.{8})___[R12]', filename)
    if not run_name_match:
        logging.warning(f"Skipping {filename}: RUN name not found in filename.")
        return

    run_name = run_name_match.group(1)
    logging.info("Extracted RUN name: %s", run_name)

    # Filter metadata based on the extracted RUN name
    run_metadata = reference_df[reference_df['RUN'].str.contains(run_name, case=False, na=False)]
    logging.info("Run metadata: %s", run_metadata)

    if run_metadata.empty:
        logging.warning("No metadata found for the current RUN.")
        return

    # Get unique_sample_names present in the run_metadata
    unique_sample_names_in_metadata = set(run_metadata['SAMPLE'].apply(extract_base_sample_name))

    # Get tags associated with the unique_sample_names
    tags_for_unique_sample_names = {}
    for unique_sample_name in unique_sample_names_in_metadata:
        tags_for_unique_sample_names[unique_sample_name] = unique_sample_dataframes[unique_sample_name]['tags'].keys()

    try:
        # Process both forward and reverse FASTQ files
        for forward_file, reverse_file in zip([forward_file], [reverse_file]):
            with gzip.open(forward_file, 'rt') as forward_handle, gzip.open(reverse_file, 'rt') as reverse_handle:
                for forward_record, reverse_record in zip(SeqIO.parse(forward_handle, format='fastq'), SeqIO.parse(reverse_handle, format='fastq')):
                    try:
                        forward_id, forward_seq = forward_record.id, forward_record.seq.lower()
                        reverse_id, reverse_seq = reverse_record.id, reverse_record.seq.lower()
        
                        # Extract tags using the new function
                        forward_tag = get_tag(str(forward_seq))
                        reverse_tag = get_tag(str(reverse_seq))
        
                        # Check if tags are None
                        if forward_tag is None or reverse_tag is None:
                            continue
                        
                        # Check if tags are different
                        if forward_tag != reverse_tag:
                            continue
                        
                        # Check if any of the tags for unique_sample_names is present in the record IDs
                        for unique_sample_name, tags in tags_for_unique_sample_names.items():
                            if forward_tag in tags and reverse_tag in tags:
                                # Append both forward and reverse sequences for the same record ID
                                unique_sample_dataframes[unique_sample_name]['tags'][forward_tag]['seqs_forward'].append(str(forward_seq))
                                unique_sample_dataframes[unique_sample_name]['tags'][reverse_tag]['seqs_reverse'].append(str(reverse_seq))

                    except NameError as e:
                        print(f"Error processing record: {str(e)}")
                        continue

    except Exception as e:
        logging.warning(f"Error processing file {filename}: {str(e)}")

def save_to_csv(unique_sample_dataframes, directory_name):
    logging.info("Saving CSVs")

    store_dir = Path('/scratch/snx3000/llampert/MED_SAMPLES_CSV') / directory_name
    store_dir.mkdir(parents=True, exist_ok=True)

    for unique_sample_name, data in tqdm(unique_sample_dataframes.items(), desc="Saving CSVs"):
        combined_df = pd.DataFrame()  # Initialize combined_df for each unique sample

        for tag, tag_data in data['tags'].items():
            forward_length = len(tag_data['seqs_forward'])
            reverse_length = len(tag_data['seqs_reverse'])
            print(f"{unique_sample_name} - {tag}: Forward length={forward_length}, Reverse length={reverse_length}")

            # Check if lengths match
            if forward_length != reverse_length:
                print(f"Lengths mismatch for {unique_sample_name} - {tag}")
                continue

            try:
                print(f"we are trying to create the dataframe...")
                # Create a DataFrame with forward and reverse sequences for the current tag
                tag_df = pd.DataFrame(data={'Forward': tag_data['seqs_forward'],
                                             'Reverse': tag_data['seqs_reverse']})

                # Debugging statements
                logging.info(f"Combined CSV DataFrame size for {unique_sample_name} - {tag}: {tag_df.shape}")

                # Append the current tag DataFrame to the combined DataFrame
                # combined_df = combined_df.append(tag_df, ignore_index=True)
        
                combined_df = pd.concat([combined_df, tag_df], ignore_index=True)

            except Exception as e:
                logging.error(f"Error creating DataFrame for {unique_sample_name} - {tag}: {str(e)}")

        try:
            # Save the combined DataFrame to a single CSV file
            save_file = store_dir / f'{unique_sample_name}.csv'
            logging.info(f"Saving combined CSV for {unique_sample_name} to {save_file}")

            combined_df = combined_df.dropna()
            combined_df = combined_df.sample(frac=1)  # randomize
            combined_df.to_csv(save_file, index=False)
            logging.info(f"Combined CSV saved for {unique_sample_name}")

        except Exception as e:
            logging.error(f"Error saving combined CSV for {unique_sample_name}: {str(e)}")



def main():
    setup_logging()

    if len(sys.argv) < 2:
        exit_with_error("Please provide the directory name as an argument.")

    directory_name = sys.argv[1]
    fastq_dir = Path(f'/store/sdsc/sd29/med_data_wp3/{directory_name}')
    store_dir = Path('/scratch/snx3000/llampert/MED_SAMPLES_CSV')

    excel_file = locate_excel_file(fastq_dir)
    if excel_file is None:
        exit_with_error("Excel file not found. Exiting.")

    reference_df = load_metadata(excel_file)
    
    # Modify: Remove entries with specified sample prefixes
    samples_to_exclude = ['other', 'OTHER', 'Other', 'CINEG']
    reference_df = reference_df[~reference_df['SAMPLE'].str.lower().str.startswith(tuple(samples_to_exclude))]

    filename_list_path = f'/users/llampert/Data_Med/Fieldworks_refs/{directory_name}.txt'
    all_filenames = read_filename_list(filename_list_path)

    # Define num_files after reading the filenames
    num_files = len(all_filenames)

    unique_sample_dataframes = {}
    logging.info(f"All Filenames: {all_filenames}, num_files: {num_files}")

    # Iterate through each unique sample in reference_df and initialize the data structure
    for _, row in reference_df.iterrows():
        sample_name = row['SAMPLE']
        tag = row['TAG']
        unique_sample_name = extract_base_sample_name(sample_name)

        if unique_sample_name not in unique_sample_dataframes:
            unique_sample_dataframes[unique_sample_name] = {'tags': {tag: {'seqs_forward': [], 'seqs_reverse': []}}}
        elif tag not in unique_sample_dataframes[unique_sample_name]['tags']:
            unique_sample_dataframes[unique_sample_name]['tags'][tag] = {'seqs_forward': [], 'seqs_reverse': []}

    logging.info("Initialized unique_sample_dataframes with all samples from reference_df.")

    # Process R1 files and store data in unique_sample_dataframes
    for i, filename in tqdm(enumerate(all_filenames), desc="Processing R1 files", total=num_files):
        if "_R1.fastq.gz" not in filename:
            # Skip files that are not R1 files
            continue

        # Process the forward and reverse FASTQ files directly within the loop
        forward_file = fastq_dir / f"{filename[:-14]}___R1.fastq.gz"
        reverse_file = fastq_dir / f"{filename[:-14]}___R2.fastq.gz"

        process_file(fastq_dir, filename, reference_df, unique_sample_dataframes, forward_file, reverse_file)


    save_to_csv(unique_sample_dataframes, directory_name)


if __name__ == "__main__":
    main()
