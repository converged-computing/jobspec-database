import argparse
import os
import zipfile

import xml.etree.ElementTree as ET
import re

import nltk
import numpy as np
import pandas as pd
from transformers import pipeline
from yaml import safe_load


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Training configuration file.')
    # parser.add_argument('--results_csv', required=True, help='Results CSV file.')
    args = parser.parse_args()
    return args


def word_positions(text):
    words = text.split()
    positions = []
    current_pos = 0
    for word in words:
        start = text.index(word, current_pos)
        end = start + len(word)
        positions.append((start, end))
        current_pos = end
    return words, positions


def convert_to_conllu(words, positions, entities):
    tagged_words = [(word, 'O') for word in words]

    for entity in entities:
        start_index = entity['start']
        end_index = entity['end']
        entity_type = entity['entity']

        for i, (word, (word_start, word_end)) in enumerate(zip(words, positions)):
            if word_end > start_index and word_start < end_index:
                if word_start == start_index:
                    prefix = 'B-'
                else:
                    prefix = 'I-'
                tagged_words[i] = (word, prefix + entity_type)

    result = '\n'.join([f"{word} {tag}" for word, tag in tagged_words])
    return result


def conll_to_df(conll_lines):
    data = {'sentence_id': [], 'words': [], 'ner_tags': []}
    current_sentence = 1

    for line in conll_lines:
        if line.startswith("-DOCSTART-") or line.strip() == '':
            current_sentence += 1
        else:
            splits = line.strip().split()
            data['sentence_id'].append(current_sentence)
            data['words'].append(splits[0])
            data['ner_tags'].append(splits[1])

    return pd.DataFrame(data)


def get_namespace(element):
    m = re.match(r'\{.*\}', element.tag)
    return m.group(0) if m else ''


def extract_text_from_page_xml(content):
    root = ET.fromstring(content)
    namespaces = {'ns': root.tag.split('}')[0].strip('{')}
    unicode_texts = root.findall('.//ns:Unicode', namespaces=namespaces)
    texts = [elem.text for elem in unicode_texts if elem.text]
    return " ".join(texts)


def main():
    args = parse_arguments()

    with open(args.config, 'r') as config_file:
        config = safe_load(config_file)

    nltk.download('punkt')
    tokenizer = nltk.data.load('tokenizers/punkt/czech.pickle')

    model1 = config["models"]["model1"]["path"]
    model2 = config["models"]["model2"]["path"]
    model3 = config["models"]["model3"]["path"]

    token_classifier1 = pipeline(
        "ner", model=model1, aggregation_strategy="simple"
    )
    token_classifier2 = pipeline(
        "ner", model=model2, aggregation_strategy="simple"
    )
    token_classifier3 = pipeline(
        "ner", model=model3, aggregation_strategy="simple"
    )

    path_raw_text = config["raw_text"]["pero"]["path"]
    head, tail = os.path.split(path_raw_text.rstrip('/\\'))
    new_path = os.path.join(head, tail + '_ner')
    if not os.path.exists(new_path):
        os.makedirs(new_path, exist_ok=True)

    for filename in os.listdir(path_raw_text):
        if filename.endswith(".zip"):
            zip_path = os.path.join(path_raw_text, filename)
            new_zip_path = os.path.join(new_path, filename)

            with zipfile.ZipFile(zip_path, 'r') as z:
                with zipfile.ZipFile(new_zip_path, 'w') as new_z:
                    for file_info in z.infolist():
                        with z.open(file_info) as file:
                            contents = file.read().decode('utf-8')
                            text = extract_text_from_page_xml(contents)

                            sentences = tokenizer.tokenize(text)
                            processed_contents = process_text(sentences, token_classifier1,
                                                              token_classifier2,
                                                              token_classifier3)
                            new_filename = file_info.filename[:-4] + '.conll'
                            new_z.writestr(new_filename,
                                           "\n".join(processed_contents).encode('utf-8'))


def process_long_sentence(sentence, classifier, max_length=512):
    # Tokenize the sentence to words
    words = sentence.split()  # Consider using a tokenizer if words need special handling

    # Split into chunks ensuring each is within the max_length limit
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_length:  # +1 for space
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1  # +1 for space

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    # Process each chunk and adjust indices
    results = []
    offset = 0

    for chunk in chunks:
        chunk_result = classifier(chunk)
        adjusted_chunk_result = []

        for entity in chunk_result:
            # Adjust 'start' and 'end' indices by the current offset
            adjusted_entity = {
                'start': entity['start'] + offset,
                'end': entity['end'] + offset,
                'entity_group': entity['entity_group'],
                'score': entity['score'],
                'word': entity['word']
            }
            adjusted_chunk_result.append(adjusted_entity)

        results.extend(adjusted_chunk_result)
        offset += len(chunk) + 1  # +1 to account for the space after each chunk

    return results


def process_text(texts, token_classifier1, token_classifier2, token_classifier3):
    global final_entity, final_score
    full_output = []
    for sentence in texts:
        sentence = sentence.strip()
        if sentence:
            result1 = process_long_sentence(sentence, token_classifier1)
            result2 = process_long_sentence(sentence, token_classifier2)
            result3 = process_long_sentence(sentence, token_classifier3)

            results = [result1, result2, result3]
            entities_info = {}

            # Shromažďování všech entit do slovníku s klíčem jako (start, end, word)
            for idx, result in enumerate(results):
                for entity in result:
                    key = (entity['start'], entity['end'], entity['word'].strip())
                    if key not in entities_info:
                        entities_info[key] = {'entity_groups': [], 'scores': []}
                    entities_info[key]['entity_groups'].append(entity['entity_group'])
                    entities_info[key]['scores'].append(entity['score'])

            # Rozhodovací logika pro vyhodnocení finální entity a score
            final_results = []
            for key, info in entities_info.items():
                entities = info['entity_groups']
                scores = info['scores']
                final_entity = ""
                final_score = ""

                # Zjistíme, jaká entita se objevuje nejčastěji
                if len(set(entities)) == 1:
                    # Všechny modely souhlasí
                    if len(entities) > 1:
                        final_entity = entities[0]
                        final_score = np.mean(scores)
                    elif scores[0] > 0.7:
                        final_entity = entities[0]
                        final_score = scores[0]
                else:
                    # Počítáme výskyty každé entity
                    entity_counts = {entity: entities.count(entity) for entity in set(entities)}
                    most_common_entity = max(entity_counts, key=entity_counts.get)
                    count_most_common = entity_counts[most_common_entity]

                    if count_most_common > 1:
                        # Existuje majorita
                        final_entity = most_common_entity
                        # Průměrné skóre pro nejčastější entitu
                        indices = [i for i, e in enumerate(entities) if e == most_common_entity]
                        final_score = np.mean([scores[i] for i in indices])
                    else:
                        # Žádná majorita, bere se entita s nejvyšším skóre
                        if max(scores) > 0.7:
                            max_index = np.argmax(scores)
                            final_entity = entities[max_index]
                            final_score = scores[max_index]
                if final_entity != "":
                    final_results.append({
                        'word': key[2],
                        'entity': final_entity,
                        'score': final_score,
                        'start': key[0],
                        'end': key[1]
                    })

            words, positions = word_positions(sentence)
            conllu_output = convert_to_conllu(words, positions, final_results)
            full_output.append(conllu_output)
            full_output.append('')
    return full_output


if __name__ == "__main__":
    main()
