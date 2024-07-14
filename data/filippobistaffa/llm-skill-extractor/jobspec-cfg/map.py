import argparse as ap
import pandas as pd
import numpy as np
import os


from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI()

def get_embedding(text, model='text-embedding-3-small'):
   text = text.replace('\n', ' ')
   return np.array(client.embeddings.create(input=[text], model=model).data[0].embedding)


def cosine_similarity(a, b):
    if len(a) != len(b):
        raise ValueError(f'Arrays of different lengths: {len(a)} != {len(b)}')
    #cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos_sim = np.dot(a, b) # assumes that np.linalg.norm(a) == np.linalg.norm(b) == 1 (holds in case of OpenAI embeddings)
    return cos_sim


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('--embeddings', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'skills-embeddings-3-small.tar.gz'))
    parser.add_argument('--skills', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/clean-0.csv'))
    parser.add_argument('--model', type=str, default='text-embedding-3-small', choices=['text-embedding-3-small', 'text-embedding-3-large'])
    parser.add_argument('-n', type=int, default=5)
    parser.add_argument('--output', type=str)
    args, additional = parser.parse_known_args()
    embeddings = pd.read_pickle(args.embeddings)
    input = pd.read_csv(args.skills, index_col=0, sep=';')
    input['embedding'] = input['skill'].apply(lambda x: get_embedding(x, model=args.model))
    output = []
    for i, row in input.iterrows():
        embeddings['cos_sim'] = embeddings[f'embedding'].apply(lambda x: cosine_similarity(x, row['embedding']))
        topn = embeddings.nlargest(args.n, 'cos_sim')
        print(row['skill'])
        print(topn)
        for _, row_top in topn.iterrows():
            output.append([i + 1, row['skill'], row_top['skill'], row_top['cos_sim']])
    if args.output is not None:
        pd.DataFrame(output, columns=['id', 'skill', 'mapping', 'value']).to_csv(args.output, sep=';', index=False)
