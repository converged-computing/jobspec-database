import argparse
import json

import numpy as np

from eval_coqa import CoQAEvaluator

def add_arguments(parser):
    parser.add_argument("--input_file", help="path to input file", required=True)
    parser.add_argument("--output_file", help="path to output file", required=True)
    parser.add_argument("--answer_threshold", help="threshold of answer", required=False, default=0.1, type=float)

def convert_coqa(input_file,
                 output_file,
                 answer_threshold):
    with open(input_file, "r") as file:
        input_data = json.load(file)
    
    output_data = []
    for key, answer in input_data.items():
        qas = key.split('_')
        id = qas[0]
        turn_id = int(qas[1])
        
        output_data.append({
            "id": id,
            "turn_id": turn_id,
            "answer": answer
        })        
    
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    convert_coqa(args.input_file, args.output_file, args.answer_threshold)
