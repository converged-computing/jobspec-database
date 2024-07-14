#!/usr/bin/env python3
import logging
import csv
import argparse
import tqdm
import torch
import json
from common import dataset
from question import Question

def create_arg_parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--alpha", help="probability for a word not to be substituted in smoothing distribution",type=float, default=0.75)
    parser.add_argument("-i", "--index", help="index of question in dataset, starting from 0", type=int, default=0)
    parser.add_argument("-N", "--smoothing_number", help="number of smoothed inputs to take",type=int, default=100)
    parser.add_argument("-m", "--quantile", help="q-th quartile to take to estimate the enclosing ball with probability \
                        1- alpha_2 : see equation 8",type=int, default=100)
    parser.add_argument("-k", "--top_k", help="number of synonyms to be considered for smoothing",type=int, default=10)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser

def sample(current_question, N : int, top : int, alpha : float, filename : str):
    """
    For each radius :  
    * Generates N smoothed questions
    * Prompts the model (T5) on each question
    * Dumps the output in a file
    Parameters : 
    - current_question : Original question in dataset
    - N                : Number of smoothed questions to be generated for each perturbed question
    - top              : number of synonyms considered when smoothing (variable "K" in the paper)
    - alpha            : Probability of not changing the original word when smoothing
    - filename         : output file name
    """
    with open(filename, "a") as f:

        torch.cuda.empty_cache() 
        from common import t5_tok, t5_qa_model, verify_vocab_in_w2v
        writer = csv.writer(f)
        writer.writerow(["question", "answer"])
        smooth_prompts = current_question.generate_smooth_N_questions(N, top, alpha)

        for _, smooth_prompt in tqdm.tqdm(enumerate(smooth_prompts)):
            
            input_ids = t5_tok(smooth_prompt, return_tensors="pt").input_ids
            gen_output = t5_qa_model.generate(input_ids)[0]
            smooth_answer = t5_tok.decode(gen_output, skip_special_tokens=True)
            
            # If all the words of the answer are not in word2vec's vocabulary, 
            # we need to roll a new question, so that we can compute Word Mover's Distance

            if not verify_vocab_in_w2v(smooth_answer):
                
                # logger.debug(f"Answer {smooth_answer} not in vocabulary of Word2Vec")
                # We re-roll a new question 10 times. If this fails, we go to the next smooth question
                # N.B. Another option is actually just to continue and write nothing for this smooth question.
                
                # continue

                i = 0
                while (i < 10) and (not verify_vocab_in_w2v(smooth_answer)):
                    new_prompt = current_question.generate_smooth_questions(top, alpha)
                    input_ids = t5_tok(new_prompt, return_tensors="pt").input_ids
                    gen_output = t5_qa_model.generate(input_ids)[0]
                    smooth_answer = t5_tok.decode(gen_output, skip_special_tokens=True)
                    i+=1
                if i == 10:
                    # logger.debug(f"Answer re-roll exceeded 10 rolls, refrain to put down answer")
                    continue
            del input_ids
            torch.cuda.empty_cache() 

            writer.writerow([smooth_prompt, smooth_answer])
        f.close()

if __name__ == "__main__":
    """
    Samples Z = {z_i} such that z_i follows a distribution f(\phi (x)) three times, twice for "smooth" step and once for "certify" step.
    For each question, outputs 3 files corresponding to the three Z samples named as question<question_number>_<1/2/3>.
    Computation of MEB and of the center of the ball is done in smooth.py, using question<question_number>_1 and question<question_number>_2.
    Computation of the max radius is done in certify.py, using question<question_number>_3.
    """
    parser = create_arg_parse()
    args = parser.parse_args()

    alpha = args.alpha
    N = args.smoothing_number
    k = args.top_k
    m = args.quantile
    i = args.index
    logger = logging.getLogger("__prompt__")

    if not args.verbose:
        logger.setLevel(logging.ERROR)
    elif args.verbose == 1:
        logger.setLevel(logging.WARNING)
    elif args.verbose == 2:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)        

    ### Prompt
    logger.debug("Reached generation loop")

    logger.debug(f"Current question : {dataset[i][1]['question']}")
    frag_filename = "question_prompt"+str(i)

    current_question = Question(dataset[i][1]['question'], dataset[i][1]['answer']['normalized_aliases'], dataset[i][1]['question_id'])
    current_question.generate_synonyms_albert(k)

    # first N sample for smooth algorithm
    first_sample_name = frag_filename + "_1"
    sample(current_question, N, k, alpha, first_sample_name)  

    # second N sample for smooth algorithm
    second_sample_name = frag_filename + "_2"
    sample(current_question, N, k, alpha, second_sample_name) 

    # first m sample for certify algorithm
    third_sample_name = frag_filename + "_3"
    sample(current_question, m, k, alpha, third_sample_name)

    config = {"alpha": args.alpha, "N": args.smoothing_number, "k" : args.top_k, "m" : args.quantile}

    with open('config_prompt.json', 'w') as f:
        json.dump(config, f)
        f.close()
    