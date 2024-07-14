import random
import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForCausalLM


def pre_process(num_shots=8, eval_records=30):
    # download dataset
    print("Loading the dataset ...")
    dataset = load_dataset("boolq")
    dataset = dataset.shuffle()  # shuffle the data

    # random sampling num_shots / 2 true and num_shots / 2 false prompts
    print(f"Randomly selecting {num_shots} prompts from dataset ...")
    true_samples, ct_true = [], 0
    false_samples, ct_false = [], 0
    rand_idx = np.random.randint(0, len(dataset['train']), num_shots * 2)
    for idx in rand_idx:
        sample = dataset['train'][int(idx)]
        if sample['answer'] and ct_true < num_shots // 2:
            true_samples.append(sample)
            ct_true += 1
        elif sample['answer'] is False and ct_false < num_shots // 2:
            false_samples.append(sample)
            ct_false += 1

    # randomly shuffling samples to avoid recency bias
    samples = true_samples + false_samples
    random.shuffle(samples)

    # random sampling eval_records prompts from the validation data
    print(f"Randomly selecting {eval_records} prompts from validation dataset ...")
    rand_idx = np.random.randint(0, len(dataset['validation']), eval_records)
    test_samples = [dataset['validation'][int(idx)] for idx in rand_idx]

    return samples, test_samples


def create_prompt_str(samples):
    prompt = ""
    for samp in samples:
        prompt += "Passage: "
        prompt += samp['passage']
        prompt += "\n"
        prompt += "Question: "
        prompt += samp['question'] + "?"
        prompt += "\n"
        prompt += "Answer: "
        prompt += "True" if samp['answer'] else "False"
        prompt += "\n"
    return prompt


def evaluate_bloomz(prompt, test_samples, model_name='bigscience/bloomz'):
    # init the bloomz tokenizer and pretrained model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    correct_preds = 0
    total_preds = len(test_samples)
    for test_sample in test_samples:
        prompt_curr_sample = prompt
        prompt_curr_sample += "Passage: "
        prompt_curr_sample += test_sample['passage']
        prompt_curr_sample += "\n"
        prompt_curr_sample += "Question: "
        prompt_curr_sample += test_sample['question'] + "?"
        prompt_curr_sample += "\n"
        prompt_curr_sample += "Answer:"

        inputs = tokenizer.encode(prompt_curr_sample, return_tensors="pt")
        outputs = model.generate(inputs, max_new_tokens=3)
        result = tokenizer.decode(outputs[0])

        bloomz_ans = result
        # remove previous prompts from the bloomz's response - take only last line containing ans when \n is encountered
        if '\n' in bloomz_ans:
            bloomz_ans = bloomz_ans[bloomz_ans.rindex('\n') + 1:]

        bloomz_pred = any(true_keyword in bloomz_ans.lower() for true_keyword in ['true', 'yes'])
        print()
        print(f"Passage: {test_sample['passage']}")
        print(f"Question: {test_sample['question']}?")
        print(f"Predicted: {bloomz_pred}, Actual: {test_sample['answer']}")
        if bloomz_pred == test_sample['answer']:
            correct_preds += 1

    print()
    print(f'Correct Preds: {correct_preds}, Total Preds: {total_preds}')
    return correct_preds / total_preds * 100


# the entry point of the program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--model", type=str, default="bigscience/bloomz-560m")

    args = parser.parse_args()
    print(f"Specified arguments: {args}")

    # load the data and preprocess for prompts and eval data
    samples, test_samples = pre_process(num_shots=8, eval_records=100)
    prompt = create_prompt_str(samples=samples)

    print(" >>>>>>>>  Sending prompts to Bloomz model ... ")
    accuracy = evaluate_bloomz(prompt=prompt, test_samples=test_samples, model_name=args.model)
    print(f'Accuracy: {accuracy}%')
