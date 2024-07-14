import torch
import random
import argparse
import evaluate
import subprocess
import numpy as np
import torch.nn as nn
from datasets import load_dataset
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from transformers import BloomTokenizerFast
from petals import DistributedBloomForCausalLM
from transformers import AutoTokenizer, get_scheduler

"""
References:
Took guidance from this colab notebook: 
https://colab.research.google.com/drive/1Ervk6HPNS6AYVr3xVdQnY5a-TjjmLCdQ?usp=sharing#scrollTo=iK_iT8J3zDC0
This is given on the official documentation and getting started with Petals and Bloom here: https://petals.ml/
"""


class PetalsFinetuned(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.distributed_layers = model.transformer.h
        self.adapter = nn.Sequential(nn.Linear(14336, 32), nn.Linear(32, 14336))
        self.head = nn.Sequential(nn.LayerNorm(14336), nn.Linear(14336, 1))

    def forward(self, embeddings):
        hidden_states = self.distributed_layers[0:6](embeddings)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.distributed_layers[6:10](hidden_states)
        pooled_states = torch.mean(hidden_states, dim=1)
        return self.head(pooled_states)


def print_gpu_memory():
    """
    Print the amount of GPU memory used by the current process
    This is useful for debugging memory issues on the GPU
    """
    # check if gpu is available
    if torch.cuda.is_available():
        print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
        print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
        print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))

        p = subprocess.check_output('nvidia-smi')
        print(p.decode("utf-8"))


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


def evaluate_petals(prompt, test_samples, model_name='bigscience/bloomz', ptuning_mode='ptune'):
    # init the petals tokenizer and pretrained model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DistributedBloomForCausalLM.from_pretrained(model_name)  # , tuning_mode=ptuning_mode)

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

        petals_ans = result
        # remove previous prompts from the petals's response - take only last line containing ans when \n is encountered
        if '\n' in petals_ans:
            petals_ans = petals_ans[petals_ans.rindex('\n') + 1:]

        petals_pred = any(true_keyword in petals_ans.lower() for true_keyword in ['true', 'yes'])
        print()
        print(f"Passage: {test_sample['passage']}")
        print(f"Question: {test_sample['question']}?")
        print(f"Predicted: {petals_pred}, Actual: {test_sample['answer']}")
        if petals_pred == test_sample['answer']:
            correct_preds += 1

    print()
    print(f'Correct Preds: {correct_preds}, Total Preds: {total_preds}')
    return correct_preds / total_preds * 100


class BoolQADataset(torch.utils.data.Dataset):
    """
    Dataset for the dataset of BoolQ questions and answers
    """

    def __init__(self, passages, questions, answers, tokenizer, max_len):
        self.passages = passages
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, index):
        """
        This function is called by the DataLoader to get an instance of the data
        :param index:
        :return:
        """

        passage = str(self.passages[index])
        question = self.questions[index]
        answer = self.answers[index]

        # this is input encoding for your model. Note, question comes first since we are doing question answering
        # and we don't wnt it to be truncated if the passage is too long
        prompt_curr_sample = ""
        prompt_curr_sample += "Passage: "
        prompt_curr_sample += passage
        prompt_curr_sample += "\n"
        prompt_curr_sample += "Question: "
        prompt_curr_sample += question + "?"
        prompt_curr_sample += "\n"
        prompt_curr_sample += "Answer:"

        # encode_plus will encode the input and return a dictionary of tensors
        encoded_review = self.tokenizer.encode(
            prompt_curr_sample,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )

        return {
            'input_ids': encoded_review['input_ids'][0],  # we only have one example in the batch
            'attention_mask': encoded_review['attention_mask'][0],
            # attention mask tells the model where tokens are padding
            'labels': torch.tensor(answer, dtype=torch.long)  # labels are the answers (yes/no)
        }


def create_dataloaders(mytokenizer, n_samples=50, batch_size=1):
    # download dataset
    dataset = load_dataset("boolq")
    dataset = dataset.shuffle()  # shuffle the data

    # use this tiny subset for debugging the implementation
    dataset_train_subset = dataset['train'][:n_samples]
    dataset_dev_subset = dataset['train'][:n_samples]
    dataset_test_subset = dataset['train'][:n_samples]

    # maximum length of the input; any input longer than this will be truncated
    # we had to do some pre-processing on the data to figure what is the length of most instances in the dataset
    max_len = 128

    print("Loding the data into DS...")
    train_dataset = BoolQADataset(
        passages=list(dataset_train_subset['passage']),
        questions=list(dataset_train_subset['question']),
        answers=list(dataset_train_subset['answer']),
        tokenizer=mytokenizer,
        max_len=max_len
    )
    validation_dataset = BoolQADataset(
        passages=list(dataset_dev_subset['passage']),
        questions=list(dataset_dev_subset['question']),
        answers=list(dataset_dev_subset['answer']),
        tokenizer=mytokenizer,
        max_len=max_len
    )
    test_dataset = BoolQADataset(
        passages=list(dataset_test_subset['passage']),
        questions=list(dataset_test_subset['question']),
        answers=list(dataset_test_subset['answer']),
        tokenizer=mytokenizer,
        max_len=max_len
    )

    print(" >>>>>>>> Initializing the data loaders ... ")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, validation_dataloader, test_dataloader


def train(mymodel, num_epochs, train_dataloader, validation_dataloader, device, lr):
    """ Train a PyTorch Module

    :param torch.nn.Module mymodel: the model to be trained
    :param int num_epochs: number of epochs to train for
    :param torch.utils.data.DataLoader train_dataloader: DataLoader containing training examples
    :param torch.utils.data.DataLoader validation_dataloader: DataLoader containing validation examples
    :param torch.device device: the device that we'll be training on
    :param float lr: learning rate
    :return None
    """

    # here, we use the AdamW optimizer. Use torch.optim.Adam.
    # instantiate it on the untrained model parameters with a learning rate of 5e-5
    print(" >>>>>>>>  Initializing optimizer")
    optimizer = torch.optim.AdamW(mymodel.parameters(), lr=lr)

    # now, we set up the learning rate scheduler
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_dataloader) * num_epochs
    )

    loss = torch.nn.CrossEntropyLoss()
    train_acc_store = []
    val_acc_store = []

    for epoch in range(num_epochs):

        # put the model in training mode (important that this is done each epoch,
        # since we put the model into eval mode during validation)
        mymodel.train()

        # load metrics
        train_accuracy = evaluate.load('accuracy')

        print(f"Epoch {epoch + 1} training:")

        for i, batch in enumerate(train_dataloader):
            """
            You need to make some changes here to make this function work.
            Specifically, you need to: 
            Extract the input_ids, attention_mask, and labels from the batch; then send them to the device. 
            Then, pass the input_ids and attention_mask to the model to get the logits.
            Then, compute the loss using the logits and the labels.
            Then, call loss.backward() to compute the gradients.
            Then, call optimizer.step()  to update the model parameters.
            Then, call lr_scheduler.step() to update the learning rate.
            Then, call optimizer.zero_grad() to reset the gradients for the next iteration.
            Then, compute the accuracy using the logits and the labels.
            """
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            output = mymodel(input_ids=input_ids, attention_mask=attention_mask, labels=labels.view((-1, 1)))
            ce_loss = loss(output, labels)
            predictions = torch.argmax(output.logits, dim=1)

            ce_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # update metrics
            train_accuracy.add_batch(predictions=predictions, references=labels)

        # print evaluation metrics
        train_acc = train_accuracy.compute()
        print(f" ===> Epoch {epoch + 1}")
        print(f" - Average training metrics: accuracy={train_acc['accuracy']}")
        train_acc_store.append(train_acc['accuracy'])

        # normally, validation would be more useful when training for many epochs
        val_accuracy = evaluate_model(mymodel, validation_dataloader, device)
        print(f" - Average validation metrics: accuracy={val_accuracy['accuracy']}")
        val_acc_store.append(val_accuracy['accuracy'])

    # Plotting epoch-wise train accuracy curve:
    plt.plot(train_acc_store, '-o', label='train_acc', color='blue')
    plt.xlabel('Epoch Number')
    plt.ylabel('Training Acc')
    plt.legend()
    plt.savefig(f'train_acc_batchsize_{train_dataloader.batch_size}_lr_{lr}.png')
    # plt.savefig('train_acc.pdf')

    # Plotting epoch-wise validation accuracy curve:
    plt.plot(val_acc_store, '-o', label='val_acc', color='green')
    plt.xlabel('Epoch Number')
    plt.ylabel('Validation Acc')
    plt.legend()
    plt.savefig(f'val_acc_batchsize_{train_dataloader.batch_size}_lr_{lr}.png')
    # plt.savefig('val_acc.pdf')


def evaluate_model(model, dataloader, device):
    """ Evaluate a PyTorch Model
    :param torch.nn.Module model: the model to be evaluated
    :param torch.utils.data.DataLoader dataloader: DataLoader containing testing examples
    :param torch.device device: the device that we'll be training on
    :return accuracy
    """
    # load metrics
    dev_accuracy = evaluate.load('accuracy')

    # turn model into evaluation mode
    model.eval()

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        output = model(input_ids=input_ids, attention_mask=attention_mask)

        predictions = output.logits
        predictions = torch.argmax(predictions, dim=1)
        dev_accuracy.add_batch(predictions=predictions, references=batch['labels'])

    # compute and return metrics
    return dev_accuracy.compute()


# the entry point of the program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="bigscience/bloomz-560m")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    print(f"Specified arguments: {args}")

    # load the data and preprocess for prompts and eval data
    samples, test_samples = pre_process(num_shots=8, eval_records=100)
    prompt = create_prompt_str(samples=samples)

    print(" >>>>>>>>  Sending prompts to Petals model ... ")
    accuracy = evaluate_petals(prompt=prompt, test_samples=test_samples, model_name=args.model,
                               ptuning_mode='ptune')
    print(f'Accuracy: {accuracy}%')

    print(" >>>>>>>>  Finetuning the Petals model ... ")
    print(" >>>>>>>>  Starting training ... ")
    model = PetalsFinetuned(model=args.model)
    tokenizer = BloomTokenizerFast.from_pretrained(args.model)
    train_dataloader, validation_dataloader, test_dataloader = create_dataloaders(mytokenizer=tokenizer, n_samples=50,
                                                                                  batch_size=args.batch_size)
    train(model, args.num_epochs, train_dataloader, validation_dataloader, args.device, args.lr)

    # print the GPU memory usage just to make sure things are alright
    print_gpu_memory()

    val_accuracy = evaluate_model(model, validation_dataloader, args.device)
    print(f" - Average DEV metrics: accuracy={val_accuracy['accuracy']}")

    test_accuracy = evaluate_model(model, test_dataloader, args.device)
    print(f" - Average TEST metrics: accuracy={test_accuracy['accuracy']}")

    accuracy = evaluate_petals(prompt=prompt, test_samples=test_samples, model_name=args.model,
                               ptuning_mode='ptune')
    print(f'Accuracy: {accuracy}%')
