import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
import evaluate as evaluate
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler, T5Tokenizer, T5EncoderModel, T5ForConditionalGeneration
import argparse
import subprocess
import matplotlib.pyplot as plt


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
        input_encoding = question + " [SEP] " + passage

        # encode_plus will encode the input and return a dictionary of tensors
        encoded_review = self.tokenizer.encode_plus(
            input_encoding,
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


class T5Pooler(nn.Module):
    def __init__(self, hidden_size, activation=nn.Tanh()):
        super().__init__()
        self.dense = nn.Linear(768, hidden_size)
        self.activation = activation

    def forward(self, hidden_states):
        # take the mean of the hidden states
        mean_tensor = torch.mean(hidden_states, dim=1)
        return self.activation(self.dense(mean_tensor))


class T5Model(nn.Module):
    def __init__(self):
        super(T5Model, self).__init__()
        self.t5 = T5EncoderModel.from_pretrained('t5-base')
        self.pooler = T5Pooler(768, nn.LeakyReLU())
        self.fc = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.t5(input_ids, attention_mask=attention_mask)
        pooled_outputs = self.pooler(outputs.last_hidden_state)
        outputs = self.fc(pooled_outputs)
        return outputs


t5_model = T5Model()


def evaluate_model(model, dataloader, device, t5=False):
    """ Evaluate a PyTorch Model
    :param torch.nn.Module model: the model to be evaluated
    :param torch.utils.data.DataLoader test_dataloader: DataLoader containing testing examples
    :param torch.device device: the device that we'll be training on
    :param bool t5: if the model is t5 or not
    :return accuracy
    """
    # load metrics
    dev_accuracy = evaluate.load('accuracy')

    # turn model into evaluation mode
    model.eval()

    if t5:
        model = t5_model
        t5_model.to(device)
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            output.logits = torch.reshape(output.logits, (output.logits.size(0), -1))
            predictions = torch.argmax(output.logits, dim=1)
            dev_accuracy.add_batch(predictions=predictions, references=batch['labels'])

    else:
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask)

            predictions = output.logits
            predictions = torch.argmax(predictions, dim=1)
            dev_accuracy.add_batch(predictions=predictions, references=batch['labels'])

    # compute and return metrics
    return dev_accuracy.compute()


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

    t5 = False
    if isinstance(mymodel, T5ForConditionalGeneration):
        t5 = True

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

            if t5:
                # mymodel = t5_model
                output = mymodel(input_ids=input_ids, attention_mask=attention_mask, labels=labels.view((-1, 1)))
                ce_loss = output.loss
                output.logits = torch.reshape(output.logits, (output.logits.size(0), -1))
                predictions = torch.argmax(output.logits, dim=1)

            else:
                output = mymodel(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                ce_loss = loss(output.logits, labels)
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
        val_accuracy = evaluate_model(mymodel, validation_dataloader, device, t5)
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


def pre_process(model_name, batch_size, device, small_subset=False):
    # download dataset
    print("Loading the dataset ...")
    dataset = load_dataset("boolq")
    dataset = dataset.shuffle()  # shuffle the data

    print("Slicing the data...")
    if small_subset:
        # use this tiny subset for debugging the implementation
        dataset_train_subset = dataset['train'][:10]
        dataset_dev_subset = dataset['train'][:10]
        dataset_test_subset = dataset['train'][:10]
    else:
        # since the dataset does not come with any validation data,
        # split the training data into "train" and "dev"
        dataset_train_subset = dataset['train'][:8000]
        dataset_dev_subset = dataset['validation']
        dataset_test_subset = dataset['train'][8000:]

    # maximum length of the input; any input longer than this will be truncated
    # we had to do some pre-processing on the data to figure what is the length of most instances in the dataset
    max_len = 128
    t5 = False
    if 't5' in model_name:
        t5 = True

    print("Loading the tokenizer...")
    if t5:
        mytokenizer = T5Tokenizer.from_pretrained(model_name)
    else:
        mytokenizer = AutoTokenizer.from_pretrained(model_name)

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

    # from Hugging Face (transformers), read their documentation to do this.
    print("Loading the model ...")
    if t5:
        pretrained_model = T5ForConditionalGeneration.from_pretrained(model_name, num_labels=2)
    else:
        pretrained_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    print("Moving model to device ..." + str(device))
    pretrained_model.to(device)
    return pretrained_model, train_dataloader, validation_dataloader, test_dataloader


# the entry point of the program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--small_subset", type=str, default=False)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")

    args = parser.parse_args()
    print(f"Specified arguments: {args}")

    # Fixing argparse for small_subset param
    small_subset = str(args.small_subset).upper()
    if small_subset == 'TRUE' or small_subset == "1":
        small_subset = True
    else:
        small_subset = False

    opti_hyperparams = False
    if opti_hyperparams:
        # Hyperparam Optimization
        lrs = [1e-4, 5e-4, 1e-3]
        epochs = [5, 7, 9]
        best_acc = -1
        best_test_acc = -1
        best_lr = -1
        best_epoch = -1

        for lr in lrs:
            for epoch in epochs:

                # load the data and models
                pretrained_model, train_dataloader, validation_dataloader, test_dataloader = pre_process(args.model,
                                                                                                         args.batch_size,
                                                                                                         args.device,
                                                                                                         small_subset)

                print(" >>>>>>>>  Starting training ... ")
                # train(pretrained_model, args.num_epochs, train_dataloader, validation_dataloader, args.device, args.lr)
                train(pretrained_model, epoch, train_dataloader, validation_dataloader, args.device, lr)

                # print the GPU memory usage just to make sure things are alright
                print_gpu_memory()

                val_accuracy = evaluate_model(pretrained_model, validation_dataloader, args.device)
                print(f" - Average DEV metrics: accuracy={val_accuracy['accuracy']}")

                test_accuracy = evaluate_model(pretrained_model, test_dataloader, args.device)
                print(f" - Average TEST metrics: accuracy={test_accuracy['accuracy']}")

                if val_accuracy['accuracy'] > best_acc:
                    best_acc = val_accuracy['accuracy']
                    best_test_acc = test_accuracy['accuracy']
                    best_lr = lr
                    best_epoch = epoch

                del pretrained_model, train_dataloader, validation_dataloader, test_dataloader

        print(
            f"Best Hyperparams - LR: {best_lr}, Epochs: {best_epoch}, Validation Acc: {best_acc}, Test Acc: {best_test_acc}")
    else:
        # load the data and models
        pretrained_model, train_dataloader, validation_dataloader, test_dataloader = pre_process(args.model,
                                                                                                 args.batch_size,
                                                                                                 args.device,
                                                                                                 small_subset)

        print(" >>>>>>>>  Starting training ... ")
        train(pretrained_model, args.num_epochs, train_dataloader, validation_dataloader, args.device, args.lr)

        # print the GPU memory usage just to make sure things are alright
        print_gpu_memory()

        val_accuracy = evaluate_model(pretrained_model, validation_dataloader, args.device)
        print(f" - Average DEV metrics: accuracy={val_accuracy['accuracy']}")

        test_accuracy = evaluate_model(pretrained_model, test_dataloader, args.device)
        print(f" - Average TEST metrics: accuracy={test_accuracy['accuracy']}")
