from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from huggingface_hub import login
login(token='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', add_to_git_credential=True)


def save_model_local(
    model_id,
    num_labels=2,
    problem_type="single_label_classification",
    model_dir="models",
):
    model_path = f"{model_dir}/{model_id}_model"
    tokenizer_path = f"{model_dir}/{model_id}_tokenizer"
    AutoModelForCausalLM.from_pretrained(
        model_id,
        #num_labels=num_labels,
        problem_type=problem_type,
        ignore_mismatched_sizes=True,
    ).save_pretrained(model_path)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(tokenizer_path)


save_model_local(model_id="meta-llama/Llama-2-7b-hf")
# save_model_local(
#     model_id="bert-base-german-cased",
#     model_path="models/bert-base-uncased_model",
#     tokenizer_path="models/bert-base-uncased_tokenizer",
#     #num_labels=2,
# )