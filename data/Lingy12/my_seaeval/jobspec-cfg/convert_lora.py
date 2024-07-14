import fire
from transformers import AutoModel, AutoModelForCausalLM
from peft.peft_model import PeftModel
import torch
import glob
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""
def convert_checkpoint(checkpoint_folder, base_model, destination):
    adapter = os.path.join(checkpoint_folder, 'pt_lora_model')

    # print('Loading parameter file = {}'.format(adapter))
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map='auto')

    # model = PeftModel.from_pretrained(adapter, "outputs")
    # model.load_adapter(adapter)
    # model.resize_token_embeddings(32001) # since <pad> is used
    # model.load_adapter(adapter)
    
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    # model.save_pretrained(destination, max_shard_size='20GB')
    # model = model.merge_and_upload()
    model.save_pretrained(destination)

if __name__ == "__main__":
    fire.Fire(convert_checkpoint)
