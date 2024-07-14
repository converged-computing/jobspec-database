# download hf sft model to .cache to have ready for dpo
import tempfile
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer

# model = GPTNeoXForCausalLM.from_pretrained("lomahony/pythia-70m-helpful-sft")
# model = GPTNeoXForCausalLM.from_pretrained("lomahony/pythia-160m-helpful-sft")
# model = GPTNeoXForCausalLM.from_pretrained("lomahony/pythia-410m-helpful-sft")
# model = GPTNeoXForCausalLM.from_pretrained("lomahony/pythia-1b-helpful-sft")
# model = GPTNeoXForCausalLM.from_pretrained("lomahony/pythia-1.4b-helpful-sft")
# model = GPTNeoXForCausalLM.from_pretrained("lomahony/pythia-2.8b-helpful-sft")


# if convert to .pt
# https://huggingface.co/openai/whisper-large-v2/discussions/61
# with tempfile.TemporaryDirectory() as tmp_dir_name:
#     model.save_pretrained(tmp_dir_name)
#     state_dict = torch.load(f"{tmp_dir_name}/pytorch_model.bin")
#     torch.save(state_dict, f".cache/laura/pythia-70m-helpful-sft/policy.pt")

state_dict = torch.load(f".cache/laura/models--lomahony--pythia-70m-helpful-sft/snapshots/951e3e7390d510470d7b5c4d4d941a940ab5aa04/pytorch_model.bin")
torch.save(state_dict, f".cache/laura/models--lomahony--pythia-70m-helpful-sft/snapshots/951e3e7390d510470d7b5c4d4d941a940ab5aa04/policy.pt")
