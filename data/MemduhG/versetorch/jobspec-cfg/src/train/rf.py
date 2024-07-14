import torch
import sys
import os
import time
from reformer_pytorch import ReformerEncDec

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.model.adafactor import Adafactor
from src.utils.utils import get_tokenizer
from src.model.model import NoamOpt
from src.data_utils.data import get_training_iterators
from src.utils.save import save_checkpoint, load_latest

save_every = 1800
save_path = "checkpoints/tur-rf"
MAX_SEQ_LEN = 1024

device = 0 if torch.cuda.device_count() > 0 else "cpu"

enc_dec = ReformerEncDec(dim=512, enc_num_tokens=32000, enc_depth=6, enc_max_seq_len=MAX_SEQ_LEN, dec_num_tokens=32000,
                         dec_depth=6, dec_max_seq_len=MAX_SEQ_LEN, ignore_index=3, pad_value=3).to(device)

opt = Adafactor(enc_dec.parameters())

tokenizer = get_tokenizer("tr")
train_iter, valid_iter, test_iter, train_idx, dev_idx, test_idx = get_training_iterators("tur")

steps = load_latest(save_path, enc_dec, opt)

last_saved = time.time()



for batch in train_iter:
    src, tgt = torch.transpose(batch.src, 0, 1).to(device), torch.transpose(batch.trg, 0, 1).to(device)
    print(src.shape, tgt.shape)
    input_mask = src != 3
    try:
        loss = enc_dec(src, tgt, return_loss=True, enc_input_mask=input_mask)
    except AssertionError:
        print("Skipped overlong sample while training", src.shape, tgt.shape)
        continue
    print(loss)
    loss.backward()
    opt.step()
    opt.zero_grad()
    steps += 1
    if time.time() - last_saved > save_every:
        print("Saving checkpoint at", steps, "steps with loss of", float(loss))
        save_checkpoint(enc_dec, opt, steps, save_path)
        last_saved = time.time()
