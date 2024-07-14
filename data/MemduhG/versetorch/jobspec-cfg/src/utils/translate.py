import argparse
import torch
import sys
import os
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data_utils.batch import rebatch_single
from src.data_utils.data import make_val_iterator
from src.model.model import make_model, greedy_decode
from src.utils.utils import get_tokenizer
from src.utils.tb import get_files, write_prose_evals, write_evals

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def translate_devset(args):
    save_to = args.output
    model = make_model(32000, 32000, N=6).to(device)
    try:
        checkpoint = torch.load(args.checkpoint)
    except RuntimeError:
        checkpoint = torch.load(args.checkpoint, map_location = torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    tokenizer = get_tokenizer(args.language)
    val_iter, val_indices = make_val_iterator(args.input, tokenizer, batch_size=128)
    pad_idx = 3
    val_iter = (rebatch_single(pad_idx, b) for b in val_iter)
    decoded = [""] * len(val_indices)
    for batch in val_iter:
        out = greedy_decode(model, batch.src, batch.src_mask, max_len=args.max_len, start_symbol=1)
        for c, decoded_row in enumerate(out):
            padded_src = batch.src[c, :].tolist()
            src_seq = []
            for item in padded_src:
                if item == 3:
                    break
                src_seq.append(item)
            index = val_indices[tuple(src_seq)]
            to_spm = []
            for item in decoded_row:
                if item == 2:
                    break
                to_spm.append(item)
            decoded_string = tokenizer.Decode(decoded_row.tolist())
            decoded[index] = decoded_string
            print(decoded_string.encode('utf8') )
        print("Decoded batch of", batch.src.shape)


    # TODO cutoff at line end and actually decode
    with open(save_to, "w", encoding="utf-8") as outfile:
        for line in decoded:
            outfile.writelines(line + "\n")

    checkpoint_time = os.stat(args.checkpoint).st_mtime
    os.utime(save_to, (checkpoint_time, checkpoint_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max_len", type=int, required=True)
    args = parser.parse_args()
    translate_devset(args)
    _, experiment, checkpoint_name = args.checkpoint.split("/")
    translation = checkpoint_name.partition(".pt")[0]
    writer = SummaryWriter()
    references, sources, prose_sources = get_files()
    ref = references[experiment]
    src = sources[experiment]
    prose_src = prose_sources[experiment]
    if "prose" not in args.output:
        file_path = "translations/{}/{}".format(experiment, translation)
        write_evals(writer, experiment, translation, file_path, ref, src)
    else:
        file_path = "prose_translations/{}/{}".format(experiment, translation)
        write_prose_evals(writer, experiment, translation, file_path, prose_src)

    writer.flush()
    writer.close()
