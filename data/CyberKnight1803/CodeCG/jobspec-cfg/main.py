import argparse

from trainer import train_nl2nl

from config import (
    AVAIL_GPUS,
    BATCH_SIZE,
    DROPOUT_RATE,
    LEARNING_RATE, 
    MAX_EPOCHS,
    MAX_SEQUENCE_LENGTH,
    NL_DECODER_BASE_MODEL,
    NL_ENCODER_BASE_MODEL,
    NUM_WORKERS,
    PADDING,
    PATH_BASE_MODELS,
    PATH_CACHE_DATASETS,
    PATH_LOGS,
    PATH_SAVE_NL_DECODER,
    PATH_SAVE_NL_ENCODER,
    PATH_SAVE_NL_LM,
    PL,
    TOKENIZER_MODEL,
    WEIGHT_DECAY
)


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="Set epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Set batch size")

    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Set learning rate")
    parser.add_argument("--g_lr", type=float, default=LEARNING_RATE, help="Set generator learning rate")
    parser.add_argument("--d_lr", type=float, default=LEARNING_RATE, help="Set discriminator learning rate")

    parser.add_argument("--wd", type=float, default=WEIGHT_DECAY, help="Set weight decay")
    parser.add_argument("--dropout", type=float, default=DROPOUT_RATE, help="Set dropout rate")

    parser.add_argument("--tokenizer", type=str, default=TOKENIZER_MODEL, help="Set Tokenizer")
    parser.add_argument("--nl_en_model", type=str, default=NL_ENCODER_BASE_MODEL, help="Set NL encoder base model")
    parser.add_argument("--nl_de_model", type=str, default=NL_DECODER_BASE_MODEL, help="Set NL decoder base model")
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQUENCE_LENGTH, help="Set max length of a sequence")
    parser.add_argument("--padding", type=str, default=PADDING, help="Set padding type")
    
    parser.add_argument("--pl", type=str, default=PL, help="Set programming language")
    parser.add_argument("--logger", type=str, default="tensorboard", help="Set logger")

    parser.add_argument("--gpus", type=int, default=AVAIL_GPUS, help="Set no. of GPUs")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Set no. of CPU Threads")

    parser.add_argument("--path_logs", type=str, default=PATH_LOGS, help="Set path to log runs")
    parser.add_argument("--path_base_models", type=str, default=PATH_BASE_MODELS, help="Set path to save base pretrained models")
    parser.add_argument("--path_cache_datasets", type=str, default=PATH_CACHE_DATASETS, help="Set path to cache datasets")
    parser.add_argument("--path_save_nl_encoder", type=str, default=PATH_SAVE_NL_ENCODER, help="Set path to save trained NL encoder")
    parser.add_argument("--path_save_nl_decoder", type=str, default=PATH_SAVE_NL_DECODER, help="Set path to save NL decoder")
    parser.add_argument("--path_save_nl_lm", type=str, default=PATH_SAVE_NL_LM, help="Set path to save lm head")

    parser.add_argument("--run_name", type=str, required=True, help="Set exp run name")
    parser.add_argument("--jobid", type=str, help="Set Job ID when running on HPC")

    args = parser.parse_args()

    # test_dm(args)
    train_nl2nl(args)

