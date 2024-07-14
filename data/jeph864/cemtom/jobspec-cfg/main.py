from cemtom.dataset import fetch_dataset
from cemtom.preprocessing import Preprocessor as Pipe
from cemtom.embedder import BertVocabEmbedder, get_word_embedding_model
from cemtom.models import Sia
from cemtom._base import Trainer, get_data, get_torch_data
from cemtom.evaluation.metrics import TopicEvaluation
from cemtom.models.vae_base import *
import argparse

from pytorch_lightning import Trainer as PyTrainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

import mlflow
def main(args):
    pretrained_dir = "./pretrained_models"
    bert_vocab_embeddings_path = f'{pretrained_dir}/bert/20NG-bert-layer12-firstword.txt'
    fasttext_vocab_embeddings_path = f'{pretrained_dir}/fasttext/wiki.en.bin'
    hf_embeddings_path = f'{pretrained_dir}/hf/20NG-mlpnet.safetensors'

    pipe, data = get_data(args.dataset)
    trainer = Trainer(
        dataset=data,
        preprocessor=pipe,
        nr_dimensions=args.use_dims,
        nr_topics=args.num_topics,
        vocab_embeddings=args.vocab_embeddings,
        vocab_embeddings_path=args.vocab_embeddings_path,
        sia_rerank=args.rerank,
        sia_weighting=args.doc_info
    )
    texts = [doc.split() for doc in pipe.data.get_corpus()]
    model = trainer.train_bertopic()
    if args.wordvec_path is None:
        args.wordvec_path = f'{pretrained_dir}/word2vec/GoogleNews-vectors-negative300.bin'
    eval = TopicEvaluation(
        model=trainer.training_output,
        texts=texts,
        dataset=data,
        word2vec_path=args.wordvec_path
    )
    coherence = eval.coherence_score(remove=('sia_npmi'))
    print(coherence)

    if args.mlflow:
        mlflow.set_tracking_uri(uri="127.0.0.1:8080")
        mlflow.create_experiment(args.model)
        with mlflow.start_run():
            mlflow.log_params({})
        for metric, value in coherence.items():
            mlflow.log_metric(metric, value)
        #model_info = mlflow.

def main_vae(settings=None):
    torch.set_float32_matmul_precision('medium')
    seed_everything(42, workers=True)

    if settings is None:
        settings = {
            'dataset': '20NewsGroup',
            'batch_size': 128,
            'max_epochs': 1,
            'topics': 20,
            'model_dir': './models/',
            'devices': -1,
            'accelerator': 'auto'

        }
    loader, text, vocab = get_torch_data(settings['dataset'], batch_size=settings['batch_size'])
    logger = TensorBoardLogger(settings['model_dir'], name='prodlda', version=settings['dataset'])
    model = ProdLDA(in_features=len(vocab), num_topics=settings['topics'])
    checkpoint = ModelCheckpoint(monitor='val/loss', mode='min', save_top_k=1, filename='{epoch}-{val/loss:.2f}')
    trainer = PyTrainer(
        max_epochs=settings['max_epochs'],
        callbacks=[checkpoint],
        accelerator=settings['accelerator'],
        default_root_dir=settings['model_dir'],
        devices=settings['devices'],
        strategy=DDPStrategy(find_unused_parameters=True),
        log_every_n_steps=1,
        logger=logger
    )
    trainer.fit(model=model, train_dataloaders=loader['train'], val_dataloaders=loader['val'])
    # get topics
    best_model = checkpoint.best_model_path
    topics = model.get_topic_words(vocab=vocab, path=best_model)
    print(topics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--vocab_embeddings", type=str, default="bert", choices=["bert", "fasttext", "glove"])
    parser.add_argument("--vocab_embeddings_path", type=str)
    parser.add_argument("--model", type=str, default="bert", help="the topic model")

    parser.add_argument("--clustering_algo", type=str,
                        choices=["KMeans", "SPKMeans", "GMM", "KMedoids", "Agglo", "DBSCAN", "Spectral", "VMFM"])

    parser.add_argument("--topics_file", type=str, default="topics.json", help="topics file")

    parser.add_argument('--use_dims', type=int)
    parser.add_argument('--num_topics', type=int, default=20)
    parser.add_argument("--doc_info", type=str, default="wgt", choices=["SVD", "DUP", "wgt", "robust", "logtfdf"])
    parser.add_argument("--rerank", type=str, default="tf", choices=["tf", "tfidf", "tfdf", "graph"])

    parser.add_argument("--dataset", type=str, default="20NewsGroup", choices=["20Newsgroup", "children", "reuters"])

    parser.add_argument("--stopwords", type=str, help="Path to stopwords")

    parser.add_argument("--vocab", required=False, type=str, nargs='+', default=[])
    parser.add_argument("--scale", type=str, required=False)

    args_ = parser.parse_args()
    return args_


if __name__ == '__main__':
    # args = parse_args()
    # main(args)
    main_vae()
