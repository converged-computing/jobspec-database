import numpy as np
from tqdm import tqdm

from src.experiments.main import *

# Steal from main.py
## Simply use XGBDataIterator and save into a numpy array. (np array is pre-allocated using np.empty)
    
import pickle
def save_pickle(f_name: str, obj: object):
    with open(f_name, 'wb') as f_obj:
        pickle.dump(obj, f_obj, protocol=pickle.HIGHEST_PROTOCOL)


class DSEmbedder():
    def __init__(self, dataset_name: str, ds_dir: str, batch_size=32):
        self.dhandler = dhandler = DataHandler(ds_name=dataset_name, ds_dir=ds_dir)
        self.dataset_name = dataset_name
        self.ds_dir = ds_dir
        self.batch_size = batch_size
        self.data = []
        self.label = []
        self.train_data = None
        self.train_label = None
        self.val_data = None
        self.val_label = None
        self.xgb_fusion = None
        self.setup()

    def setup(self):
        # Setup datasets
        train_text_ds: DataHandler.TextModalityDS = self.dhandler.load_text_ds("train_text.json")
        val_text_ds: DataHandler.TextModalityDS = self.dhandler.load_text_ds("val_text.json")
        val_img_ds: DataHandler.ImageModalityDS = self.dhandler.load_img_ds("val")
        train_img_ds: DataHandler.ImageModalityDS = self.dhandler.load_img_ds("train")
        self.train_set: DataHandler.BimodalDS = self.dhandler.BimodalDS(
            image_ds=train_img_ds, text_ds=train_text_ds
        )
        self.val_set: DataHandler.BimodalDS = self.dhandler.BimodalDS(
            image_ds=val_img_ds, text_ds=val_text_ds
        )

        self.N_train = len(self.train_set)
        self.N_val = len(self.val_set)

        # Setup models
        models: Tuple[text_clf_t, img_clf_t] = experiment1(
            bimodal_val_ds=self.val_set,
            train_text_ds=train_text_ds,
            load_bert=True,
            dataset_name=self.dataset_name
        )
        text_clf: text_clf_t = models[0]
        img_clf: img_clf_t = models[1]
        self.xgb_fusion = MultimodalModel.XGBFusion(
            img_clf, text_clf, n_jobs=1, cache_dir=self.ds_dir, enable_subsample=False
        )
        self.M = sum([clf.M for clf in self.xgb_fusion.classifiers])

    def _input_data(self, data, label, training_set=True):
        ix = self.ix
        start_ix = ix*self.batch_size
        end_ix = start_ix + self.batch_size   # It is okay that this is too large. Numpy corrects the index.
        assert end_ix > start_ix
        if training_set:
            self.train_data[start_ix:end_ix,:] = data
            self.train_label[start_ix:end_ix,0] = label
        else:
            self.val_data[start_ix:end_ix,:] = data
            self.val_label[start_ix:end_ix,0] = label

    def embed(self):
        # Preallocate memory
        self.train_data = np.empty((self.N_train, self.M), dtype=np.float16)
        self.val_data = np.empty((self.N_val, self.M), dtype=np.float16)
        self.train_label = np.empty((self.N_train, 1), dtype=np.uint16)
        self.val_label = np.empty((self.N_val, 1), dtype=np.uint16)

        # Iterate through dataset while adding the embeddings here
        def it_gen(training_set, it):
            self.ix = 0
            status = 1
            while status:
                status = it.next(lambda data, label: self._input_data(data, label, training_set=training_set))  # Status=0 when no more data
                yield status
                self.ix += 1

        it_train = MultimodalModel.XGBDataIterator(self.xgb_fusion, bimodal_dataloader(self.train_set, shuffle=True, batch_size=self.batch_size)).reset()
        it_val = MultimodalModel.XGBDataIterator(self.xgb_fusion, bimodal_dataloader(self.val_set, shuffle=True, batch_size=self.batch_size)).reset()
        for _ in tqdm(it_gen(True, it_train), total=it_train._len):  pass   # Embed data
        for _ in tqdm(it_gen(False, it_val), total=it_val._len):     pass   # Embed data
        return self

    def save_embeddings(self, f_name_prefix: str=None):
        if f_name_prefix is None:
            f_name_prefix = self.dataset_name
        save_pickle(os.path.join(self.ds_dir, f_name_prefix, 'train.pkl'), {'X': self.train_data, 'y': self.train_label})
        save_pickle(os.path.join(self.ds_dir, f_name_prefix, 'val.pkl'), {'X': self.val_data, 'y': self.val_label})
        return self


@click.command()
@click.option(
    "--dataset",
    default="CMPlaces",
    help="Dataset to use. Either 'CMPlaces' or 'ImageNet'"
)
@click.option(
    "--ds_dir",
    default="/work3/s184399",
    help="The directory in which the datasets are located"
)
def main(*args, **kwargs):
    dataset_name = kwargs["dataset"]
    ds_dir = kwargs['ds_dir']
    _ = DSEmbedder(dataset_name=dataset_name, ds_dir=ds_dir).embed().save_embeddings()


if __name__ == '__main__':
    main()