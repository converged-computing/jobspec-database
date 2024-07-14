from ClipFineTuner import CLIPFineTuner
from transformers import CLIPProcessor, CLIPTokenizer, CLIPImageProcessor
from torch.utils.data import DataLoader
import wandb
from load_data import load_google_data
from PIL import Image
from datasets import load_dataset, concatenate_datasets
import sys
import copy
import torch
import random

climate_zone_descriptions = {
    "Af": "Tropical rainforest climate",
    "Am": "Tropical monsoon climate",
    "Aw": "Tropical savanna climate (wet-dry)",
    "BWh": "Hot desert climate",
    "BWk": "Cold desert climate",
    "BSh": "Hot semi-arid climate",
    "BSk": "Cold semi-arid climate",
    "Csa": "Hot-summer Mediterranean climate",
    "Csb": "Warm-summer Mediterranean climate",
    "Csc": "Cold-summer Mediterranean climate",
    "Cwa": "Monsoon-influenced humid subtropical climate",
    "Cwb": "Subtropical highland climate with dry winters",
    "Cwc": "Cold subtropical highland climate with dry winters",
    "Cfa": "Humid subtropical climate",
    "Cfb": "Oceanic climate",
    "Cfc": "Subpolar oceanic climate",
    "Dsa": "Hot-summer Mediterranean continental climate",
    "Dsb": "Warm-summer Mediterranean continental climate",
    "Dsc": "Cold-summer Mediterranean continental climate",
    "Dsd": "Cold-summer Mediterranean continental climate with extremely cold winters",
    "Dwa": "Monsoon-influenced hot-summer humid continental climate",
    "Dwb": "Monsoon-influenced warm-summer humid continental climate",
    "Dwc": "Monsoon-influenced subarctic climate",
    "Dwd": "Monsoon-influenced subarctic climate with extremely cold winters",
    "Dfa": "Hot-summer humid continental climate",
    "Dfb": "Warm-summer humid continental climate",
    "Dfc": "Subarctic climate with cool summers",
    "Dfd": "Subarctic climate with extremely cold winters",
    "ET": "Tundra climate",
    "EF": "Ice cap climate"
}


class ImbagClip():
    def __init__(self, batch_size=8, learning_rate=1e-3, epochs=3, mode="country_first"):
        """
        Take dataset configuration
        Load dataset in that configuration
        Initialize dataloader
        Initialize a CLIPFineTuner
        Train CLIPFineTuner on the dataset

        Args:
        mode (str): "country_first" for Country -> Climate -> Geocell, "climate_first" for Climate -> Country -> Geocell, "separate" for mixed in separate captions, "combined" for a single combined caption
        """
        self.mode = mode
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        print(f"Hyperparameters: {self.batch_size}, {self.learning_rate}, {self.epochs}")
        self.model = CLIPFineTuner(batch_size=self.batch_size,learning_rate=self.learning_rate,epochs=self.epochs) # TODO: add parameters
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14-336")
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

        wandb.init() # TODO: add project name


    def process_data(self, examples, argument):
        """
        Process the data for CLIP model

        Args:
        batch: batch of data

        Returns:
        batch: processed batch
        """

        if argument == "combined":
            captions = []
            images = []
            for country, climate, geocell, file_path in zip(examples["Country"], examples["Climate Zone"], examples["Geocell"], examples["File Path"]):
                # Load image using PIL
                img = Image.open(file_path)
                images.append(img)

                caption = "this image was taken in " + country + " within geocell " + str(geocell) + " in " + climate_zone_descriptions.get(climate).lower()
                captions.append(caption)

            return self.processor(text=captions, images=images, return_tensors="pt", padding="max_length", max_length=32, truncation=True)
        
        if argument == "combined_mixed":
            captions = []
            images = []
            for country, climate, geocell, file_path in zip(examples["Country"], examples["Climate Zone"], examples["Geocell"], examples["File Path"]):
                # Load image using PIL
                img = Image.open(file_path)
                images.append(img)

                beginning = "this image was taken"
                country = f" in {country}"
                geocell = f" within geocell {str(geocell)}"
                climate = f" in {climate_zone_descriptions.get(climate).lower()}"
                random_order = [country, geocell, climate]

                random.shuffle(random_order)

                caption = [beginning] + random_order

                caption = "".join(caption)
                captions.append(caption)

            return self.processor(text=captions, images=images, return_tensors="pt", padding="max_length", max_length=32, truncation=True)

        else:
            country_captions = []
            climate_captions = []
            geocell_captions = []
            images = []
            for country, climate, geocell, file_path in zip(examples["Country"], examples["Climate Zone"], examples["Geocell"], examples["File Path"]):
                # Load image using PIL
                img = Image.open(file_path)
                images.append(img)

                climate_caption = "this image was taken in " + climate_zone_descriptions.get(climate).lower()

                country_caption = "this image was taken in " + country
                geocell_caption = "this image was taken within geocell " + str(geocell)

                country_captions.append(country_caption)
                climate_captions.append(climate_caption)
                geocell_captions.append(geocell_caption)

            country_tok = self.tokenizer(country_captions, return_tensors="pt", padding="max_length", max_length=20, truncation=True)
            climate_tok = self.tokenizer(climate_captions, return_tensors="pt", padding="max_length", max_length=20, truncation=True)
            geocell_tok = self.tokenizer(geocell_captions, return_tensors="pt", padding="max_length", max_length=20, truncation=True)
            images_proc = self.image_processor(images, return_tensors="pt")

            images_captions = {
                "Country Caption": country_tok["input_ids"],
                "Country Attention Mask": country_tok["attention_mask"],
                "Climate Zone Caption": climate_tok["input_ids"],
                "Climate Zone Attention Mask": climate_tok["attention_mask"],
                "Geocell Caption": geocell_tok["input_ids"],
                "Geocell Attention Mask": geocell_tok["attention_mask"],
                "pixel_values": images_proc["pixel_values"]
            }

            return images_captions   

        
    def preprocess(self):
        """
        Using CLIP Processor to preprocess the dataset

        Returns:
        dataset: dataset object
        """
        dataset = load_google_data("panos_dataset.hf")
        self.train_dataset, self.validation_dataset = dataset['train'], dataset['validation']
        # self.train_dataset = self.train_dataset.select([random.randint(0, len(self.train_dataset)) for _ in range(10)])
        # self.validation_dataset = self.validation_dataset.select([random.randint(0, len(self.train_dataset)) for _ in range(10)])

        workers = torch.cuda.device_count()

        if self.mode == "combined":
            self.train_dataset = self.train_dataset.map(lambda x: self.process_data(x, "combined"), batched=True, batch_size=500, remove_columns=["File Path", "Climate Zone", "Country", "Geocell"])

        elif self.mode == "combined_mixed":
            self.train_dataset = self.train_dataset.map(lambda x: self.process_data(x, "combined_mixed"), batched=True, batch_size=500, remove_columns=["File Path", "Climate Zone", "Country", "Geocell"])

        else:
            self.train_dataset = self.train_dataset.map(lambda x: self.process_data(x, "Country"), batched=True, batch_size=500, remove_columns=["File Path", "Climate Zone", "Country", "Geocell"])

            country_dataset_train = self.train_dataset.rename_column("Country Caption", "input_ids").rename_column("Country Attention Mask", "attention_mask").remove_columns(["Climate Zone Caption", "Climate Zone Attention Mask", "Geocell Caption", "Geocell Attention Mask"])
            climate_dataset_train = self.train_dataset.rename_column("Climate Zone Caption", "input_ids").rename_column("Climate Zone Attention Mask", "attention_mask").remove_columns(["Country Caption", "Country Attention Mask", "Geocell Caption", "Geocell Attention Mask"])
            
            if self.mode == "country_first":
                self.train_dataset = concatenate_datasets([country_dataset_train, climate_dataset_train])

            elif self.mode == "climate_first":
                self.train_dataset = concatenate_datasets([climate_dataset_train, country_dataset_train])

            elif self.mode == "separate":
                self.train_dataset = concatenate_datasets([country_dataset_train, climate_dataset_train])

            elif self.mode == "geocell_only":
                geocell_dataset_train = self.train_dataset.rename_column("Geocell Caption", "input_ids").rename_column("Geocell Attention Mask", "attention_mask").remove_columns(["Country Caption", "Country Attention Mask", "Climate Zone Caption", "Climate Zone Attention Mask"])
                self.train_dataset = concatenate_datasets([geocell_dataset_train])


    def prepare_dataloader(self):
        """
        Prepare dataloader for training
        """
        if self.mode == "separate":
            shuffle = True
        else:
            shuffle = False

        batch_size = self.batch_size

        self.train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'pixel_values'])
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


    def train(self):
        """
        Train the model on the dataset

        Returns:
        model: trained model
        """
        self.model.train(self.train_loader, self.validation_dataset)


if __name__ == "__main__":
    argument = sys.argv[1]
    epochs = int(sys.argv[2])
    print(f"Running ImbagClip with {argument}...")
    imbag_clip = ImbagClip(epochs=epochs, mode=argument)
    imbag_clip.preprocess()
    imbag_clip.prepare_dataloader()
    imbag_clip.train()