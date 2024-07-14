import os
import torch
from torch.utils.data import Dataset
from transformers import ViTForImageClassification, AutoProcessor, Trainer, TrainingArguments
from PIL import Image
from sklearn.metrics import mean_squared_error
import argparse
from tqdm import tqdm


class RegressionDatasetMemory(Dataset):
    def __init__(self, data_dir, labels_file, processor, target_size=(384, 384)):
        self.data_dir = data_dir
        self.processor = processor
        self.labels = {}
        self.target_size=target_size

        with open(labels_file, 'r') as file:
            for line in file:
                img_id, label = line.strip().split('\t')
                self.labels[img_id] = float(label)

        min_inlink = min(list(self.labels.values()))
        max_inlink = max(list(self.labels.values()))

        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
        
        self.processed_image_files = []

        for img in tqdm(self.image_files, desc="preprocessing images"):
            img_id = img.split('.')[0]
            image_path = os.path.join(self.data_dir, img)
            image = Image.open(image_path)
            label = self.labels[img_id]

            image = image.resize(self.target_size, Image.Resampling.LANCZOS)

            inputs = self.processor(images=image, return_tensors="pt")
            self.processed_image_files.append({"pixel_values": inputs["pixel_values"].squeeze(), "labels": (label-min_inlink)/(max_inlink-min_inlink)})


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        return self.processed_image_files[idx]
    
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default='google/vit-base-patch32-384', type=str, required=False,
                        help="Base ViT model to fine tune.")
    parser.add_argument("--train_path", default="data/train", type=str, required=False,)
    parser.add_argument("--eval_path", default='data/val', type=str, required=False)
    parser.add_argument("--test_path", default='data/test', type=str, required=False)                    
    parser.add_argument("--labels_path", default='data/labels.tsv', type=str, required=False)                    
    parser.add_argument("--output_model_path", default='./models', type=str, required=True,
                        help="Path for trained model and checkpoints.")
    parser.add_argument("--save_every_n_steps", default=0, type=int, required=False,
                        help="Save every N steps. (recommended 10000)")
    parser.add_argument("--logging_steps", default=1, type=int, required=False,
                        help="Logging steps parameter.")
    parser.add_argument("--eval_steps", default=5, type=int, required=False,
                        help="Logging steps parameter.")
    parser.add_argument("--per_device_train_batch_size", default=32, type=int, required=False,
                        help="Per device batch size parameter.")
    parser.add_argument("--per_device_eval_batch_size", default=32, type=int, required=False,
                        help="Per device batch size parameter.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, required=False,
                        help="Gradient accumulation parameter.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, required=False,
                        help="Learning rate parameter.")
    parser.add_argument("--epochs", default=10, type=int, required=False,
                        help="Number of epochs to train")
    parser.add_argument("--warmup_steps", default=10, type=int, required=False,
                        help="Number of warmup steps")
    
    args = parser.parse_args()

    # Paths to data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch32-384", num_labels=1, ignore_mismatched_sizes=True).to(device)
    processor = AutoProcessor.from_pretrained("google/vit-base-patch32-384")

    # Create datasets
    train_dataset = RegressionDatasetMemory(args.train_path, args.labels_path, processor)
    val_dataset = RegressionDatasetMemory(args.eval_path, args.labels_path, processor)
    test_dataset = RegressionDatasetMemory(args.test_path, args.labels_path, processor)

    training_args = TrainingArguments(
        output_dir=args.output_model_path,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps, 
        report_to="wandb",
        run_name=args.output_model_path.split("/")[-1]
    )

    # Hugging Face Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=None,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate on test data
    trainer.evaluate(test_dataset, metric_key_prefix="test")

    # Save the model
    model_save_path = os.path.join(training_args.output_dir)
    trainer.save_model(model_save_path)

if __name__ == '__main__':
    main()