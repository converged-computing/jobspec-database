import torch
import os
from dataset_loaders.load_dataset import get_loader
from models.models import get_model

from dataset_loaders.datasset_config import DatasetConfig

from utils.data_utils import get_dataset_constant
from utils.model_utils import get_model_constant, load_model, freeze_layers_except_last
from utils.env_utils import get_env_constant

import argparse
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from torchvision import transforms
from PIL import ImageDraw, ImageFont

def convert_tensor_to_PIL_image(tensor):
    transform = transforms.ToPILImage()
    return transform(tensor)

def save_image(predicted_class, image_matrix, image_path):
    inv_normalize = transforms.Normalize(
        mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5],
        std=[1/0.5, 1/0.5, 1/0.5]
    )

    im = convert_tensor_to_PIL_image(inv_normalize(image_matrix))

    mf = ImageFont.truetype('arial.ttf', 30)
    ImageDraw.Draw(im).text((0,0), predicted_class, (255,0,0), font=mf)

    im.save(image_path)

def generate_images(model, val_loader:DataLoader, config: DatasetConfig, images_path):
    model.eval()

    class_already_generated = {}
    id_to_label = config.get_id_to_label()
    total_generated = 0
    total_classes = len([id for id in id_to_label.keys()])

    with torch.no_grad():
        for images, labels in val_loader:
            if labels[0].item() in class_already_generated and labels[-1].item() in class_already_generated:
                continue
            
            if total_generated >= total_classes:
                break

            images, labels = images.to('cuda'), labels.to('cuda')

            with autocast(dtype=torch.float16):
                outputs = model(images)
                predicted = torch.argmax(torch.nn.functional.softmax(outputs, dim=1), dim=1)
            
            for i, idx in enumerate(labels):
                idx = idx.item()
                if idx in class_already_generated:
                    continue
                
                image_file_name = id_to_label[idx] + '.png'
                class_already_generated[idx] = True
                total_generated += 1

                save_image(id_to_label[predicted[i].item()], images[i], os.path.join(images_path, image_file_name))
                total_generated += 1


    print(f"All images generated. Check directory:", images_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet", "imagenet_real", "flowers102", "oxford_iiit"], default="imagenet_real",
                        help="Which model to evaluate on.", required=True)
    
    parser.add_argument("--model_type", choices=["vit_l_16", "vit_h_14", "eff_net_l2", "bit_l_res"], default="vit_l_16", help="Which variant to use.", required=True)
    
    parser.add_argument("--env", choices=["local", "newton"], default="local", help="Which variant to use.", required=True)

    parser.add_argument("--pretrained_dir", type=str, default=None, help="Where to search for pretrained ViT models.")
    
    parser.add_argument("--image_folder", default="models/images", type=str, help="The output directory where images will be written.", required=True)

    
    args = parser.parse_args()

    dataset_name = get_dataset_constant(args.dataset)
    model_name = get_model_constant(args.model_type)
    env_name = get_env_constant(args.env)
    
    dataset_config = DatasetConfig(dataset_name, env_name)
    model = get_model(model_name, dataset_name)

    if args.pretrained_dir:
        model = load_model(model, args.pretrained_dir)

    model = freeze_layers_except_last(model)

    val_loader = get_loader(dataset_config, False)

    generate_images(model, val_loader, dataset_config, args.image_folder)

if __name__ == '__main__':
    main()