import torch
import torchvision.utils as vutils
from tqdm import tqdm
import os
from attack_util import get_mnist_instance, attack1, attack2, attack3, attack4, get_model_and_processor, get_target, get_mnist_torchvision, llava_id
import argparse

def generate_adversarial_dataset(model, processor, mnist, id, data_range, alpha):
    if torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()} GPUs detected")
        model = torch.nn.DataParallel(model)
    for i in tqdm(data_range):
        inputs, label_id = get_mnist_instance(mnist[i], processor)
        if id == 0:
            target_id = get_target(model, processor, inputs, label_id)
            img, _, _ = attack1(model, inputs, target_id)
        elif id==1:
            img, _, _ = attack2(model, inputs, label_id)
        elif id==2:
            target_id = get_target(model, processor, inputs, label_id)
            img, _, _ = attack3(model, inputs, target_id, alpha=alpha)
        elif id==3:
            img, _, _ = attack4(model, inputs, label_id, alpha=alpha)

        if img is not None:
            torch.save(img, f'{data_dir}/{i}-{processor.decode(label_id)}.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--n', type=int, default=3000)
    parser.add_argument('--alpha', type=float, default=100)
    args = parser.parse_args()

    data_dir = f'rad_data/tensors-{args.id}'
    os.makedirs(f'{data_dir}', exist_ok = True)

    model, processor = get_model_and_processor(llava_id)
    data_range = range(args.n)
    mnist = get_mnist_torchvision()
    generate_adversarial_dataset(model, processor, mnist, args.id, data_range, args.alpha)