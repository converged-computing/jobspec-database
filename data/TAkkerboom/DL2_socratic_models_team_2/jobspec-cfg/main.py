from src.dataset import Raven
from src.model import VLM, LM, OpenCV, CLIP
from src.const import SHAPES, ANGLES, SIZES, COLORS
from transformers import AutoModelForSeq2SeqLM
from os.path import exists
from tqdm import tqdm
import numpy as np
import argparse
import torch
import gdown
import zipfile
from sklearn.metrics import classification_report
import json


URL = "https://drive.google.com/file/d/111swnEzAY2NfZgeyAhVwQujMjRUfeyuY/view?usp=sharing"


class SM:
    def __init__(self):
        self.PATH = "C:/Users/maart/Downloads/RAVEN-10000-release/RAVEN-10000"
        fig_types = ['center_single']

        self.test_set = Raven(self.PATH, 'test', fig_types[0])
        self.test_set.load_data()

        self.VLM = VLM()
        self.LM = LM("google/flan-t5-large", 'cpu', AutoModelForSeq2SeqLM)
        self.prompt = '''You are given a logic puzzle from the RAVEN dataset. The first shape on the first row is {}, the second shape on the first row is {}, the third item on the first row is {}. The first shape on the second row is {}, the second shape on the second row is a {}, the third shape on the second row is {}. The first shape on the third row is {}, the second shape is {}. Based on this, what is the third shape on the third row? You can only choose between: {}, {}, {}, {}, {}, {}, {}, {}'''


    def VLM_pred_attributes(self):
        answers = []

        for i in range(self.test_set.len()):
            puzzle = self.test_set.get_puzzle(i)
            puzzle_answers = []
            for j, c_image in enumerate(puzzle):
                angle = self.VLM.forward(c_image, 'At what angle is the figure located? The options are: -135, 90, -45, 0, 45, 90, 135, 180')
                color = self.VLM.forward(c_image, 'What is the color value of the figure?')
                size = self.VLM.forward(c_image, 'What is the size of the figure? The options are: 0.4, 0.5, 0.6, 0.7, 0.8, 0.9')
                type = self.VLM.forward(c_image, 'What is the shape of the figure? The options are: triangle, square, pentagon, hexagon, circle')
                puzzle_answers.append([angle, color, size, type])

            answers.append(puzzle_answers)

        return answers

    def OpenCV_pred_attributes(self):
        answers = []
        opencvmodel = OpenCV()
        for i in range(self.test_set.len()):
            puzzle = self.test_set.get_puzzle(i)
            puzzle_answers = []
            for j, c_image in enumerate(puzzle):
                puzzle_answers.append(opencvmodel.detect_shape(c_image))
            answers.append(puzzle_answers)
        return answers

    def generate_prompts(self, attributes):
        return [self.prompt.format(*[generate_single_description(*shape) for shape in puzzle]) for puzzle in attributes]

    def get_descriptions(self, attributes):
        descriptions = []
        for puzzle in attributes:
            shape_desc = []
            for shape in puzzle[-8:]:
                for angle, color, size, type in shape:
                    shape_desc.append(generate_single_description(angle, color, size, type))
            descriptions.append(shape_desc)
        return descriptions

    def inference(self, prompts, descriptions, batch_size, samples):
        predictions = []
        j = 0
        prompts = [prompts[x:x+batch_size] for x in range(0, len(prompts), batch_size)]
        descriptions = [prompts[x:x+batch_size] for x in range(0, len(descriptions), batch_size)]
        N  = int(samples / batch_size)
        for description_batch, prompt_batch in zip(descriptions[:N], prompts[:N]):
            out = self.LM.forward(prompt_batch)
            for output, description in zip(out, description_batch):
                if output in description:
                    pred = description.index(output)
                else:
                    pred = 8
                predictions.append(pred)
            j += 1
            if j % 1 == 0:
                print('currently at ', j * 10)

        return predictions

    def forward(self):
        attributes = self.VLM_pred_attributes()
        # attributes = self.OpenCV_pred_attributes()
        prompts = self.generate_prompts(attributes)
        descriptions = self.get_descriptions(attributes)
        self.inference(prompts, descriptions, 10, 1000)


class Demo:
    def __init__(self, device='cpu', prompt=None):
        self.device = device
        self.prompt = '''You are given a logic puzzle from the RAVEN dataset. The first shape on the first row is {}, the second shape on the first row is {}, the third item on the first row is {}. The first shape on the second row is {}, the second shape on the second row is a {}, the third shape on the second row is {}. The first shape on the third row is {}, the second shape is {}. Based on this, what is the third shape on the third row? You can only choose between: {}, {}, {}, {}, {}, {}, {}, {}'''

    def load_VLM(self, model):
        print(f'loading {model}...')

        if model == 'openai/clip-vit-base-patch32':
            self.VLM = CLIP(model, self.device)

            # Templates for predictions
            self.template_type = ['The figure is shaped as a {}'.format(value) for value in SHAPES.values()]
            self.template_color = ['The color of the figure is {}'.format(value) for value in COLORS.values()]
            self.template_angle = ['The figure is rotated by {} degrees'.format(value) for value in ANGLES.values()]
            self.template_sizes = ['The percentage of the image covered by the figure is {}'.format(value) for value in SIZES.values()]

        else:
            self.VLM = VLM(model, self.device)

            # Templates for predictions
            self.template_type = 'What is the shape of the figure?'
            self.template_color = 'What is the color of the figure?'
            self.template_angle = 'At what angle is the figure located?'
            self.template_sizes = 'What is the size of the figure?'

        print('VLM is ready to use')

    def load_LM(self, model, model_class=AutoModelForSeq2SeqLM):
        print(f'loading {model}...')
        self.LM = LM(model, model_class, self.device)

        print('LM is ready to use')

    def get_attributes(self, puzzle):
        answers = []

        for c_image in puzzle:
            if isinstance(self.template_angle, list):
                angle = (self.VLM.forward(c_image, self.template_angle)).split()[-2]
                color = (self.VLM.forward(c_image, self.template_color)).split()[-1]
                size = (self.VLM.forward(c_image, self.template_sizes)).split()[-1]
                type = (self.VLM.forward(c_image, self.template_type)).split()[-1]
            else:
                angle = self.VLM.forward(c_image, self.template_angle)
                color = self.VLM.forward(c_image, self.template_color)
                size = self.VLM.forward(c_image, self.template_sizes)
                type = self.VLM.forward(c_image, self.template_type)

            answers.append([angle, color, size, type])

        return answers

    def generate_prompts(self, attributes):
        desc = [generate_single_description(*attr) for attr in attributes]
        prompt = self.prompt.format(*desc)

        return desc, prompt

    def OpenCV_pred_attributes(self, puzzle):
        answers = []
        opencvmodel = OpenCV()

        for c_image in puzzle:
            answers.append(opencvmodel.detect_shape(c_image))

        return answers

    def get_descriptions(self, attributes):
        descriptions = []
        for puzzle in attributes:
            shape_desc = []
            for shape in puzzle[-8:]:
                for angle, color, size, type in shape:
                    shape_desc.append(generate_single_description(angle, color, size, type))
            descriptions.append(shape_desc)
        return descriptions

    def solve(self, prompt):
        pred = self.LM.forward(prompt)

        return pred

    def forward(self, puzzle, ClassicOpenCV):
        if ClassicOpenCV:
            attributes = self.OpenCV_pred_attributes(puzzle)
        else:
            attributes = self.get_attributes(puzzle)
        desc, prompt = self.generate_prompts(attributes)
        pred = self.solve([prompt])[0]

        try:
            output = pred.split(',')[0]
        except:
            output = pred

        answers = desc[8:16]

        if output in answers:
            index = answers.index(output)
        else:
            index = 8

        return index, output

def generate_single_description(angle, color, size, type):
    template = 'a {} {} of size {} at a {} degrees angle'.format(color, type, size, angle)

    return template

def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_model(output_dir, name, groundtruths, preds):
    y_true = [x[0] for x in groundtruths]
    y_pred = [y[0] for y in preds]

    cr = classification_report(y_true, y_pred, output_dict=True) # correct
    with open(output_dir + f'{name}_cr.txt', 'w') as f:
        print(cr, file=f)

    # cr = classification_report(y_pred, y_true, output_dict=True) # wrong
    # with open(output_dir + f'old_{name}_cr.txt', 'w') as f:
    #     print(cr, file=f)

def eval_model_from_file(output_dir, name, groundtruths_path, preds_path):
    with open(groundtruths_path, 'r') as gt, open(preds_path, 'r') as p:
        groundtruths, preds = [], []
        for gt_line, pred_line in zip(gt, p):
            gt_content, pred_content = gt_line.strip(), pred_line.strip()

            if gt_content[2] == ',':
                id = int(gt_content[1])
                desc = json.loads(gt_content[3:-1].replace("'", '"'))
            else:
                id = int(gt_content[1:3])
                desc = json.loads(gt_content[4:-1].replace("'", '"'))

            groundtruths.append((id, desc))

            if pred_content[2] == ',':
                id = int(pred_content[1])
                desc = json.loads(pred_content[3:-1].replace("'", '"'))
            else:
                id = int(pred_content[1:3])
                desc = json.loads(pred_content[4:-1].replace("'", '"'))

            preds.append((id, desc))

        evaluate_model(output_dir, name, groundtruths, preds)


def main(name, seed, data_dir, split, type, ClassicOpenCV, vlm, lm, eval=True):
    """
    Main function for testing the model.
    """
    # Choose seed for reproducibility
    set_seed(seed)
    # Pick GPU if available
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Download the RAVEN dataset
    output = f"{data_dir}/RAVEN10000.zip"
    if not exists(output):
        gdown.download(URL, output, quiet=False, fuzzy=True)
        # Unzip the archive
        with zipfile.ZipFile(output, "r") as zip_ref:
            print('unpacking...')
            zip_ref.extractall(data_dir)
            print('unpacking done')
    else:
        print('Dataset already exists, moving further...')

    # Load the data
    test_set = Raven(f'{data_dir}/RAVEN-10000/', split, type)
    test_set.load_data()

    # Prepare the Socratic Model
    model = Demo(device)
    model.load_VLM(vlm)
    model.load_LM(lm)

    output_dir = f'./output/'

    groundtruths = []
    preds = []

    # if eval:
    #     eval_model_from_file(output_dir, name, './output/gt_centresingle_test.txt', './output/BLIP_FT5XL.txt')

    with open(output_dir + f'{name}_results.txt', 'w') as file:
        # Inference
        for i in tqdm(range(test_set.len())):
            if not ClassicOpenCV:
                puzzle = test_set.get_puzzle(i)
            else:
                puzzle = test_set.items[i].images[0:16, :, :]

            index, output = model.forward(puzzle,ClassicOpenCV)
            file.write(f"{index, output}\n")

            if eval:
                id = test_set.items[i].target_id
                groundtruths.append((id, test_set.items[i].symbolic[id]))
                preds.append((index, output))

        if eval:
            evaluate_model(output_dir, name, groundtruths, preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='experiment1_BLIP_xl', type=str,
                        help='How to name the results')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='./data', type=str,
                        help='Data directory where to find dataset.')
    parser.add_argument('--split', default='test', type=str,
                        help='Data split to use.')
    parser.add_argument('--type', default='center_single', type=str,
                        help='Puzzle type to use.')
    parser.add_argument('--ClassicOpenCV', default=False, type=bool,
                        help='Use OpenCV or not')
    parser.add_argument('--vlm', default='Salesforce/blip-vqa-base', type=str,
                        help='VLM weights to use.')
    parser.add_argument('--lm', default='google/flan-t5-xl', type=str,
                        help='LM weights to use.')
    parser.add_argument('--eval', default=True, type=bool,
                        help='Evaluate SM model after inference.')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
