import math
import requests
import json
import numpy as np
from scenedetect import detect, AdaptiveDetector
from PIL import Image
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import tensorflow as tf
from transformers import AutoProcessor, AutoTokenizer, Blip2ForConditionalGeneration, TFCLIPModel

class VideoAnalyzer():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # caption
        self.blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-6.7b")
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-6.7b", torch_dtype=torch.float16)
        self.blip_model.to(self.device)

        # sentiment
        self.clip_model =  TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.sentiment_labels = ['affection',  'cheerfullness',  'confusion',  'contentment','disappointment', 'disgust','enthrallment','envy',
	        'exasperation','gratitude','horror',  'irritabilty', 'lust','neglect','nervousness','optimism','pride','rage',
	        'relief', 'sadness','shame',  'suffering', 'surprise', 'sympathy', 'zest']
        CLIP_E_WEIGHTS = '/work/07016/cw38637/ls6/nlp/WEBemo_25cat_classification.hdf5'
        self.clip_e_model = self.CLIP_e_crossentropy()
        self.clip_e_model.load_weights(CLIP_E_WEIGHTS)

    def analyze(self, name):
        indexes = self.get_frame_indexes(name)
        cap = cv2.VideoCapture(name)
        captions = []
        sentiments = np.zeros((len(self.sentiment_labels), ))
        for idx in indexes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            captions.append(self.get_caption(image))
            sentiments += self.get_sentiment(image)
        cap.release()
        result = {
            'indexes': indexes,
            'captions': captions,
            'sentiments': [
                self.sentiment_labels,
                sentiments.tolist()
            ]
        }
        with open(name.replace('.mp4', '-video.txt'), 'w') as f:
            json.dump(result, f)

    def get_frame_indexes(self, name, total_scenes=15):
        scenes=detect(name, AdaptiveDetector())
        indexes = []
        duration = scenes[-1][1].get_frames()
        if len(scenes) < total_scenes:
            for scene in scenes:
                start = scene[0].get_frames()
                end = scene[1].get_frames()
                num_frames = math.ceil((end - start) / duration * total_scenes)
                for split in range(1, num_frames + 1):
                    indexes.append(start + round((end - start) / (num_frames + 1) * split))
        else:
            for scene in scenes:
                start = scene[0].get_frames()
                end = scene[1].get_frames()
                indexes.append(round((start + end) / 2))
        return indexes
    
    def get_caption(self, image):
        prompt = 'Question: what is happening in the photo? Answer:'
        inputs = self.blip_processor(image, text=prompt, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.blip_model.generate(**inputs, max_length=100)
        generated_text = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text

    def get_sentiment(self, image):
        inputs = self.clip_processor(images=image, return_tensors="tf")
        image_features = self.clip_model.get_image_features(**inputs)
        image_features = image_features.numpy()
        preds = self.clip_e_model.predict(image_features, verbose=0)
        preds = np.squeeze(preds)
        return preds
    
    def CLIP_e_crossentropy(self, num_classes=25, activation='softmax'):
        IMG_FEATURES_SIZE = 512
        INPUT_img = tf.keras.layers.Input(shape=(IMG_FEATURES_SIZE),name='input_img_features')
        fc = tf.keras.layers.Dense(512, activation='relu', name='img_fc1')(INPUT_img)
        preds = tf.keras.layers.Dense(num_classes,activation=activation,name='preds')(fc)
        model = tf.keras.models.Model(inputs=INPUT_img, outputs=preds)
        return model


BASE_DIR = '/work/07016/cw38637/ls6/nlp'
videos = [
    'family-man',
    'harry-potter',
    'matrix',
    'perfect-storm',
    'polar-express'
]

video_analyzer = VideoAnalyzer()
for video in videos:
    video_analyzer.analyze(f'{BASE_DIR}/{video}.mp4')