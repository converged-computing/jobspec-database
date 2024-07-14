import argparse
from detect_ import detect
from utils.torch_utils import TracedModel
from utils.general import increment_path
from models.experimental import attempt_load
import os
import torch
import cv2
import datetime
from pathlib import Path
from mail import email_sender

def split_video(vidSource, outputPath):
    vidPath = vidSource
    shotsPath = outputPath

    cap = cv2.VideoCapture(vidPath)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    tot_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    vid_length = 300
    num_vids = int(tot_length/(vid_length*fps))+1
    
    
    segRange =[(vid_length*fps*i,vid_length*fps*(i+1)) for i in range(num_vids)] # a list of starting/ending frame indices pairs

    for idx,(begFidx,endFidx) in enumerate(segRange):
        writer = cv2.VideoWriter(shotsPath%idx,fourcc,fps,size)
        cap.set(cv2.CAP_PROP_POS_FRAMES,begFidx)
        ret = True # has frame returned
        while(cap.isOpened() and ret and writer.isOpened()):
            ret, frame = cap.read()
            frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
            if frame_number < endFidx:
                writer.write(frame)
            else:
                break
        writer.release()
        
        

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov7-helmet.pt', help='path to model weights')
    parser.add_argument('--source', type=str, default='inference/videos/20220928_142037.mp4', help='path to video')
    parser.add_argument('--project', default=None, help='save results to project folder. Defaults to current time and date')
    parser.add_argument('--stream', action='store_true', help='is source a stream?')
    parser.add_argument('--mail', action='store_true', help='whether to send a mail of the results or not')
    opt = parser.parse_args()
    
    if opt.project is None:
        project = 'runs/%s' %datetime.datetime.today().strftime('%Y_%m_%d/%H')
    else:
        project = opt.project
    
    
    save_dir = Path(increment_path(Path(project), exist_ok=True))
    (save_dir / 'videos').mkdir(parents=True, exist_ok=True)
    
    if not opt.stream:
        split_video(opt.source, str(save_dir.absolute())+'/videos/%d.mp4')
    
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    # Load model
    imgsz = 640
    model = attempt_load(opt.model, map_location=device)  # load FP32 model
    model = TracedModel(model, device, imgsz)
    if device != 'cpu':
        model.half()  # to FP16
    if opt.stream:
        detect(model, source = opt.source, name = '', project = project)
    else:
        source = str(save_dir.absolute())+'/videos'
        detect(model, source = source, name = '', project = project)
    if opt.mail:
        email_sender()

