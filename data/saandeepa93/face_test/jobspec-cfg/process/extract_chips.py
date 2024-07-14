import os 
import io 

import torch 
from torchvision import transforms
from torchvision.transforms import ToPILImage

from decord import VideoReader, cpu
import webdataset as wds

from icecream import ic 
from sys import exit as e 
from transforms import * 
import pickle
import json
from tqdm import tqdm





class DataAugmentationForVideoMAE(object):
  def __init__(self):
    self.input_mean = [0.5, 0.5, 0.5]  # IMAGENET_DEFAULT_MEAN
    self.input_std = [0.5, 0.5, 0.5]  # IMAGENET_DEFAULT_STD
    normalize = GroupNormalize(self.input_mean, self.input_std)
    self.train_augmentation = GroupScale((224, 224))
    self.transform = transforms.Compose([                            
        self.train_augmentation,
        Stack(roll=False),
        # ADD ANY AUG BEFORE THIS LINE
        ToTorchFormatTensor(div=True),
    ])

  def __call__(self, images):
    process_data = self.transform(images)
    return process_data

  def __repr__(self):
    repr = "(DataAugmentationForVideoMAE,\n"
    repr += "  transform = %s,\n" % str(self.transform)
    repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
    repr += ")"
    return repr

class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()  # Convert tensors to lists
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)

def load_pickle(subject, root):
  fpath = os.path.join(root, f"{subject}_rel.pickle")
  with open(fpath, 'rb') as inp:
    pickl_file = pickle.load(inp)
  return pickl_file

def save_images(cfg, process_data, subject, mode):
  imgs = [ ToPILImage()(process_data[:, vid, :, :].cpu().clamp(-1.,1.)) for vid in range(process_data.shape[1])  ]
  for id, im in enumerate(imgs):
    im.save(f"{cfg.PATHS.VIS_PATH}/{mode}_{subject}_rec_img{id}.jpg")



def _bboxCrop(img_arr, bbox):
    x1, y1, x2, y2 = bbox
    x1 = int(max(0, x1))
    y1 = int(max(0, y1))
    x2 = int(max(0, x2))
    y2 = int(max(0, y2))
    crop_img = img_arr[y1:y2, x1:x2, :]
    crop_img = Image.fromarray(np.array(crop_img)[:, :, ::-1]).convert('RGB')
    return crop_img

def _loadFrames(fpaths, bboxes):
  cropped_images = []
  for fpath, bbox in zip(fpaths, bboxes):
    image = Image.open(fpath).convert('RGB')
    img_arr = np.array(image)
    crop_img = _bboxCrop(img_arr, bbox)
    cropped_images.append(crop_img)
  return cropped_images

def _loadFieldVid(fpath, frames, bboxes):
  decord_vr = VideoReader(fpath, num_threads=1, ctx=cpu(0))
  buffer = decord_vr.get_batch(frames).asnumpy()
  cropped_images = []
  for img_arr, bbox in zip(buffer, bboxes):
    crop_img = _bboxCrop(img_arr, bbox)
    cropped_images.append(crop_img)
  return cropped_images


def image_encoder(image, format="PNG"):
    """Encode a PIL image to bytes."""
    img_byte_arr = io.BytesIO()
    # image.save(img_byte_arr, format=format)
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

if __name__ == "__main__":
  fname_train = f"all_files_train.pkl"
  with open(f'/shares/rra_sarkar-2135-1003-00/faces/face_verification/data/{fname_train}', 'rb') as f:
    all_files_train = pickle.load(f)

  fname_val = f"all_files_val.pkl"
  with open(f'/shares/rra_sarkar-2135-1003-00/faces/face_verification/data/{fname_val}', 'rb') as f:
    all_files_val = pickle.load(f)

  dest_dir = "./data/face_chips/shard-%06d.tar"
  annot_path = "/shares/rra_sarkar-2135-1003-00/faces/BGC_pickles_v4b/"
  root_dir = "/shares/rra_sarkar-2135-1003-00/BRIAR_RAW/"


  transform = DataAugmentationForVideoMAE()

  output_shard_pattern = "./data/face_chips/50_frames/shard-%06d.tar"
  sink =  wds.ShardWriter(output_shard_pattern, maxcount=5, maxsize=10e9, verbose=0)

  controlled_process_data_dict = {}
  pbar = tqdm(range(len(all_files_train)))

  completed_subjects = []
  for idx in pbar:
    subject_phase_video = all_files_train[idx]
    subject_phase, selected_vid = subject_phase_video.split('-')
    selected_vid = int(selected_vid)
    subject, phase = subject_phase.split('_')
    subject_dict = load_pickle(f"{subject_phase}", annot_path )
    pbar.set_description(subject)


    # ****LOAD CONTROLLED FRAMES****
    if subject_phase not in controlled_process_data_dict:
      controlled_fpaths = subject_dict['controlled']['fpath']
      controlled_bboxes = subject_dict['controlled']['bbox']
      field_type =  subject_dict['controlled']['type']
      
      all_controlled_frames = np.arange(len(controlled_fpaths))
      selected_controlled_fpaths = [os.path.join(root_dir, controlled_fpaths[k]) for k in all_controlled_frames]
      selected_controlled_bboxes = [controlled_bboxes[k] for k in all_controlled_frames]

      images = _loadFrames(selected_controlled_fpaths, selected_controlled_bboxes)
      process_data = transform(images)
      process_data_g = process_data.contiguous().view((len(all_controlled_frames), 3) + process_data.size()[-2:]).transpose(0,1)

      controlled_process_data_dict[subject_phase] = process_data_g
    
    else:
      process_data_g = controlled_process_data_dict[subject_phase]


    cls_g =[f"{phase}/{subject}/{field_type}"]

     # ****LOAD FIELD VIDEO****
    field_fpath = os.path.join(root_dir, subject_dict['field']['fpath'][selected_vid])
    all_field_frames = subject_dict['field']['frame_num'][selected_vid]
    all_field_bboxes = subject_dict['field']['bbox'][selected_vid]
    field_type =  subject_dict['field']['type'][selected_vid]
    video_name = field_fpath.split('/')[-1].split('.')[0]
    video_dist = field_fpath.split('/')[-3]
    
    all_field_ind = np.arange(len(all_field_frames))

    if len(all_field_ind) > 100:
      selected_field_ind = random.sample(list(all_field_ind), 100)
    else:
      selected_field_ind = all_field_ind

    selected_field_frames = [all_field_frames[k] for k in selected_field_ind]
    selected_field_bboxes = [all_field_bboxes[k] for k in selected_field_ind]

    images = _loadFieldVid(field_fpath, selected_field_frames, selected_field_bboxes)
    process_data = transform(images)
    process_data_p = process_data.contiguous().view((len(selected_field_ind), 3) + process_data.size()[-2:]).transpose(0,1)

    cls_p = f"{phase}/{subject}/{video_dist}/{field_type}/{video_name}"

    label = 0 #int(all_files_train[subject_phase_video])

    # print(process_data_g.size(), process_data_p.size())
    buffer_g = io.BytesIO()
    torch.save(process_data_g.contiguous().cpu().detach(), buffer_g)  # Move tensor to CPU before serialization
    image_bytes_g = buffer_g.getvalue()
    
    buffer_p = io.BytesIO()
    torch.save(process_data_p.contiguous().cpu().detach(), buffer_p)  # Move tensor to CPU before serialization
    image_bytes_p = buffer_p.getvalue()

    meta_info = {
      "gallery": selected_controlled_fpaths,
      "probe": cls_p, 
      "probe_frames": selected_field_frames 
    }
    # meta_info['probe_frames'] = [int(k.item()) for k in meta_info['probe_frames']]
    json_str = json.dumps(meta_info, cls=TensorEncoder)
    json_bytes = json_str.encode("utf-8")

    data_dict = {
          "__key__": f"sample{idx:06d}",
          'image_g.pt': image_bytes_g,
          'image_p.pt': image_bytes_p,
          'json': json_bytes
    }
    sink.write(data_dict)
    completed_subjects.append(subject_phase_video)

    if idx == 10:
      break

  with open('./data/completed_tars.pkl', 'wb') as file:
    # Use pickle.dump() to serialize and save the list to the file
    pickle.dump(completed_subjects, file)
  sink.close()
  print("DONE!!")