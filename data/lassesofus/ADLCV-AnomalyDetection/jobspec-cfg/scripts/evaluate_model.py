"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
   
import matplotlib.pyplot as plt
import argparse
import os
# from visdom import Visdom
# viz = Visdom(port=8850)
from skimage.filters import threshold_otsu
import sys
sys.path.append("..")
sys.path.append(".")
from guided_diffusion.new_bratsloader import BRATSDataset, load_file
import torch.nn.functional as F
import numpy as np
import torch as th
import pdb
import torch.distributed as dist
from guided_diffusion.image_datasets import load_data
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_classifier,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
L = [100, 250, 500, 750, 999]

def visualize(img):
    _min = img.min()
    _max = img.max()
    if _max > 0:
        normalized_img = (img - _min)/ (_max - _min)
    else:
        normalized_img = img
    return normalized_img

def save_image(img, filename):
    # Move the tensor to CPU if it's not already
    img = img.cpu()

    # Check if the image is 2D (grayscale) or 3D (color)
    if img.dim() == 2:
        # For a 2D image (grayscale), display it as a grayscale image
        plt.imshow(img, cmap='gray')
    elif img.dim() == 3:
        # If it's a 3D tensor with 3 channels, permute to make it (H, W, C)
        img = img.permute(1, 2, 0)
        plt.imshow(img)
    else:
        raise ValueError("Unsupported image dimensions")

    plt.axis('off')  # Remove axis ticks and labels
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

#make Otsu thresholding
def get_otsu_threshold(images:np.ndarray):
    images = images.flatten()
    hist, edges = np.histogram(images, bins = 256)
    # hist = hist[1:]
    # hist = hist/np.sum(hist)
    # print(hist)
    # sigma_t = lambda t: np.var(hist[:t])*np.sum(hist[:t]) + np.var(hist[t:])*np.sum(hist[t:])
    # threshold = edges[np.argmin(np.array([sigma_t(t) for t in range(1, 255)])) + 1]
    threshold = threshold_otsu(hist = (hist[1:], edges[2:]))
    return threshold

def dice_score(output, ground_truth):
    intersection = np.logical_and(output, ground_truth)
    # print(F"DICE insides\nIntersection: {intersection.sum()}\nUnion: {output.sum() + ground_truth.sum()}\nOuput: {output.sum()}\nGT: {ground_truth.sum()}")
    # print(output.shape)
    # print(ground_truth.shape)
    # print(intersection.shape)
    # print(union.shape)
    return 2. * intersection.sum() / (output.sum() + ground_truth.sum())

def iou_score(output, ground_truth):
    intersection = np.logical_and(output, ground_truth)
    union = np.logical_or(output, ground_truth)
    # print(union.shape, union.max(), union.min)
    # print(F"IOU insides\nIntersection: {intersection.sum()}\nUnion: {union.sum()}\nOuput: {output.sum()}\nGT: {ground_truth.sum()}")
    # print(output.shape)
    # print(ground_truth.shape)
    # print(intersection.shape)
    # print(union.shape)
    return intersection.sum() / union.sum()

def accuracy(d:dict):
    denominator = sum(d.values())
    return (d["TP"] + d["TN"])/denominator if denominator > 0 else 0
    
def sensitivity(d:dict):
    denominator = d["TP"] + d["FN"]
    return d["TP"]/denominator if denominator > 0 else 0

def specificity(d:dict):
    denominator = d["TN"] + d["FP"]
    return d["TN"]/denominator if denominator > 0 else 0

def calculate_dict(seg_maps_pred, labels):
    d = {"TP":0, "TN":0, "FP":0, "FN":0}
    for i in range(len(labels)):
        label_pred = seg_maps_pred[i,:,:].any()
        key = F"{'T' if label_pred == labels[i] else 'F'}{'P' if label_pred else 'N'}"
        d[key] += 1
    return d


# def auroc_score(output, ground_truth):
#     #read about how this works
#     from sklearn.metrics import roc_auc_score
#     return roc_auc_score(ground_truth, output)


def save_heatmap(data, filename):
    try:
        data = data.cpu()
    except AttributeError:
        pass
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.axis('off')  # Turn off the axis
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    ds = BRATSDataset(args.data_dir, eval_slices=["120"])
    datal = th.utils.data.DataLoader(
    ds,
    batch_size=args.batch_size,
    shuffle=False)
   
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location= th.device("cuda" if th.cuda.is_available() else 'cpu'))
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    print("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path)
    )
    classifier.to(dist_util.dev())

    print('loaded classifier')
    p1 = np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()
    p2 = np.array([np.array(p.shape).prod() for p in classifier.parameters()]).sum()
    print('pmodel', p1, 'pclass', p2)


    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()


    def cond_fn(x, t,  y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            a=th.autograd.grad(selected.sum(), x_in)[0]
            return  a, a * args.classifier_scale




    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    # logger.log("sampling...")
    # all_images = []
    # all_labels = []


    # has_anomaly = lambda diff: (diff > FINAL_THRESHOLD).any()
    # results = []
    # patient_ID = "295"
    # datalist = [load_file(os.path.expanduser(f"~/Desktop/ADLCV-AnomalyDetection/data/brats/val/BraTS20_Training_{patient_ID}/brats_train_{patient_ID}_120.nii.gz"))]
    # for img, _, label, seg_map, number in datal:
    #     model_kwargs = {}
        #img = next(data)  # should return an image from the dataloader "data"
        #pdb.set_trace()
        #print('img', img[0].shape, img[1]) 
        # if args.dataset=='brats':
        #     #Labelmask = th.where(img[3] > 0, 1, 0)
        #     #number=img[4][0]
        #     #if img[2]==0:
        #     #    continue    #take only diseased images as input
        #     # Make folder for image number for saving images
        #     if not os.path.exists('results/plots/'+patient_ID):
        #         os.makedirs('results/plots/'+patient_ID)
        #     save_image(visualize(img[0][0, 0, ...]), 'results/plots/'+patient_ID+'/input 0.png')
        #     save_image(visualize(img[0][0, 1, ...]), 'results/plots/'+patient_ID+'/input 1.png')
        #     save_image(visualize(img[0][0, 2, ...]), 'results/plots/'+patient_ID+'/input 2.png')
        #     save_image(visualize(img[0][0, 3, ...]), 'results/plots/'+patient_ID+'/input 3.png')
        #     save_image(visualize(label), 'results/plots/'+patient_ID+'/ground truth.png')    
        # else:
        #     viz.image(visualize(img[0][0, ...]), opts=dict(caption="img input"))
        #     print('img1', img[1])
        #     number=img[1]["path"]
        #     print('number', number)
    model_kwargs = {}
    # if args.class_cond:
    #     classes = th.randint(
    #         low=0, high=1, size=(args.batch_size,), device=dist_util.dev()
    #     )
    #     model_kwargs["y"] = classes
    #     print('y', model_kwargs["y"])

    sample_fn = (
        diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
    )

    for l in L:
        logger.log("_____________________")
        logger.log(l)
        outputs = []
        labels = []
        seg_maps = []
        i = 0
        for img, _, label, seg_map, number in datal:
            img = th.unsqueeze(img, 0)
            # model_kwargs = {"y" :th.tensor([0,], device = dist_util.dev())}
            if args.class_cond:
                classes = th.randint(
                    low=0, high=1, size=(args.batch_size,), device=dist_util.dev()
                )
                model_kwargs["y"] = classes
            sample, x_noisy, org = sample_fn(
            model_fn,
            (args.batch_size, 4, args.image_size, args.image_size), img, org=img,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
            noise_level = l
            )
            difftot=abs(org[0, :4,...]-sample[0, ...]).sum(dim=0)
            outputs.append(difftot.detach().cpu().numpy())
            seg_maps.append(seg_map.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())

            # save_heatmap(visualize(difftot), 'results/plots/val/' + F'{l} - {number[0]} - anomaly.png')
            # save_heatmap(visualize(th.squeeze(seg_map)), 'results/plots/val/' + F'{l} - {number[0]} - gt.png')
            


            # i += 1
            # if i > 4:
            #     break
        
        outputs = np.array(outputs)
        seg_maps = np.squeeze(np.array(seg_maps, dtype = int))
        labels = np.array(labels)
        logger.log(labels)

        otsu_threshold = get_otsu_threshold(outputs)
        seg_maps_pred = np.array(outputs > otsu_threshold, dtype = int)
        print(otsu_threshold)
        # for index in range(seg_maps_pred.shape[0]):
        #     save_heatmap(visualize(seg_maps_pred[index,:,:]), 'results/plots/val/' + F'{l} - {index} - prediction.png')

        logger.log("L: ", l)
        logger.log("DICE: ", dice_score(seg_maps_pred, seg_maps))
        logger.log("IOU: ", iou_score(seg_maps_pred, seg_maps))
        d = calculate_dict(seg_maps_pred, labels)
        logger.log("Accuracy: ", accuracy(d))
        logger.log("Sensitivity: ", sensitivity(d))
        logger.log("Specificity: ", specificity(d))








            
        
            # save_image(visualize(x_noisy[0, 0, ...]), 'results/plots/'+patient_ID+F'/noisy 0 {l}.png')
            # save_image(visualize(x_noisy[0, 1, ...]), 'results/plots/'+patient_ID+F'/noisy 1 {l}.png')
            # save_image(visualize(x_noisy[0, 2, ...]), 'results/plots/'+patient_ID+F'/noisy 2 {l}.png')
            # save_image(visualize(x_noisy[0, 3, ...]), 'results/plots/'+patient_ID+F'/noisy 3 {l}.png')
       


def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,
        num_samples=10,
        batch_size=1,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=100,
        noise_level=250,
        dataset='brats'
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()

