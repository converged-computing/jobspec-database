from templates import *
from templates_cls import *
from experiment_classifier import ClsModel
import matplotlib.pyplot as plt
from torchvision.utils import *
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
import seaborn as sns
import numpy as np
import cv2 as cv2
import torch
import torchvision.transforms as transforms
from ssim import *

# ----------------- cluster spezifisch remote -------------------- 
torch.set_printoptions(threshold=torch.inf)
print(plt.get_backend())
plt.switch_backend('agg')
print(plt.get_backend())

# ---------------- Diffusion Autoencoder Laden ----------------
device = 'cuda:0'
conf = mri_autoenc()
model = LitModel(conf)
state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)

# ------------------ Classifier Laden -------------------------
cls_conf = mri_autoenc_cls()
cls_model = ClsModel(cls_conf)
state = torch.load(f'checkpoints/{cls_conf.name}/last.ckpt',
                    map_location='cpu')
print('latent step:', state['global_step'])
cls_model.load_state_dict(state['state_dict'], strict=False)
cls_model.to(device)


# ---------------- Multiclass Classifier Evaluation -------------------
test_dir = ImageDataset('/home/yv312705/Code/diffusion_autoenc/FastMri/test_classifier', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False, sort_names=True)
test_size = test_dir.__len__()
test_data_dir = '/home/yv312705/Code/diffusion_autoenc/FastMri/test_classifier/'

subdirs = [subdir for subdir in sorted(os.listdir(test_data_dir)) if os.path.isdir(os.path.join(test_data_dir, subdir))]
label_map = {subdir: i for i, subdir in enumerate(subdirs)}

print(MriAttrDataset.id_to_cls)

labels = []
y_predictedlabel = []
y_confpredlabel = []
y_truelabel = []


for subdir in subdirs:
    subdir_path = os.path.join(test_data_dir, subdir)
    for filename in os.listdir(subdir_path):
        labels.append(label_map[subdir])

for i in range (0, test_size):
    test_batch = test_dir[i]['img'][None]
    cond = model.encode(test_batch.to(device))
    cond = cls_model.normalize(cond)
    pred = cls_model.classifier.forward(cond)
    #print(pred)
    _ , confpred = torch.max(pred, dim=1)
    confpred = confpred.item()
    y_confpredlabel.append(confpred)

    pred = torch.softmax(pred, dim=1)
    pred = pred[0].tolist()

    # ROC Metrics
    y_predictedlabel.append(pred)
    #if i < 10:
        #y_truelabel.append(1 - labels[i])
    #else:
    y_truelabel.append(labels[i])

y_truelabel_bin = label_binarize(y_truelabel, classes=list(range(len(MriAttrDataset.id_to_cls))))
y_truelabel_bin = torch.tensor(y_truelabel_bin)
y_predictedlabel = torch.tensor(y_predictedlabel)

# ----------------- ROC AUC -------------------

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(MriAttrDataset.id_to_cls)):
    fpr[i], tpr[i], _ = roc_curve(y_truelabel_bin[:, i], y_predictedlabel[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# ------------------ Plot des ROC AUC -------------------------
plt.figure()

colors = ['blue', 'red', 'green', 'orange', 'yellow','purple'] 

for i, color in zip(range(len(MriAttrDataset.id_to_cls)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (AUC = {1:0.2f})'.format(MriAttrDataset.id_to_cls[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tick_params(axis='x', labelsize=14)  
plt.tick_params(axis='y', labelsize=14)
plt.xlabel('False Positive Rate', fontweight='bold', fontsize=14)
plt.ylabel('True Positive Rate', fontweight='bold', fontsize=14)
plt.title(f'ROC / Number of testdata: {test_size}', fontsize=16)
plt.legend(loc="lower right", fontsize=11)

antwort = input("Möchten Sie die Figur des multiklassen ROC Plots speichern? (ja/nein)")

# Wenn die Antwort "Ja" lautet, speichern Sie die Figur ab
if antwort.lower() == "ja":

    pfad = "/home/yv312705/Code/diffusion_autoenc/eval_plots/mri_nine/"

    if not os.path.exists(pfad):
        os.makedirs(pfad)

    plt.savefig(pfad + "ROC_plot.png")
    print("Figur wurde gespeichert!")
else:
    print("Figur wurde nicht gespeichert.")


# ----------------- Confusionmatrix -------------------
cm = confusion_matrix(y_truelabel, y_confpredlabel)

# ----------------- Plot der Confusionmatrix -------------------
plt.figure()
ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
tick_marks = np.arange(len(MriAttrDataset.id_to_cls))
ax.set_xticklabels(['cor_pd', 'cor_pd_fs', 'cor_t1'], fontsize=12)
ax.set_yticklabels(['cor_pd', 'cor_pd_fs', 'cor_t1'], fontsize=12)
ax.set_xlabel('Predicted Sequence', fontweight='bold', fontsize=14)
ax.set_ylabel('Actual Sequence', fontweight='bold', fontsize=14)  
ax.tick_params(axis='x', labelsize=14)  
ax.tick_params(axis='y', labelsize=14)
plt.title(f'Confusion Matrix / Number of testdata: {test_size}', fontsize=16)

antwort = input("Möchten Sie die Figur der Confusionmatrix speichern? (ja/nein)")

# Wenn die Antwort "Ja" lautet, speichern Sie die Figur ab
if antwort.lower() == "ja":

    pfad = "/home/yv312705/Code/diffusion_autoenc/eval_plots/mri_nine/"

    if not os.path.exists(pfad):
        os.makedirs(pfad)

    plt.savefig(pfad + "Confusionmatrix_plot.png")
    print("Figur wurde gespeichert!")
else:
    print("Figur wurde nicht gespeichert.")


'''
# ------------------ Binary Classifier mit ROC Plot testen -----------------------

test_dir = ImageDataset('/home/yv312705/Code/diffusion_autoenc/FastMri/test_classifier', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)
test_size = test_dir.__len__()
test_data_dir = '/home/yv312705/Code/diffusion_autoenc/FastMri/test_classifier/'

subdirs = [subdir for subdir in sorted(os.listdir(test_data_dir)) if os.path.isdir(os.path.join(test_data_dir, subdir))]
label_map = {subdir: i for i, subdir in enumerate(subdirs)}

print(MriAttrDataset.id_to_cls)

labels = []

y_predictedlabel = []
y_confpredlabel = []
y_truelabel = []

for subdir in subdirs:
    subdir_path = os.path.join(test_data_dir, subdir)
    for filename in os.listdir(subdir_path):
        labels.append(label_map[subdir])

for i in range (0, test_size):
    test_batch = test_dir[i]['img'][None]
    cond = model.encode(test_batch.to(device))
    cond = cls_model.normalize(cond)
    pred = cls_model.classifier.forward(cond)
    #print(pred)
    _ , confpred = torch.max(pred, dim=1)
    confpred = confpred.item()
    y_confpredlabel.append(confpred)

    pred = torch.softmax(pred, dim=1)
    pred = pred[0,1].cpu().item()

    # ROC Metrics
    y_predictedlabel.append(pred)
    #if i < 10:
        #y_truelabel.append(1 - labels[i])
    #else:
    y_truelabel.append(labels[i])


fpr, tpr, thresholds = roc_curve(y_truelabel, y_predictedlabel)

#Confusionsmatrix berechnen
cm = confusion_matrix(y_truelabel, y_confpredlabel)

auc_score = "AUC Score: " + str(auc(fpr, tpr).item())
print(auc_score)

# -------------- Plot the ROC curve ---------------
import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr, tpr, color='orange', label='ROC curve (AUC = %0.2f)' % auc(fpr,tpr).item())
plt.plot([0, 1], [0, 1], linestyle='--', color='blue', label='Baseline')
plt.title(f'ROC Curve / Number of testdata: {test_size}')
plt.xlabel('False Positive Rate', fontweight='bold')
plt.ylabel('True Positive Rate', fontweight='bold')
plt.legend()
#plt.show()

antwort = input("Möchten Sie die Figur des ROC Plots speichern? (ja/nein)")

# Wenn die Antwort "Ja" lautet, speichern Sie die Figur ab
if antwort.lower() == "ja":

    pfad = "/home/yv312705/Code/diffusion_autoenc/eval_plots/mri_seven/"

    if not os.path.exists(pfad):
        os.makedirs(pfad)

    plt.savefig(pfad + "ROC_plot.png")
    print("Figur wurde gespeichert!")
else:
    print("Figur wurde nicht gespeichert.")


# ----------------- Plot der Confusionmatrix -------------------
plt.figure()
ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
ax.set_xticklabels(['Negative', 'Positive'])
ax.set_yticklabels(['Negative', 'Positive']) 
ax.set_xlabel('Predicted Sequence', fontweight='bold')
ax.set_ylabel('Actual Sequence', fontweight='bold')  
plt.title(f'Confusion Matrix / Number of testdata: {test_size}')

antwort = input("Möchten Sie die Figur der Confusionmatrix speichern? (ja/nein)")

# Wenn die Antwort "Ja" lautet, speichern Sie die Figur ab
if antwort.lower() == "ja":

    pfad = "/home/yv312705/Code/diffusion_autoenc/eval_plots/mri_seven/"

    if not os.path.exists(pfad):
        os.makedirs(pfad)

    plt.savefig(pfad + "Confusionmatrix_plot.png")
    print("Figur wurde gespeichert!")
else:
    print("Figur wurde nicht gespeichert.")
######################################################################
'''
  
# ------------- Originale Bilder Laden -------------
data = ImageDataset('/home/yv312705/Code/diffusion_autoenc/FastMri/test_classifier/c_cor_t1', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)
for p in range(30):

    batch = data[p]['img'][None]
    ori = (batch + 1) / 2
    ori_dat = (ori *255).byte()
    ori_np = np.array(ori_dat[0,0])
    ori_np = Image.fromarray(ori_np)
    ori_np = ori_np.resize((ori_np.size[0]*2, ori_np.size[1]*2))
    ori_np = np.array(ori_np)

    # ------------- Edge Image des originalen Bildes -----------
    edge_img = cv2.Canny(ori_np, threshold1=100, threshold2=150)
    cv2.imwrite('/home/yv312705/Code/diffusion_autoenc/eval_plots/edges.png', edge_img)


    # -------------------- Encoder ------------------
    cond = model.encode(batch.to(device))
    xT = model.encode_stochastic(batch.to(device), cond, T=250)

    # --------------------- Classifier Test ----------
    cond = cls_model.normalize(cond)
    pred = cls_model.classifier.forward(cond)
    print('pred:', pred)
    cond = cls_model.denormalize(cond)

    # ----------------- Plot Original u. Encodiert ----------
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
    ax[1].imshow(xT[0].permute(1, 2, 0).cpu())

    # ----------- Auswahl der zu manipulierenden Attribute -----------
    print(MriAttrDataset.id_to_cls)

    # ----------- Eingabe des zu manipulierenden Attributs -----------
    cls_id = MriAttrDataset.cls_to_id['cor_pd_fs']

    # ----------- Edge Detection Algorithm -------------------
    images = []
    edge_images = []
    stepsizes = []

    stepsize = 0.01
    num_steps = 60

    for j in range(num_steps):
        cond_class = cls_model.normalize(cond)
        cond_class = cond_class + stepsize * math.sqrt(512) * F.normalize(cls_model.classifier.weight[cls_id][None, :], dim=1)

        prediction = cls_model.classifier.forward(cond_class)
        print('pred:', prediction)

        cond_class = cls_model.denormalize(cond_class)
        img = model.render(xT, cond_class, T=100)
        stepsizes.append(stepsize)
        cond_class = 0

        img = (img *255).byte()
        img = np.array(img[0,0].cpu())
        img = Image.fromarray(img)
        img = img.resize((img.size[0]*2, img.size[1]*2))
        img = np.array(img)

        images.append(img)

        if stepsize == 0.6:
            break
        stepsize += 0.01

# ----------------- function to classify the edges ------------------- 
    def edgeclassifier(images, original_edgeimg, step_size):

        original_edgetensor = torch.from_numpy(original_edgeimg).unsqueeze(0)
        diff_tensors = []

        # ---------- edge images generated ---------
        for i in range(num_steps):
            edge_image = cv2.Canny(images[i], threshold1=100, threshold2=150)
            edge_images.append(edge_image)

        # ------- Abs. Diff images  ----------
        for j in range(num_steps):
            edge_tensor = torch.from_numpy(edge_images[j]).unsqueeze(0)
            diff_tensor = torch.abs(edge_tensor.permute(1, 2, 0).cpu() - original_edgetensor.permute(1, 2, 0).cpu())
            diff_tensors.append(diff_tensor)

        return diff_tensors, edge_images

    diff_tens, edge_images = edgeclassifier(images, edge_img, stepsizes)

    # ----------- compute the lowest difference between original and manip in edges -------------
    counter_list = []
    threshhold = 0.4

    for diff_t in diff_tens:
        counter = 0
        for pixel in torch.flatten(diff_t):
            if pixel != 0:
                counter += 1

        counter_list.append(counter)

    sorted_counter = sorted(enumerate(counter_list), key=lambda x: x[1])

    for k in range(num_steps):
        smallest_index = sorted_counter[k][0]
        selected_stepsize = stepsizes[smallest_index]
        if selected_stepsize > threshhold:
            break

    print(selected_stepsize)


    # --------------- Big Plot of ori, edges, diff_edges and best image -------------------
    fig, ax = plt.subplots(4, num_steps, figsize=(5*30, 10))

    for i in range(num_steps):
        if i == 0:
            ax[0,0].imshow(ori[0].permute(1, 2, 0).cpu())
            ax[0,0].set_title('Original Image pdw', fontsize= 10)
            ax[1,0].imshow(edge_img, cmap='gray')
            ax[1,0].set_title('Canny Edge Detection Original', fontsize= 10)
            ax[3,0].imshow(images[smallest_index], cmap='gray')
            ax[3,0].axis('off')
            ax[3,0].set_title('Best Image !!!!', fontsize= 10)

        else:     
            ax[0,i].imshow(images[i], cmap='gray')
            ax[0,i].axis('off')
            ax[0,i].set_title(str(i), fontsize= 10)
            ax[1,i].imshow(edge_images[i-1], cmap='gray')
            ax[1,i].axis('off')
            ax[1,i].set_title(str(i), fontsize= 10)
            ax[2,i].imshow(diff_tens[i-1].sum(2), cmap='gray', vmin=0, vmax=100)
            ax[2,i].axis('off')
            ax[2,i].set_title(str(i), fontsize= 10)

    fig.suptitle(f'Manipulate Mode Testing', fontsize= 17)

    #antwort = input("Möchten Sie die Figur des Big Plot speichern? (ja/nein)")
    antwort = "ja"

    if antwort.lower() == "ja":

        pfad = f'/home/yv312705/Code/diffusion_autoenc/eval_plots/mri_nine/{MriAttrDataset.id_to_cls[2]}/{MriAttrDataset.id_to_cls[1]}/'

        if not os.path.exists(pfad):
            os.makedirs(pfad)

        plt.savefig(pfad + f'classifier_big_{p}{MriAttrDataset.id_to_cls[1]}.png')
        print("Figur wurde gespeichert!")
    else:
        print("Figur wurde nicht gespeichert.")



    # ----------------------- Small Plot of ori and best image ----------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
    ax[0].set_title(f'Original Image | {MriAttrDataset.id_to_cls[2]}', fontsize= 14)
    ax[0].axis('off')
    ax[1].imshow(images[smallest_index], cmap='gray')
    ax[1].set_title(f'Converted Image | {MriAttrDataset.id_to_cls[1]}', fontsize= 14)
    ax[1].axis('off')
    fig.suptitle(f'Sequence Conversion | weight factor: {round(selected_stepsize,2)}', fontsize= 16, fontweight='bold')

    #antwort = input("Möchten Sie die Figur des Small Plot speichern? (ja/nein)")
    antwort = "ja"

    if antwort.lower() == "ja":

        pfad = f'/home/yv312705/Code/diffusion_autoenc/eval_plots/mri_nine/{MriAttrDataset.id_to_cls[2]}/{MriAttrDataset.id_to_cls[1]}/'

        if not os.path.exists(pfad):
            os.makedirs(pfad)

        plt.savefig(pfad + f'classifier_small_{p}{MriAttrDataset.id_to_cls[1]}.png')
        print("Figur wurde gespeichert!")
    else:
        print("Figur wurde nicht gespeichert.")


    # ------------------ Gif erstellen ----------------------
    frames = []

    for i in range(smallest_index):
        numpy_array = (images[i])
        image = Image.fromarray(numpy_array)
        img_resized = image.resize((image.size[0]*2, image.size[1]*2))
        frames.append(img_resized)

    frames[0].save(f'/home/yv312705/Code/diffusion_autoenc/eval_plots/mri_nine/{MriAttrDataset.id_to_cls[2]}/{MriAttrDataset.id_to_cls[1]}/classifier_gif_{p}{MriAttrDataset.id_to_cls[1]}.gif', format='GIF', save_all=True, append_images=frames[1:], duration=150, loop=0)

