import datetime
import numpy as np
import numpy.linalg as LA
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.externals import six
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from bofw import Bofw
import sys
from sklearn.metrics import roc_curve, auc

print("Start time is ", datetime.datetime.now())

DATA_DIR = "data/processed"
N_COMPONENT = 2

subjects = range(1, 13)

X =  np.concatenate([np.load("{0}/{1}/subj{2}_train_data.npy".format(DATA_DIR, N_COMPONENT, subject)) for subject in subjects])
y =  np.concatenate([np.load("{0}/{1}/subj{2}_train_labels.npy".format(DATA_DIR, N_COMPONENT, subject)) for subject in subjects])

X_test =  np.concatenate([np.load("{0}/{1}/subj{2}_val_data.npy".format(DATA_DIR, N_COMPONENT, subject)) for subject in subjects])
y_test =  np.concatenate([np.load("{0}/{1}/subj{2}_val_labels.npy".format(DATA_DIR, N_COMPONENT, subject)) for subject in subjects])

y = y[:, 2]
y_test = y_test[:,2]

print(X.shape, y.shape)
print(X_test.shape, y_test.shape)

clf = svm.SVC(kernel='linear',C=1,probability=True)
myBofw = Bofw()
pca = PCA(n_components=0.9)
scaler = StandardScaler()

bofw_pipeline = Pipeline([('myown', myBofw), ('bofw_pca', pca), ('bofw_scaling', scaler), ('svm', clf)])

estimator=bofw_pipeline.set_params(myown__num_clusters=2**8, svm__C=2**4).fit(X, y)

score = estimator.score(X_test, y_test)
predictions = estimator.predict(X_test)
print(score)

y_binary = label_binarize(y_test,classes=[1,2,3,4,5,6])
predictions_binary=label_binarize(predictions,classes=[1,2,3,4,5,6])

aucTotal = 0
y_predict_prob=estimator.predict_proba(X_test)
for i in range(0,6):
    singleAuc=roc_auc_score(y_binary[:,i],y_predict_prob[:,i])
    aucTotal+=singleAuc
    print("for label",i,"auc=",singleAuc)
totalAUC=aucTotal/6
print "totalAUC",totalAUC



#Compute ROC curve and ROC area for each class
n_classes=y_binary.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_binary[:, i], y_predict_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
plt.figure()
ax = plt.subplot(111)
plt.plot(fpr[0], tpr[0], label='event 1 (area = %0.2f)' % roc_auc[0])
plt.plot(fpr[1], tpr[1], label='event 2 (area = %0.2f)' % roc_auc[1])
plt.plot(fpr[2], tpr[2], label='event 3 (area = %0.2f)' % roc_auc[2])
plt.plot(fpr[3], tpr[3], label='event 4 (area = %0.2f)' % roc_auc[3])
plt.plot(fpr[4], tpr[4], label='event 5 (area = %0.2f)' % roc_auc[4])
plt.plot(fpr[5], tpr[5], label='event 6 (area = %0.2f)' % roc_auc[5])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.legend(loc=4)
plt.legend(loc="lower right")
plt.savefig("ROC6Partials.pdf")
plt.clf()


from scipy import interp

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_binary.ravel(), y_predict_prob.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

##############################################################################
# Plot ROC curves for the multiclass problem

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         linewidth=2)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         linewidth=2)

#for i in range(n_classes):
#    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
#                                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")


plt.savefig("ROCMean.pdf")



from sklearn.metrics import confusion_matrix

print confusion_matrix(y_test,predictions)
