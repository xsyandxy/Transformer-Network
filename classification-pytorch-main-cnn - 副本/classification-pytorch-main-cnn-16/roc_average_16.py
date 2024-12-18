import pandas as pd
from sklearn import metrics
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np
from numpy import interp

CSV_file = './swin_16.csv'
fontdict = {'color': 'black',
             'family':'Times New Roman',
            'weight': 'light',
            'size': 20}
file_name=CSV_file.split('_')[0].split('/')[1]
p_csv = pd.read_csv(CSV_file,usecols=['r0','0','r1','1','r2','2','r3','3','r4','4','r5','5','r6','6','r7','7','r8','8','r9','9','r10','10','r11','11','r12','12','r13','13','r14','14','r15','15'])

fpr = dict()
tpr = dict()
roc_auc = dict()


for i in range(16):
    fpr['{}'.format(i)], tpr['{}'.format(i)], thersholds = metrics.roc_curve(p_csv['r{}'.format(i)].to_numpy(), p_csv['{}'.format(i)].to_numpy())
    roc_auc['{}'.format(i)] = auc(fpr['{}'.format(i)], tpr['{}'.format(i)])
    print(roc_auc['{}'.format(i)])
#
# y_label4 = p_csv['r4'].to_numpy()
# y_pre4 = p_csv['4'].to_numpy()
# fpr['4'], tpr['4'], thersholds = metrics.roc_curve(y_label4, y_pre4)
# roc_auc['4'] = auc(fpr['4'], tpr['4'])
# print(roc_auc['4'])

###########################################################################
classes_n=16
all_fpr = np.unique(np.concatenate([fpr['{}'.format(i)] for i in range(classes_n)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(classes_n):
    mean_tpr += interp(all_fpr, fpr['{}'.format(i)], tpr['{}'.format(i)])

# Finally average it and compute AUC
mean_tpr /= classes_n
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

############################################################################

colors = ['aqua', 'darkorange', 'deeppink','navy','cornflowerblue','aliceblue','antiquewhite','blueviolet','cadetblue','chocolate','deepskyblue','ghostwhite','greenyellow','lightslategray','mediumaquamarine','mediumpurple']
lw=2
plt.figure(figsize=(15,8),dpi=100)


n_classes = ['0', '1', '2','3','4','5','6','7','8','9','10','11','12','13','14','15']
label_list=['Atelectasis','Cardiomegaly','Consolidation','COVID','Edema','Effusion','Emphysema','Fibrosis','Hernia','Infiltration','Mass','Nodule','Normal','Pleural_Thickening','Pneumonia','Pneumothorax']  ####timesformer#####




# plt.plot(fpr['1'], tpr['1'], color=colors[int('1')], lw=lw,
#             label='ROC curve of class {0} (area = {1:0.2f})'
#             ''.format('1', roc_auc['1']))
for i in n_classes:
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], color=colors[int(i)], lw=lw,
             label='ROC curve of {0} (AUC = {1:0.2f})'
             .format(label_list[int(i)], roc_auc[i]))
# plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
plt.plot(fpr['macro'], tpr['macro'], color=colors[int(4)], lw=lw,
         label='macro-average ROC curve (area = {0:0.2f})'
               .format(roc_auc["macro"])
         )

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontdict=fontdict)
plt.ylabel('True Positive Rate',fontdict=fontdict)
plt.title('{}ROC curve'.format(file_name),fontdict=fontdict)
# plt.legend(loc="lower right")
plt.legend(loc="lower right", numpoints=5,prop = {'size':0.3})
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=10)


plt.grid(linestyle='-.')
plt.grid(True)
plt.savefig('{}_16.png'.format(file_name))
plt.show()

