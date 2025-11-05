import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from transformers import VivitImageProcessor
from transformers import VivitForVideoClassification
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import auc, roc_curve, roc_auc_score, classification_report
import numpy as np
import glob
import matplotlib.pyplot as plt

save = True
n_classes = 2
batch_size = 14
num_workers = 8
folds = [1, 2, 3, 4, 5]

def get_train_validation_files(fold_num):
    datasets = ['train', 'test']
    train_files = []
    validation_files = []

    for dataset in datasets:
        dir_0 = glob.glob('/your_directory/Macro/'+str(fold_num)+'fold/'+dataset+'/0/*/*')
        for file in dir_0:
            label = [1, 0]
            if dataset == "train":
                train_files.append((file, label))
            elif dataset == "test":
                validation_files.append((file, label))
        dir_1 = glob.glob('/your_directory/Macro/'+str(fold_num)+'fold/'+dataset+'/1/*/*')
        for file in dir_1:
            label = [0, 1]
            if dataset == "train":
                train_files.append((file, label))
            elif dataset == "test":
                validation_files.append((file, label))

    return train_files, validation_files

class IntracranialDataset(Dataset):
    def __init__(self, df, mode='test'):
        self.data = df
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx][0]
        video = torch.tensor(np.load(img_name))
        labels = torch.tensor(self.data[idx][1])
        return {'image': video, 'labels': labels}

class MYVIVIT(nn.Module):
    def __init__(self, n_classes):
        super(MYVIVIT, self).__init__()
        self.model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")
        self.fc = nn.Linear(400, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x.logits)
        return x

def multi_acc(y_pred, y_test):
    _, y_pred_tags = torch.max(y_pred, dim=1)
    correct_pred = (y_pred_tags == y_test).float()
    trueval = correct_pred.sum().tolist()
    return int(trueval)

def epochVal(model, dataLoader, criterion, fold_n):
    model.eval()
    valLoss = 0
    valAcc = 0
    valprediction = []
    valy = []

    for i, batch in enumerate(dataLoader):
        input = batch["image"].float().cuda()
        target = batch["labels"].long().cuda()
        output = model(input)
        valprediction += output.tolist()
        valy += target.tolist()
        lossvalue = criterion(F.softmax(output, dim=1), torch.max(target, 1)[1])
        acc = multi_acc(F.softmax(output, dim=1), torch.max(target, 1)[1])
        valLoss += lossvalue.item()
        valAcc += acc
        del input, target, output

    xpre = np.array(valprediction)
    yclass = np.array(valy)

    print(classification_report(np.argmax(yclass, axis=1), np.argmax(xpre, axis=1),
                                target_names=['normal', 'abnormal']))

    np.save(f'/your_directory/checkpoint/Macro/{fold_n}fold/{fold_n}fold_pred.npy', xpre)
    np.save(f'/your_directory/checkpoint/Macro/{fold_n}fold/{fold_n}fold_label.npy', yclass)

tpr = []
fpr = []
totalauc = []
colors = ['g', 'r', 'k', 'C0', 'C1', 'b']
plt.figure(1)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('False positive rate', fontsize=14)
plt.ylabel('True positive rate', fontsize=14)

mean_fpr = np.linspace(0, 1, 100)
tprlist = []

for fold_n in folds:
    savedir = f'/your_directory/checkpoint/Macro/{fold_n}fold/'

    net = MYVIVIT(n_classes)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net).cuda()
    checkpoint = torch.load(f'/your_directory/checkpoint/Macro/{fold_n}fold/{fold_n}fold.pth')
    net.load_state_dict(checkpoint['model_state_dict'])

    _, testdata = get_train_validation_files(fold_n)
    valdataset = IntracranialDataset(testdata, mode='test')
    valloader = DataLoader(valdataset, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=False)
    criterion = nn.CrossEntropyLoss()

    epochVal(net, valloader, criterion, fold_n)

    del net

    pred = np.load(f'/your_directory/checkpoint/Macro/{fold_n}fold/{fold_n}fold_pred.npy')
    target = np.load(f'/your_directory/checkpoint/Macro/{fold_n}fold/{fold_n}fold_label.npy')

    roc_auc = roc_auc_score(target[:, 1], pred[:, 1])
    f, t, _ = roc_curve(target[:, 1], pred[:, 1])

    fpr.append(f)
    tpr.append(t)
    totalauc.append(roc_auc)
    print('AUC:', roc_auc)

    plt.plot(f, t, color=colors[fold_n - 1], label=f'{fold_n}-fold (AUC = {roc_auc:.3f})', lw=2, alpha=.3)

    tprlist.append(np.interp(mean_fpr, f, t))

mean_tpr = np.mean(tprlist, axis=0)
mean_tpr[0] = 0
mean_tpr[-1] = 1.0
mean_auc = np.mean(totalauc)

plt.plot(mean_fpr, mean_tpr, color=colors[0], label=f'Mean ROC (AUC = {mean_auc:.3f})', lw=2)
plt.plot([0, 1], [0, 1], '--', color='navy', linewidth=0.5)
plt.legend(loc="lower right", fontsize=11)
plt.savefig(f'{savedir}{fold_n}fold_auc.png', bbox_inches='tight', pad_inches=0)
plt.close()