import os
import random
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from transformers import VivitImageProcessor
from huggingface_hub import hf_hub_download
from transformers import VivitForVideoClassification
import torch.nn as nn
import torch
import copy
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from sklearn.metrics import auc, roc_curve, roc_auc_score
import numpy as np
import sys, time
from torchvision.models import resnet
from sklearn.metrics import classification_report
import glob

save = True
n_classes = 2
batch_size = 16
n_epochs = 50
epoch_lr = [10, 20, 30, 40]
num_workers = 32
folds = [1, 2, 3, 4, 5]

def get_train_validation_files(fold_num):
    datasets = ['train', 'test']
    train_files = []
    validation_files = []

    for dataset in datasets:
        dir_0 = glob.glob('/your_directory/Macro/'+str(fold_num)+'fold/'+dataset+'/0/*/*')[:8]
        for file in dir_0:
            label = [1, 0]
            if dataset == "train":
                train_files.append((file, label))
            elif dataset == "test":
                validation_files.append((file, label))
        dir_1 = glob.glob('/your_directory/Macro/'+str(fold_num)+'fold/'+dataset+'/1/*/*')[:4]
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
        self.image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")

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
        self.relu = nn.ReLU()
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

def epochVal(model, dataLoader, criterion, ep_num):
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

    valprediction = np.array(valprediction)
    valy = np.array(valy)
    if np.isnan(valprediction).any() or np.isnan(valy).any():
        valprediction = np.nan_to_num(valprediction, nan=0.0)
        valy = np.nan_to_num(valy, nan=0.0)

    xpre = valprediction
    yclass = valy
    print(classification_report(np.argmax(yclass, axis=1), np.argmax(xpre, axis=1),
                                target_names=['normal', 'abnormal']))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    finaly = np.array(yclass)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(np.array(finaly[:, i]), (np.array(valprediction))[:, i])
        roc_auc[i] = roc_auc_score(np.array(finaly[:, i]), (np.array(valprediction))[:, i])

    fpr["micro"], tpr["micro"], _ = roc_curve(np.array(finaly).ravel(), np.array(valprediction).ravel())
    roc_auc["micro"] = roc_auc_score(np.array(finaly).ravel(), np.array(valprediction).ravel())

    valauc = roc_auc[1]
    roc_auc = valauc
    valLoss = valLoss / len(dataLoader)
    valAcc = valAcc / len(valy)
    return valAcc, valLoss, roc_auc, fpr, tpr

totallist = []
auclist = []
fold_acc = []

for fold_n in folds:
    savedir = '/your_directory/Macro/'+str(fold_n)+'fold/'
    os.makedirs(savedir, exist_ok=True)

    net = MYVIVIT(n_classes)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net).cuda()

    traindata, testdata = get_train_validation_files(fold_n)
    trndataset = IntracranialDataset(traindata, mode='train')
    valdataset = IntracranialDataset(testdata, mode='test')
    trnloader = DataLoader(trndataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    valloader = DataLoader(valdataset, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=False)
    criterion = nn.CrossEntropyLoss()
    lr = 0.01
    plist = [{'params': net.parameters(), 'lr': lr}]
    optimizer = optim.SGD(plist, lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=epoch_lr, gamma=0.1)

    for epoch in range(n_epochs):
        start_time = time.time()
        net.train().cuda()
        tr_loss = 0
        tr_acc = 0

        for step, batch in enumerate(trnloader):
            inputs = batch["image"].float().cuda()
            labels = batch["labels"].long().cuda()
            outputs = net(inputs)
            loss = criterion(F.softmax(outputs, dim=1), torch.max(labels, 1)[1])
            acc = multi_acc(F.softmax(outputs, dim=1), torch.max(labels, 1)[1])
            tr_loss += loss.item()
            tr_acc += acc
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            sys.stdout.flush()
            del inputs, labels, outputs

        scheduler.step()
        val_acc, val_loss, val_auc, fp, tp = epochVal(net, valloader, criterion, epoch)
        epoch_time = time.time() - start_time
        tr_loss = tr_loss / len(trnloader)
        tr_acc = tr_acc / len(traindata)
        print('Epoch: {} epoch, Trn loss: {}, trn acc: {}, val loss: {}, val acc: {}, val auc: {}'.format(epoch, round(tr_loss, 4),
                                                                                        round(tr_acc, 4),
                                                                                        round(val_loss, 4),
                                                                                        round(val_acc, 4),
                                                                                        round(val_auc, 4)))

    last_model_wts = copy.deepcopy(net.state_dict())
    if save:
        torch.save({'model_state_dict': last_model_wts, 'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(savedir, f"{fold_n}fold_last.pth"))

    del net