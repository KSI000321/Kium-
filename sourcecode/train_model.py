import random
import pandas as pd
import numpy as np
import os
import re
import gc
from glob import glob
import cv2

import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from tqdm.auto import tqdm
import warnings
warnings.filterwarnings(action='ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

M_CFG = {
    'IMG_SIZE':224,
    'EPOCHS':100,
    'LEARNING_RATE':1e-3,
    'BATCH_SIZE':32,
    'SEED':42,
    'ANT_CLASS_NUM':5,
    'POS_CLASS_NUM':6,
    'ANT_LIST':['ICA', 'AntChor', 'ACA', 'ACOM', 'MCA'],
    'POS_LIST':['VA', 'PICA', 'SCA', 'BA', 'PCA', 'PCOM'],
    'SUBMISSION_LIST':['Index', 'Aneurysm', 'L_ICA', 'R_ICA', 'L_PCOM', 'R_PCOM', 'L_AntChor', 'R_AntChor', 'L_ACA', 'R_ACA', 'L_ACOM', 'R_ACOM', 'L_MCA', 'R_MCA', 'L_VA', 'R_VA', 'L_PICA', 'R_PICA', 'L_SCA', 'R_SCA', '(L)BA', '(R)BA', 'L_PCA', 'R_PCA'],
    'SUBMISSION_L_POS_LIST':['L_PCOM', 'L_VA', 'L_PICA', 'L_SCA', '(L)BA', 'L_PCA'],
    'SUBMISSION_R_POS_LIST':['R_PCOM', 'R_VA', 'R_PICA', 'R_SCA', '(R)BA', 'R_PCA'],
    'SUBMISSION_L_ANT_LIST':['L_ICA', 'L_AntChor', 'L_ACA', 'L_ACOM', 'L_MCA'],
    'SUBMISSION_R_ANT_LIST':['R_ICA', 'R_AntChor', 'R_ACA', 'R_ACOM', 'R_MCA']
}

B_CFG = {
    'DEVICE':torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    'IMG_SIZE':724,
    'EPOCHS':100,
    'LEARNING_RATE':1e-4,
    'BATCH_SIZE':32,
    'SEED':42,
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(M_CFG['SEED']) # Seed 고정

class CustomDataset(Dataset):
    def __init__(self, df, phase='train', transforms=None):
        self.phase = phase
        self.path = df['path']
        self.transforms = transforms

        if self.phase=='train':
            self.labels = df.iloc[:, 1:-2].values

    def __getitem__(self, index):
        img_path = self.path[index]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        if self.phase=='train':
            labels = self.labels[index]
            return image, labels
        else:
            return image

    def __len__(self):
        return len(self.path)

class AsymmetricLossOptimized(nn.Module):

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = x
        self.xs_neg = 1.0 - x

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()

def train(model, optimizer, train_loader, val_loader, scheduler, device, label_list=M_CFG['ANT_LIST']):
    model.to(device)
    criterion = AsymmetricLossOptimized()

    best_score = 0
    best_model = None
    early_stopping_patience = 0

    for epoch in range(1, M_CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.to(device).float()
            labels = labels.to(device).float()

            optimizer.zero_grad()

            output = model(imgs)
            output = output.reshape(M_CFG['BATCH_SIZE'], -1)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        _val_loss, _val_score = validation(model, criterion, val_loader, device, label_list)
        _train_loss = np.mean(train_loss)

        if scheduler is not None:
            scheduler.step(_val_score)

        if best_score <= _val_score:
            best_score = _val_score
            best_model = model
            early_stopping_patience = 0
        else:
            early_stopping_patience += 1

        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val f1 : [{_val_score:.5f}] early_stopping_patience : {early_stopping_patience}/5')

        if early_stopping_patience == 5:
            break

    return best_model

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

def validation(model, criterion, val_loader, device, label_list=M_CFG['ANT_LIST']):
    model.eval()
    val_loss = []
    preds_list, labels_list = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.to(device).float()
            labels = labels.to(device).float()

            preds = model(imgs)
            preds = preds.reshape(M_CFG['BATCH_SIZE'], -1)

            loss = criterion(preds, labels)

            preds = preds.detach().cpu().numpy()
            preds = np.where(preds>=0.5, 1, 0)
            labels = labels.detach().cpu().numpy()

            preds_list += preds.tolist()
            labels_list += labels.tolist()
            val_loss.append(loss.item())

        label_names = label_list

        _val_loss = np.mean(val_loss)
        _val_score = f1_score(labels_list, preds_list, average='weighted')
        print(classification_report(labels, preds, target_names=label_names))

    return _val_loss, _val_score

from torch.utils.data.sampler import WeightedRandomSampler

def get_weighted_sampler(df):
    class_counts = df['sum'].value_counts().to_list()
    num_samples = sum(class_counts)
    labels = df['sum'].to_list()

    class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]

    weights = [class_weights[labels[i]] for i in range(int(num_samples))]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))

    return sampler

class SwinNet(nn.Module):
    def __init__(self, out_features, inp_channels=1, pretrained=True):
        super(SwinNet, self).__init__()
        self.model = timm.create_model('swinv2_cr_tiny_ns_224', pretrained=True,
                                      in_chans=inp_channels)
        self.classifier = nn.Linear(1000, out_features)

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        x = F.sigmoid(x)
        return x

class ResNet18(nn.Module):
    def __init__(self, output_dim):
        super(ResNet18, self).__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone = models.resnet18(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, output_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.backbone(x)
        x = self.classifier(x)
        x = F.sigmoid(x)

        return x

class binary_CustomDataset(Dataset):
    def __init__(self, df, phase='train', transforms=None):
        self.phase = phase
        self.path = df['path']
        self.transforms = transforms

        if self.phase=='train':
            self.labels = df.loc[:, 'sum'].values

    def __getitem__(self, index):
        img_path = self.path[index]
        image = self.preprocessing(img_path)

        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        if self.phase=='train':
            labels = self.labels[index]
            return image, labels
        else:
            return image

    def __len__(self):
        return len(self.path)

    def preprocessing(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.bitwise_not(image)
        image = image.astype(np.float32)
        image = cv2.medianBlur(image, 3)
        return image

def binary_train(model, criterion, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = criterion

    best_score = -1
    best_model = None
    early_stopping_patience = 0

    for epoch in range(1, B_CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.to(device).float()
            labels = labels.to(device).float()

            optimizer.zero_grad()

            output = model(imgs)
            output = output.reshape(B_CFG['BATCH_SIZE'])
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        _val_loss, _val_score = binary_validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)

        if scheduler is not None:
            scheduler.step(_val_score)

        if best_score <= _val_score:
            best_score = _val_score
            best_model = model
            early_stopping_patience = 0
        else:
            early_stopping_patience += 1

        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val roc auc : [{_val_score:.5f}] early_stopping_patience : {early_stopping_patience}/5')

        if early_stopping_patience == 5:
            break

    return best_model

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

def binary_validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    preds_list, labels_list = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.to(device).float()
            labels = labels.to(device).float()

            preds = model(imgs)
            preds = preds.reshape(B_CFG['BATCH_SIZE'])

            loss = criterion(preds, labels)

            preds = preds.detach().cpu().numpy()
            preds = np.where(preds>=0.5, 1, 0)
            labels = labels.detach().cpu().numpy()

            preds_list += preds.tolist()
            labels_list += labels.tolist()
            val_loss.append(loss.item())

        _val_loss = np.mean(val_loss)
        _val_score = roc_auc_score(labels_list, preds_list, average='weighted')
        print(classification_report(labels_list, preds_list))

    return _val_loss, _val_score

from torch.utils.data.sampler import WeightedRandomSampler

def get_weighted_sampler(df):
    class_counts = df['sum'].value_counts().to_list()
    num_samples = sum(class_counts)
    labels = df['sum'].to_list()

    class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]

    weights = [class_weights[labels[i]] for i in range(int(num_samples))]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))

    return sampler

class MedNet_resnet18(nn.Module):
  def __init__(self, in_dim, out_dim, weights_path, device=B_CFG['DEVICE']):
    super(MedNet_resnet18, self).__init__()
    self.backbone = models.resnet18(weights=torch.load(weights_path, map_location=B_CFG['DEVICE']))
    self.backbone.conv1 = nn.Conv2d(in_dim, 64, kernel_size=(7, 7),
                                    stride=(2, 2), padding=(3, 3),
                                    bias=False)
    self.backbone.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.Dropout(0.3),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128,16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )

  def forward(self, x):
    x = self.backbone(x)
    x = F.sigmoid(x)
    return x


file_path = '/content/drive/MyDrive/2023_k_ium_composition' #1 kium 파일 주소

data = pd.read_csv(os.path.join(file_path, 'train_set', 'train_set _text_x/train.csv')) #2 - 학습에 사용되는 train.csv 주소 (train 이미지와 같은 곳에 위치)
data['sum'] = data.iloc[:, 2:].sum(axis=1)

train_images = glob('/content/drive/MyDrive/2023_k_ium_composition/train_set/train_set _text_x/*.jpg') #3 - 학습에 사용되는 모든 train 이미지 주소들
train_images = sorted(train_images)

# multilabel 분류 dataset
anterior_path = glob(os.path.join(file_path, 'train_set', 'train_set _text_x/*I-*.jpg'))
posterior_path = glob(os.path.join(file_path, 'train_set', 'train_set _text_x/*V-*.jpg'))
anterior_path = sorted(anterior_path)
posterior_path = sorted(posterior_path)

L_anterior = ['Index', 'L_ICA', 'L_AntChor', 'L_ACA', 'L_ACOM', 'L_MCA']
R_anterior = ['Index', 'R_ICA', 'R_AntChor', 'R_ACA', 'R_ACOM', 'R_MCA']
L_posterior = ['Index', 'L_VA', 'L_PICA', 'L_SCA', 'BA', 'L_PCA', 'L_PCOM']
R_posterior = ['Index', 'R_VA', 'R_PICA', 'R_SCA', 'BA', 'R_PCA', 'R_PCOM']

L_anterior_df = data[L_anterior]
R_anterior_df = data[R_anterior]
L_posterior_df = data[L_posterior]
R_posterior_df = data[R_posterior]

L_anterior_df.rename(columns={'L_ICA':'ICA', 'L_AntChor':'AntChor', 'L_ACA':'ACA', 'L_ACOM':'ACOM', 'L_MCA':'MCA'}, inplace=True)
R_anterior_df.rename(columns={'R_ICA':'ICA', 'R_AntChor':'AntChor', 'R_ACA':'ACA', 'R_ACOM':'ACOM', 'R_MCA':'MCA'}, inplace=True)

L_posterior_df.rename(columns={'L_VA':'VA', 'L_PICA':'PICA', 'L_SCA':'SCA', 'L_PCA':'PCA', 'L_PCOM':'PCOM'}, inplace=True)
R_posterior_df.rename(columns={'R_VA':'VA', 'R_PICA':'PICA', 'R_SCA':'SCA', 'R_PCA':'PCA', 'R_PCOM':'PCOM'}, inplace=True)

# multilabel 분류 dataset
anterior = pd.DataFrame(index=range(4*len(data)), columns=L_anterior_df.columns)
anterior.iloc[::4, :] = L_anterior_df.values
anterior.iloc[2::4, :] = R_anterior_df.values
anterior.fillna(method='ffill', inplace=True)
anterior['sum'] = anterior.iloc[:, 1:].sum(axis=1)
anterior['path'] = anterior_path

posterior = pd.DataFrame(index=range(4*len(data)), columns=L_posterior_df.columns)
posterior.iloc[::4, :] = L_posterior_df.values
posterior.iloc[2::4, :] = R_posterior_df.values
posterior.fillna(method='ffill', inplace=True)
posterior['sum'] = posterior.iloc[:, 1:].sum(axis=1)
posterior['path'] = posterior_path

# 이진분류 dataset

binary_anterior = anterior.copy()
binary_anterior.drop(['ICA', 'AntChor', 'ACA', 'ACOM', 'MCA'], axis=1, inplace=True)
binary_anterior['sum'] = anterior.iloc[:, 1:].sum(axis=1)
binary_anterior['sum'] = binary_anterior['sum'].map(lambda x: 1 if x > 0 else 0)

binary_posterior = posterior.copy()
binary_posterior.drop(['VA'	,'PICA'	,'SCA'	,'BA'	,'PCA'	,'PCOM'], axis=1, inplace=True)
binary_posterior['sum'] = posterior.iloc[:, 1:].sum(axis=1)
binary_posterior['sum'] = binary_posterior['sum'].map(lambda x: 1 if x > 0 else 0)

# multi label dataloader

mul_transform = A.Compose([A.Resize(M_CFG['IMG_SIZE'],M_CFG['IMG_SIZE']),
                       A.Normalize(mean=(0.5), std=(0.5), max_pixel_value=255.0, always_apply=False, p=1.0),
                       ToTensorV2()
                       ])

# multi label anterior(cariotid) dataloader
mul_train_anterior, mul_val_anterior = train_test_split(anterior, test_size=0.2, stratify=anterior['sum'], random_state=42)

mul_train_anterior.reset_index(drop=True, inplace=True)
mul_val_anterior.reset_index(drop=True, inplace=True)

mul_train_anterior_sampler = get_weighted_sampler(mul_train_anterior)
mul_val_anterior_sampler = get_weighted_sampler(mul_val_anterior)

mul_train_anterior_dataset = CustomDataset(mul_train_anterior, 'train', mul_transform)
mul_train_anterior_loader = DataLoader(mul_train_anterior_dataset, batch_size=M_CFG['BATCH_SIZE'], drop_last=True, sampler=mul_train_anterior_sampler, num_workers=4)

mul_val_anterior_dataset = CustomDataset(mul_val_anterior, 'train', mul_transform)
mul_val_anterior_loader = DataLoader(mul_val_anterior_dataset, batch_size=M_CFG['BATCH_SIZE'], drop_last=True, sampler=mul_val_anterior_sampler, num_workers=4)

# multi label posterior(vertebral) dataloader
mul_train_posterior, mul_val_posterior = train_test_split(posterior, test_size=0.2, stratify=anterior['sum'], random_state=42)

mul_train_posterior.reset_index(drop=True, inplace=True)
mul_val_posterior.reset_index(drop=True, inplace=True)

mul_train_posterior_sampler = get_weighted_sampler(mul_train_posterior)
mul_val_posterior_sampler = get_weighted_sampler(mul_val_posterior)

mul_train_posterior_dataset = CustomDataset(mul_train_posterior, 'train', mul_transform)
mul_train_posterior_loader = DataLoader(mul_train_posterior_dataset, batch_size=M_CFG['BATCH_SIZE'], drop_last=True, sampler=mul_train_posterior_sampler, num_workers=4)

mul_val_posterior_dataset = CustomDataset(mul_val_posterior, 'train', mul_transform)
mul_val_posterior_loader = DataLoader(mul_val_posterior_dataset, batch_size=M_CFG['BATCH_SIZE'], drop_last=True, sampler=mul_val_posterior_sampler, num_workers=4)

# multi label anterior(cariotid) dataloader
binary_transform = A.Compose([A.Resize(B_CFG['IMG_SIZE'], B_CFG['IMG_SIZE']),
                       A.Normalize(mean=(0.5), std=(0.5), max_pixel_value=255.0, always_apply=False, p=1.0),
                       ToTensorV2()
                       ])

binary_train_anterior, binary_val_anterior = train_test_split(binary_anterior, test_size=0.2, stratify=anterior['sum'], random_state=42)

binary_train_anterior.reset_index(drop=True, inplace=True)
binary_val_anterior.reset_index(drop=True, inplace=True)

binary_train_anterior_sampler = get_weighted_sampler(binary_train_anterior)

binary_train_anterior_dataset = binary_CustomDataset(binary_train_anterior, 'train', binary_transform)
binary_train_anterior_loader = DataLoader(binary_train_anterior_dataset, batch_size=B_CFG['BATCH_SIZE'], drop_last=True, sampler=binary_train_anterior_sampler, num_workers=4)

binary_val_anterior_dataset = binary_CustomDataset(binary_val_anterior, 'train', binary_transform)
binary_val_anterior_loader = DataLoader(binary_val_anterior_dataset, batch_size=B_CFG['BATCH_SIZE'], drop_last=True, num_workers=4)

# binary label posterior(vertebral) dataloader
binary_train_posterior, binary_val_posterior = train_test_split(binary_posterior, test_size=0.2, stratify=posterior['sum'], random_state=42)

binary_train_posterior.reset_index(drop=True, inplace=True)
binary_val_posterior.reset_index(drop=True, inplace=True)

binary_train_posterior_sampler = get_weighted_sampler(binary_train_posterior)

binary_train_posterior_dataset = binary_CustomDataset(binary_train_posterior, 'train', binary_transform)
binary_train_posterior_loader = DataLoader(binary_train_posterior_dataset, batch_size=B_CFG['BATCH_SIZE'], drop_last=True, sampler=binary_train_posterior_sampler, num_workers=4)

binary_val_posterior_dataset = binary_CustomDataset(binary_val_posterior, 'train', binary_transform)
binary_val_posterior_loader = DataLoader(binary_val_posterior_dataset, batch_size=B_CFG['BATCH_SIZE'], drop_last=True, num_workers=4)

# multi label anterior(cariotid) model
mul_ant_model = SwinNet(M_CFG['ANT_CLASS_NUM'])
mul_ant_model.eval()
optimizer = torch.optim.AdamW(params=mul_ant_model.parameters(), lr = M_CFG["LEARNING_RATE"], weight_decay=1e-5, eps=1e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, min_lr=1e-8, verbose=True)
multi_ant_infer_model = train(mul_ant_model, optimizer, mul_train_anterior_loader, mul_val_anterior_loader, scheduler, device, M_CFG['ANT_LIST'])

# multi label posterior(vertebral) model
mul_pos_model = ResNet18(M_CFG['POS_CLASS_NUM'])
mul_pos_model.eval()
optimizer = torch.optim.AdamW(params=mul_pos_model.parameters(), lr = M_CFG["LEARNING_RATE"], weight_decay=1e-5, eps=1e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, min_lr=1e-8, verbose=True)
multi_pos_infer_model = train(mul_pos_model, optimizer, mul_train_posterior_loader, mul_val_posterior_loader, scheduler, device, M_CFG['POS_LIST'])

# binary label anterior(cariotid) model
binary_ant_model = MedNet_resnet18(1, 1, '/content/drive/MyDrive/resnet_18.pth', B_CFG['DEVICE']) #4 MedNet 가중치 주소
binary_ant_model.eval()

criterion = nn.BCELoss(reduction='mean')

optimizer = torch.optim.AdamW(params=binary_ant_model.parameters(), lr = B_CFG["LEARNING_RATE"], weight_decay=1e-5, eps=1e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, min_lr=1e-8, verbose=True)
binary_ant_infer_model = binary_train(binary_ant_model, criterion, optimizer, binary_train_anterior_loader, binary_val_anterior_loader, scheduler, B_CFG['DEVICE'])

# binary label anterior(vertebral) model
binary_pos_model = MedNet_resnet18(1, 1, '/content/drive/MyDrive/resnet_18.pth') #4

criterion = nn.BCELoss(reduction='mean')

optimizer = torch.optim.AdamW(params=binary_pos_model.parameters(), lr = B_CFG["LEARNING_RATE"], weight_decay=1e-5, eps=1e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, min_lr=1e-8, verbose=True)
binary_pos_infer_model = binary_train(binary_pos_model, criterion, optimizer, binary_train_posterior_loader, binary_val_posterior_loader, scheduler, B_CFG['DEVICE'])

