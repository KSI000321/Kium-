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

class submission_process:
  def __init__(self, ant_model, pos_model, test_ant_df, test_pos_df, ant_test_transform, pos_test_transform, device):
    self.device = device
    self.ant_model = ant_model.to(self.device)
    self.pos_model = pos_model.to(self.device)

    # ant dataset
    test_anterior_dataset = CustomDataset(test_ant_df, 'test', ant_test_transform)
    self.test_anterior_loader = DataLoader(test_anterior_dataset, batch_size= 64, shuffle=False, drop_last=False, num_workers=4)

    # pos dataset
    test_posterior_dataset = CustomDataset(test_pos_df, 'test', pos_test_transform)
    self.test_posterior_loader = DataLoader(test_posterior_dataset, batch_size= 64, shuffle=False, drop_last=False, num_workers=4)

    self.data_len = int(len(test_anterior_dataset) / 4) # 모델 input을 위한 DataFrame의 행 갯수는 사진의 갯수이지만, 제출 DataFrame의 행 갯수는 사람의 수 이므로 4로 나눔 (왜 8이 아닌 4? => ant, pos로 나눠서 학습했기 때문에)


  def infer(self): # multilabel 예측
    self.ant_model.eval()
    self.pos_model.eval()

    ant_prediction_list = []
    pos_prediction_list = []

    # ant predict
    with torch.no_grad():
      for x in tqdm(self.test_anterior_loader):
        x = x.to(self.device).float()
        ant_pred = self.ant_model(x)
        ant_pred = ant_pred.detach().cpu().tolist()
        ant_prediction_list += ant_pred

    # pos predict
    with torch.no_grad():
      for x in tqdm(self.test_posterior_loader):
        x = x.to(self.device).float()
        pos_pred = self.pos_model(x)
        pos_pred = pos_pred.detach().cpu().tolist()
        pos_prediction_list += pos_pred

    ant_prediction_list = np.array(ant_prediction_list)
    pos_prediction_list = np.array(pos_prediction_list)

    return ant_prediction_list, pos_prediction_list


  def get_output_df(self, threshold=0.5): # infer함수의 반환값들로 제출 형식에 맞게 csv파일 만들기
    output_csv = pd.DataFrame(index=range(self.data_len), columns=M_CFG['SUBMISSION_LIST'])
    '''
    A/B 이미지에 대한 예측값은 가장 큰 값으로 대체
    동일한 위치(L 혹은 R)에 조영제를 주입하였지만 각각의 방향(A/B)에서 바라보았을 때 강조되는 부분이 다를 것이기 때문에 예측값이 다를 수 있음
    그래서 A, B 두 이미지 가장 큰 확률을 가지는 값으로 판단하려고 합니다
    '''

    ant_prediction_list, pos_prediction_list = self.infer()

    i = 0
    ant_arrays = ant_prediction_list
    for csv_index in range(0, self.data_len):
      ant_array = ant_arrays[i:i+4, :]
      output_csv.loc[csv_index, M_CFG['SUBMISSION_L_ANT_LIST']] = np.max(ant_array[0:2, :], axis=0)
      output_csv.loc[csv_index, M_CFG['SUBMISSION_R_ANT_LIST']] = np.max(ant_array[2:4], axis=0)
      i+=4

    i = 0
    pos_arrays = pos_prediction_list
    for csv_index in range(0, self.data_len):
      pos_array = pos_arrays[i:i+4, :]
      output_csv.loc[csv_index, M_CFG['SUBMISSION_L_POS_LIST']] = np.max(pos_array[0:2, :], axis=0)
      output_csv.loc[csv_index, M_CFG['SUBMISSION_R_POS_LIST']] = np.max(pos_array[2:4], axis=0)
      i+=4

    output_csv["Index"] = range(self.data_len)
    output_csv["Aneurysm"] = 0
    output_csv['BA'] = np.max(output_csv.loc[:, ['(L)BA', '(R)BA']].values, axis=1) # 좌우 구분이 없는 BA는 따로 처리
    output_csv.drop(['(L)BA', '(R)BA'], axis=1, inplace=True)
    output_csv = output_csv.iloc[:, 2:].astype(np.float64)

    return output_csv

def get_binary_classification(ant_model, pos_model, ant_transform, pos_transform, ant_df, pos_df, device):
    ant_model.eval()
    pos_model.eval()

    ant_prediction_list = []
    pos_prediction_list = []

    ant_dataset = binary_CustomDataset(ant_df, phase='test', transforms=ant_transform)
    ant_loader = DataLoader(ant_dataset, batch_size=B_CFG['BATCH_SIZE'], shuffle=False, num_workers=4)

    pos_dataset = binary_CustomDataset(pos_df, phase='test', transforms=pos_transform)
    pos_loader = DataLoader(pos_dataset, batch_size=B_CFG['BATCH_SIZE'], shuffle=False, num_workers=4)


    # ant predict
    with torch.no_grad():
      for x in tqdm(ant_loader):
        x = x.to(device).float()
        ant_pred = ant_model(x)
        ant_pred = ant_pred.detach().cpu().tolist()
        ant_prediction_list += ant_pred

    with torch.no_grad():
      for x in tqdm(pos_loader):
        x = x.to(device).float()
        pos_pred = pos_model(x)
        pos_pred = pos_pred.detach().cpu().tolist()
        pos_prediction_list += pos_pred

    ant_list = []
    pos_list = []

    for i in range(0, len(ant_prediction_list), 4):
      ant_list.append(ant_prediction_list[i:i+4])
      pos_list.append(pos_prediction_list[i:i+4])

    ant_prediction_list = np.array(ant_list)
    ant_prediction_list = np.mean(ant_prediction_list, axis=1)
    pos_prediction_list = np.array(pos_list)
    pos_prediction_list = np.mean(pos_prediction_list, axis=1)

    pred_array = np.concatenate([ant_prediction_list, pos_prediction_list], axis=1)
    prediction = np.max(pred_array, axis=1)

    return prediction

class MedNet_resnet18(nn.Module):
  def __init__(self, in_dim=1, out_dim=1):
    super(MedNet_resnet18, self).__init__()
    self.backbone = models.resnet18()
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
        self.backbone = models.resnet18()
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

file_path = './키움화이팅_제출파일' #1 제출파일 주소

data = pd.read_csv(os.path.join(file_path, '/sourcecode/test', '/test.csv')) #2 - 추론에 사용되는 test.csv 주소
data['sum'] = data.iloc[:, 2:].sum(axis=1)

train_images = glob(os.path.join(file_path, '/sourcecode/test', '/*.jpg'))
train_images = sorted(train_images)

# multilabel 분류 dataset
anterior_path = glob(os.path.join(file_path, '/sourcecode/test', '/*I-*.jpg')) #3 - 추론에 사용되는 모든 test 이미지 주소들 (I)
posterior_path = glob(os.path.join(file_path, '/sourcecode/test', '/*V-*.jpg')) #3 - 추론에 사용되는 모든 test 이미지 주소들 (V)
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

import albumentations as A
from albumentations.pytorch import ToTensorV2

mul_transform = A.Compose([A.Resize(M_CFG['IMG_SIZE'], M_CFG['IMG_SIZE']),
                           A.Normalize(mean=(0.5), std=(0.5), max_pixel_value=255.0, always_apply=False, p=1.0),
                           ToTensorV2()
                           ])

binary_transform = A.Compose([A.Resize(B_CFG['IMG_SIZE'], B_CFG['IMG_SIZE']),
                              A.Normalize(mean=(0.5), std=(0.5), max_pixel_value=255.0, always_apply=False, p=1.0),
                              ToTensorV2()
                              ])

mul_ant_model = torch.load(os.path.join(file_path, '/model/SwinNet_ant_multilabel.pt'))#4 multi label classificaion anterior pretrained model 주소
mul_pos_model = torch.load(os.path.join(file_path, '/model/SwinNet_ant_multilabel.pt'))#5 multi label classificaion posterior pretrained model 주소

binary_ant_model = torch.load(os.path.join(file_path, '/model/MedNet_ant_binary.pt'))#6 binary classificaion anterior pretrained model 주소
binary_pos_model = torch.load(os.path.join(file_path, '/model/MedNet_pos_binary.pt'))#7 binary classificaion posterior pretrained model 주소

Aneurysm = get_binary_classification(binary_ant_model, binary_pos_model, binary_transform, binary_transform, binary_anterior, binary_posterior, B_CFG['DEVICE'])

infer_process = submission_process(mul_ant_model, mul_pos_model, anterior, posterior, mul_transform, mul_transform, device)
output_csv = infer_process.get_output_df(threshold=0.5)
output_csv['Index'] = data['Index']
output_csv['Aneurysm'] = Aneurysm
sub_cols = ['Index', 'Aneurysm', 'L_ICA', 'R_ICA', 'L_PCOM', 'R_PCOM', 'L_AntChor', 'R_AntChor', 'L_ACA', 'R_ACA', 'L_ACOM', 'R_ACOM', 'L_MCA', 'R_MCA', 'L_VA', 'R_VA', 'L_PICA', 'R_PICA', 'L_SCA', 'R_SCA', 'BA', 'L_PCA', 'R_PCA']
output_csv = output_csv[sub_cols]

# 임계점들은 train.csv에 대한 예측값을 기준으로 정함

# 임계점 못 넘는 Aneurysm의 해당하는 위치들은 모두 0으로 변환
zero_index = output_csv[output_csv['Aneurysm'] <= 0.65].index
output_csv.iloc[zero_index, 2:] = 0

# Aneurysm이 0이 아닌 위치들에도 각 열에 대한 임계점으로 0, 1 변환 - 90백분위수 기준
location_threshold = [0.6356, 0.6347, 0.6183, 0.6208, 0.627 , 0.6252, 0.548 , 0.5475,
                      0.5911, 0.5887, 0.6467, 0.6463, 0.4094, 0.4105, 0.5806, 0.5885,
                      0.9278, 0.9153, 0.8203, 0.8055, 0.3095]

output_csv.iloc[:, 2:] = np.where(output_csv.iloc[:, 2:].astype(np.float64).to_numpy() >= location_threshold, 1, 0)

output_csv.to_csv(os.path.join(file_path, "/sorcecode/output.csv"), index=False) #8 output.csv 저장 위치
