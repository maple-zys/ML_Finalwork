import pandas as pd
import torch
import cv2
import albumentations as A
from trans import get_transforms
import numpy as np
import torch.nn as nn
from PIL import Image
import os
from torchvision import transforms, models
from tqdm import tqdm


df = pd.read_csv('./data/test.csv')
df['label'] = 0
# print(df['id'][3453])
# print(df.shape[0])

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
normMean = [0.5964188, 0.4566936, 0.3908954]
normStd = [0.2590655, 0.2314241, 0.2269535]


for k in range(5):
    path = './logs/exp13/fold' + str(k) + '/'
    for _, _, files in os.walk('./logs/exp13/fold' + str(k)):
        for p in files:
            if 'best' in p:
                path += p
    # path = './logs/exp13/fold' + str(k) + '/last_checkpoint_ep74.pth'
    my_model = models.resnext50_32x4d()
    my_model.load_state_dict(torch.load(path)['state_dict}'])
    # my_model.load_state_dict(torch.load('./logs/exp5/best_checkpoint_ep15.pth')['state_dict}'])
    # my_model.load_state_dict(torch.load('./logs/exp5/last_checkpoint_ep46.pth')['state_dict}'])
    # my_model.load_state_dict(torch.load('./logs/k_4/highest_valid_acc.pth')['state_dict}'])
    my_model.to(device)
    my_model.eval()
    with torch.no_grad():
        for i in tqdm(range(df.shape[0])):
            img = cv2.imread(os.path.join('./data/test/', str(df['id'][i]) + '.jpg'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res1 = A.Compose([A.HorizontalFlip(p=1), A.Resize(200, 200),
                              A.Normalize(mean=(0.5964188, 0.4566936, 0.3908954),
                                          std=(0.2590655, 0.2314241, 0.2269535))])(image=img)
            img1 = res1['image'].astype(np.float32)
            img1 = torch.tensor(img1.transpose(2, 0, 1))
            res2 = A.Compose([A.OneOf(
                [A.MedianBlur(blur_limit=5), A.GaussianBlur(blur_limit=5), A.GaussNoise(var_limit=(5.0, 30.0)), ], p=1),
                              A.Resize(200, 200), A.Normalize(mean=(0.5964188, 0.4566936, 0.3908954),
                                                              std=(0.2590655, 0.2314241, 0.2269535))])(image=img)
            img2 = res2['image'].astype(np.float32)
            img2 = torch.tensor(img2.transpose(2, 0, 1))
            res3 = A.Compose([A.CoarseDropout(p=1, max_holes=1, max_height=75, max_width=75), A.Resize(200, 200),
                              A.Normalize(mean=(0.5964188, 0.4566936, 0.3908954),
                                          std=(0.2590655, 0.2314241, 0.2269535))])(image=img)
            img3 = res3['image'].astype(np.float32)
            img3 = torch.tensor(img3.transpose(2, 0, 1))
            res4 = A.Compose([A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=1),
                              A.Resize(200, 200),
                              A.Normalize(mean=(0.5964188, 0.4566936, 0.3908954),
                                          std=(0.2590655, 0.2314241, 0.2269535))])(image=img)
            img4 = res4['image'].astype(np.float32)
            img4 = torch.tensor(img4.transpose(2, 0, 1))
            res5 = A.Compose([A.CLAHE(clip_limit=4.0, p=1), A.Resize(200, 200),
                              A.Normalize(mean=(0.5964188, 0.4566936, 0.3908954),
                                          std=(0.2590655, 0.2314241, 0.2269535))])(image=img)
            img5 = res5['image'].astype(np.float32)
            img5 = torch.tensor(img5.transpose(2, 0, 1))
            res6 = A.Normalize(mean=(0.5964188, 0.4566936, 0.3908954), std=(0.2590655, 0.2314241, 0.2269535))(image=img)
            img6 = res6['image'].astype(np.float32)
            img6 = torch.tensor(img6.transpose(2, 0, 1))
            res7 = A.Compose([A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1),
                              A.Normalize(mean=(0.5964188, 0.4566936, 0.3908954),
                                          std=(0.2590655, 0.2314241, 0.2269535))])(image=img)
            img7 = res7['image'].astype(np.float32)
            img7 = torch.tensor(img7.transpose(2, 0, 1))
            # img1 = valid_transformer(img)
            # img2 = transforms.RandomHorizontalFlip(p=1)(img)
            # img2 = valid_transformer(img2)
            # img3 = transforms.Compose([transforms.Resize((150, 150)), transforms.Resize((200, 200))])(img)
            # img3 = valid_transformer(img3)
            # img4 = transforms.RandomRotation((-20, 20), resample=False, expand=False, center=None)(img)
            # img4 = valid_transformer(img4)
            # img5 = transforms.Compose([transforms.Resize((150, 150)), transforms.Resize((200, 200))])(img)
            # img5 = valid_transformer(img5)
            image = torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0)
            image = image.to(device)
            out = my_model(image)
            out = torch.mean(out, dim=0)
            out = out.view(1, -1)
            _, pred = torch.max(out.data.view(1, -1), 1)
            df['label'][i] = pred
    df.to_csv('./vote/output_fold_' + str(k) + '_tta5_best.csv', index=False)
    print('output_fold_' + str(k) + '_tta5_best.csv finished')
