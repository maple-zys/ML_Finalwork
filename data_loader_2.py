from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import numpy as np
import cv2


class Train_Dataset(Dataset):
    def __init__(self, file_path, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform
        self.data = pd.read_csv(file_path).values
        self.x_data = self.data[:, 0]
        self.y_data = self.data[:, 1]
        # self.x_data = self.data[0:1440, 0]
        # self.y_data = self.data[0:1440, 1]

    def __getitem__(self, index):
        x, y = self.x_data[index], self.y_data[index]
        # img = Image.open(os.path.join(self.image_path, str(x) + '.jpg')).convert('RGB')
        img = cv2.imread(os.path.join(self.image_path, str(x) + '.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        # if self.transform is not None:
        #     res = self.transform(image=img)
        #     img = res['image'].astype(np.float32)
        # else:
        #     img = img.astype(np.float32)
        # img = img.transpose(2, 0, 1)
        return img, y

    def __len__(self):
        return len(self.x_data)


class Test_Dataset(Dataset):
    def __init__(self, file_path, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform
        self.data = pd.read_csv(file_path, encoding='utf8').values
        self.x_data = self.data[:, 0]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        x = self.x_data[index]
        img = Image.open(os.path.join(self.image_path, str(x) + '.jpg')).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return x, img

if __name__ == '__main__':
    train_transformer = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = Train_Dataset('./data/train.csv', './data/train/', transform=train_transformer)
    train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
    # for x, y in train_loader:
    #     print(x.shape, y)
