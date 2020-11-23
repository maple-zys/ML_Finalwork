from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch
import pandas as pd
import data_loader
from tqdm import tqdm

# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument("-tta", type=bool, default=False)
# args = parser.parse_args()

df = pd.read_csv('./data/test.csv')
df['label'] = 0
# data = [0]*5708
# df['label'] = data
# df.to_csv('./data/test.csv', index=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
normMean = [0.5964188, 0.4566936, 0.3908954]
normStd = [0.2590655, 0.2314241, 0.2269535]
valid_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normMean, normStd)
    ])
# tta_transformer = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.RandomChoice([
#         transforms.ColorJitter(brightness=[0.8, 1.2], contrast=[0.8, 1.2], saturation=[0.8, 1.2], hue=[-0.2, 0.2]),
#         transforms.Compose([transforms.Resize((150, 150)), transforms.Resize((200, 200))]),
#         transforms.Compose([transforms.RandomResizedCrop(150, scale=(0.78, 1), ratio=(0.90, 1.10), interpolation=2),
#                             transforms.Resize((200, 200))]),
#         transforms.RandomHorizontalFlip(p=1),
#         transforms.RandomRotation((-20, 20), resample=False, expand=False, center=None)
#     ]),
#     transforms.ToTensor()
# ])
test_dataset = data_loader.Test_Dataset('./data/test.csv', './data/test', valid_transformer)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
my_model = models.resnext50_32x4d()
# my_model = Res2Net50_v1b()
# my_model = varGFaceNet()
# print(torch.load('./logs/resnet50/best_checkpoint_ep17.pt'))
my_model.load_state_dict(torch.load('./logs/exp14/resnext50+da+normlize+pre/highest_valid_acc.pth')['state_dict}'])
my_model.to(device)
my_model.eval()
count = 0
with torch.no_grad():
    for step, data in tqdm(enumerate(test_loader)):
        index, img = data
        img = img.to(device)
        out = my_model(img)
        # if args.tta:
        #     img = tta_transformer(img)
        #     out1 = my_model(img)
        #     out = (out + out1) / 2
        _, pred = torch.max(out.data, 1)
        for i in range(pred.cpu().numpy().shape[0]):
            df['label'][count] = pred[i]
            count += 1
print(count)
df.to_csv('./data/resnext50+da+pre.csv', index=False)



